"""
Multi-provider LLM wrapper.

Supports: Groq, OpenAI, DeepSeek (OpenAI-compatible), and Anthropic.
Each provider's SDK differences are handled here so the rest of the
codebase can keep calling ``call_llm`` with a uniform interface.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel

from neuralresearcher.config import Config, LLMProvider
import os
import time
import json
import random
from typing import Protocol, TypeVar, Type
from pydantic import ValidationError

from neuralresearcher.errors import LLMError, SchemaError


# ---------------------------------------------------------------------------
# Shared response models (provider-agnostic)
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]  # {"name": ..., "arguments": ...}


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "assistant"
    content: Optional[str] = None
    tool_calls: List[ToolCall] = []
    usage: Usage


class ProviderCapabilities(BaseModel):
    supports_tool_calls: bool = True
    supports_json_schema: bool = True
    supports_json_object: bool = True
    supports_seed: bool = True
    supports_streaming: bool = True
    max_output_tokens: Optional[int] = None
    max_context_window: Optional[int] = None
    supports_timeout: bool = True

class ProviderAdapter(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities:
        ...

    def get_client(self) -> Any:
        ...

    def generate(self, config: Config, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], response_format: Optional[Dict[str, Any]], tool_choice: str) -> LLMResponse:
        ...

# ---------------------------------------------------------------------------
# Client factories & Adapters
# ---------------------------------------------------------------------------

_CLIENTS = {}

class OpenAICompatibleAdapter:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._capabilities = self._init_capabilities()
        
    def _init_capabilities(self) -> ProviderCapabilities:
        if self.provider == LLMProvider.GROQ:
            return ProviderCapabilities(supports_json_schema=False, max_output_tokens=8192)
        elif self.provider == LLMProvider.DEEPSEEK:
            return ProviderCapabilities(supports_json_schema=True, max_output_tokens=8192)
        return ProviderCapabilities(max_output_tokens=4096)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities
        
    def get_client(self) -> Any:
        if self.provider not in _CLIENTS:
            if self.provider == LLMProvider.GROQ:
                from groq import Groq
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise LLMError("GROQ_API_KEY is not set.")
                _CLIENTS[self.provider] = Groq(api_key=api_key)
            elif self.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise LLMError("OPENAI_API_KEY is not set.")
                _CLIENTS[self.provider] = OpenAI(api_key=api_key)
            elif self.provider == LLMProvider.DEEPSEEK:
                from openai import OpenAI
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise LLMError("DEEPSEEK_API_KEY is not set.")
                _CLIENTS[self.provider] = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return _CLIENTS[self.provider]

    def generate(self, config: Config, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], response_format: Optional[Dict[str, Any]], tool_choice: str) -> LLMResponse:
        client = self.get_client()
        kwargs: Dict[str, Any] = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "timeout": 60.0,
        }
        if config.seed is not None:
            kwargs["seed"] = config.seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            if response_format.get("type") == "json_schema" and not self.capabilities.supports_json_schema:
                # Fallback to json_object
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = response_format

        response = None
        for attempt in range(config.api_retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if config.provider == LLMProvider.GROQ:
                    import groq
                    if isinstance(e, (groq.AuthenticationError, groq.PermissionDeniedError, groq.BadRequestError)):
                        raise LLMError(f"Non-retryable API Error ({config.provider.value}): {str(e)}")
                    if isinstance(e, (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)) or getattr(e, 'status_code', 200) >= 500 or getattr(e, 'status_code', 200) == 429:
                        if attempt < config.api_retries:
                            time.sleep((2 ** attempt) + random.uniform(0, 1))
                            continue
                else:
                    import openai
                    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError)):
                        raise LLMError(f"Non-retryable API Error ({config.provider.value}): {str(e)}")
                    if isinstance(e, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)) or getattr(e, 'status_code', 200) >= 500 or getattr(e, 'status_code', 200) == 429:
                        if attempt < config.api_retries:
                            time.sleep((2 ** attempt) + random.uniform(0, 1))
                            continue
                raise LLMError(f"Error calling LLM API ({config.provider.value}): {str(e)}")

        if response is None:
            raise LLMError(f"Exceeded max retries calling {config.provider.value} API.")

        choice = response.choices[0]
        message = choice.message
        parsed_tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                parsed_tool_calls.append(ToolCall(
                    id=tc.id,
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                ))

        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return LLMResponse(content=message.content, tool_calls=parsed_tool_calls, usage=usage)


class AnthropicAdapter:
    def __init__(self):
        self.provider = LLMProvider.ANTHROPIC
        self._capabilities = ProviderCapabilities(supports_json_schema=False, max_output_tokens=4096)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities
        
    def get_client(self) -> Any:
        if self.provider not in _CLIENTS:
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set.")
            _CLIENTS[self.provider] = Anthropic(api_key=api_key)
        return _CLIENTS[self.provider]

    def generate(self, config: Config, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], response_format: Optional[Dict[str, Any]], tool_choice: str) -> LLMResponse:
        client = self.get_client()
        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                user_messages.append(m)

        kwargs: Dict[str, Any] = {
            "model": config.model_name,
            "messages": user_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "timeout": 60.0,
        }

        anthropic_tools = []
        if tools:
            for t in tools:
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        structured_tool_name = None
        if response_format:
            if response_format.get("type") == "json_schema":
                js = response_format.get("json_schema", {})
                structured_tool_name = js.get("name", "structured_output")
                schema = js.get("schema", {"type": "object"})
                anthropic_tools.append({
                    "name": structured_tool_name,
                    "description": "Provide the final structured JSON response matching the schema.",
                    "input_schema": schema,
                })
                kwargs["tool_choice"] = {"type": "tool", "name": structured_tool_name}
            elif response_format.get("type") == "json_object":
                system_text += "\nIMPORTANT: Reply with valid JSON only. No markdown fences."

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if system_text.strip():
            kwargs["system"] = system_text.strip()

        response = None
        for attempt in range(config.api_retries + 1):
            try:
                response = client.messages.create(**kwargs)
                break
            except Exception as e:
                import anthropic
                if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError)):
                    raise LLMError(f"Non-retryable API Error (Anthropic): {str(e)}")
                if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)) or getattr(e, 'status_code', 200) >= 500 or getattr(e, 'status_code', 200) == 429:
                    if attempt < config.api_retries:
                        time.sleep((2 ** attempt) + random.uniform(0, 1))
                        continue
                raise LLMError(f"Error calling Anthropic API: {str(e)}")

        if response is None:
            raise LLMError("Exceeded max retries calling Anthropic API.")

        content_text = ""
        parsed_tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if structured_tool_name and block.name == structured_tool_name:
                    content_text = json.dumps(block.input)
                else:
                    parsed_tool_calls.append(ToolCall(
                        id=block.id,
                        function={"name": block.name, "arguments": json.dumps(block.input)},
                    ))

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(content=content_text if content_text else None, tool_calls=parsed_tool_calls, usage=usage)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call_llm(
    config: Config,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tool_choice: str = "auto",
    store: Optional[Any] = None,
    task_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> LLMResponse:
    """
    Unified LLM call that dispatches to the correct provider.
    """

    # --- Dispatch ---
    adapter: ProviderAdapter
    if config.provider in (LLMProvider.GROQ, LLMProvider.OPENAI, LLMProvider.DEEPSEEK):
        adapter = OpenAICompatibleAdapter(config.provider)
    elif config.provider == LLMProvider.ANTHROPIC:
        adapter = AnthropicAdapter()
    else:
        raise LLMError(f"Unsupported provider: {config.provider}")

    result = adapter.generate(config, messages, tools, response_format, tool_choice)

    # --- Transcript logging ---
    if store and task_id and agent_name:
        usage_dict = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if result.content:
            assistant_msg["content"] = result.content
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": tc.function}
                for tc in result.tool_calls
            ]
        store.save_transcript(task_id, agent_name, messages, usage_dict, assistant_msg)

    return result

T = TypeVar("T", bound=BaseModel)

def generate_structured(
    messages: List[Dict[str, Any]],
    output_model: Type[T],
    config: Config,
    agent_name: str,
    store: Optional[Any] = None,
    task_id: Optional[str] = None
) -> T:
    schema = output_model.model_json_schema()
    name = schema.get("title", "structured_output").lower()
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True
        }
    }
    
    system_prompt_addition = f"\n\nIMPORTANT: You must return a JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
    
    # Inject schema into system prompt to assist fallback modes
    modified_messages = [dict(msg) for msg in messages]
    for msg in modified_messages:
        if msg["role"] == "system":
            msg["content"] = msg["content"] + system_prompt_addition
            break
    else:
        modified_messages.insert(0, {"role": "system", "content": system_prompt_addition})
        
    last_error = None
    for attempt in range(config.schema_repair_attempts + 1):
        try:
            response = call_llm(
                config=config,
                messages=modified_messages,
                response_format=response_format,
                store=store,
                task_id=task_id,
                agent_name=agent_name
            )
            content = response.content
            if not content:
                raise SchemaError("Empty response content.")
            return output_model.model_validate_json(content)
        except ValidationError as e:
            last_error = e
            # Telemetry/transcripts can log repair count automatically because call_llm is logged
            modified_messages.append({"role": "assistant", "content": content if 'content' in locals() else ""})
            modified_messages.append({
                "role": "user",
                "content": f"Schema validation failed. Please fix the JSON. Errors:\n{str(e)}"
            })
        except LLMError:
            raise
        except Exception as e:
            last_error = e
            raise SchemaError(f"Unexpected error during structured generation: {str(e)}")
            
    raise SchemaError(f"Exhausted {config.schema_repair_attempts} schema repair attempts. Last error: {str(last_error)}")

