"""
Multi-provider LLM wrapper.

Supports: Groq, OpenAI, DeepSeek (OpenAI-compatible), and Anthropic.
Each provider's SDK differences are handled here so the rest of the
codebase can keep calling ``call_llm`` with a uniform interface.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel

from neuralresearcher.config import Config, LLMProvider
from neuralresearcher.errors import LLMError
import os, time, json


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


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _get_groq_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def _get_openai_client(base_url: Optional[str] = None):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set.")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _get_deepseek_client():
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMError("DEEPSEEK_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _get_anthropic_client():
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Provider-specific call helpers
# ---------------------------------------------------------------------------

def _call_openai_compatible(client, config: Config, messages, tools, response_format, tool_choice):
    """Works for Groq, OpenAI, and DeepSeek (all OpenAI-compatible SDKs)."""

    kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format

    response = None
    for attempt in range(config.max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            err_msg = str(e)
            if "Rate limit" in err_msg or "429" in err_msg:
                if attempt < config.max_retries:
                    time.sleep(2 ** attempt + 2)
                    continue
            raise LLMError(f"Error calling LLM API ({config.provider.value}): {err_msg}")

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


def _call_anthropic(client, config: Config, messages, tools, response_format):
    """Adapts our uniform interface to Anthropic's Messages API."""

    # Anthropic uses a top-level ``system`` param — strip it from messages.
    system_text = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            user_messages.append(m)

    # If response_format asks for JSON, append an instruction.
    if response_format and response_format.get("type") == "json_object":
        system_text += "\nIMPORTANT: Reply with valid JSON only. No markdown fences."

    kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "messages": user_messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if system_text.strip():
        kwargs["system"] = system_text.strip()

    # Convert tools to Anthropic schema
    if tools:
        anthropic_tools = []
        for t in tools:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        kwargs["tools"] = anthropic_tools

    response = None
    for attempt in range(config.max_retries + 1):
        try:
            response = client.messages.create(**kwargs)
            break
        except Exception as e:
            err_msg = str(e)
            if "rate" in err_msg.lower() or "429" in err_msg:
                if attempt < config.max_retries:
                    time.sleep(2 ** attempt + 2)
                    continue
            raise LLMError(f"Error calling Anthropic API: {err_msg}")

    if response is None:
        raise LLMError("Exceeded max retries calling Anthropic API.")

    # Parse Anthropic response blocks
    content_text = ""
    parsed_tool_calls: List[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            parsed_tool_calls.append(ToolCall(
                id=block.id,
                function={"name": block.name, "arguments": json.dumps(block.input)},
            ))

    usage = Usage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    return LLMResponse(
        content=content_text if content_text else None,
        tool_calls=parsed_tool_calls,
        usage=usage,
    )


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
    if config.provider == LLMProvider.GROQ:
        client = _get_groq_client()
        result = _call_openai_compatible(client, config, messages, tools, response_format, tool_choice)

    elif config.provider == LLMProvider.OPENAI:
        client = _get_openai_client()
        result = _call_openai_compatible(client, config, messages, tools, response_format, tool_choice)

    elif config.provider == LLMProvider.DEEPSEEK:
        client = _get_deepseek_client()
        result = _call_openai_compatible(client, config, messages, tools, response_format, tool_choice)

    elif config.provider == LLMProvider.ANTHROPIC:
        client = _get_anthropic_client()
        result = _call_anthropic(client, config, messages, tools, response_format)

    else:
        raise LLMError(f"Unsupported provider: {config.provider}")

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
