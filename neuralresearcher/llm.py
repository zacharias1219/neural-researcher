from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel
from groq import Groq
from groq.types.chat import ChatCompletionMessageToolCall

from neuralresearcher.config import Config, Settings
from neuralresearcher.errors import LLMError
import os

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]  # name and arguments

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMResponse(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "assistant"
    content: Optional[str]
    tool_calls: List[ToolCall] = []
    usage: Usage

def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)

import time

def call_llm(
    config: Config,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tool_choice: str = "auto"
) -> LLMResponse:
    client = _get_client()
    
    kwargs = {
        "model": config.model_name,
        "messages": messages,
    }
    
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
            if "Rate limit reached" in err_msg or "429" in err_msg:
                if attempt < config.max_retries:
                    time.sleep(2 ** attempt + 2)  # Exponential backoff + offset
                    continue
            raise LLMError(f"Error calling Groq API: {err_msg}")
            
    if response is None:
        raise LLMError("Exceeded max retries calling Groq API.")
        
    choice = response.choices[0]
    message = choice.message
    
    parsed_tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            parsed_tool_calls.append(ToolCall(
                id=tc.id,
                function={"name": tc.function.name, "arguments": tc.function.arguments}
            ))
            
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )
    
    return LLMResponse(
        content=message.content,
        tool_calls=parsed_tool_calls,
        usage=usage
    )
