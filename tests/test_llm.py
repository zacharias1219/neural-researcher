import pytest
import os
from unittest.mock import patch, MagicMock
from neuralresearcher.llm import call_llm, OpenAICompatibleAdapter, AnthropicAdapter
from neuralresearcher.config import Config, LLMProvider
from neuralresearcher.errors import LLMError, SchemaError
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Groq provider tests
# ---------------------------------------------------------------------------

def test_missing_groq_api_key():
    from neuralresearcher.llm import _CLIENTS
    _CLIENTS.clear()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="GROQ_API_KEY is not set."):
            adapter = OpenAICompatibleAdapter(LLMProvider.GROQ)
            adapter.get_client()


@patch("neuralresearcher.llm.time.sleep")
@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_groq_rate_limit_retry(mock_get_client, mock_sleep):
    import openai
    import httpx
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, request=request)
    rate_limit_err = openai.RateLimitError(
        message="Rate limit reached 429", response=response, body=None
    )

    mock_client.chat.completions.create.side_effect = [
        rate_limit_err,
        rate_limit_err,
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="success", tool_calls=[]))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
    ]

    config = Config(provider=LLMProvider.GROQ, max_retries=3)
    response = call_llm(config=config, messages=[])

    assert response.content == "success"
    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("neuralresearcher.llm.time.sleep")
@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_groq_max_retries_exceeded(mock_get_client, mock_sleep):
    import openai
    import httpx
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, request=request)
    rate_limit_err = openai.RateLimitError(
        message="Rate limit reached 429", response=response, body=None
    )

    mock_client.chat.completions.create.side_effect = rate_limit_err

    config = Config(provider=LLMProvider.GROQ, api_retries=2)
    with pytest.raises(LLMError, match="Error calling LLM API"):
        call_llm(config=config, messages=[])

    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------

def test_missing_openai_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="OPENAI_API_KEY is not set."):
            adapter = OpenAICompatibleAdapter(LLMProvider.OPENAI)
            adapter.get_client()


@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_openai_call(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai result", tool_calls=[]))],
        usage=MagicMock(prompt_tokens=5, completion_tokens=15, total_tokens=20),
    )

    config = Config(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    response = call_llm(config=config, messages=[{"role": "user", "content": "hello"}])

    assert response.content == "openai result"
    assert response.usage.total_tokens == 20


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------

def test_missing_anthropic_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set."):
            adapter = AnthropicAdapter()
            adapter.get_client()


@patch("neuralresearcher.llm.AnthropicAdapter.get_client")
def test_anthropic_call(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "anthropic result"

    mock_client.messages.create.return_value = MagicMock(
        content=[text_block],
        usage=MagicMock(input_tokens=5, output_tokens=15),
    )

    config = Config(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")
    response = call_llm(
        config=config,
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ],
    )

    assert response.content == "anthropic result"
    assert response.usage.total_tokens == 20

    # Verify system prompt was extracted and passed separately
    call_kwargs = mock_client.messages.create.call_args
    assert "system" in call_kwargs.kwargs
    # Messages passed to Anthropic should NOT include the system message
    for m in call_kwargs.kwargs["messages"]:
        assert m["role"] != "system"


@patch("neuralresearcher.llm.AnthropicAdapter.get_client")
def test_anthropic_tool_use(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_123"
    tool_block.name = "search_papers"
    tool_block.input = {"keywords": ["mamba"]}

    mock_client.messages.create.return_value = MagicMock(
        content=[tool_block],
        usage=MagicMock(input_tokens=10, output_tokens=30),
    )

    config = Config(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")
    response = call_llm(
        config=config,
        messages=[{"role": "user", "content": "search"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "Search for papers",
                "parameters": {"type": "object", "properties": {"keywords": {"type": "array"}}},
            },
        }],
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function["name"] == "search_papers"
    assert response.content is None  # No text blocks


@patch("neuralresearcher.llm.AnthropicAdapter.get_client")
def test_anthropic_structured_output_json_schema(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_schema_1"
    tool_block.name = "structured_output"
    tool_block.input = {"domain": "ML", "subfields": ["NLP"]}

    mock_client.messages.create.return_value = MagicMock(
        content=[tool_block],
        usage=MagicMock(input_tokens=15, output_tokens=25),
    )

    config = Config(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")
    response = call_llm(
        config=config,
        messages=[{"role": "user", "content": "analyze"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": {"type": "object", "properties": {"domain": {"type": "string"}}},
            },
        },
    )

    # Tool call should be converted to content text (not external tool calls)
    assert response.content == '{"domain": "ML", "subfields": ["NLP"]}'
    assert len(response.tool_calls) == 0
    assert response.usage.total_tokens == 40


# ---------------------------------------------------------------------------
# Structured generation tests
# ---------------------------------------------------------------------------

class DummyOutput(BaseModel):
    name: str
    age: int

@patch("neuralresearcher.llm.call_llm")
def test_generate_structured_success(mock_call_llm):
    from neuralresearcher.llm import generate_structured
    
    mock_call_llm.return_value = MagicMock(content='{"name": "Alice", "age": 30}')
    config = Config(provider=LLMProvider.OPENAI)
    
    result = generate_structured(
        messages=[],
        output_model=DummyOutput,
        config=config,
        agent_name="test"
    )
    
    assert result.name == "Alice"
    assert result.age == 30
    assert mock_call_llm.call_count == 1


@patch("neuralresearcher.llm.call_llm")
def test_generate_structured_malformed_then_success(mock_call_llm):
    from neuralresearcher.llm import generate_structured
    
    mock_call_llm.side_effect = [
        MagicMock(content='{"name": "Alice"}'), # Missing age
        MagicMock(content='{"name": "Alice", "age": 30}')
    ]
    
    config = Config(provider=LLMProvider.OPENAI, schema_repair_attempts=2)
    messages = [{"role": "user", "content": "Hello"}]
    
    result = generate_structured(
        messages=messages,
        output_model=DummyOutput,
        config=config,
        agent_name="test"
    )
    
    assert result.name == "Alice"
    assert result.age == 30
    assert mock_call_llm.call_count == 2


@patch("neuralresearcher.llm.call_llm")
def test_generate_structured_repair_exhaustion(mock_call_llm):
    from neuralresearcher.llm import generate_structured
    
    mock_call_llm.return_value = MagicMock(content='{"name": "Alice", "age": "thirty"}') # type error
    
    config = Config(provider=LLMProvider.OPENAI, schema_repair_attempts=2)
    
    with pytest.raises(SchemaError, match="Exhausted 2 schema repair attempts"):
        generate_structured(
            messages=[],
            output_model=DummyOutput,
            config=config,
            agent_name="test"
        )
        
    assert mock_call_llm.call_count == 3 # 1 initial + 2 retries


