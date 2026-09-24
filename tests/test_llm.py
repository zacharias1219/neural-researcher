import pytest
import os
from unittest.mock import patch, MagicMock
from neuralresearcher.llm import call_llm, _get_groq_client, _get_openai_client, _get_anthropic_client
from neuralresearcher.config import Config, LLMProvider
from neuralresearcher.errors import LLMError


# ---------------------------------------------------------------------------
# Groq provider tests
# ---------------------------------------------------------------------------

def test_missing_groq_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="GROQ_API_KEY is not set."):
            _get_groq_client()


@patch("neuralresearcher.llm.time.sleep")
@patch("neuralresearcher.llm._get_groq_client")
def test_groq_rate_limit_retry(mock_get_client, mock_sleep):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.chat.completions.create.side_effect = [
        Exception("Rate limit reached 429"),
        Exception("Rate limit reached 429"),
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
@patch("neuralresearcher.llm._get_groq_client")
def test_groq_max_retries_exceeded(mock_get_client, mock_sleep):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.chat.completions.create.side_effect = Exception("Rate limit reached 429")

    config = Config(provider=LLMProvider.GROQ, max_retries=2)
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
            _get_openai_client()


@patch("neuralresearcher.llm._get_openai_client")
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
            _get_anthropic_client()


@patch("neuralresearcher.llm._get_anthropic_client")
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


@patch("neuralresearcher.llm._get_anthropic_client")
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


@patch("neuralresearcher.llm._get_anthropic_client")
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

