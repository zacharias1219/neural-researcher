import pytest
from unittest.mock import patch, MagicMock

from neuralresearcher.llm import OpenAICompatibleAdapter, AnthropicAdapter, call_llm
from neuralresearcher.config import Config, LLMProvider
from neuralresearcher.errors import LLMError


def test_capabilities_matrix():
    groq_adapter = OpenAICompatibleAdapter(LLMProvider.GROQ)
    assert not groq_adapter.capabilities.supports_json_schema
    assert groq_adapter.capabilities.supports_json_object

    openai_adapter = OpenAICompatibleAdapter(LLMProvider.OPENAI)
    assert openai_adapter.capabilities.supports_json_schema

    anthropic_adapter = AnthropicAdapter()
    assert not anthropic_adapter.capabilities.supports_json_schema


@patch("neuralresearcher.llm.AnthropicAdapter.get_client")
def test_anthropic_seed_omitted(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.messages.create.return_value = MagicMock(
        content=[], usage=MagicMock(input_tokens=5, output_tokens=15)
    )
    config = Config(provider=LLMProvider.ANTHROPIC, seed=42)
    call_llm(config=config, messages=[{"role": "user", "content": "hello"}])

    kwargs = mock_client.messages.create.call_args.kwargs
    assert "seed" not in kwargs


@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_supported_seed_passed(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="hello",
                    tool_calls=[]))],
        usage=MagicMock(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20))

    config = Config(provider=LLMProvider.OPENAI, seed=42)
    call_llm(config=config, messages=[{"role": "user", "content": "hello"}])

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs.get("seed") == 42


@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_native_json_schema_fallback(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="{}", tool_calls=[]))],
        usage=MagicMock()
    )

    config = Config(provider=LLMProvider.GROQ)
    call_llm(
        config=config,
        messages=[],
        response_format={"type": "json_schema", "json_schema": {"schema": {}}}
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs.get("response_format") == {"type": "json_object"}


@patch("neuralresearcher.llm.OpenAICompatibleAdapter.get_client")
def test_auth_errors_not_retried(mock_get_client):
    import openai
    import httpx
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock an openai AuthenticationError
    request = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(401, request=request)
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="401 Unauthorized", response=response, body=None)

    config = Config(provider=LLMProvider.OPENAI, api_retries=2)
    with pytest.raises(LLMError, match="Non-retryable API Error"):
        call_llm(config=config, messages=[])

    # Should only be called once, not retried
    assert mock_client.chat.completions.create.call_count == 1
