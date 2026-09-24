import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from neuralresearcher.errors import ConfigError


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"


# Default models per provider
DEFAULT_MODELS = {
    LLMProvider.GROQ: "openai/gpt-oss-120b",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.DEEPSEEK: "deepseek-chat",
}

# Environment variable names per provider
API_KEY_ENV_VARS = {
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
}

# Human-friendly display names
PROVIDER_DISPLAY_NAMES = {
    LLMProvider.GROQ: "Groq",
    LLMProvider.OPENAI: "OpenAI",
    LLMProvider.ANTHROPIC: "Anthropic",
    LLMProvider.DEEPSEEK: "DeepSeek",
}


class Config(BaseModel):
    provider: LLMProvider = LLMProvider.GROQ
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 3
    tool_timeout_seconds: int = 60
    strict_mode: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: int | None = None
    
    # Guardrail knobs
    min_papers: int = 3
    relevance_threshold: float = 0.2
    coverage_warning_threshold: float = 0.6


def load_config(
    provider_override: Optional[LLMProvider] = None,
    model_override: str | None = None,
    strict_override: bool | None = None,
) -> Config:
    provider = provider_override or LLMProvider.GROQ

    # Validate API key is present
    env_var = API_KEY_ENV_VARS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ConfigError(
            f"{env_var} environment variable is missing. "
            f"Set it or paste the key when prompted."
        )

    model = model_override or DEFAULT_MODELS[provider]

    config = Config(provider=provider, model_name=model)
    if strict_override is not None:
        config.strict_mode = strict_override

    return config
