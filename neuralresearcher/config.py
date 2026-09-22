import os
from pydantic import BaseModel, Field

from neuralresearcher.errors import ConfigError

class Settings(BaseModel):
    groq_api_key: str = Field(..., description="Groq API key required for LLM calls.")

class Config(BaseModel):
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 3
    tool_timeout_seconds: int = 60
    strict_mode: bool = False

def load_config(model_override: str | None = None, strict_override: bool | None = None) -> Config:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ConfigError("GROQ_API_KEY environment variable is missing.")
    
    # Ideally Settings is instantiated here to validate but we can just hold it
    # Settings(groq_api_key=api_key)
    
    config = Config()
    if model_override:
        config.model_name = model_override
    if strict_override is not None:
        config.strict_mode = strict_override
        
    return config
