from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    data_dir: str = Field(default="research", validation_alias="NEURALRESEARCHER_DATA_DIR")
    transport: str = Field(default="stdio", validation_alias="NEURALRESEARCHER_MCP_TRANSPORT")
    host: str = Field(default="127.0.0.1", validation_alias="NEURALRESEARCHER_MCP_HOST")
    port: int = Field(default=8000, validation_alias="NEURALRESEARCHER_MCP_PORT")
    path: str = Field(default="/mcp", validation_alias="NEURALRESEARCHER_MCP_PATH")
    auth_token: Optional[str] = Field(default=None, validation_alias="NEURALRESEARCHER_MCP_AUTH_TOKEN")
    max_concurrent_runs: int = Field(default=2, validation_alias="NEURALRESEARCHER_MAX_CONCURRENT_RUNS")
    log_level: str = Field(default="INFO", validation_alias="NEURALRESEARCHER_MCP_LOG_LEVEL")
    max_result_bytes: int = Field(default=1024 * 1024 * 10, validation_alias="NEURALRESEARCHER_MCP_MAX_RESULT_BYTES") # 10MB default
    model_config = SettingsConfigDict(populate_by_name=True)
