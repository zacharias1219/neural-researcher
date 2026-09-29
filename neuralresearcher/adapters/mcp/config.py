from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class MCPSettings(BaseModel):
    data_dir: str = Field(default="research", alias="NEURALRESEARCHER_DATA_DIR")
    transport: str = Field(default="stdio", alias="NEURALRESEARCHER_MCP_TRANSPORT")
    host: str = Field(default="127.0.0.1", alias="NEURALRESEARCHER_MCP_HOST")
    port: int = Field(default=8000, alias="NEURALRESEARCHER_MCP_PORT")
    path: str = Field(default="/mcp", alias="NEURALRESEARCHER_MCP_PATH")
    auth_token: Optional[str] = Field(default=None, alias="NEURALRESEARCHER_MCP_AUTH_TOKEN")
    max_concurrent_runs: int = Field(default=2, alias="NEURALRESEARCHER_MAX_CONCURRENT_RUNS")
    log_level: str = Field(default="INFO", alias="NEURALRESEARCHER_MCP_LOG_LEVEL")
    max_result_bytes: int = Field(default=1024 * 1024 * 10, alias="NEURALRESEARCHER_MCP_MAX_RESULT_BYTES") # 10MB default
    model_config = ConfigDict(populate_by_name=True)
