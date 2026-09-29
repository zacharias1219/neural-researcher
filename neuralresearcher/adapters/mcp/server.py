import asyncio
import logging
from typing import Optional

from mcp.server.mcpserver import MCPServer

from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.application.run_manager import RunManager
from neuralresearcher.adapters.mcp.tools import register_tools
from neuralresearcher.adapters.mcp.resources import register_resources
from neuralresearcher.adapters.mcp.prompts import register_prompts
from neuralresearcher.adapters.mcp.security import BearerAuthMiddleware

def create_mcp_server(
    service: RunManager,
    settings: MCPSettings,
) -> MCPServer:
    server = MCPServer(
        name="neuralresearcher",
        version="0.1.0",
        instructions="MCP server for Neural Researcher agentic pipeline."
    )

    register_tools(server, service)
    register_resources(server, service)
    register_prompts(server)

    return server

def start_mcp_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp"):
    import os
    from neuralresearcher.logging import configure_console
    if transport == "stdio":
        configure_console(stderr=True)
        
    settings = MCPSettings(transport=transport, host=host, port=port, path=path)
    
    # Configure logging
    log_level_name = settings.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    service = RunManager(
        data_dir=settings.data_dir,
        max_concurrent_runs=settings.max_concurrent_runs
    )
    server = create_mcp_server(service, settings)

    if settings.transport == "stdio":
        asyncio.run(server.run_stdio_async())
    elif settings.transport == "streamable-http":
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        
        # Check authentication constraint
        if settings.host != "127.0.0.1" and not settings.auth_token:
            raise RuntimeError("Authentication must be configured when binding to a non-loopback interface.")
            
        app = server.streamable_http_app(endpoint=settings.path)
        
        if settings.auth_token:
            app.add_middleware(BearerAuthMiddleware, auth_token=settings.auth_token)
            
        uvicorn.run(app, host=settings.host, port=settings.port)
    else:
        raise ValueError(f"Unknown transport: {settings.transport}")

def main():
    import typer
    typer.run(start_mcp_server)

if __name__ == "__main__":
    main()
