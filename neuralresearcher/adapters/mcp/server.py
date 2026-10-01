import asyncio
import logging

from mcp.server.mcpserver import MCPServer

from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.adapters.mcp.prompts import register_prompts
from neuralresearcher.adapters.mcp.resources import register_resources
from neuralresearcher.adapters.mcp.security import BearerAuthMiddleware
from neuralresearcher.adapters.mcp.tools import register_tools
from neuralresearcher.application.run_manager import RunManager


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

def start_mcp_server(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
):
    from neuralresearcher.logging import configure_console

    overrides = {
        key: value
        for key, value in {
            "transport": transport,
            "host": host,
            "port": port,
            "path": path,
        }.items()
        if value is not None
    }
    from typing import Any, cast
    settings = MCPSettings(**cast(dict[str, Any], overrides))

    if settings.transport == "stdio":
        configure_console(stderr=True)

    # Configure logging
    log_level_name = settings.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    service = RunManager(
        data_dir=settings.data_dir,
        max_concurrent_runs=settings.max_concurrent_runs,
        max_result_bytes=settings.max_result_bytes
    )
    server = create_mcp_server(service, settings)

    import contextlib

    if settings.transport == "stdio":
        async def run_stdio():
            try:
                await server.run_stdio_async()
            finally:
                await service.shutdown()
        asyncio.run(run_stdio())
    elif settings.transport == "streamable-http":
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        # Check authentication constraint
        if settings.host != "127.0.0.1" and not settings.auth_token:
            raise RuntimeError("Authentication must be configured when binding to a non-loopback interface.")

        mcp_app = server.streamable_http_app()
        original_lifespan = mcp_app.router.lifespan_context

        @contextlib.asynccontextmanager
        async def app_lifespan(app_instance):
            async with original_lifespan(app_instance):
                yield
                await service.shutdown()

        mount_path = settings.path
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"

        app = Starlette(
            routes=[
                Mount(mount_path, app=mcp_app),
            ],
            lifespan=app_lifespan,
        )

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
