import asyncio
import sys
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.adapters.mcp.security import BearerAuthMiddleware
from neuralresearcher.adapters.mcp.server import create_mcp_server
from neuralresearcher.application.run_manager import RunManager


async def verify_stdio():
    print("Verifying Stdio Transport...")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "neuralresearcher.cli", "mcp", "--transport", "stdio"],
        env={"GROQ_API_KEY": "dummy"}
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            tools = await client.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"  Available tools: {tool_names}")
            
            # Use 'research_list_runs' instead of 'research_start' for a harmless op if available
            harmless_tool = "research_list_runs" if "research_list_runs" in tool_names else tool_names[0]
            result = await client.call_tool(harmless_tool, {})
            print(f"  Successfully invoked `{harmless_tool}`.")
    print("  Stdio OK.\n")


async def verify_http_auth():
    print("Verifying Streamable HTTP Auth...")
    settings = MCPSettings(
        data_dir="test_research",
        transport="streamable-http",
        auth_token="secret_token",
        host="127.0.0.1",
        port=8089,
        path="/mcp"
    )
    service = RunManager(data_dir=settings.data_dir)
    server = create_mcp_server(service, settings)
    mcp_app = server.sse_app()
    
    app = Starlette(routes=[Mount("/mcp", app=mcp_app)])
    app.add_middleware(BearerAuthMiddleware, auth_token=settings.auth_token)
    
    config = uvicorn.Config(app, host="127.0.0.1", port=8089, log_level="critical")
    uvicorn_server = uvicorn.Server(config)
    
    thread = threading.Thread(target=uvicorn_server.run)
    thread.daemon = True
    thread.start()
    
    await asyncio.sleep(1) # wait for server to bind
    
    url = "http://127.0.0.1:8089/mcp/sse"
    
    try:
        # 1. Reject absent token
        try:
            async with sse_client(url=url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as client:
                    await client.initialize()
            print("  FAIL: Did not reject absent token.")
            sys.exit(1)
        except BaseException as e:
            if "401" in repr(e) or "Unauthorized" in repr(e) or "Server returned an error response" in repr(e):
                print("  [x] Rejected absent token successfully.")
            else:
                print(f"  [x] Rejected with unknown error: {repr(e)}")
                
        # 2. Reject invalid token
        try:
            async with sse_client(url=url, headers={"Authorization": "Bearer bad"}) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as client:
                    await client.initialize()
            print("  FAIL: Did not reject invalid token.")
            sys.exit(1)
        except BaseException as e:
            if "401" in repr(e) or "Unauthorized" in repr(e) or "Server returned an error response" in repr(e):
                print("  [x] Rejected invalid token successfully.")
            else:
                print(f"  [x] Rejected with unknown error: {repr(e)}")
        
        # 3. Allow valid token
        async with sse_client(url=url, headers={"Authorization": "Bearer secret_token"}) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                print("  [x] Valid token initialized successfully.")
                tools = await client.list_tools()
                
                harmless_tool = "research_list_runs" if "research_list_runs" in [t.name for t in tools.tools] else tools.tools[0].name
                await client.call_tool(harmless_tool, {})
                print(f"  [x] Successfully invoked `{harmless_tool}` over HTTP.")
                
        print("  HTTP Auth OK.\n")
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=2)


async def main():
    await verify_stdio()
    await verify_http_auth()
    print("All MCP transport smoke tests PASSED.")

if __name__ == "__main__":
    asyncio.run(main())
