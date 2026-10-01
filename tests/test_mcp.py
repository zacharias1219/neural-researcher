import json
import subprocess
import sys

import pytest
from mcp import Client

from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.adapters.mcp.server import create_mcp_server
from neuralresearcher.application.run_manager import RunManager


@pytest.fixture
def mcp_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    settings = MCPSettings(
        data_dir=str(
            tmp_path / "research"),
        max_concurrent_runs=1)

    # We use a mocked RunManager to avoid external calls
    class StubRunManager(RunManager):
        async def start_run(self, req):
            from neuralresearcher.application.models import RunHandle
            return RunHandle(
                run_id="test_id",
                topic=req.topic,
                status="INIT",
                current_stage=None,
                created_at="now",
                status_resource_uri="research://runs/test_id/status",
                result_resource_uri="research://runs/test_id/result",
                plan_resource_uri="research://runs/test_id/plan")

        async def cancel_run(self, run_id):
            from neuralresearcher.application.models import RunStatus
            return RunStatus(
                run_id=run_id,
                topic="test topic",
                provider="groq",
                model="",
                status="ACTIVE",
                current_stage="INIT",
                success=False,
                terminal=False,
                artifact_count=0,
                cancellation_requested=True)

    service = StubRunManager(
        data_dir=settings.data_dir,
        max_concurrent_runs=settings.max_concurrent_runs)
    return create_mcp_server(service, settings)


@pytest.mark.asyncio
async def test_mcp_client_in_memory(mcp_server):
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools.tools]
        assert "research_start" in tool_names

        result = await client.call_tool("research_start", {"topic": "test topic", "provider": "groq"})

        # Verify JSON returned
        if hasattr(result, "content") and len(result.content) > 0:
            text_result = result.content[0].text
        else:
            text_result = result[0].text if hasattr(
                result, "__getitem__") else str(result)

        assert "test_id" in text_result, f"Result was: {text_result}"
        data = json.loads(text_result)
        assert data["run_id"] == "test_id"


@pytest.mark.asyncio
async def test_mcp_client_resources_and_prompts(mcp_server):
    async with Client(mcp_server) as client:
        prompts = await client.list_prompts()
        assert "create_research_plan" in [p.name for p in prompts.prompts]

        await client.list_resources()
        # Templates are fetched differently, list_resources might be empty for templates,
        # but we know it connects and negotiates protocol.


def test_stdio_subprocess_cleanliness(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    # Instead of launching a full client which is async, let's just check the
    # output doesn't contain stray print logs
    cmd = [sys.executable, "-m", "neuralresearcher.cli", "mcp"]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    # Send a JSON-RPC initialization request
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }) + "\n"

    p.stdin.write(init_req)
    p.stdin.flush()

    out_line = p.stdout.readline()

    # Assert stdout is pure JSON-RPC
    assert out_line.strip().startswith("{"), "Stdout contained non-JSON data: " + out_line
    response = json.loads(out_line)
    assert response.get("jsonrpc") == "2.0"

    # Send initialized notification
    init_notif = json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }) + "\n"
    p.stdin.write(init_notif)
    p.stdin.flush()

    # Tool discovery
    list_tools_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }) + "\n"
    p.stdin.write(list_tools_req)
    p.stdin.flush()

    tools_line = p.stdout.readline()
    assert tools_line.strip().startswith("{")
    tools_resp = json.loads(tools_line)
    assert "result" in tools_resp
    assert "tools" in tools_resp["result"]

    # Status invocation (we just query a random ID, should fail gracefully or say not found)
    call_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "research_status",
            "arguments": {
                "run_id": "dummy_run_id"
            }
        }
    }) + "\n"
    p.stdin.write(call_req)
    p.stdin.flush()

    call_line = p.stdout.readline()
    assert call_line.strip().startswith("{")
    call_resp = json.loads(call_line)
    assert "result" in call_resp
    assert "content" in call_resp["result"]

    p.kill()


@pytest.mark.asyncio
async def test_streamable_http_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    settings = MCPSettings(
        data_dir=str(
            tmp_path / "research"),
        max_concurrent_runs=1,
        transport="streamable-http",
        auth_token="secret",
        host="127.0.0.1",
        port=8001,
        path="/mcp")
    service = RunManager(data_dir=settings.data_dir)
    server = create_mcp_server(service, settings)
    mcp_app = server.streamable_http_app()
    from starlette.applications import Starlette
    from starlette.routing import Mount

    app = Starlette(
        routes=[
            Mount("/mcp", app=mcp_app),
        ]
    )

    from neuralresearcher.adapters.mcp.security import BearerAuthMiddleware
    app.add_middleware(BearerAuthMiddleware, auth_token=settings.auth_token)

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        # Missing token -> 401
        res = client.get("/mcp/sse")
        assert res.status_code == 401

        # Invalid token -> 401
        res = client.get("/mcp/sse", headers={"Authorization": "Bearer bad"})
        assert res.status_code == 401

        # Valid token -> reaches MCP
        res = client.get(
            "/mcp/sse",
            headers={
                "Authorization": "Bearer secret"})
        assert res.status_code != 401

        # Test root path doesn't expose MCP
        res_root = client.get("/", headers={"Authorization": "Bearer secret"})
        assert res_root.status_code == 404
