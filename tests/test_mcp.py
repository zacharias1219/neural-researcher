import pytest
import asyncio
from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.adapters.mcp.server import create_mcp_server
from neuralresearcher.application.run_manager import RunManager

@pytest.fixture
def mcp_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    settings = MCPSettings(data_dir=str(tmp_path / "research"), max_concurrent_runs=1)
    service = RunManager(data_dir=settings.data_dir, max_concurrent_runs=settings.max_concurrent_runs)
    return create_mcp_server(service, settings)

@pytest.mark.asyncio
async def test_mcp_server_tools_registered(mcp_server):
    tools_call = mcp_server.list_tools()
    if asyncio.iscoroutine(tools_call):
        tools = await tools_call
    else:
        tools = tools_call
    tool_names = [t.name for t in tools]
    
    expected_tools = [
        "research_start",
        "research_resume",
        "research_status",
        "research_cancel",
        "research_result",
        "research_list_runs",
        "research_list_artifacts"
    ]
    for ext in expected_tools:
        assert ext in tool_names

@pytest.mark.asyncio
async def test_mcp_server_resources_registered(mcp_server):
    resources_call = mcp_server.list_resource_templates()
    if asyncio.iscoroutine(resources_call):
        resources = await resources_call
    else:
        resources = resources_call
    resource_uris = [r.uri_template for r in resources]
    
    expected_uris = [
        "research://runs/{run_id}/manifest",
        "research://runs/{run_id}/status",
        "research://runs/{run_id}/result",
        "research://runs/{run_id}/plan",
        "research://runs/{run_id}/papers",
        "research://runs/{run_id}/claims",
        "research://runs/{run_id}/gaps",
        "research://runs/{run_id}/directions",
        "research://runs/{run_id}/review",
        "research://runs/{run_id}/coverage",
        "research://runs/{run_id}/artifacts/{artifact_name}"
    ]
    for ext in expected_uris:
        assert ext in resource_uris

@pytest.mark.asyncio
async def test_mcp_server_prompts_registered(mcp_server):
    prompts_call = mcp_server.list_prompts()
    if asyncio.iscoroutine(prompts_call):
        prompts = await prompts_call
    else:
        prompts = prompts_call
    prompt_names = [p.name for p in prompts]
    
    expected_prompts = [
        "create_research_plan",
        "review_research_plan",
        "explore_research_gap"
    ]
    for ext in expected_prompts:
        assert ext in prompt_names

@pytest.mark.asyncio
async def test_research_start_and_status(mcp_server):
    # Call the tool directly
    import json
    
    res = await mcp_server.call_tool("research_start", {"topic": "test topic", "provider": "groq"})
    
    if hasattr(res, "content") and hasattr(res.content[0], "text"):
        res = res.content[0].text
    elif not isinstance(res, str):
        res = res[0].text if hasattr(res, "__getitem__") and hasattr(res[0], "text") else str(res)
    
    try:
        data = json.loads(res)
        run_id = data["run_id"]
        assert run_id is not None
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON returned: {res}")

    # Check status
    res_status = await mcp_server.call_tool("research_status", {"run_id": run_id})
    if hasattr(res_status, "content") and hasattr(res_status.content[0], "text"):
        res_status = res_status.content[0].text
    elif not isinstance(res_status, str):
        res_status = res_status[0].text if hasattr(res_status, "__getitem__") and hasattr(res_status[0], "text") else str(res_status)
    status_data = json.loads(res_status)
    assert status_data["status"] in ["INIT", "ACTIVE", "STARTING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "HALTED"]

@pytest.mark.asyncio
async def test_mcp_server_cancel_tool(mcp_server):
    import json
    res = await mcp_server.call_tool("research_start", {"topic": "cancel test", "provider": "groq"})
    if hasattr(res, "content") and hasattr(res.content[0], "text"):
        res = res.content[0].text
    elif not isinstance(res, str):
        res = res[0].text if hasattr(res, "__getitem__") and hasattr(res[0], "text") else str(res)
    data = json.loads(res)
    run_id = data["run_id"]

    res_cancel = await mcp_server.call_tool("research_cancel", {"run_id": run_id})
    if hasattr(res_cancel, "content") and hasattr(res_cancel.content[0], "text"):
        res_cancel = res_cancel.content[0].text
    elif not isinstance(res_cancel, str):
        res_cancel = res_cancel[0].text if hasattr(res_cancel, "__getitem__") and hasattr(res_cancel[0], "text") else str(res_cancel)
    cancel_data = json.loads(res_cancel)
    assert cancel_data["cancellation_requested"] is True
