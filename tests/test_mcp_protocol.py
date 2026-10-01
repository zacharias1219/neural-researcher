"""
MCP protocol tests using an in-memory client.
These tests verify the MCP server surface without making external provider/network calls.
"""
import json

import pytest
from mcp import Client

from neuralresearcher.adapters.mcp.config import MCPSettings
from neuralresearcher.adapters.mcp.server import create_mcp_server
from neuralresearcher.application.models import (
    ArtifactMetadata,
    ResearchResult,
    RunHandle,
    RunStatus,
    RunSummary,
)


class InMemoryService:
    """A fully in-memory mock service for MCP testing. No external calls."""

    def __init__(self):
        self._runs = {}
        self.shutting_down = False

    async def start_run(self, req):
        return RunHandle(
            run_id="mcp-test-001",
            topic=req.topic,
            status="INIT",
            current_stage=None,
            created_at="2026-01-01T00:00:00",
            status_resource_uri="research://runs/mcp-test-001/status",
            result_resource_uri="research://runs/mcp-test-001/result",
            plan_resource_uri="research://runs/mcp-test-001/plan",
        )

    async def resume_run(self, req):
        return RunHandle(
            run_id="mcp-test-002",
            topic="Resumed Topic",
            status="RESUMED",
            current_stage=None,
            created_at="2026-01-01T00:00:00",
            status_resource_uri="research://runs/mcp-test-002/status",
            result_resource_uri="research://runs/mcp-test-002/result",
            plan_resource_uri="research://runs/mcp-test-002/plan",
        )

    async def get_status(self, run_id):
        return RunStatus(
            run_id=run_id,
            topic="Test",
            provider="groq",
            model="test-model",
            status="REPORT_READY",
            current_stage="REPORT_READY",
            created_at="2026-01-01",
            success=True,
            terminal=True,
            halt_code=None,
            failed_stage=None,
            error_message=None,
            duration_seconds=10.0,
            artifact_count=1,
            cancellation_requested=False,
        )

    async def cancel_run(self, run_id):
        return RunStatus(
            run_id=run_id,
            topic="Test",
            provider="groq",
            model="test-model",
            status="ACTIVE",
            current_stage="SCOPED",
            created_at="2026-01-01",
            success=False,
            terminal=False,
            halt_code=None,
            failed_stage=None,
            error_message=None,
            duration_seconds=5.0,
            artifact_count=0,
            cancellation_requested=True,
        )

    async def get_result(self, run_id):
        return ResearchResult(
            run_id=run_id,
            final_state="REPORT_READY",
            success=True,
            topic="Test",
            halt_code=None,
            failed_stage=None,
            artifact_resource_uris={"plan": f"research://runs/{run_id}/artifacts/research_plan.md"},
        )

    async def list_runs(self, status=None, provider=None, topic=None, created_after=None, limit=50):
        return [
            RunSummary(run_id="run-a", topic="Topic A", provider="groq", status="ACTIVE", created_at="2026-01-01"),
            RunSummary(run_id="run-b", topic="Topic B", provider="openai", status="HALTED", created_at="2026-01-02"),
        ]

    async def list_artifacts(self, run_id):
        return [
            ArtifactMetadata(
                name="plan", path="research_plan.md", mime_type="text/markdown",
                size_bytes=1024, modified_at="2026-01-01",
                resource_uri=f"research://runs/{run_id}/artifacts/research_plan.md",
            ),
        ]

    async def read_public_resource(self, run_id, resource_name):
        if resource_name == "manifest.json":
            return json.dumps({"run_id": run_id, "topic": "Test", "status": "REPORT_READY"}).encode("utf-8")
        raise ValueError("RESOURCE_NOT_FOUND")

    async def read_generated_artifact(self, run_id, artifact_name):
        if artifact_name == "research_plan.md":
            return b"# Research Plan\n\nThis is a test plan."
        raise ValueError("RESOURCE_NOT_FOUND")

    async def shutdown(self, grace_period=5.0):
        self.shutting_down = True


@pytest.fixture
def mcp_service():
    return InMemoryService()


@pytest.fixture
def mcp_test_server(mcp_service):
    settings = MCPSettings(data_dir="test_research", max_concurrent_runs=1)
    return create_mcp_server(mcp_service, settings)


@pytest.mark.asyncio
async def test_mcp_discover_tools(mcp_test_server):
    async with Client(mcp_test_server) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert "research_start" in tool_names
        assert "research_status" in tool_names
        assert "research_result" in tool_names
        assert "research_cancel" in tool_names
        assert "research_list_runs" in tool_names
        assert "research_list_artifacts" in tool_names
        assert "research_resume" in tool_names


@pytest.mark.asyncio
async def test_mcp_discover_prompts(mcp_test_server):
    async with Client(mcp_test_server) as client:
        prompts = await client.list_prompts()
        prompt_names = {p.name for p in prompts.prompts}
        assert "create_research_plan" in prompt_names
        assert "review_research_plan" in prompt_names
        assert "explore_research_gap" in prompt_names


@pytest.mark.asyncio
async def test_mcp_call_research_start(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_start", {"topic": "test topic", "provider": "groq"})
        text = result.content[0].text
        data = json.loads(text)
        assert data["run_id"] == "mcp-test-001"
        assert data["topic"] == "test topic"
        assert data["status"] == "INIT"


@pytest.mark.asyncio
async def test_mcp_call_research_status(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_status", {"run_id": "test-run"})
        text = result.content[0].text
        data = json.loads(text)
        assert data["run_id"] == "test-run"
        assert data["success"] is True
        assert data["terminal"] is True


@pytest.mark.asyncio
async def test_mcp_call_research_result(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_result", {"run_id": "test-run"})
        text = result.content[0].text
        data = json.loads(text)
        assert data["run_id"] == "test-run"
        assert data["success"] is True
        assert "plan" in data["artifact_resource_uris"]


@pytest.mark.asyncio
async def test_mcp_call_research_cancel(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_cancel", {"run_id": "test-run"})
        text = result.content[0].text
        data = json.loads(text)
        assert data["cancellation_requested"] is True


@pytest.mark.asyncio
async def test_mcp_call_research_list_runs(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_list_runs", {})
        text = result.content[0].text
        data = json.loads(text)
        # MCP SDK may serialize list items as individual dicts or as an array
        if isinstance(data, list):
            assert len(data) == 2
            assert data[0]["run_id"] == "run-a"
        else:
            # Single dict means it's the first item
            assert "run_id" in data


@pytest.mark.asyncio
async def test_mcp_call_research_list_artifacts(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_list_artifacts", {"run_id": "test-run"})
        text = result.content[0].text
        data = json.loads(text)
        # MCP SDK may serialize list items as individual dicts or as an array
        if isinstance(data, list):
            assert len(data) == 1
            assert data[0]["name"] == "plan"
        else:
            assert data["name"] == "plan"


@pytest.mark.asyncio
async def test_mcp_invalid_provider_returns_error(mcp_test_server):
    async with Client(mcp_test_server) as client:
        result = await client.call_tool("research_start", {"topic": "test", "provider": "invalid_provider"})
        text = result.content[0].text
        data = json.loads(text)
        assert data.get("error") == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_mcp_tool_responses_are_json(mcp_test_server):
    """All tool responses should be valid JSON (typed Pydantic models)."""
    async with Client(mcp_test_server) as client:
        for tool_name, args in [
            ("research_start", {"topic": "test", "provider": "groq"}),
            ("research_status", {"run_id": "test"}),
            ("research_result", {"run_id": "test"}),
            ("research_cancel", {"run_id": "test"}),
            ("research_list_runs", {}),
            ("research_list_artifacts", {"run_id": "test"}),
        ]:
            result = await client.call_tool(tool_name, args)
            text = result.content[0].text
            # Must be valid JSON
            parsed = json.loads(text)
            assert isinstance(parsed, (dict, list)), f"{tool_name} returned non-dict/list"
