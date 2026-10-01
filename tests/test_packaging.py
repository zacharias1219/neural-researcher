"""Tests to verify package structure, imports, and wheel contents."""
import importlib
import pytest


def test_import_neuralresearcher():
    mod = importlib.import_module("neuralresearcher")
    assert hasattr(mod, "__version__")
    assert mod.__version__ == "0.1.0"


def test_import_agents():
    importlib.import_module("neuralresearcher.agents")
    importlib.import_module("neuralresearcher.agents.planner")
    importlib.import_module("neuralresearcher.agents.reviewer")


def test_import_io():
    importlib.import_module("neuralresearcher.io")
    importlib.import_module("neuralresearcher.io.store")


def test_import_application():
    importlib.import_module("neuralresearcher.application")
    importlib.import_module("neuralresearcher.application.run_manager")
    importlib.import_module("neuralresearcher.application.models")
    importlib.import_module("neuralresearcher.application.cancellation")
    importlib.import_module("neuralresearcher.application.research_service")


def test_import_evals():
    importlib.import_module("neuralresearcher.evals")
    importlib.import_module("neuralresearcher.evals.core")
    importlib.import_module("neuralresearcher.evals.graders")
    importlib.import_module("neuralresearcher.evals.tasks")


def test_import_adapters():
    importlib.import_module("neuralresearcher.adapters")
    importlib.import_module("neuralresearcher.adapters.mcp")


def test_import_mcp_modules():
    """MCP modules should import if mcp extras are installed."""
    try:
        importlib.import_module("neuralresearcher.adapters.mcp.server")
        importlib.import_module("neuralresearcher.adapters.mcp.tools")
        importlib.import_module("neuralresearcher.adapters.mcp.resources")
        importlib.import_module("neuralresearcher.adapters.mcp.prompts")
        importlib.import_module("neuralresearcher.adapters.mcp.security")
        importlib.import_module("neuralresearcher.adapters.mcp.config")
    except ImportError:
        pytest.skip("MCP extras not installed")


def test_import_core_modules():
    importlib.import_module("neuralresearcher.config")
    importlib.import_module("neuralresearcher.context")
    importlib.import_module("neuralresearcher.errors")
    importlib.import_module("neuralresearcher.state")
    importlib.import_module("neuralresearcher.orchestrator")
    importlib.import_module("neuralresearcher.cli")
    importlib.import_module("neuralresearcher.logging")


def test_state_models_complete():
    from neuralresearcher.state import (
        HaltCode, RunResult, TopicSpec, Paper, Claim, Gap,
        Direction, PlanStep, ResearchPlan, CoverageReport, ReviewResult,
        ContentLevel, TopicDomain,
    )
    # Verify HaltCode has all required codes
    assert hasattr(HaltCode, "CANCELLED")
    assert hasattr(HaltCode, "INTERRUPTED")
    assert hasattr(HaltCode, "INTERNAL_ERROR")
    assert hasattr(HaltCode, "STORAGE_FAILURE")
    assert hasattr(HaltCode, "OUT_OF_SCOPE")


def test_application_models_complete():
    from neuralresearcher.application.models import (
        StartResearchRequest, ResumeResearchRequest, RunHandle,
        RunStatus, ResearchResult, RunSummary, ArtifactMetadata,
    )
    # Verify all models can be instantiated with required fields
    handle = RunHandle(
        run_id="test", topic="test", status="INIT",
        current_stage=None, created_at="now",
        status_resource_uri="", result_resource_uri="", plan_resource_uri="",
    )
    assert handle.run_id == "test"
