"""Tests for the new CLI commands: runs, status, artifacts, version, providers."""
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner

from neuralresearcher.cli import app
from neuralresearcher.application.models import RunSummary, RunStatus, ArtifactMetadata

runner = CliRunner()


@patch("neuralresearcher.cli.RunManager")
def test_cli_runs_command(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_list(*args, **kwargs):
        return [
            RunSummary(run_id="run-1", topic="Test Topic", provider="groq", status="ACTIVE", created_at="2026-01-01"),
            RunSummary(run_id="run-2", topic="Other Topic", provider="openai", status="HALTED", created_at="2026-01-02"),
        ]
    mock_rm.list_runs = mock_list

    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0
    assert "run-1" in result.stdout
    assert "run-2" in result.stdout
    assert "Test Topic" in result.stdout


@patch("neuralresearcher.cli.RunManager")
def test_cli_runs_empty(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_list(*args, **kwargs):
        return []
    mock_rm.list_runs = mock_list

    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0
    assert "No runs found" in result.stdout


@patch("neuralresearcher.cli.RunManager")
def test_cli_status_command(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_status(run_id):
        return RunStatus(
            run_id=run_id,
            topic="Test Topic",
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
            duration_seconds=12.5,
            artifact_count=2,
            cancellation_requested=False,
        )
    mock_rm.get_status = mock_status

    result = runner.invoke(app, ["status", "test-run-123"])
    assert result.exit_code == 0
    assert "test-run-123" in result.stdout
    assert "Test Topic" in result.stdout
    assert "groq" in result.stdout


@patch("neuralresearcher.cli.RunManager")
def test_cli_status_unknown_run(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_status(run_id):
        raise ValueError("UNKNOWN_RUN")
    mock_rm.get_status = mock_status

    result = runner.invoke(app, ["status", "nonexistent"])
    assert result.exit_code != 0
    assert "UNKNOWN_RUN" in result.stdout


@patch("neuralresearcher.cli.RunManager")
def test_cli_artifacts_command(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_artifacts(run_id):
        return [
            ArtifactMetadata(
                name="plan",
                path="research_plan.md",
                mime_type="text/markdown",
                size_bytes=2048,
                modified_at="2026-01-01",
                resource_uri=f"research://runs/{run_id}/artifacts/research_plan.md",
            ),
        ]
    mock_rm.list_artifacts = mock_artifacts

    result = runner.invoke(app, ["artifacts", "test-run-123"])
    assert result.exit_code == 0
    assert "research_plan.md" in result.stdout
    assert "plan" in result.stdout


@patch("neuralresearcher.cli.RunManager")
def test_cli_artifacts_empty(mock_rm_cls, tmp_path):
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm

    async def mock_artifacts(run_id):
        return []
    mock_rm.list_artifacts = mock_artifacts

    result = runner.invoke(app, ["artifacts", "test-run-123"])
    assert result.exit_code == 0
    assert "No artifacts found" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "neuralresearcher" in result.stdout
    assert "v" in result.stdout


def test_cli_providers():
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "Groq" in result.stdout
    assert "OpenAI" in result.stdout
    assert "Anthropic" in result.stdout
    assert "DeepSeek" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "neuralresearcher" in result.stdout.lower()
