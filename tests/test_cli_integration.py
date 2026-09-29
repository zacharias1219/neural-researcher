from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from neuralresearcher.cli import app
from neuralresearcher.orchestrator import OrchestratorState

runner = CliRunner()

@patch("neuralresearcher.cli.RunManager")
def test_cli_success_path(mock_rm_cls, tmp_path):
    import os
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm
    
    mock_handle = MagicMock()
    mock_handle.run_id = "test-task-123"
    
    async def mock_start(*args, **kwargs):
        return mock_handle
    mock_rm.start_run = mock_start

    async def mock_status(*args, **kwargs):
        status = MagicMock()
        status.terminal = True
        return status
    mock_rm.get_status = mock_status

    async def mock_get_result(*args, **kwargs):
        res = MagicMock()
        res.success = True
        res.halt_code = None
        res.failed_stage = None
        res.artifact_resource_uris = {"plan": f"research://runs/test-task-123/artifacts/research_plan.md"}
        res.run_id = "test-task-123"
        return res
    mock_rm.get_result = mock_get_result
    
    result = runner.invoke(app, ["run", "test topic", "--no-interactive"])
    
    assert result.exit_code == 0
    assert "Research plan generated" in result.stdout

@patch("neuralresearcher.cli.RunManager")
def test_cli_halt_path(mock_rm_cls, tmp_path):
    import os
    os.environ["GROQ_API_KEY"] = "fake"
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm
    
    mock_handle = MagicMock()
    mock_handle.run_id = "test-task-456"
    
    async def mock_start(*args, **kwargs):
        return mock_handle
    mock_rm.start_run = mock_start

    async def mock_status(*args, **kwargs):
        status = MagicMock()
        status.terminal = True
        return status
    mock_rm.get_status = mock_status

    async def mock_get_result(*args, **kwargs):
        res = MagicMock()
        res.success = False
        res.halt_code = "INVALID_MODEL_OUTPUT"
        res.failed_stage = "planner"
        res.artifact_resource_uris = {}
        res.run_id = "test-task-456"
        return res
    mock_rm.get_result = mock_get_result
    
    result = runner.invoke(app, ["run", "test topic", "--no-interactive"])
    
    assert result.exit_code != 0
    assert "Pipeline Halted" in result.stdout
    assert "INVALID_MODEL_OUTPUT" in result.stdout
    assert "Research plan generated" not in result.stdout

@patch("neuralresearcher.cli.RunManager")
def test_cli_resume_path(mock_rm_cls, tmp_path):
    import os
    os.environ["GROQ_API_KEY"] = "fake"
    
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm
    
    mock_handle = MagicMock()
    mock_handle.run_id = "test-task-789"
    
    async def mock_resume(*args, **kwargs):
        return mock_handle
    mock_rm.resume_run = mock_resume

    async def mock_status(*args, **kwargs):
        status = MagicMock()
        status.terminal = True
        return status
    mock_rm.get_status = mock_status

    async def mock_get_result(*args, **kwargs):
        res = MagicMock()
        res.success = True
        res.halt_code = None
        res.failed_stage = None
        res.artifact_resource_uris = {"plan": f"research://runs/test-task-789/artifacts/research_plan.md"}
        res.run_id = "test-task-789"
        return res
    mock_rm.get_result = mock_get_result

    result = runner.invoke(app, ["run", "test topic", "--resume", "test-task-789", "--no-interactive"])
    assert result.exit_code == 0
    assert "test-task-789" in result.stdout

@patch("neuralresearcher.cli.RunManager")
def test_cli_resume_unknown_path(mock_rm_cls, tmp_path):
    import os
    os.environ["GROQ_API_KEY"] = "fake"
    
    mock_rm = MagicMock()
    mock_rm_cls.return_value = mock_rm
    
    async def mock_resume(*args, **kwargs):
        raise Exception("UNKNOWN_RUN")
    mock_rm.resume_run = mock_resume
    
    result = runner.invoke(app, ["run", "test topic", "--resume", "unknown-task", "--no-interactive"])
    assert result.exit_code != 0
    assert "Cannot resume run" in result.stdout

def test_security_api_keys_sanitized():
    import os
    os.environ["OPENAI_API_KEY"] = "sk-super-secret-key-123"
    os.environ["GROQ_API_KEY"] = "sk-super-secret-key-123"
    result = runner.invoke(app, ["run", "test topic", "--no-interactive"])
    assert "sk-super-secret-key-123" not in result.stdout
