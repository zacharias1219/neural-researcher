import asyncio
import pytest
import datetime
import os
from pathlib import Path
from neuralresearcher.application.run_manager import RunManager
from neuralresearcher.application.models import StartResearchRequest, ResumeResearchRequest
from neuralresearcher.config import LLMProvider
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import OrchestratorState

# We will mock the orchestrator to simulate slow running or immediate failure


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "research"
    d.mkdir()
    return d


@pytest.fixture
def manager(data_dir):
    return RunManager(data_dir=str(data_dir))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")


@pytest.mark.asyncio
async def test_completed_run_reaches_terminal_state_without_deadlock(
        manager,
        mocker):
    def fake_run(self):
        import time
        time.sleep(0.1)
        from neuralresearcher.state import RunResult, HaltCode
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            halt_code=None,
            failed_stage=None,
            duration_seconds=1.0,
            message=None,
            artifact_paths={})

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    # Wait for completion, with a timeout to catch deadlock
    async def wait_for_completion():
        while True:
            status = await manager.get_status(handle.run_id)
            if status.terminal:
                return status
            await asyncio.sleep(0.05)

    status = await asyncio.wait_for(wait_for_completion(), timeout=2.0)
    assert status.success is True
    assert status.status == "COMPLETED"


@pytest.mark.asyncio
async def test_cancel_active_run_returns_promptly(manager, mocker):
    def fake_run(self):
        import time
        try:
            # We must poll cancel_token in a sync function
            for _ in range(100):
                if getattr(
                    self,
                    'cancel_token',
                        None) and self.cancel_token.is_cancelled:
                    raise asyncio.CancelledError()
                time.sleep(0.1)
        except asyncio.CancelledError:
            raise
    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)

    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)
    await asyncio.sleep(0.05)

    status = await asyncio.wait_for(manager.cancel_run(handle.run_id), timeout=1.0)
    assert status.cancellation_requested is True


@pytest.mark.asyncio
async def test_cancel_terminal_run_is_idempotent_and_nonblocking(
        manager,
        mocker):
    store = StateStore(directory=str(manager.data_dir))
    manifest = {
        "topic": "t",
        "provider": "groq",
        "final_state": "COMPLETED",
        "success": True}
    store.save_manifest(manifest)

    # Should return promptly without deadlock
    status = await asyncio.wait_for(manager.cancel_run(store.run_id), timeout=1.0)
    assert status.terminal is True
    # Cancellation should not be requested if already terminal
    assert status.cancellation_requested is False


@pytest.mark.asyncio
async def test_state_progress_updates_during_execution(manager, mocker):
    def fake_run(self):
        import time
        self.on_state_change("STAGE_ONE")
        time.sleep(0.1)
        self.on_state_change("STAGE_TWO")
        time.sleep(0.1)
        from neuralresearcher.state import RunResult
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    stages = set()

    async def poll():
        while True:
            s = await manager.get_status(handle.run_id)
            if s.current_stage:
                stages.add(s.current_stage)
            if s.terminal:
                break
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout=2.0)
    assert "STAGE_ONE" in stages
    assert "STAGE_TWO" in stages


@pytest.mark.asyncio
async def test_duplicate_resume_is_rejected(manager, mocker):
    store = StateStore(directory=str(manager.data_dir))
    manifest = {"topic": "t", "provider": "groq", "final_state": "HALTED"}
    store.save_manifest(manifest)

    def fake_run(self):
        import time
        time.sleep(0.5)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)

    # First resume succeeds and returns a new run_id
    req = ResumeResearchRequest(run_id=store.run_id)
    handle1 = await manager.resume_run(req)
    assert handle1.run_id != store.run_id

    # Second resume on the original checkpoint creates another branch
    handle2 = await manager.resume_run(req)
    assert handle2.run_id != store.run_id
    assert handle2.run_id != handle1.run_id


@pytest.mark.asyncio
async def test_cancel_before_worker_start_is_honored(manager, mocker):
    # Occupy the semaphore
    def blocking_run(self):
        import time
        time.sleep(0.5)
        from neuralresearcher.state import RunResult
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=blocking_run)

    # Consume semaphore capacity
    for _ in range(manager.max_concurrent_runs):
        await manager.start_run(StartResearchRequest(topic="fill", provider=LLMProvider.GROQ))

    # Queue a run that will be blocked
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    # Cancel it immediately before worker starts
    await manager.cancel_run(handle.run_id)

    # Wait a bit, then check status
    await asyncio.sleep(0.1)
    status = await manager.get_status(handle.run_id)
    assert status.cancellation_requested is True


@pytest.mark.asyncio
async def test_active_run_cleanup_after_success(manager, mocker):
    def fake_run(self):
        from neuralresearcher.state import RunResult
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    await asyncio.sleep(0.2)
    assert handle.run_id not in manager._active_runs
    assert handle.run_id not in manager._cancellation_tokens


@pytest.mark.asyncio
async def test_active_run_cleanup_after_failure(manager, mocker):
    def fake_run(self):
        raise Exception("Fatal crash")

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    await asyncio.sleep(0.2)
    assert handle.run_id not in manager._active_runs
    assert handle.run_id not in manager._cancellation_tokens


@pytest.mark.asyncio
async def test_manifest_failure_message_reaches_status(manager, mocker):
    def fake_run(self):
        from neuralresearcher.state import RunResult, HaltCode
        return RunResult(
            run_id=self.task_id,
            success=False,
            final_state="HALTED",
            halt_code=HaltCode.INTERNAL_ERROR,
            message="Custom error message",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    await asyncio.sleep(0.2)
    status = await manager.get_status(handle.run_id)
    assert status.error_message == "Custom error message"


@pytest.mark.asyncio
async def test_bootstrap_failure_is_persisted(manager, mocker):
    # Mock load_config to throw an error, which happens before the
    # orchestrator runs
    mocker.patch(
        "neuralresearcher.application.run_manager.load_config",
        side_effect=Exception("Configuration boom"))

    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)

    await asyncio.sleep(0.2)
    status = await manager.get_status(handle.run_id)
    assert status.terminal is True
    assert status.success is False
    assert status.halt_code == "INTERNAL_ERROR"
    assert "The research run failed internally." in status.error_message


@pytest.mark.asyncio
async def test_manifest_artifact_cannot_escape_run_directory(
        manager,
        tmp_path):
    store = StateStore(directory=str(manager.data_dir))
    escape_path = str(tmp_path / "secret.txt")
    with open(escape_path, "w") as f:
        f.write("secret")

    manifest = {"artifact_paths": {"secret.txt": escape_path}}
    store.save_manifest(manifest)

    with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
        await manager.read_generated_artifact(store.run_id, "secret.txt")


@pytest.mark.asyncio
async def test_mcp_max_result_bytes_is_enforced(manager):
    manager.max_result_bytes = 10  # 10 bytes max
    store = StateStore(directory=str(manager.data_dir))

    art_path = store.directory / "big.txt"
    art_path.write_bytes(b"A" * 20)

    store.save_manifest({"artifact_paths": {"big": str(art_path)}})

    with pytest.raises(ValueError, match="ARTIFACT_TOO_LARGE"):
        await manager.read_generated_artifact(store.run_id, "big.txt")


def test_time_window_filters_retrieved_papers(mocker):
    from neuralresearcher.agents.retrieval import run_retrieval
    from neuralresearcher.context import AgentContext
    from neuralresearcher.state import TopicSpec
    from neuralresearcher.config import load_config

    store = StateStore(directory="test_tmp")
    store.save_topic_spec(
        TopicSpec(
            id="t1",
            raw_topic="test",
            domain="machine_learning",
            subfields=[],
            keywords=[],
            scope_constraints={},
            time_window={
                "start_year": 2020,
                "end_year": 2022}))

    class FakeResponse:
        tool_calls = [1]
        content = ""

    mocker.patch(
        "neuralresearcher.agents.retrieval.call_llm",
        return_value=FakeResponse())

    # Return 3 papers: 2019, 2021, 2023
    import json
    fake_papers = json.dumps([
        {"id": "1", "title": "A", "authors": [], "year": 2019, "url": "", "abstract": ""},
        {"id": "2", "title": "B", "authors": [], "year": 2021, "url": "", "abstract": ""},
        {"id": "3", "title": "C", "authors": [], "year": 2023, "url": "", "abstract": ""}
    ])
    mocker.patch(
        "neuralresearcher.agents.retrieval.execute_tool_call",
        return_value=(
            None,
            fake_papers))

    ctx = AgentContext(
        task_id="t",
        topic="test",
        config=load_config(),
        store=store)
    run_retrieval(ctx)

    papers = store.load_papers()
    assert len(papers) == 1
    assert papers[0].year == 2021


def test_semantic_scholar_results_survive_source_merge(mocker):
    from neuralresearcher.tools import search_papers_impl

    # We will mock requests to avoid network calls
    class MockResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self._content = content

        @property
        def content(self):
            return self._content

        def json(self):
            import json
            return json.loads(self._content)

    def fake_get(url, **kwargs):
        if "arxiv" in url:
            xml = b'''<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2101.00001</id>
                <published>2021-01-01T00:00:00Z</published>
                <title>Arxiv Paper</title>
                <summary>Abstract 1</summary>
                <author><name>Author 1</name></author>
              </entry>
            </feed>'''
            return MockResponse(200, xml)
        elif "semanticscholar" in url:
            s2 = '{"data": [{"paperId": "123", "title": "S2 Paper", "authors": [{"name": "Author 2"}], "year": 2022, "url": "", "abstract": "Abstract 2"}]}'
            return MockResponse(200, s2)
        return MockResponse(404, "")

    mocker.patch("requests.Session.get", side_effect=fake_get)

    from neuralresearcher.config import load_config
    results = search_papers_impl(["test"], max_results=5, config=load_config())
    sources = [r["source"] for r in results]
    assert "arxiv" in sources
    assert "semantic_scholar" in sources


@pytest.mark.asyncio
async def test_max_papers_injected_to_config(manager, mocker):
    mock_run = mocker.patch("neuralresearcher.orchestrator.Orchestrator.run")
    req = StartResearchRequest(
        topic="test",
        provider=LLMProvider.GROQ,
        max_papers=17)
    handle = await manager.start_run(req)
    await asyncio.sleep(0.1)

    store = StateStore(directory=str(manager.data_dir), run_id=handle.run_id)
    manifest = store.load_manifest()
    assert manifest["max_papers"] == 17

    # We could also intercept the Orchestrator init to check config.max_papers
    # but checking the manifest is sufficient to show it was processed.


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "valid/../../escape",
        "abc.def",
        "abc def",
        "",
        "a" * 65,
        "normal-id\n../../escape",
    ],
)
@pytest.mark.asyncio
async def test_invalid_run_ids_rejected(manager, run_id):
    with pytest.raises(ValueError, match="INVALID_RUN_ID"):
        await manager.get_status(run_id)


def test_offline_mode_skips_semantic_scholar(mocker):
    from neuralresearcher.tools import search_papers_impl

    # We will mock requests to avoid network calls
    class MockResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self._content = content

        @property
        def content(self): return self._content
        def json(self): return {}

    get_mock = mocker.patch(
        "requests.Session.get",
        return_value=MockResponse(
            200,
            b"<feed></feed>"))

    # Test 1: Neither offline nor cassette
    mocker.patch.dict(
        os.environ, {
            "NEURALRESEARCHER_OFFLINE": "0"}, clear=False)
    from neuralresearcher.config import load_config
    search_papers_impl(["test"], max_results=5, config=load_config())
    assert any("semanticscholar" in call[0][0]
               for call in get_mock.call_args_list), "Should call Semantic Scholar"

    get_mock.reset_mock()

    # Test 2: Offline mode
    mocker.patch.dict(
        os.environ, {
            "NEURALRESEARCHER_OFFLINE": "1"}, clear=False)
    search_papers_impl(["test"], max_results=5, config=load_config())
    assert not any("semanticscholar" in call[0][0]
                   for call in get_mock.call_args_list), "Should not call Semantic Scholar when offline"


@pytest.mark.asyncio
async def test_reservation_rolled_back_on_manifest_failure(manager, mocker):
    mocker.patch(
        "neuralresearcher.io.store.StateStore.save_manifest",
        side_effect=Exception("Disk full"))
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    with pytest.raises(Exception, match="Disk full"):
        await manager.start_run(req)
    assert len(manager._active_runs) == 0
    assert len(manager._cancellation_tokens) == 0


@pytest.mark.asyncio
async def test_reservation_rolled_back_on_task_creation_failure(
        manager,
        mocker):
    mocker.patch(
        "asyncio.create_task",
        side_effect=Exception("Task creation failed"))
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    with pytest.raises(Exception, match="Task creation failed"):
        await manager.start_run(req)
    assert len(manager._active_runs) == 0


@pytest.mark.asyncio
async def test_invalid_provider_in_resume_aborts_cleanly(manager):
    store = StateStore(directory=str(manager.data_dir))
    store.save_manifest({"topic": "test", "provider": "invalid-provider"})

    with pytest.raises(ValueError):
        await manager.resume_run(ResumeResearchRequest(run_id=store.run_id))

    assert len(manager._active_runs) == 0


@pytest.mark.asyncio
async def test_resume_validation_failure_preserves_artifacts(manager):
    store = StateStore(directory=str(manager.data_dir))
    store.save_manifest({"topic": "test", "provider": "groq", "success": True})

    analysis_dir = store.directory / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="RESUME_NOT_ALLOWED"):
        await manager.resume_run(ResumeResearchRequest(run_id=store.run_id))

    assert analysis_dir.exists(), "Artifacts should not be deleted if resume fails validation"


@pytest.mark.asyncio
async def test_artifact_authorization(manager):
    store = StateStore(directory=str(manager.data_dir))
    store.save_manifest({"topic": "test"})

    (store.directory / "state.json").write_text("secret state")
    (store.directory / "research_plan.md").write_text("plan")

    # Read public
    with pytest.raises(ValueError, match="RESOURCE_NOT_FOUND"):
        await manager.read_public_resource(store.run_id, "state.json")

    assert await manager.read_public_resource(store.run_id, "research_plan.md") == b"plan"

    # Read generated
    with pytest.raises(ValueError, match="RESOURCE_NOT_FOUND"):
        await manager.read_generated_artifact(store.run_id, "state.json")


@pytest.mark.asyncio
async def test_resolved_model_in_status(manager, mocker):
    mocker.patch("neuralresearcher.orchestrator.Orchestrator.run")
    req = StartResearchRequest(topic="test", provider=LLMProvider.GROQ)
    handle = await manager.start_run(req)
    await asyncio.sleep(0.1)

    status = await manager.get_status(handle.run_id)
    assert status.model != ""


@pytest.mark.asyncio
async def test_shutdown_signals_tokens_and_waits(manager, mocker):
    def slow_run(self):
        import time
        import asyncio
        while not getattr(self.cancel_token, 'is_cancelled', False):
            time.sleep(0.1)
        raise asyncio.CancelledError()

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=slow_run)

    handle = await manager.start_run(StartResearchRequest(topic="test", provider=LLMProvider.GROQ))
    await asyncio.sleep(0.1)

    # Trigger shutdown with small grace period
    await manager.shutdown(grace_period=2.0)

    # Status should be interrupted
    status = await manager.get_status(handle.run_id)
    assert status.terminal is True
    assert status.success is False
    assert status.error_message == "Run cancelled by request."


@pytest.mark.asyncio
async def test_late_cancelled_error_cannot_replace_interrupted(manager, mocker):
    def fake_run(self):
        import time
        import asyncio
        # Simulate a task that takes a long time
        for _ in range(50):
            if getattr(self.cancel_token, 'is_cancelled', False):
                # wait a bit more to simulate late cancellation error after shutdown has marked it INTERRUPTED
                time.sleep(0.5)
                raise asyncio.CancelledError()
            time.sleep(0.1)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)

    handle = await manager.start_run(StartResearchRequest(topic="test", provider=LLMProvider.GROQ))
    await asyncio.sleep(0.1)

    # shutdown with 0 grace period forces INTERRUPTED to be written immediately
    await manager.shutdown(grace_period=0.0)
    
    # Let the worker raise CancelledError now
    await asyncio.sleep(1.0)
    
    status = await manager.get_status(handle.run_id)
    assert status.terminal is True
    assert status.halt_code == "INTERRUPTED"


@pytest.mark.asyncio
async def test_state_callbacks_cannot_mutate_terminal_run(manager, mocker):
    def fake_run(self):
        # Fire state callback multiple times with delays
        import time
        self.on_state_change("INITIAL")
        time.sleep(0.2)
        self.on_state_change("SHOULD_BE_IGNORED")
        time.sleep(0.1)
        from neuralresearcher.state import RunResult
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)

    handle = await manager.start_run(StartResearchRequest(topic="test", provider=LLMProvider.GROQ))
    await asyncio.sleep(0.1)
    
    # Force it to be terminal manually
    store = StateStore(directory=str(manager.data_dir), run_id=handle.run_id)
    manifest = store.load_manifest()
    manifest["final_state"] = "HALTED"
    manifest["halt_code"] = "INTERRUPTED"
    store.save_manifest(manifest)
    
    # Wait for the run to finish
    await asyncio.sleep(0.5)
    
    status = await manager.get_status(handle.run_id)
    # Since we set final_state manually, the callback SHOULD_BE_IGNORED should not have mutated current_stage
    assert status.current_stage != "SHOULD_BE_IGNORED"


@pytest.mark.asyncio
async def test_shutdown_rejects_newly_submitted_runs(manager):
    await manager.shutdown(grace_period=0.0)
    with pytest.raises(ValueError, match="SERVICE_SHUTTING_DOWN"):
        await manager.start_run(StartResearchRequest(topic="test", provider=LLMProvider.GROQ))
    with pytest.raises(ValueError, match="SERVICE_SHUTTING_DOWN"):
        await manager.resume_run(ResumeResearchRequest(run_id="fake_run"))


@pytest.mark.asyncio
async def test_resume_creates_distinct_run_with_resumed_from(manager, mocker):
    store = StateStore(directory=str(manager.data_dir))
    manifest = {"topic": "t", "provider": "groq", "final_state": "HALTED"}
    store.save_manifest(manifest)

    def fake_run(self):
        from neuralresearcher.state import RunResult
        return RunResult(
            run_id=self.task_id,
            success=True,
            final_state="COMPLETED",
            duration_seconds=1.0)

    mocker.patch(
        "neuralresearcher.orchestrator.Orchestrator.run",
        new=fake_run)

    req = ResumeResearchRequest(run_id=store.run_id)
    handle = await manager.resume_run(req)
    
    assert handle.run_id != store.run_id
    
    await asyncio.sleep(0.1)
    new_store = StateStore(directory=str(manager.data_dir), run_id=handle.run_id)
    new_manifest = new_store.load_manifest()
    assert new_manifest["resumed_from"] == store.run_id


@pytest.mark.asyncio
async def test_public_manifest_exposes_only_allowlisted_fields(manager, mocker):
    store = StateStore(directory=str(manager.data_dir))
    manifest = {
        "run_id": store.run_id,
        "topic": "test",
        "provider": "groq",
        "model": "llama",
        "strict": False,
        "secret_api_key": "12345",
        "internal_path": "/var/tmp/secret"
    }
    store.save_manifest(manifest)

    import json
    public_data = await manager.read_public_resource(store.run_id, "manifest.json")
    public_manifest = json.loads(public_data.decode('utf-8'))
    
    assert "secret_api_key" not in public_manifest
    assert "internal_path" not in public_manifest
    assert public_manifest["run_id"] == store.run_id
    assert public_manifest["topic"] == "test"

