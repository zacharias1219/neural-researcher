
import pytest

from neuralresearcher.application.cancellation import CancellationToken
from neuralresearcher.application.run_manager import RunManager
from neuralresearcher.errors import StateCorruptionError
from neuralresearcher.io.store import StateStore


@pytest.fixture
def run_manager(tmp_path):
    return RunManager(data_dir=str(tmp_path))

@pytest.mark.asyncio
async def test_list_runs_does_not_create_missing_run_dirs(run_manager, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Create an empty directory simulating a missing manifest
    bogus_dir = runs_dir / "bogus_run"
    bogus_dir.mkdir()

    # Also a file that is not a directory
    (runs_dir / "not_a_dir.txt").write_text("hello")

    runs = await run_manager.list_runs()
    assert len(runs) == 0

    # Check that it didn't create a manifest in bogus_run or otherwise alter it
    assert not (bogus_dir / "manifest.json").exists()

@pytest.mark.asyncio
async def test_list_runs_skips_or_flags_corrupt_run_without_failing_all(run_manager, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Good run
    good_store = StateStore(directory=str(tmp_path))
    good_store.save_manifest({"run_id": good_store.run_id, "final_state": "SUCCESS"})

    # 2. Corrupt run
    corrupt_store = StateStore(directory=str(tmp_path))
    corrupt_store.manifest_file.write_text("{corrupt json")

    # 3. Good run 2
    good_store2 = StateStore(directory=str(tmp_path))
    good_store2.save_manifest({"run_id": good_store2.run_id, "final_state": "ACTIVE"})

    runs = await run_manager.list_runs()
    assert len(runs) == 3

    statuses = {r.status for r in runs}
    assert "SUCCESS" in statuses
    assert "ACTIVE" in statuses
    assert "CORRUPTED" in statuses

    corrupt_run_summary = next(r for r in runs if r.status == "CORRUPTED")
    assert corrupt_run_summary.topic == "CORRUPTED RUN"
    assert corrupt_run_summary.run_id == corrupt_store.run_id

@pytest.mark.asyncio
async def test_open_existing_store_returns_unknown_only_for_missing_run(run_manager, tmp_path):
    # Missing directory -> ValueError("UNKNOWN_RUN")
    with pytest.raises(ValueError, match="UNKNOWN_RUN"):
        run_manager._open_existing_store("nonexistent-1234")

    # Missing manifest -> ValueError("UNKNOWN_RUN")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    empty_run = runs_dir / "empty-run-1234"
    empty_run.mkdir()

    with pytest.raises(ValueError, match="UNKNOWN_RUN"):
        run_manager._open_existing_store("empty-run-1234")

@pytest.mark.asyncio
async def test_manifest_corruption_maps_to_storage_failure(run_manager, tmp_path):
    corrupt_store = StateStore(directory=str(tmp_path))
    corrupt_store.manifest_file.write_text("{invalid json...]")

    # This should bubble up the underlying StateCorruptionError (storage failure), not UNKNOWN_RUN
    with pytest.raises(StateCorruptionError):
        run_manager._open_existing_store(corrupt_store.run_id)

@pytest.mark.asyncio
async def test_cancel_run_releases_lock_before_status_lookup(run_manager, tmp_path):
    # Setup a valid run
    store = StateStore(directory=str(tmp_path))
    store.save_manifest({"run_id": store.run_id, "topic": "Lock Test"})
    run_manager._active_runs.add(store.run_id)
    run_manager._cancellation_tokens[store.run_id] = CancellationToken()

    status = await run_manager.cancel_run(store.run_id)
    assert status.cancellation_requested is True

    # Verify lock is not held
    lock = run_manager._get_lock(store.run_id)
    assert not lock.locked()
