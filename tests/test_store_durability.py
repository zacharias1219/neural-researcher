import pytest
from pathlib import Path
from neuralresearcher.io.store import StateStore
from neuralresearcher.errors import StateCorruptionError
from unittest.mock import patch

def test_store_serialization_failure_preserves_state(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    store.save_papers([]) # Valid state
    
    class UnserializablePaper:
        pass
        
    with pytest.raises(Exception):
        # Should fail serialization
        store.save_papers([UnserializablePaper()]) # type: ignore
        
    # State file should still be valid []
    loaded = store.load_papers()
    assert loaded == []

def test_store_failure_before_atomic_replace(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    store.save_papers([]) # Valid state
    
    with patch('os.replace') as mock_replace:
        mock_replace.side_effect = Exception("Atomic replace failed")
        
        with pytest.raises(Exception):
            store.save_papers([])
            
    loaded = store.load_papers()
    assert loaded == [] # the old valid state remains untouched

def test_store_corrupt_json_raises_and_preserves_corrupt_file(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    store.directory.mkdir(parents=True, exist_ok=True)
    with open(store.state_file, "w", encoding="utf-8") as f:
        f.write("{invalid json")
        
    with pytest.raises(StateCorruptionError) as exc:
        store.load_papers()
        
    # Corrupt file is preserved as backup
    assert "Saved backup to" in str(exc.value)
    
    corrupt_backups = list(store.directory.glob("state.corrupt.*.json"))
    assert len(corrupt_backups) == 1
    with open(corrupt_backups[0], "r", encoding="utf-8") as f:
        assert f.read() == "{invalid json"

def test_unsupported_schema_version_fail(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    store.directory.mkdir(parents=True, exist_ok=True)
    import json
    with open(store.state_file, "w", encoding="utf-8") as f:
        json.dump({"version": 9999, "papers": []}, f)
        
    with pytest.raises(StateCorruptionError) as exc:
        store.load_papers()
    
    assert "Unsupported state version: 9999" in str(exc.value)
