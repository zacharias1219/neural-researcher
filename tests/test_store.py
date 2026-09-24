import pytest
from pathlib import Path
from neuralresearcher.io.store import StateStore
from neuralresearcher.state import Paper, ReviewResult, CoverageReport

def test_store_idempotent_save(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    
    # Save coverage
    cov = CoverageReport(id="1", clusters=[], warnings=["Test"])
    store.save_coverage_report(cov)
    loaded_cov = store.load_coverage_report()
    assert loaded_cov.warnings == ["Test"]
    
    # Save review
    rev = ReviewResult(id="1", passed=True, suggestions=["test"])
    store.save_review_result(rev)
    loaded_rev = store.load_review_result()
    assert loaded_rev.passed is True
    
    # Reload coverage to ensure it wasn't overwritten
    loaded_cov2 = store.load_coverage_report()
    assert loaded_cov2.warnings == ["Test"]

def test_store_update_papers(tmp_path: Path):
    store = StateStore(directory=str(tmp_path))
    
    p1 = Paper(id="1", title="A", authors=[], venue="v", url="u", abstract="ab", year=2024)
    p2 = Paper(id="2", title="B", authors=[], venue="v", url="u", abstract="ab", year=2024)
    store.save_papers([p1, p2])
    
    # Update p1 with methods
    p1_update = Paper(id="1", title="A", authors=[], venue="v", url="u", abstract="ab", year=2024, methods=["M1"])
    store.update_papers([p1_update])
    
    loaded = store.load_papers()
    assert len(loaded) == 2
    p1_loaded = next(p for p in loaded if p.id == "1")
    assert p1_loaded.methods == ["M1"]
    
    # Update p2 with datasets
    p2_update = Paper(id="2", title="B", authors=[], venue="v", url="u", abstract="ab", year=2024, datasets=["D1"])
    store.update_papers([p2_update])
    
    loaded = store.load_papers()
    p1_loaded = next(p for p in loaded if p.id == "1")
    p2_loaded = next(p for p in loaded if p.id == "2")
    assert p1_loaded.methods == ["M1"] # Should not be overwritten
    assert p2_loaded.datasets == ["D1"]
