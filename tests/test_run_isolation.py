from neuralresearcher.io.store import StateStore
from neuralresearcher.state import ResearchPlan
from pathlib import Path

def test_two_runs_different_ids(tmp_path: Path):
    store1 = StateStore(directory=str(tmp_path))
    store2 = StateStore(directory=str(tmp_path))
    
    assert store1.run_id != store2.run_id
    assert store1.directory != store2.directory
    
def test_states_remain_separate(tmp_path: Path):
    store1 = StateStore(directory=str(tmp_path))
    store2 = StateStore(directory=str(tmp_path))
    
    plan1 = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h1", expected_contribution="c1", steps=[])
    store1.save_plan(plan1, [])
    
    plan2, _ = store2.load_plan()
    assert plan2 is None
    
def test_explicit_resume_loads_requested_run(tmp_path: Path):
    store1 = StateStore(directory=str(tmp_path))
    plan1 = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h1", expected_contribution="c1", steps=[])
    store1.save_plan(plan1, [])
    
    run_id = store1.run_id
    
    store_resume = StateStore(directory=str(tmp_path), run_id=run_id)
    plan_resume, _ = store_resume.load_plan()
    assert plan_resume is not None
    assert plan_resume.hypothesis == "h1"

def test_new_run_cannot_see_stale_data(tmp_path: Path):
    store1 = StateStore(directory=str(tmp_path))
    plan1 = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h1", expected_contribution="c1", steps=[])
    store1.save_plan(plan1, [])
    
    store2 = StateStore(directory=str(tmp_path))
    plan2, _ = store2.load_plan()
    assert plan2 is None
