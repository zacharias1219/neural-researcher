"""Tests for the reviewer agent's structural validation checks."""
import pytest
import tempfile
import shutil
from pathlib import Path

from neuralresearcher.state import PlanStep, ResearchPlan, ReviewResult
from neuralresearcher.io.store import StateStore
from neuralresearcher.config import Config
from neuralresearcher.errors import WorkflowError


class MockOrchestrator:
    """Minimal orchestrator mock for reviewer tests."""
    def __init__(self, store, strict=False):
        self.store = store
        self.config = Config(strict_mode=strict)


def _make_step(id, label="Step", type="experiment", risk_level="medium",
               inputs=None, outputs=None, dependencies=None,
               compute_hours=10.0, human_hours=2.0, **kwargs):
    return PlanStep(
        id=id, label=label, type=type, risk_level=risk_level,
        inputs=["input_1"] if inputs is None else inputs,
        outputs=["output_1"] if outputs is None else outputs,
        dependencies=[] if dependencies is None else dependencies,
        estimated_cost={"compute_hours": compute_hours, "human_hours": human_hours},
        **kwargs
    )


def _make_plan(primary_gap_ids=None, timeline_weeks=12):
    return ResearchPlan(
        id="plan_01", topic_spec_id="topic_01",
        primary_gap_ids=["gap_01"] if primary_gap_ids is None else primary_gap_ids,
        hypothesis="Test hypothesis",
        expected_contribution="Test contribution",
        timeline_weeks=timeline_weeks,
    )


@pytest.fixture
def tmp_store(tmp_path):
    store = StateStore(directory=str(tmp_path / "research"))
    return store


class TestReviewerChecks:
    def test_valid_plan_passes(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="data", label="Preprocess data"),
            _make_step("step_02", type="experiment", label="Train baseline vs M3",
                       dependencies=["step_01"]),
            _make_step("step_03", type="ablation", label="Ablate patch sizes",
                       dependencies=["step_02"]),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result is not None
        assert result.passed is True
        assert len(result.issues) == 0

    def test_empty_primary_gap_ids_flagged(self, tmp_store):
        plan = _make_plan(primary_gap_ids=[])
        steps = [
            _make_step("step_01", type="experiment"),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("primary_gap_ids" in i for i in result.issues)

    def test_no_experiment_steps_flagged(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="data"),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("experiment" in i for i in result.issues)

    def test_no_ablation_steps_flagged(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="experiment"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("ablation" in i for i in result.issues)

    def test_empty_inputs_on_experiment_flagged(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="experiment", inputs=[]),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("no inputs" in i for i in result.issues)

    def test_zero_compute_on_experiment_flagged(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="experiment", compute_hours=0.0),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("compute_hours=0" in i for i in result.issues)

    def test_invalid_dependency_flagged(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="experiment", dependencies=["step_nonexistent"]),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert result.passed is False
        assert any("does not exist" in i for i in result.issues)

    def test_no_baseline_suggestion(self, tmp_store):
        plan = _make_plan()
        steps = [
            _make_step("step_01", type="experiment", label="Train model"),
            _make_step("step_02", type="ablation", label="Ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert any("baseline" in s.lower() for s in result.suggestions)

    def test_strict_mode_raises_on_failure(self, tmp_store):
        plan = _make_plan(primary_gap_ids=[])
        steps = [
            _make_step("step_01", type="data"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store, strict=True)
        from neuralresearcher.agents.reviewer import run_reviewer
        
        with pytest.raises(WorkflowError):
            run_reviewer(orch)

    def test_zero_timeline_suggestion(self, tmp_store):
        plan = _make_plan(timeline_weeks=0)
        steps = [
            _make_step("step_01", type="experiment"),
            _make_step("step_02", type="ablation"),
        ]
        tmp_store.save_plan(plan, steps)
        
        orch = MockOrchestrator(tmp_store)
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(orch)
        
        result = tmp_store.load_review_result()
        assert any("timeline" in s.lower() for s in result.suggestions)
