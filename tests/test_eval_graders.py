from unittest.mock import MagicMock
from neuralresearcher.evals.graders import grade_completion, grade_correctness
from neuralresearcher.evals.core import EvalTask
from neuralresearcher.orchestrator import OrchestratorState


def test_grader_halt_code_and_stage_pass():
    orch = MagicMock()
    orch.state = OrchestratorState.HALTED
    orch.store.load_manifest.return_value = {
        "halt_code": "OUT_OF_SCOPE",
        "failed_stage": "topic_scope"
    }

    task = EvalTask(
        id="t1", topic="finance", description="",
        success_criteria={
            "expect_halt": True,
            "expected_halt_code": "OUT_OF_SCOPE",
            "expected_failed_stage": "topic_scope"
        }
    )

    comp = grade_completion(orch, task)
    corr = grade_correctness(orch, task)

    assert comp.passed
    assert corr.passed


def test_grader_wrong_halt_code_fails():
    orch = MagicMock()
    orch.state = OrchestratorState.HALTED
    orch.store.load_manifest.return_value = {
        "halt_code": "PROVIDER_ERROR",  # Wrong code
        "failed_stage": "topic_scope"
    }

    task = EvalTask(
        id="t1", topic="finance", description="",
        success_criteria={
            "expect_halt": True,
            "expected_halt_code": "OUT_OF_SCOPE",
            "expected_failed_stage": "topic_scope"
        }
    )

    comp = grade_completion(orch, task)
    corr = grade_correctness(orch, task)

    assert not comp.passed
    assert not corr.passed


def test_grader_wrong_stage_fails():
    orch = MagicMock()
    orch.state = OrchestratorState.HALTED
    orch.store.load_manifest.return_value = {
        "halt_code": "OUT_OF_SCOPE",
        "failed_stage": "retrieval"  # Wrong stage
    }

    task = EvalTask(
        id="t1", topic="finance", description="",
        success_criteria={
            "expect_halt": True,
            "expected_halt_code": "OUT_OF_SCOPE",
            "expected_failed_stage": "topic_scope"
        }
    )

    comp = grade_completion(orch, task)
    assert not comp.passed
