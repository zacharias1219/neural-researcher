from pathlib import Path
from typing import Dict, Any, List

from neuralresearcher.evals.core import EvalTask, Outcome
from neuralresearcher.orchestrator import Orchestrator, OrchestratorState


def grade_completion(orchestrator: Orchestrator, task: EvalTask) -> Outcome:
    expect_halt = task.success_criteria.get("expect_halt", False)
    
    if expect_halt:
        if orchestrator.state == OrchestratorState.HALTED:
            return Outcome(score=1.0, passed=True, details="Halted as expected for negative task")
        else:
            return Outcome(score=0.0, passed=False, details=f"Expected HALTED, got {orchestrator.state}")
    
    if orchestrator.state == OrchestratorState.REPORT_READY:
        return Outcome(score=1.0, passed=True, details="Reached REPORT_READY")
    
    # Fractional score for partial pipeline progress
    pipeline_order = [
        OrchestratorState.INIT,
        OrchestratorState.SCOPED,
        OrchestratorState.RETRIEVED,
        OrchestratorState.READ,
        OrchestratorState.MAPPED,
        OrchestratorState.GAPS_IDENTIFIED,
        OrchestratorState.DIRECTIONS_PROPOSED,
        OrchestratorState.PLAN_DRAFTED,
        OrchestratorState.PLAN_REVIEWED,
        OrchestratorState.REPORT_READY,
    ]
    if orchestrator.state in pipeline_order:
        idx = pipeline_order.index(orchestrator.state)
        progress = round(idx / (len(pipeline_order) - 1), 2)
        return Outcome(
            score=progress,
            passed=False,
            details=f"Stuck at {orchestrator.state.name} ({progress * 100:.0f}% pipeline progress)"
        )

    return Outcome(score=0.0, passed=False, details=f"Failed to reach REPORT_READY, stuck in {orchestrator.state}")


def grade_correctness(orchestrator: Orchestrator, task: EvalTask) -> Outcome:
    expect_halt = task.success_criteria.get("expect_halt", False)
    if expect_halt:
        if orchestrator.state == OrchestratorState.HALTED:
            return Outcome(score=1.0, passed=True, details="Halted as expected for negative task")
        return Outcome(score=0.0, passed=False, details=f"Negative task failed to halt (got {orchestrator.state})")
        
    papers = orchestrator.store.load_papers()
    gaps = orchestrator.store.load_gaps()
    plan, steps = orchestrator.store.load_plan()
    review = orchestrator.store.load_review_result()

    criteria_scores: List[float] = []
    issues: List[str] = []

    # 1. Papers criteria
    min_papers = task.success_criteria.get("min_papers", 0)
    if min_papers > 0:
        p_ratio = min(1.0, len(papers) / min_papers)
        criteria_scores.append(p_ratio)
        if len(papers) < min_papers:
            issues.append(f"Found {len(papers)} papers (expected {min_papers})")
    else:
        criteria_scores.append(1.0 if papers else 0.5)

    # 2. Gaps criteria
    min_gaps = task.success_criteria.get("min_gaps", 0)
    if min_gaps > 0:
        g_ratio = min(1.0, len(gaps) / min_gaps)
        criteria_scores.append(g_ratio)
        if len(gaps) < min_gaps:
            issues.append(f"Found {len(gaps)} gaps (expected {min_gaps})")
    else:
        criteria_scores.append(1.0 if gaps else 0.5)

    # 3. Plan & steps existence
    if plan and steps:
        criteria_scores.append(1.0)
    elif plan or steps:
        criteria_scores.append(0.5)
        issues.append("Partial plan without steps generated")
    else:
        criteria_scores.append(0.0)
        issues.append("No research plan generated")

    # 4. Required keyword check
    required_keyword = task.success_criteria.get("required_keyword")
    if required_keyword:
        plan_file = orchestrator.store.directory / "research_plan.md"
        plan_text = ""
        if plan_file.exists():
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_text = f.read()
                
        if plan_text and required_keyword.lower() in plan_text.lower():
            criteria_scores.append(1.0)
        else:
            criteria_scores.append(0.0)
            issues.append(f"Required keyword '{required_keyword}' not found in plan")

    # 5. Review check
    if review and review.passed:
        criteria_scores.append(1.0)
    elif review:
        criteria_scores.append(0.7)
    else:
        criteria_scores.append(0.5)

    total_score = round(sum(criteria_scores) / len(criteria_scores), 2)
    passed = total_score >= 0.8 and len(issues) == 0

    details = "All criteria met" if passed else f"Score {total_score}: " + "; ".join(issues)
    return Outcome(score=total_score, passed=passed, details=details)


def grade_efficiency(
    duration_sec: float,
    total_tokens: int,
    max_duration_sec: float = 180.0,
    max_tokens: int = 50000
) -> Outcome:
    """Grade trial efficiency against latency and token bounds."""
    dur_score = 1.0 if duration_sec <= max_duration_sec else max(0.0, 1.0 - (duration_sec - max_duration_sec) / max_duration_sec)
    tok_score = 1.0 if total_tokens <= max_tokens else max(0.0, 1.0 - (total_tokens - max_tokens) / max_tokens)
    
    efficiency_score = round((dur_score + tok_score) / 2.0, 2)
    passed = duration_sec <= max_duration_sec and total_tokens <= max_tokens
    
    details = f"Duration: {duration_sec:.1f}s (max {max_duration_sec}s), Tokens: {total_tokens} (max {max_tokens})"
    return Outcome(score=efficiency_score, passed=passed, details=details)
