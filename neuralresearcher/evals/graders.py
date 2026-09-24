from neuralresearcher.evals.core import EvalTask, Outcome
from neuralresearcher.orchestrator import Orchestrator, OrchestratorState
from pathlib import Path

def grade_completion(orchestrator: Orchestrator, task: EvalTask) -> Outcome:
    expect_halt = task.success_criteria.get("expect_halt", False)
    
    if expect_halt:
        if orchestrator.state == OrchestratorState.HALTED:
            return Outcome(score=1.0, passed=True, details="Halted as expected")
        else:
            return Outcome(score=0.0, passed=False, details=f"Expected HALTED, got {orchestrator.state}")
    
    if orchestrator.state == OrchestratorState.REPORT_READY:
        return Outcome(score=1.0, passed=True, details="Reached REPORT_READY")
    return Outcome(score=0.0, passed=False, details=f"Failed to reach REPORT_READY, stuck in {orchestrator.state}")

def grade_correctness(orchestrator: Orchestrator, task: EvalTask) -> Outcome:
    expect_halt = task.success_criteria.get("expect_halt", False)
    if expect_halt:
        return Outcome(score=1.0, passed=True, details="N/A for negative task")
        
    papers = orchestrator.store.load_papers()
    gaps = orchestrator.store.load_gaps()
    plan, steps = orchestrator.store.load_plan()
    
    min_papers = task.success_criteria.get("min_papers", 0)
    if len(papers) < min_papers:
        return Outcome(score=0.0, passed=False, details=f"Found {len(papers)} papers, expected {min_papers}")
        
    min_gaps = task.success_criteria.get("min_gaps", 0)
    if len(gaps) < min_gaps:
        return Outcome(score=0.0, passed=False, details=f"Found {len(gaps)} gaps, expected {min_gaps}")
        
    if not plan:
        return Outcome(score=0.0, passed=False, details="No research plan generated")
        
    required_keyword = task.success_criteria.get("required_keyword")
    if required_keyword:
        plan_file = orchestrator.store.directory / "research_plan.md"
        plan_text = ""
        if plan_file.exists():
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_text = f.read()
        elif Path("research_plan.md").exists():
            with open("research_plan.md", "r", encoding="utf-8") as f:
                plan_text = f.read()
                
        if not plan_text or required_keyword.lower() not in plan_text.lower():
            return Outcome(score=0.0, passed=False, details=f"Required keyword '{required_keyword}' not found in plan")
            
    return Outcome(score=1.0, passed=True, details="All criteria met")
