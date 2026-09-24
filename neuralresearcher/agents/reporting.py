from typing import Any
from neuralresearcher.io.plan_writer import render_markdown_plan

def run_reporting(orchestrator: Any) -> None:
    plan, steps = orchestrator.store.load_plan()
    if not plan:
        return
        
    papers = orchestrator.store.load_papers()
    gaps = orchestrator.store.load_gaps()
    directions = orchestrator.store.load_directions()
    claims = orchestrator.store.load_claims()
    coverage_report = orchestrator.store.load_coverage_report()
    review_result = orchestrator.store.load_review_result()
    
    render_markdown_plan(
        plan=plan,
        steps=steps,
        papers=papers,
        gaps=gaps,
        directions=directions,
        claims=claims,
        coverage_report=coverage_report,
        review_result=review_result,
        output_path=str(orchestrator.store.directory / "research_plan.md")
    )
    
    from neuralresearcher.logging import log_success
    log_success("Generated research_plan.md successfully.")
