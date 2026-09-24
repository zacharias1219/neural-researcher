from neuralresearcher.context import AgentContext
from neuralresearcher.io.plan_writer import render_markdown_plan
from neuralresearcher.logging import log_success


def run_reporting(context: AgentContext) -> None:
    plan, steps = context.store.load_plan()
    if not plan:
        return
        
    papers = context.store.load_papers()
    gaps = context.store.load_gaps()
    directions = context.store.load_directions()
    claims = context.store.load_claims()
    coverage_report = context.store.load_coverage_report()
    review_result = context.store.load_review_result()
    
    render_markdown_plan(
        plan=plan,
        steps=steps,
        papers=papers,
        gaps=gaps,
        directions=directions,
        claims=claims,
        coverage_report=coverage_report,
        review_result=review_result,
        output_path=str(context.store.directory / "research_plan.md")
    )
    
    log_success("Generated research_plan.md successfully.")
