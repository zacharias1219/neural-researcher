from typing import List
from pathlib import Path

from neuralresearcher.state import (
    ResearchPlan, PlanStep, Paper, Gap, Direction
)

def render_markdown_plan(
    plan: ResearchPlan,
    steps: List[PlanStep],
    papers: List[Paper],
    gaps: List[Gap],
    directions: List[Direction],
    output_path: str = "research_plan.md"
) -> None:
    lines = []
    
    # Title
    lines.append(f"# Research Plan")
    lines.append("")
    
    # Overview
    lines.append("## Overview")
    lines.append(f"**Hypothesis:** {plan.hypothesis}")
    lines.append(f"**Expected Contribution:** {plan.expected_contribution}")
    lines.append("")
    
    # Gaps Addressed
    lines.append("## Primary Gaps Addressed")
    for gap_id in plan.primary_gap_ids:
        gap = next((g for g in gaps if g.id == gap_id), None)
        if gap:
            lines.append(f"- **{gap.gap_type}**: {gap.description}")
    lines.append("")
    
    # Execution Steps (SOP)
    lines.append("## Standard Operating Procedure (SOP)")
    
    phases = ["data", "implementation", "experiment", "ablation", "analysis", "writing"]
    for phase in phases:
        phase_steps = [s for s in steps if s.type == phase]
        if phase_steps:
            lines.append(f"### {phase.capitalize()} Phase")
            for step in phase_steps:
                lines.append(f"#### {step.label} ({step.id})")
                lines.append(f"- **Risk Level**: {step.risk_level}")
                lines.append(f"- **Dependencies**: {', '.join(step.dependencies) if step.dependencies else 'None'}")
                lines.append(f"- **Inputs**: {', '.join(step.inputs) if step.inputs else 'None'}")
                lines.append(f"- **Outputs**: {', '.join(step.outputs) if step.outputs else 'None'}")
                lines.append("")
                
    # Key Papers
    lines.append("## Key Literature References")
    for paper in papers:
        lines.append(f"- {paper.title} ({paper.year}) - {paper.url}")
        
    lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
