from typing import List, Optional
from pathlib import Path

from neuralresearcher.state import (
    ResearchPlan, PlanStep, Paper, Gap, Direction,
    Claim, CoverageReport, ReviewResult
)


def render_markdown_plan(
    plan: ResearchPlan,
    steps: List[PlanStep],
    papers: List[Paper],
    gaps: List[Gap],
    directions: List[Direction],
    claims: Optional[List[Claim]] = None,
    coverage_report: Optional[CoverageReport] = None,
    review_result: Optional[ReviewResult] = None,
    output_path: str = "research_plan.md"
) -> None:
    claims = claims or []
    lines = []
    
    # ================================================================
    # Section 1: Overview
    # ================================================================
    lines.append("# Research Plan")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append(f"**Note:** This plan focuses on a specific direction selected from multiple identified gaps in the broader topic.")
    lines.append("")
    lines.append(f"**Hypothesis:** {plan.hypothesis}")
    lines.append(f"**Expected Contribution:** {plan.expected_contribution}")
    if plan.target_venue:
        lines.append(f"**Target Venue:** {plan.target_venue}")
    if plan.timeline_weeks > 0:
        lines.append(f"**Estimated Timeline:** {plan.timeline_weeks} weeks")
    lines.append("")
    
    # ================================================================
    # Section 2: Field Summary
    # ================================================================
    lines.append("## 2. Field Summary")
    lines.append("")
    
    if coverage_report and coverage_report.clusters:
        for cluster in coverage_report.clusters:
            if cluster.domain == "uncategorized":
                continue
            cluster_papers = [p for p in papers if p.id in cluster.paper_ids]
            if not cluster_papers:
                continue
            lines.append(f"### {cluster.domain.replace('_', ' ').title()}")
            for p in cluster_papers:
                methods_str = ", ".join(p.methods) if p.methods else "—"
                lines.append(f"- **{p.title}** ({p.year}): {methods_str}")
            lines.append("")
    else:
        # Fallback: list all papers without clustering
        for p in papers:
            methods_str = ", ".join(p.methods) if p.methods else "—"
            lines.append(f"- **{p.title}** ({p.year}): {methods_str}")
        lines.append("")
    
    # ================================================================
    # Section 3: Gap Analysis
    # ================================================================
    lines.append("## 3. Gap Analysis")
    lines.append("")
    
    # Primary gaps
    primary_gaps = [g for g in gaps if g.id in plan.primary_gap_ids]
    other_gaps = [g for g in gaps if g.id not in plan.primary_gap_ids]
    
    if primary_gaps:
        lines.append("### Primary Gaps Addressed")
        lines.append("")
        for gap in primary_gaps:
            _render_gap(lines, gap, papers, claims)
    
    # Coverage warnings
    if coverage_report and coverage_report.warnings:
        lines.append("### Coverage Warnings")
        lines.append("")
        for w in coverage_report.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")
    
    # ================================================================
    # Section 4: Experiment Design
    # ================================================================
    lines.append("## 4. Experiment Design")
    lines.append("")
    
    # Extract datasets from data-phase steps
    data_steps = [s for s in steps if s.type == "data"]
    experiment_steps = [s for s in steps if s.type == "experiment"]
    impl_steps = [s for s in steps if s.type == "implementation"]
    
    if data_steps:
        lines.append("### Datasets")
        lines.append("")
        for step in data_steps:
            outputs_str = ", ".join(step.outputs) if step.outputs else "—"
            lines.append(f"- **{step.label}**: {outputs_str}")
        lines.append("")
    
    if impl_steps:
        lines.append("### Model Configurations")
        lines.append("")
        for step in impl_steps:
            lines.append(f"- **{step.label}**")
            if step.description:
                lines.append(f"  - {step.description}")
        lines.append("")
    
    if experiment_steps:
        lines.append("### Experiments & Baselines")
        lines.append("")
        for step in experiment_steps:
            lines.append(f"- **{step.label}**")
            if step.description:
                lines.append(f"  - {step.description}")
            if step.inputs:
                lines.append(f"  - Inputs: {', '.join(step.inputs)}")
            if step.metrics:
                lines.append(f"  - Metrics: {', '.join(step.metrics)}")
        lines.append("")
    
    # Evaluation metrics (from papers)
    all_metrics = set()
    for p in papers:
        all_metrics.update(p.metrics)
    if all_metrics:
        lines.append("### Evaluation Metrics")
        lines.append("")
        for m in sorted(all_metrics):
            lines.append(f"- {m}")
        lines.append("")
    
    # Hardware assumptions (from step assumptions)
    hw_assumptions = set()
    for step in steps:
        for a in step.assumptions:
            if any(kw in a.lower() for kw in ["gpu", "tpu", "hardware", "memory", "storage", "cpu", "ram", "compute"]):
                hw_assumptions.add(a)
    if hw_assumptions:
        lines.append("### Hardware Assumptions")
        lines.append("")
        for a in sorted(hw_assumptions):
            lines.append(f"- {a}")
        lines.append("")
    
    # ================================================================
    # Section 5: Standard Operating Procedure (SOP)
    # ================================================================
    lines.append("## 5. Standard Operating Procedure (SOP)")
    lines.append("")
    
    phases = ["data", "implementation", "experiment", "ablation", "analysis", "writing"]
    for phase in phases:
        phase_steps = [s for s in steps if s.type == phase]
        if phase_steps:
            lines.append(f"### {phase.capitalize()} Phase")
            lines.append("")
            for step in phase_steps:
                status_badge = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}.get(step.status, "⬜")
                lines.append(f"#### {status_badge} {step.label} (`{step.id}`)")
                if step.description:
                    lines.append(f"> {step.description}")
                    lines.append("")
                lines.append(f"| Field | Value |")
                lines.append(f"|-------|-------|")
                lines.append(f"| **Status** | `{step.status}` |")
                lines.append(f"| **Risk Level** | {step.risk_level} |")
                lines.append(f"| **Dependencies** | {', '.join(f'`{d}`' for d in step.dependencies) if step.dependencies else 'None'} |")
                lines.append(f"| **Inputs** | {', '.join(step.inputs) if step.inputs else 'None'} |")
                lines.append(f"| **Outputs** | {', '.join(step.outputs) if step.outputs else 'None'} |")
                if step.metrics:
                    lines.append(f"| **Metrics** | {', '.join(step.metrics)} |")
                
                compute_h = step.estimated_cost.get("compute_hours", 0.0)
                human_h = step.estimated_cost.get("human_hours", 0.0)
                lines.append(f"| **Compute Hours** | {compute_h} |")
                lines.append(f"| **Human Hours** | {human_h} |")
                
                if step.assumptions:
                    lines.append(f"| **Assumptions** | {'; '.join(step.assumptions)} |")
                
                lines.append("")
    
    # ================================================================
    # Section 6: Timeline & Resources
    # ================================================================
    lines.append("## 6. Timeline & Resources")
    lines.append("")
    
    total_compute = plan.resource_summary.get("total_compute_hours", 0.0)
    total_human = plan.resource_summary.get("total_human_hours", 0.0)
    
    lines.append(f"**Total Compute Hours:** {total_compute:.1f}")
    lines.append(f"**Total Human Hours:** {total_human:.1f}")
    if plan.timeline_weeks > 0:
        lines.append(f"**Estimated Duration:** {plan.timeline_weeks} weeks")
    lines.append("")
    
    # Per-phase breakdown
    lines.append("### Per-Phase Cost Breakdown")
    lines.append("")
    lines.append("| Phase | Steps | Compute Hours | Human Hours |")
    lines.append("|-------|-------|---------------|-------------|")
    for phase in phases:
        phase_steps = [s for s in steps if s.type == phase]
        if phase_steps:
            p_compute = sum(s.estimated_cost.get("compute_hours", 0.0) for s in phase_steps)
            p_human = sum(s.estimated_cost.get("human_hours", 0.0) for s in phase_steps)
            lines.append(f"| {phase.capitalize()} | {len(phase_steps)} | {p_compute:.1f} | {p_human:.1f} |")
    lines.append("")
    
    # ================================================================
    # Section 7: Manuscript Outline
    # ================================================================
    lines.append("## 7. Manuscript Outline")
    lines.append("")
    
    writing_steps = [s for s in steps if s.type == "writing"]
    if writing_steps:
        for step in writing_steps:
            lines.append(f"- **{step.label}**")
            if step.description:
                lines.append(f"  - {step.description}")
    else:
        # Default outline
        lines.append("- Introduction (problem statement + gaps)")
        lines.append("- Related Work")
        lines.append("- Methods")
        lines.append("- Experiments")
        lines.append("- Results and Analysis")
        lines.append("- Discussion and Future Work")
        lines.append("- Conclusion")
    lines.append("")
    
    # ================================================================
    # Section 8: Review Summary
    # ================================================================
    lines.append("## 8. Review Summary")
    lines.append("")
    
    if review_result:
        status_str = "✅ PASSED" if review_result.passed else "❌ FAILED"
        lines.append(f"**Status:** {status_str}")
        lines.append("")
        
        if review_result.issues:
            lines.append("### Issues")
            lines.append("")
            for issue in review_result.issues:
                lines.append(f"- ✗ {issue}")
            lines.append("")
        
        if review_result.suggestions:
            lines.append("### Suggestions")
            lines.append("")
            for sug in review_result.suggestions:
                lines.append(f"- ⚠ {sug}")
            lines.append("")
    else:
        lines.append("*No review has been performed yet.*")
        lines.append("")
    
    # ================================================================
    # Section 9: Key Literature References
    # ================================================================
    lines.append("## 9. Key Literature References")
    lines.append("")
    
    if coverage_report and coverage_report.clusters:
        # Group references by domain cluster
        for cluster in coverage_report.clusters:
            if cluster.domain == "uncategorized":
                continue
            cluster_papers = [p for p in papers if p.id in cluster.paper_ids]
            if not cluster_papers:
                continue
            lines.append(f"### {cluster.domain.replace('_', ' ').title()}")
            lines.append("")
            for p in cluster_papers:
                methods_str = f" — Methods: {', '.join(p.methods)}" if p.methods else ""
                datasets_str = f" — Datasets: {', '.join(p.datasets)}" if p.datasets else ""
                lines.append(
                    f"- {p.title} ({p.year}, {p.venue}){methods_str}{datasets_str}\n"
                    f"  {p.url}"
                )
            lines.append("")
        
        # Handle uncategorized papers
        uncategorized = [p for p in papers if p.id in 
                         [pid for c in coverage_report.clusters 
                          if c.domain == "uncategorized" for pid in c.paper_ids]]
        if uncategorized:
            lines.append("### Other")
            lines.append("")
            for p in uncategorized:
                lines.append(f"- {p.title} ({p.year}) — {p.url}")
            lines.append("")
    else:
        # Fallback: flat list
        for paper in papers:
            methods_str = f" — Methods: {', '.join(paper.methods)}" if paper.methods else ""
            datasets_str = f" — Datasets: {', '.join(paper.datasets)}" if paper.datasets else ""
            lines.append(f"- {paper.title} ({paper.year}, {paper.venue}){methods_str}{datasets_str}")
            lines.append(f"  {paper.url}")
        lines.append("")
    
    # ================================================================
    # Section 10: Future Projects (Alternative Directions)
    # ================================================================
    other_gaps = [g for g in gaps if g.id not in plan.primary_gap_ids]
    if other_gaps:
        lines.append("## 10. Future Projects (Alternative Directions)")
        lines.append("")
        lines.append("The following gaps were identified but not selected for the current plan. They represent potential follow-up projects.")
        lines.append("")
        for gap in other_gaps:
            _render_gap(lines, gap, papers, claims)
            
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _render_gap(
    lines: list,
    gap: Gap,
    papers: List[Paper],
    claims: List[Claim],
) -> None:
    """Render a single gap with its related papers, claims, and dimensions."""
    lines.append(f"#### {gap.gap_type.replace('_', ' ').title()}: {gap.description}")
    lines.append("")
    lines.append(f"- **Novelty:** {gap.novelty_estimate}")
    if gap.feasibility_notes:
        lines.append(f"- **Feasibility:** {gap.feasibility_notes}")
    
    # Related papers
    if gap.related_papers:
        related = [p for p in papers if p.id in gap.related_papers]
        if related:
            lines.append("- **Related Papers:**")
            for p in related:
                lines.append(f"  - {p.title} ({p.year})")
    
    # Supporting claims
    if gap.supporting_claims:
        supporting = [c for c in claims if c.id in gap.supporting_claims]
        if supporting:
            lines.append("- **Supporting Claims:**")
            for c in supporting:
                lines.append(f"  - [{c.type}] {c.text}")
    
    # Dimensions table
    if gap.dimensions:
        lines.append("- **Dimensions:**")
        lines.append("")
        lines.append("  | Dimension | Value |")
        lines.append("  |-----------|-------|")
        for dim, val in gap.dimensions.items():
            lines.append(f"  | {dim} | {val} |")
    
    lines.append("")
