import hashlib
import uuid
from datetime import datetime, timezone

from neuralresearcher.context import AgentContext
from neuralresearcher.state import ReviewResult
from neuralresearcher.logging import log_info
from neuralresearcher.errors import WorkflowError


def run_reviewer(context: AgentContext) -> None:
    plan, steps = context.store.load_plan()
    if not plan:
        log_info("Reviewer: no plan found to review.")
        return
    
    issues = []
    suggestions = []
    
    # --- Check 1: primary_gap_ids must be non-empty ---
    if not plan.primary_gap_ids:
        issues.append("Plan has no primary_gap_ids — it is not connected to any identified gap.")
    
    # --- Check 2: must have at least one experiment step ---
    step_types = [s.type for s in steps]
    if "experiment" not in step_types:
        issues.append("No steps of type 'experiment' — the plan has no experimental evaluation.")
    
    # --- Check 3: must have at least one ablation step ---
    if "ablation" not in step_types:
        issues.append("No steps of type 'ablation' — ablation studies are missing.")
    
    # --- Check 4: experiment/data steps must have inputs and outputs ---
    for step in steps:
        if step.type in ("experiment", "data"):
            if not step.inputs and step.type != "data":
                issues.append(
                    f"Step '{step.label}' ({step.id}) of type '{step.type}' has no inputs."
                )
            if not step.outputs:
                issues.append(
                    f"Step '{step.label}' ({step.id}) of type '{step.type}' has no outputs."
                )
    
    # --- Check 5: experiment steps must have non-zero compute_hours ---
    for step in steps:
        if step.type == "experiment":
            compute = step.estimated_cost.get("compute_hours", 0.0)
            if compute <= 0.0:
                issues.append(
                    f"Experiment step '{step.label}' ({step.id}) has compute_hours=0 — "
                    "cost estimate is missing."
                )
    
    # --- Check 6: timeline should be specified ---
    if plan.timeline_weeks <= 0:
        suggestions.append(
            "Plan timeline_weeks is 0 — consider specifying an estimated timeline."
        )
    
    # --- Check 7: look for baselines in step labels/descriptions ---
    baseline_keywords = ["baseline", "compare", "comparison", "benchmark", "vs", "versus"]
    has_baseline_mention = False
    for step in steps:
        searchable = (step.label + " " + step.description).lower()
        if any(kw in searchable for kw in baseline_keywords):
            has_baseline_mention = True
            break
    if not has_baseline_mention:
        suggestions.append(
            "No step mentions baselines or comparisons — ensure at least one baseline "
            "per domain is planned."
        )
    
    # --- Check 8: dependency validation (no circular / unresolvable refs) ---
    step_ids = {s.id for s in steps}
    for step in steps:
        for dep in step.dependencies:
            if dep not in step_ids:
                issues.append(
                    f"Step '{step.label}' ({step.id}) depends on '{dep}' which does not exist."
                )
    
    # Simple cycle detection via topological sort attempt
    adj = {s.id: set(s.dependencies) for s in steps}
    visited = set()
    in_stack = set()
    has_cycle = False
    
    def _dfs(node):
        nonlocal has_cycle
        if node in in_stack:
            has_cycle = True
            return
        if node in visited:
            return
        in_stack.add(node)
        for dep in adj.get(node, []):
            if dep in step_ids:
                _dfs(dep)
        in_stack.discard(node)
        visited.add(node)
    
    for sid in step_ids:
        _dfs(sid)
    
    if has_cycle:
        issues.append("Circular dependency detected among plan steps.")
    
    # --- Build result ---
    passed = len(issues) == 0
    if context.config.seed is not None:
        review_id = f"review_{hashlib.sha1(f'{context.topic}_{context.config.seed}_review'.encode()).hexdigest()[:8]}"
        timestamp = "2024-01-01T00:00:00+00:00"
    else:
        review_id = f"review_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

    result = ReviewResult(
        id=review_id,
        passed=passed,
        issues=issues,
        suggestions=suggestions,
        timestamp=timestamp
    )
    
    context.store.save_review_result(result)
    
    # Log summary
    if passed:
        log_info(f"Reviewer: plan PASSED with {len(suggestions)} suggestion(s).")
    else:
        log_info(f"Reviewer: plan has {len(issues)} issue(s) and {len(suggestions)} suggestion(s).")
        for issue in issues:
            log_info(f"  [x] {issue}")
    for sug in suggestions:
        log_info(f"  [!] {sug}")


