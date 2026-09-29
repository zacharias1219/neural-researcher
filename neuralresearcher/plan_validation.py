from typing import List, Literal, Optional, Tuple, Dict
from pydantic import BaseModel
from neuralresearcher.state import ResearchPlan, PlanStep

class ValidationIssue(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    step_id: Optional[str] = None
    repairable: bool

def normalize_plan(plan: ResearchPlan, steps: List[PlanStep]) -> Tuple[ResearchPlan, List[PlanStep]]:
    """Mechanically normalizes plan and steps."""
    # Deduplicate IDs, set missing statuses, fix formatting
    seen_ids = set()
    normalized_steps = []
    
    for i, step in enumerate(steps):
        # Set missing default status
        if not step.status:
            step.status = "pending"
            
        # Deduplicate step IDs
        if not step.id or step.id in seen_ids:
            step.id = f"step_{i+1}"
            
        seen_ids.add(step.id)
        
        # Format dependencies
        step.dependencies = [dep.strip() for dep in step.dependencies if dep.strip()]
        
        normalized_steps.append(step)
    
    return plan, normalized_steps

def validate_plan(plan: ResearchPlan, steps: List[PlanStep]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    
    step_map = {step.id: step for step in steps}
    
    # Check for duplicate step IDs
    if len(step_map) != len(steps):
        issues.append(ValidationIssue(
            code="DUPLICATE_ID",
            severity="error",
            message="Duplicate step IDs found.",
            repairable=True
        ))
        
    producers: Dict[str, str] = {}
    consumers: Dict[str, List[str]] = {}
    
    for step in steps:
        for out in step.outputs:
            if out in producers:
                issues.append(ValidationIssue(
                    code="DUPLICATE_OUTPUT",
                    severity="error",
                    message=f"Output {out} is produced by multiple steps.",
                    step_id=step.id,
                    repairable=False
                ))
            producers[out] = step.id
            
        for inp in step.inputs:
            if inp not in consumers:
                consumers[inp] = []
            consumers[inp].append(step.id)

    # Dependency validation
    for step in steps:
        for dep in step.dependencies:
            if dep == step.id:
                issues.append(ValidationIssue(
                    code="SELF_DEPENDENCY",
                    severity="error",
                    message="Step depends on itself.",
                    step_id=step.id,
                    repairable=True
                ))
            elif dep not in step_map:
                issues.append(ValidationIssue(
                    code="MISSING_DEPENDENCY",
                    severity="error",
                    message=f"Step depends on missing ID {dep}.",
                    step_id=step.id,
                    repairable=True
                ))

        for inp in step.inputs:
            # required inputs have an upstream producer unless explicitly marked external
            if inp not in producers and not inp.startswith("external:"):
                issues.append(ValidationIssue(
                    code="MISSING_INPUT_PRODUCER",
                    severity="error",
                    message=f"Input {inp} has no upstream producer and is not external.",
                    step_id=step.id,
                    repairable=False
                ))

    # Cycle detection
    visited = set()
    path = set()
    def visit(node: str) -> bool:
        if node in path:
            return True # Cycle
        if node in visited:
            return False
        
        visited.add(node)
        path.add(node)
        
        if node in step_map:
            for dep in step_map[node].dependencies:
                if visit(dep):
                    return True
        path.remove(node)
        return False
        
    for step in steps:
        if visit(step.id):
            issues.append(ValidationIssue(
                code="DEPENDENCY_CYCLE",
                severity="error",
                message="Cycle detected in dependencies.",
                step_id=step.id,
                repairable=False
            ))
            break # only report once
            
    # Substantive checks
    has_experiment = any(s.type == "experiment" for s in steps)
    has_analysis = any(s.type == "analysis" for s in steps)
    has_writing = any(s.type == "writing" for s in steps)
    
    if not has_experiment:
        issues.append(ValidationIssue(
            code="MISSING_EXPERIMENT",
            severity="error",
            message="Plan lacks substantive experimental steps.",
            repairable=False
        ))
        
    # Analysis consumes experiment results
    analysis_consumes_exp = False
    writing_consumes_analysis = False
    
    exp_outputs = set(out for s in steps if s.type in ("experiment", "ablation") for out in s.outputs)
    analysis_outputs = set(out for s in steps if s.type == "analysis" for out in s.outputs)
    
    for s in steps:
        if s.type == "analysis":
            if any(inp in exp_outputs for inp in s.inputs) or any(step_map[d].type in ("experiment", "ablation") for d in s.dependencies if d in step_map):
                analysis_consumes_exp = True
        elif s.type == "writing":
            if any(inp in analysis_outputs for inp in s.inputs) or any(step_map[d].type == "analysis" for d in s.dependencies if d in step_map):
                writing_consumes_analysis = True
                
    if has_experiment and has_analysis and not analysis_consumes_exp:
        issues.append(ValidationIssue(
            code="DISCONNECTED_ANALYSIS",
            severity="error",
            message="Analysis step does not consume experimental results.",
            repairable=False
        ))
        
    if has_analysis and has_writing and not writing_consumes_analysis:
        issues.append(ValidationIssue(
            code="DISCONNECTED_WRITING",
            severity="warning",
            message="Writing step does not consume analysis outputs.",
            repairable=False
        ))

    # Orphaned intermediate artifacts
    for out, prod_id in producers.items():
        if out not in consumers and step_map[prod_id].type != "writing":
            issues.append(ValidationIssue(
                code="ORPHANED_ARTIFACT",
                severity="warning",
                message=f"Output {out} is never consumed by downstream steps.",
                step_id=prod_id,
                repairable=False
            ))

    return issues
