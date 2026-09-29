import hashlib
import uuid
from datetime import datetime, timezone

from neuralresearcher.context import AgentContext
from neuralresearcher.state import ReviewResult
from typing import List
from neuralresearcher.logging import log_info
from neuralresearcher.plan_validation import validate_plan


def run_reviewer(context: AgentContext) -> None:
    plan, steps = context.store.load_plan()
    if not plan:
        log_info("Reviewer: no plan found to review.")
        return
    
    issues = []
    suggestions = []
    
    validation_issues = validate_plan(plan, steps)
    for vi in validation_issues:
        if vi.severity == "error":
            issues.append(vi.message)
        else:
            suggestions.append(vi.message)
            
    # Claim consistency review step
    claims = context.store.load_claims()
    if claims:
        from neuralresearcher.llm import generate_structured
        from pydantic import BaseModel, Field

        
        class ReviewVerificationResponse(BaseModel):
            issues: List[str] = Field(default_factory=list)
            
        claims_text = "\n".join([f"- {c.id}: {c.text} (Source: {c.evidence_ref})" for c in claims[:10]]) # limit to 10 for review
        system_prompt = (
            "You are a consistency review agent. Perform a claim_consistency_review over these claims. "
            "Return a JSON object with 'issues' (list of strings) for any logical inconsistencies. Do not assign VERIFIED/SUPPORTED status without a stored evidence excerpt."
        )
        data = generate_structured(
            config=context.config,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Claims to verify:\n{claims_text}"}
            ],
            output_model=ReviewVerificationResponse,
            store=context.store,
            task_id=context.task_id,
            agent_name="claim_consistency_review"
        )
        try:
            issues.extend(data.issues)
        except Exception:
            pass
            
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


