import hashlib
import uuid

from neuralresearcher.context import AgentContext
from neuralresearcher.state import Gap
from neuralresearcher.llm import generate_structured
from neuralresearcher.errors import SchemaError
from neuralresearcher.logging import log_info
from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class GapOutput(BaseModel):
    description: str
    gap_type: Literal["unexplored_axis", "limitation", "contradiction", "missing_combination"]
    novelty_estimate: Literal["low", "medium", "high"]
    feasibility_notes: str = ""
    related_paper_ids: List[str] = Field(default_factory=list)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    dimensions: Dict[str, str] = Field(default_factory=dict)

class GapsResponse(BaseModel):
    gaps: List[GapOutput]


def run_gaps(context: AgentContext) -> None:
    claims = context.store.load_claims()
    papers = context.store.load_papers()
    if not claims:
        return
    
    # Build a paper summary block so the LLM can reference real IDs
    paper_summaries = []
    for p in papers:
        methods_str = ", ".join(p.methods) if p.methods else "not specified"
        datasets_str = ", ".join(p.datasets) if p.datasets else "not specified"
        limitations_str = "; ".join(p.limitations) if p.limitations else "none stated"
        paper_summaries.append(
            f"- Paper ID: {p.id} | Title: {p.title} | Methods: {methods_str} | "
            f"Datasets: {datasets_str} | Limitations: {limitations_str}"
        )
    papers_block = "\n".join(paper_summaries)
    
    # Build a claim block with IDs
    claim_summaries = []
    for c in claims[:20]:  # limit for context window
        claim_summaries.append(
            f"- Claim ID: {c.id} | Paper: {c.paper_id} | Type: {c.type} | Text: {c.text}"
        )
    claims_block = "\n".join(claim_summaries)
    
    # Collect valid IDs for post-processing validation
    valid_paper_ids = {p.id for p in papers}
    valid_claim_ids = {c.id for c in claims}
        
    system_prompt = (
        "You are a gap identification agent. Identify research gaps by analyzing the claims and papers provided.\n"
        "IMPORTANT: Only use paper IDs and claim IDs that appear in the provided data. "
        "Identify at least 3 gaps. Focus on gaps that represent genuine research opportunities."
    )
    
    user_prompt = f"Papers:\n{papers_block}\n\nClaims:\n{claims_block}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = generate_structured(
        messages=messages,
        output_model=GapsResponse,
        config=context.config,
        agent_name="gaps",
        store=context.store,
        task_id=context.task_id
    )
    
    all_gaps = []
    try:
        raw_gaps = data.gaps
        if not raw_gaps:
            raise SchemaError("Gaps response contained empty 'gaps' list.")
            
        for idx, g_obj in enumerate(raw_gaps):
            g = g_obj.model_dump()
            if context.config.seed is not None:
                g['id'] = f"gap_{hashlib.sha1(f'{context.topic}_{context.config.seed}_gap_{idx}'.encode()).hexdigest()[:8]}"
            else:
                g['id'] = f"gap_{uuid.uuid4().hex[:8]}"
                
            g['gap_type'] = g.get('gap_type', 'unexplored_axis')
            g['novelty_estimate'] = g.get('novelty_estimate', 'medium')
            g.setdefault('description', 'Unspecified gap')
            g.setdefault('feasibility_notes', '')
            
            # Validate and set related_papers (filter out hallucinated IDs)
            raw_paper_ids = g.pop('related_paper_ids', [])
            g['related_papers'] = [pid for pid in raw_paper_ids if pid in valid_paper_ids]
            if len(g['related_papers']) < len(raw_paper_ids):
                dropped = len(raw_paper_ids) - len(g['related_papers'])
                log_info(f"Gap '{g['description'][:40]}...': dropped {dropped} invalid paper ID(s)")
            
            # Validate and set supporting_claims (filter out hallucinated IDs)
            raw_claim_ids = g.pop('supporting_claim_ids', [])
            g['supporting_claims'] = [cid for cid in raw_claim_ids if cid in valid_claim_ids]
            if len(g['supporting_claims']) < len(raw_claim_ids):
                dropped = len(raw_claim_ids) - len(g['supporting_claims'])
                log_info(f"Gap '{g['description'][:40]}...': dropped {dropped} invalid claim ID(s)")
            
            g.setdefault('dimensions', {})
            all_gaps.append(Gap(**g))
            
    except Exception as e:
        raise SchemaError(f"Failed to parse gaps: {e}")
        
    context.store.save_gaps(all_gaps)
