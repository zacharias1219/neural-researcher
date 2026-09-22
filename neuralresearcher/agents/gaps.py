from typing import Any
import json
import uuid
from neuralresearcher.state import Gap
from neuralresearcher.llm import call_llm

def run_gaps(orchestrator: Any) -> None:
    claims = orchestrator.store.load_claims()
    if not claims:
        return
        
    system_prompt = (
        "You are a gap identification agent. Identify research gaps from the provided claims. "
        "Return a JSON object with a 'gaps' list. "
        "Each gap must strictly have this structure: "
        '{"description": "string", "gap_type": "unexplored_axis|limitation|contradiction|missing_combination", "novelty_estimate": "low|medium|high", "feasibility_notes": "string"}'
    )
    
    claims_text = "\n".join([f"- {c.text} (from {c.paper_id})" for c in claims[:10]]) # limit for context
    user_prompt = f"Claims:\n{claims_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        config=orchestrator.config,
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    all_gaps = []
    try:
        data = json.loads(response.content)
        for g in data.get("gaps", []):
            g['id'] = f"gap_{uuid.uuid4().hex[:8]}"
            g['gap_type'] = g.get('gap_type', 'unexplored_axis')
            g['novelty_estimate'] = g.get('novelty_estimate', 'medium')
            g.setdefault('description', 'Unspecified gap')
            all_gaps.append(Gap(**g))
    except Exception as e:
        from neuralresearcher.logging import log_error
        log_error(f"Error parsing gaps: {e}")
        
    orchestrator.store.save_gaps(all_gaps)
