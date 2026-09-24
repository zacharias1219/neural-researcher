from typing import Any
import json
import uuid
from neuralresearcher.state import Direction
from neuralresearcher.llm import call_llm

def run_directions(orchestrator: Any) -> None:
    gaps = orchestrator.store.load_gaps()
    if not gaps:
        return
        
    system_prompt = (
        "You are a research directions agent. Based on the gaps, propose research directions. "
        "Return a JSON object with a 'directions' list. "
        "Each direction must strictly have this structure: "
        '{"primary_gap_id": "string", "hypothesis": "string", "justification": "string", "expected_contribution_type": "string", "novelty_assessment": "low|medium|high"}'
    )
    
    gaps_text = "\n".join([f"- {g.description} (id: {g.id})" for g in gaps[:5]])
    user_prompt = f"Gaps:\n{gaps_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        config=orchestrator.config,
        messages=messages,
        response_format={"type": "json_object"},
        store=orchestrator.store,
        task_id=orchestrator.task_id,
        agent_name="directions"
    )
    
    all_directions = []
    try:
        data = json.loads(response.content)
        for d in data.get("directions", []):
            d['id'] = f"dir_{uuid.uuid4().hex[:8]}"
            if 'primary_gap_id' not in d:
                d['primary_gap_id'] = gaps[0].id if gaps else "unknown"
            d.setdefault('hypothesis', 'Unspecified hypothesis')
            d.setdefault('justification', 'Unspecified justification')
            d.setdefault('expected_contribution_type', 'methodology')
            d['novelty_assessment'] = d.get('novelty_assessment', 'medium')
            all_directions.append(Direction(**d))
    except Exception as e:
        from neuralresearcher.logging import log_error
        log_error(f"Error parsing directions: {e}")
        
    orchestrator.store.save_directions(all_directions)
