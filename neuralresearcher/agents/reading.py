from typing import Any
import json
from neuralresearcher.state import Paper, Claim
from neuralresearcher.llm import call_llm
import uuid

def run_reading(orchestrator: Any) -> None:
    papers = orchestrator.store.load_papers()
    if not papers:
        return
        
    system_prompt = (
        "You are a reading agent. Extract claims (result, method, assumption, limitation, future_work) "
        "from the provided paper abstracts. Return a JSON object with a 'claims' list. "
        "Each claim must strictly have this structure: "
        '{"type": "result|method|assumption|limitation|future_work", "text": "string", "section": "string"}'
    )
    
    all_claims = []
    for paper in papers:
        user_prompt = f"Extract claims from this paper:\nTitle: {paper.title}\nAbstract: {paper.abstract}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = call_llm(
            config=orchestrator.config,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            data = json.loads(response.content)
            for c in data.get("claims", []):
                c['id'] = f"claim_{uuid.uuid4().hex[:8]}"
                c['paper_id'] = paper.id
                c['evidence_ref'] = paper.url
                c.setdefault('section', 'abstract')
                c.setdefault('type', 'result')
                all_claims.append(Claim(**c))
        except Exception as e:
            from neuralresearcher.logging import log_error
            log_error(f"Error parsing claims: {e}")
            
    orchestrator.store.save_claims(all_claims)
