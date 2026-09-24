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
        "You are a reading agent. For each paper, extract TWO things:\n"
        "1. Structured paper metadata\n"
        "2. Individual claims\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "paper_metadata": {\n'
        '    "methods": ["list of methods/techniques used"],\n'
        '    "datasets": ["list of datasets mentioned"],\n'
        '    "metrics": ["list of evaluation metrics used"],\n'
        '    "limitations": ["list of stated limitations"],\n'
        '    "explicit_future_work": ["list of future work directions"]\n'
        "  },\n"
        '  "claims": [\n'
        "    {\n"
        '      "type": "result|method|assumption|limitation|future_work",\n'
        '      "text": "the claim text",\n'
        '      "section": "which section this comes from",\n'
        '      "datasets": ["datasets relevant to this claim"],\n'
        '      "metrics": ["metrics relevant to this claim"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Extract at least one claim of each type if the information is available. "
        "Be specific in methods and datasets — use exact names, not generic descriptions."
    )
    
    all_claims = []
    updated_papers = []
    
    for paper in papers:
        user_prompt = (
            f"Extract structured metadata and claims from this paper:\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:5])}\n"
            f"Year: {paper.year}\n"
            f"Abstract: {paper.abstract}"
        )
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
            agent_name="reading"
        )
        
        try:
            data = json.loads(response.content)
            
            # --- Enrich paper metadata ---
            meta = data.get("paper_metadata", {})
            if meta.get("methods"):
                paper.methods = meta["methods"]
            if meta.get("datasets"):
                paper.datasets = meta["datasets"]
            if meta.get("metrics"):
                paper.metrics = meta["metrics"]
            if meta.get("limitations"):
                paper.limitations = meta["limitations"]
            if meta.get("explicit_future_work"):
                paper.explicit_future_work = meta["explicit_future_work"]
            
            # --- Extract claims ---
            for c in data.get("claims", []):
                c['id'] = f"claim_{uuid.uuid4().hex[:8]}"
                c['paper_id'] = paper.id
                c['evidence_ref'] = paper.url
                c.setdefault('section', 'abstract')
                c.setdefault('type', 'result')
                c.setdefault('datasets', [])
                c.setdefault('metrics', [])
                all_claims.append(Claim(**c))
                
        except Exception as e:
            from neuralresearcher.logging import log_error
            log_error(f"Error parsing reading output for '{paper.title}': {e}")
        
        updated_papers.append(paper)
    
    # Save enriched papers back to store
    orchestrator.store.save_papers(updated_papers)
    orchestrator.store.save_claims(all_claims)
