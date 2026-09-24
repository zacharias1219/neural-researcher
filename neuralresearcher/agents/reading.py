import hashlib
import json
import uuid
from neuralresearcher.context import AgentContext
from neuralresearcher.state import Paper, Claim
from neuralresearcher.llm import call_llm
from neuralresearcher.errors import SchemaError
from neuralresearcher.logging import log_error


def run_reading(context: AgentContext) -> None:
    papers = context.store.load_papers()
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
            config=context.config,
            messages=messages,
            response_format={"type": "json_object"},
            store=context.store,
            task_id=context.task_id,
            agent_name="reading"
        )
        
        try:
            data = json.loads(response.content)
            
            # --- Enrich paper metadata ---
            meta = data.get("paper_metadata", {})
            if meta.get("methods"):
                paper.methods = list(dict.fromkeys(paper.methods + meta["methods"]))
            if meta.get("datasets"):
                paper.datasets = list(dict.fromkeys(paper.datasets + meta["datasets"]))
            if meta.get("metrics"):
                paper.metrics = list(dict.fromkeys(paper.metrics + meta["metrics"]))
            if meta.get("limitations"):
                paper.limitations = list(dict.fromkeys(paper.limitations + meta["limitations"]))
            if meta.get("explicit_future_work"):
                paper.explicit_future_work = list(dict.fromkeys(paper.explicit_future_work + meta["explicit_future_work"]))
            
            # --- Extract claims ---
            for idx, c in enumerate(data.get("claims", [])):
                if context.config.seed is not None:
                    claim_id = f"claim_{hashlib.sha1(f'{paper.id}_{context.config.seed}_{idx}'.encode()).hexdigest()[:8]}"
                else:
                    claim_id = f"claim_{uuid.uuid4().hex[:8]}"
                c['id'] = claim_id
                c['paper_id'] = paper.id
                c['evidence_ref'] = paper.url
                c.setdefault('section', 'abstract')
                c.setdefault('type', 'result')
                c.setdefault('datasets', [])
                c.setdefault('metrics', [])
                all_claims.append(Claim(**c))

                
        except Exception as e:
            raise SchemaError(f"Failed to parse reading output for '{paper.title}': {e}")
        
        updated_papers.append(paper)
    
    # Save enriched papers back to store
    context.store.update_papers(updated_papers)
    context.store.save_claims(all_claims)
