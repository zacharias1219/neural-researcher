import hashlib
import uuid
from neuralresearcher.context import AgentContext
from neuralresearcher.state import Claim
from neuralresearcher.llm import generate_structured
from neuralresearcher.errors import SchemaError
from pydantic import BaseModel, Field
from typing import List, Literal


class PaperMetadata(BaseModel):
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    explicit_future_work: List[str] = Field(default_factory=list)

class ReadingClaim(BaseModel):
    type: Literal["result", "method", "assumption", "limitation", "future_work"]
    text: str
    section: str
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

class ReadingResponse(BaseModel):
    paper_metadata: PaperMetadata
    claims: List[ReadingClaim] = Field(default_factory=list)

def run_reading(context: AgentContext) -> None:
    papers = context.store.load_papers()
    if not papers:
        return
        
    # schema = ReadingResponse.model_json_schema()
    system_prompt = (
        "You are a reading agent. For each paper, extract structured metadata and individual claims.\n"
        "Extract at least one claim of each type if the information is available. "
        "Be specific in methods and datasets — use exact names, not generic descriptions."
    )
    
    all_claims = []
    updated_papers = []
    
    for paper in papers:
        full_text_context = f"Abstract: {paper.abstract}"
            
        user_prompt = (
            f"Extract structured metadata and claims from this paper:\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:5])}\n"
            f"Year: {paper.year}\n"
            f"Text content:\n{full_text_context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        data = generate_structured(
            messages=messages,
            output_model=ReadingResponse,
            config=context.config,
            agent_name="reading",
            store=context.store,
            task_id=context.task_id
        )
        
        try:
            # --- Enrich paper metadata ---
            meta = data.paper_metadata
            if meta.methods:
                paper.methods = list(dict.fromkeys(paper.methods + meta.methods))
            if meta.datasets:
                paper.datasets = list(dict.fromkeys(paper.datasets + meta.datasets))
            if meta.metrics:
                paper.metrics = list(dict.fromkeys(paper.metrics + meta.metrics))
            if meta.limitations:
                paper.limitations = list(dict.fromkeys(paper.limitations + meta.limitations))
            if meta.explicit_future_work:
                paper.explicit_future_work = list(dict.fromkeys(paper.explicit_future_work + meta.explicit_future_work))
            
            # --- Extract claims ---
            for idx, c in enumerate(data.claims):
                if context.config.seed is not None:
                    claim_id = f"claim_{hashlib.sha1(f'{paper.id}_{context.config.seed}_{idx}'.encode()).hexdigest()[:8]}"
                else:
                    claim_id = f"claim_{uuid.uuid4().hex[:8]}"
                
                c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
                c_dict['id'] = claim_id
                c_dict['paper_id'] = paper.id
                c_dict['evidence_ref'] = paper.url
                
                if paper.content_level != "FULL_TEXT":
                    c_dict['section'] = 'abstract'
                
                c_dict.setdefault('type', 'result')
                c_dict.setdefault('datasets', [])
                c_dict.setdefault('metrics', [])
                all_claims.append(Claim(**c_dict))

                
        except Exception as e:
            raise SchemaError(f"Failed to parse reading output for '{paper.title}': {e}")
        
        updated_papers.append(paper)
    
    # Save enriched papers back to store
    context.store.update_papers(updated_papers)
    context.store.save_claims(all_claims)
