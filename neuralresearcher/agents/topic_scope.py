import hashlib
import uuid
from neuralresearcher.context import AgentContext
from neuralresearcher.state import TopicSpec, TopicDomain
from neuralresearcher.llm import generate_structured
from pydantic import BaseModel, Field
from typing import List, Dict


class TopicSpecResponse(BaseModel):
    domain: TopicDomain
    subfields: List[str]
    time_window: Dict[str, int] = Field(default_factory=lambda: {"start_year": 2000, "end_year": 2024})
    scope_constraints: Dict[str, str] = Field(default_factory=dict)
    keywords: List[str]

def run_topic_scope(context: AgentContext) -> None:
    system_prompt = "You are a specialized agent that takes a raw research topic and outputs a refined TopicSpec."
    user_prompt = f"Raw Topic: {context.topic}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = generate_structured(
        messages=messages,
        output_model=TopicSpecResponse,
        config=context.config,
        agent_name="topic_scope",
        store=context.store,
        task_id=context.task_id
    )
    
    if context.config.seed is not None:
        tid = f"topic_{hashlib.sha1(f'{context.topic}_{context.config.seed}'.encode()).hexdigest()[:8]}"
    else:
        tid = f"topic_{uuid.uuid4().hex[:8]}"
        
    topic_spec = TopicSpec(
        id=tid,
        raw_topic=context.topic,
        domain=data.domain,
        subfields=data.subfields,
        time_window=data.time_window,
        scope_constraints=data.scope_constraints,
        keywords=data.keywords
    )
    context.store.save_topic_spec(topic_spec)

