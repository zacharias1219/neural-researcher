import hashlib
import uuid
from typing import Dict, List

from pydantic import BaseModel, Field

from neuralresearcher.context import AgentContext
from neuralresearcher.llm import generate_structured
from neuralresearcher.state import TopicDomain, TopicSpec


class TopicSpecResponse(BaseModel):
    domain: TopicDomain
    subfields: List[str]
    time_window: Dict[str, int] = Field(
        default_factory=lambda: {
            "start_year": 2000,
            "end_year": __import__('datetime').datetime.now().year,
        }
    )
    scope_constraints: Dict[str, str] = Field(default_factory=dict)
    keywords: List[str]

def run_topic_scope(context: AgentContext) -> None:
    manifest = context.store.load_manifest()
    tw_start = manifest.get("time_window_start")
    tw_end = manifest.get("time_window_end")

    system_prompt = "You are a specialized agent that takes a raw research topic and outputs a refined TopicSpec."
    if tw_start or tw_end:
        system_prompt += f"\nConstraint: The user requested literature bounded between years {tw_start or 'Any'} and {tw_end or 'Any'}. Ensure the output time_window strictly respects this boundary."

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

