import hashlib
import uuid
from typing import List, Literal

from pydantic import BaseModel

from neuralresearcher.context import AgentContext
from neuralresearcher.errors import SchemaError
from neuralresearcher.llm import generate_structured
from neuralresearcher.state import Direction


class DirectionOutput(BaseModel):
    primary_gap_id: str
    hypothesis: str
    justification: str
    expected_contribution_type: str
    novelty_assessment: Literal["low", "medium", "high"]

class DirectionsResponse(BaseModel):
    directions: List[DirectionOutput]


def run_directions(context: AgentContext) -> None:
    gaps = context.store.load_gaps()
    if not gaps:
        return

    system_prompt = (
        "You are a research directions agent. Based on the gaps, propose research directions. "
    )

    gaps_text = "\n".join([f"- {g.description} (id: {g.id})" for g in gaps[:5]])
    user_prompt = f"Gaps:\n{gaps_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    data = generate_structured(
        messages=messages,
        output_model=DirectionsResponse,
        config=context.config,
        agent_name="directions",
        store=context.store,
        task_id=context.task_id
    )

    all_directions = []
    try:
        raw_dirs = data.directions
        if not raw_dirs:
            raise SchemaError("Directions response contained empty 'directions' list.")

        for idx, d_obj in enumerate(raw_dirs):
            d = d_obj.model_dump()
            if context.config.seed is not None:
                d['id'] = f"dir_{hashlib.sha1(f'{context.topic}_{context.config.seed}_dir_{idx}'.encode()).hexdigest()[:8]}"
            else:
                d['id'] = f"dir_{uuid.uuid4().hex[:8]}"

            if 'primary_gap_id' not in d:
                d['primary_gap_id'] = gaps[0].id if gaps else "unknown"
            d.setdefault('hypothesis', 'Unspecified hypothesis')
            d.setdefault('justification', 'Unspecified justification')
            d.setdefault('expected_contribution_type', 'methodology')
            d['novelty_assessment'] = d.get('novelty_assessment', 'medium')
            all_directions.append(Direction(**d))
    except Exception as e:
        raise SchemaError(f"Failed to parse directions: {e}")

    context.store.save_directions(all_directions)
