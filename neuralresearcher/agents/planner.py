from typing import Any
import json
import uuid
from neuralresearcher.state import ResearchPlan, PlanStep
from neuralresearcher.llm import call_llm

def run_planner(orchestrator: Any) -> None:
    directions = orchestrator.store.load_directions()
    if not directions:
        return
        
    direction = directions[0] # Pick the first one for simplicity
    
    system_prompt = (
        "You are a planner agent. Create a ResearchPlan and PlanSteps for the given direction. "
        "Return a JSON object with 'plan' and 'steps'. "
        "'plan' must have: {'hypothesis': 'string', 'expected_contribution': 'string'} "
        "'steps' must be a list where each step has: "
        '{"label": "string", "type": "data|implementation|experiment|ablation|analysis|writing", "risk_level": "low|medium|high"}'
    )
    
    user_prompt = f"Direction Hypothesis: {direction.hypothesis}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        config=orchestrator.config,
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    plan = None
    try:
        data = json.loads(response.content)
        p_data = data.get("plan", {})
        p_data['id'] = f"plan_{uuid.uuid4().hex[:8]}"
        p_data['topic_spec_id'] = orchestrator.store.load_topic_spec().id
        p_data['hypothesis'] = p_data.get('hypothesis', direction.hypothesis)
        p_data['expected_contribution'] = p_data.get('expected_contribution', direction.expected_contribution_type)
        plan = ResearchPlan(**p_data)
        
        s_data = data.get("steps", [])
        parsed_steps = []
        for s in s_data:
            s['id'] = f"step_{uuid.uuid4().hex[:8]}"
            s['plan_id'] = plan.id
            s['status'] = 'pending'
            s.setdefault('label', 'Unspecified step')
            s.setdefault('type', 'experiment')
            s['risk_level'] = s.get('risk_level', 'medium')
            parsed_steps.append(PlanStep(**s))
            
        orchestrator.store.save_plan(plan, parsed_steps)
    except Exception as e:
        from neuralresearcher.logging import log_error
        log_error(f"Failed to parse plan: {e}")
        return
