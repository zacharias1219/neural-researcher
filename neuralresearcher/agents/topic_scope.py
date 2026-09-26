import hashlib
import json
import uuid
from neuralresearcher.context import AgentContext
from neuralresearcher.state import TopicSpec
from neuralresearcher.llm import call_llm
from neuralresearcher.errors import SchemaError


def run_topic_scope(context: AgentContext) -> None:
    system_prompt = (
        "You are a specialized agent that takes a raw research topic and outputs a refined TopicSpec. "
        "Return a JSON object strictly matching this schema:\n"
        "{\n"
        "  \"domain\": \"string (e.g., ML/CS)\",\n"
        "  \"subfields\": [\"string\"],\n"
        "  \"keywords\": [\"string\"]\n"
        "}"
    )
    user_prompt = f"Raw Topic: {context.topic}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_format = {"type": "json_object"} 
    
    response = call_llm(
        config=context.config,
        messages=messages,
        response_format=response_format,
        store=context.store,
        task_id=context.task_id,
        agent_name="topic_scope"
    )
    
    try:
        data = json.loads(response.content)
        if context.config.seed is not None:
            data['id'] = f"topic_{hashlib.sha1(f'{context.topic}_{context.config.seed}'.encode()).hexdigest()[:8]}"
        else:
            data['id'] = f"topic_{uuid.uuid4().hex[:8]}"
        data['raw_topic'] = context.topic
        topic_spec = TopicSpec(**data)
        context.store.save_topic_spec(topic_spec)
    except Exception as e:
        raise SchemaError(f"Failed to parse TopicSpec: {str(e)}")

