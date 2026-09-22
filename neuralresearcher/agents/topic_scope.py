import json
import uuid
from typing import Any

from neuralresearcher.state import TopicSpec
from neuralresearcher.llm import call_llm

def run_topic_scope(orchestrator: Any) -> None:
    system_prompt = (
        "You are a specialized agent that takes a raw research topic and outputs a refined TopicSpec. "
        "Return a JSON object strictly matching this schema:\n"
        "{\n"
        "  \"domain\": \"string (e.g., ML/CS)\",\n"
        "  \"subfields\": [\"string\"],\n"
        "  \"keywords\": [\"string\"]\n"
        "}"
    )
    user_prompt = f"Raw Topic: {orchestrator.topic}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # We pass the schema of TopicSpec as response format
    # For a real Groq call we might need a precise JSON schema object here
    response_format = {"type": "json_object"} 
    
    response = call_llm(
        config=orchestrator.config,
        messages=messages,
        response_format=response_format
    )
    
    try:
        data = json.loads(response.content)
        data['id'] = f"topic_{uuid.uuid4().hex[:8]}"
        data['raw_topic'] = orchestrator.topic
        topic_spec = TopicSpec(**data)
        orchestrator.store.save_topic_spec(topic_spec)
    except Exception as e:
        from neuralresearcher.errors import SchemaError
        raise SchemaError(f"Failed to parse TopicSpec: {str(e)}")
