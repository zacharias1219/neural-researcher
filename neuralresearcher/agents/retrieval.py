import json
from neuralresearcher.context import AgentContext
from neuralresearcher.llm import call_llm
from neuralresearcher.tools import TOOL_SCHEMAS, execute_tool_call
from neuralresearcher.state import Paper
from neuralresearcher.errors import ToolError, WorkflowError
from neuralresearcher.logging import log_error


def run_retrieval(context: AgentContext) -> None:
    topic_spec = context.store.load_topic_spec()
    if not topic_spec:
        raise WorkflowError("TopicSpec not found in store.")
        
    system_prompt = "You are a retrieval agent. Use the search_papers tool to find papers."
    user_prompt = (
        f"Search for papers related to: {topic_spec.raw_topic}. "
        f"Available keywords from scope: {', '.join(topic_spec.keywords)}\n"
        "Pass a list of 2 or 3 core keywords to the search_papers tool (e.g. ['mamba', 'optimization']). "
        "Do NOT pass long sentences or many words, as it will break the search."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        config=context.config,
        messages=messages,
        tools=[s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_papers"],
        store=context.store,
        task_id=context.task_id,
        agent_name="retrieval"
    )
    
    papers_data = []
    if response.tool_calls:
        for tc in response.tool_calls:
            try:
                args, result_json = execute_tool_call(tc)
                papers_data.extend(json.loads(result_json))
            except ToolError as e:
                log_error(str(e))
                
    papers = []
    for p_data in papers_data:
        paper = Paper(
            id=p_data["id"],
            title=p_data["title"],
            authors=p_data["authors"],
            venue="arXiv",
            year=p_data.get("year", 2024),
            url=p_data["url"],
            abstract=p_data["abstract"]
        )
        papers.append(paper)
        
    context.store.save_papers(papers)
