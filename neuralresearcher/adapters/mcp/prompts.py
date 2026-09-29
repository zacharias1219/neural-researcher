from mcp.types import PromptMessage, TextContent
from typing import Optional

def register_prompts(server):

    @server.prompt()
    async def create_research_plan(
        topic: str,
        scope: Optional[str] = None,
        time_window: Optional[str] = None,
        emphasis: Optional[str] = None,
        strict: Optional[str] = None
    ) -> list[PromptMessage]:
        """Guide the host to create a research plan."""
        content = f"""To research "{topic}", please follow these steps:
1. Call the `research_start` tool with the topic and any additional parameters (scope: {scope}, time_window: {time_window}, emphasis: {emphasis}, strict: {strict}).
2. Preserve the returned run ID.
3. Poll the `research_status` tool at reasonable intervals (e.g. every 5-10 seconds).
4. Stop polling when the status indicates success, failure, or cancellation.
5. Call the `research_result` tool to get the summary.
6. Read the plan resource (`research://runs/{{run_id}}/plan`).
7. Report any warnings and limitations (e.g., if evidence is abstract-only).
"""
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=content)
            )
        ]

    @server.prompt()
    async def review_research_plan(
        run_id: str,
        review_emphasis: Optional[str] = None
    ) -> list[PromptMessage]:
        """Guide the host to inspect and review a plan."""
        content = f"""Please review the research plan for run_id: {run_id}.
Guide:
1. Inspect the run status using `research_status`.
2. Read the plan resource (`research://runs/{run_id}/plan`).
3. Review existing review resource (`research://runs/{run_id}/review`).
4. Look at gaps (`research://runs/{run_id}/gaps`) and directions (`research://runs/{run_id}/directions`).
5. Check baseline coverage (`research://runs/{run_id}/coverage`).
6. Identify evidence levels, noting that abstract-only evidence has not been full-text verified.
7. Assess the artifact dependency graph and resource assumptions.
8. Note any failure warnings.
Emphasis for this review: {review_emphasis or 'General completeness and validity'}.
"""
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=content)
            )
        ]

    @server.prompt()
    async def explore_research_gap(
        run_id: str,
        gap_id: str
    ) -> list[PromptMessage]:
        """Guide the host to explore a specific gap."""
        content = f"""To explore gap {gap_id} in run {run_id}, please read:
- Gap state (`research://runs/{run_id}/gaps`)
- Related papers (`research://runs/{run_id}/papers`)
- Supporting claims (`research://runs/{run_id}/claims`)
- Directions (`research://runs/{run_id}/directions`)

Then identify:
- Why the gap may matter
- Existing evidence
- Contradictory evidence
- Missing validation
- A feasible next experiment
"""
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=content)
            )
        ]
