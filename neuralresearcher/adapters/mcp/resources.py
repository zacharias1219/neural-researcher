import json
from neuralresearcher.application.research_service import ResearchService

def register_resources(server, service: ResearchService):

    @server.resource("research://runs/{run_id}/manifest", mime_type="application/json")
    async def get_manifest(run_id: str) -> str:
        """Get run manifest."""
        try:
            return (await service.read_artifact(run_id, "manifest.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/status", mime_type="application/json")
    async def get_status(run_id: str) -> str:
        try:
            status = await service.get_status(run_id)
            return status.model_dump_json()
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/result", mime_type="application/json")
    async def get_result(run_id: str) -> str:
        try:
            result = await service.get_result(run_id)
            return result.model_dump_json()
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/plan", mime_type="text/markdown")
    async def get_plan(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "research_plan.md")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/papers", mime_type="application/json")
    async def get_papers(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "papers.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/claims", mime_type="application/json")
    async def get_claims(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "claims.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/gaps", mime_type="application/json")
    async def get_gaps(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "gaps.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/directions", mime_type="application/json")
    async def get_directions(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "directions.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/review", mime_type="application/json")
    async def get_review(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "review.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/coverage", mime_type="application/json")
    async def get_coverage(run_id: str) -> str:
        try:
            return (await service.read_artifact(run_id, "coverage.json")).decode("utf-8")
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}")

    @server.resource("research://runs/{run_id}/artifacts/{artifact_name}")
    async def get_artifact(run_id: str, artifact_name: str) -> str | bytes:
        try:
            content = await service.read_artifact(run_id, artifact_name)
            if artifact_name.endswith(".json"):
                return content.decode("utf-8")
            if artifact_name.endswith(".md"):
                return content.decode("utf-8")
            return content # return bytes for binary if needed
        except ValueError:
            raise ValueError(f"Resource not found: run_id={run_id}, artifact={artifact_name}")
