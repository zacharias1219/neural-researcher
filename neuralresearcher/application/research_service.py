from typing import Optional, Protocol

from neuralresearcher.application.models import (
    ArtifactMetadata,
    ResearchResult,
    ResumeResearchRequest,
    RunHandle,
    RunStatus,
    RunSummary,
    StartResearchRequest,
)


class ResearchService(Protocol):
    async def start_run(self, request: StartResearchRequest) -> RunHandle:
        ...

    async def resume_run(self, request: ResumeResearchRequest) -> RunHandle:
        ...

    async def get_status(self, run_id: str) -> RunStatus:
        ...

    async def cancel_run(self, run_id: str) -> RunStatus:
        ...

    async def get_result(self, run_id: str) -> ResearchResult:
        ...

    async def list_runs(
        self,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        topic: Optional[str] = None,
        created_after: Optional[str] = None,
        limit: int = 50,
    ) -> list[RunSummary]:
        ...

    async def list_artifacts(self, run_id: str) -> list[ArtifactMetadata]:
        ...

    async def read_public_resource(self, run_id: str, resource_name: str) -> bytes:
        ...

    async def read_generated_artifact(self, run_id: str, artifact_name: str) -> bytes:
        ...

    async def shutdown(self, grace_period: float = 2.0) -> None:
        ...
