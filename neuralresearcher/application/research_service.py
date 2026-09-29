from typing import Protocol, Optional
from neuralresearcher.application.models import (
    StartResearchRequest,
    ResumeResearchRequest,
    RunHandle,
    RunStatus,
    ResearchResult,
    RunSummary,
    ArtifactMetadata,
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

    async def read_artifact(self, run_id: str, artifact_name: str) -> bytes:
        ...
