import logging
from typing import Optional

from pydantic import BaseModel

from neuralresearcher.application.models import (
    ArtifactMetadata,
    ResearchResult,
    ResumeResearchRequest,
    RunHandle,
    RunStatus,
    RunSummary,
    StartResearchRequest,
)
from neuralresearcher.application.research_service import ResearchService
from neuralresearcher.config import LLMProvider

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    error: str
    message: str

def register_tools(server, service: ResearchService):

    @server.tool()
    async def research_start(
        topic: str,
        provider: str = "groq",
        model: Optional[str] = None,
        strict: bool = False,
        seed: Optional[int] = None,
        max_papers: Optional[int] = None,
        time_window_start: Optional[int] = None,
        time_window_end: Optional[int] = None
    ) -> RunHandle | ErrorResponse:
        """Start a new research run asynchronously."""
        try:
            prov_enum = LLMProvider(provider)
        except ValueError:
            return ErrorResponse(error="INVALID_ARGUMENT", message=f"Unknown provider {provider}")

        req = StartResearchRequest(
            topic=topic,
            provider=prov_enum,
            model=model,
            strict=strict,
            seed=seed,
            max_papers=max_papers,
            time_window_start=time_window_start,
            time_window_end=time_window_end
        )
        try:
            return await service.start_run(req)
        except ValueError as e:
            logger.error(f"Error starting research: {e}")
            return ErrorResponse(error="CAPACITY_EXCEEDED" if "capacity" in str(e).lower() else "INVALID_ARGUMENT", message=str(e))
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_resume(
        run_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> RunHandle | ErrorResponse:
        """Resume an explicitly selected persisted run. Note: this currently restarts the run from INIT and clears incompatible artifacts."""
        prov_enum = None
        if provider:
            try:
                prov_enum = LLMProvider(provider)
            except ValueError:
                return ErrorResponse(error="INVALID_ARGUMENT", message=f"Unknown provider {provider}")

        req = ResumeResearchRequest(
            run_id=run_id,
            provider=prov_enum,
            model=model
        )
        try:
            return await service.resume_run(req)
        except ValueError as e:
            return ErrorResponse(error=str(e).split(":")[0], message=str(e))
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_status(run_id: str) -> RunStatus | ErrorResponse:
        """Return current or terminal status of a run."""
        try:
            return await service.get_status(run_id)
        except ValueError as e:
            return ErrorResponse(error=str(e), message="Run not found")
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_cancel(run_id: str) -> RunStatus | ErrorResponse:
        """Request cooperative cancellation."""
        try:
            return await service.cancel_run(run_id)
        except ValueError as e:
            return ErrorResponse(error=str(e), message="Run not found")
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_result(run_id: str) -> ResearchResult | ErrorResponse:
        """Return a structured summary of a completed or failed run."""
        try:
            return await service.get_result(run_id)
        except ValueError as e:
            err = str(e)
            return ErrorResponse(error=err, message="Result unavailable" if err == "RUN_NOT_READY" else "Run not found")
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_list_runs(
        status: Optional[str] = None,
        provider: Optional[str] = None,
        topic: Optional[str] = None,
        created_after: Optional[str] = None,
        limit: int = 50
    ) -> list[RunSummary] | ErrorResponse:
        """List summaries of recent research runs."""
        try:
            limit = min(max(1, limit), 100)
            return await service.list_runs(status, provider, topic, created_after, limit)
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")

    @server.tool()
    async def research_list_artifacts(run_id: str) -> list[ArtifactMetadata] | ErrorResponse:
        """List artifacts generated by a run."""
        try:
            return await service.list_artifacts(run_id)
        except ValueError as e:
            return ErrorResponse(error=str(e), message="Run not found")
        except Exception:
            logger.exception("Internal MCP operation failure")
            return ErrorResponse(error="INTERNAL_ERROR", message="The operation failed internally.")
