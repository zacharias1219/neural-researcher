import asyncio
import os
import datetime
from pathlib import Path
from typing import Optional, Dict

from pydantic import ValidationError

from neuralresearcher.config import LLMProvider, load_config
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator, OrchestratorState
from neuralresearcher.state import RunResult as CoreRunResult, HaltCode
from neuralresearcher.errors import NeuralResearcherError

from neuralresearcher.application.models import (
    StartResearchRequest,
    ResumeResearchRequest,
    RunHandle,
    RunStatus,
    ResearchResult,
    RunSummary,
    ArtifactMetadata,
)
from neuralresearcher.application.research_service import ResearchService
from neuralresearcher.application.cancellation import CancellationToken
from neuralresearcher.logging import log_info, log_error, log_warning
import re

class RunManager(ResearchService):
    def __init__(self, data_dir: str = "research", max_concurrent_runs: int = 2, max_result_bytes: int = 10 * 1024 * 1024):
        self.data_dir = Path(data_dir)
        self.max_concurrent_runs = max_concurrent_runs
        self.max_result_bytes = max_result_bytes
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._run_locks: Dict[str, asyncio.Lock] = {}
        self._cancellation_tokens: Dict[str, CancellationToken] = {}
        self._active_runs: set[str] = set()
        self._tasks: Dict[str, asyncio.Task] = {}
        
        # Reconciliation on startup
        self._reconcile_runs()

    def _reconcile_runs(self):
        runs_dir = self.data_dir / "runs"
        if not runs_dir.exists():
            return
        
        for run_id in os.listdir(runs_dir):
            store = StateStore(directory=str(self.data_dir), run_id=run_id)
            manifest = store.load_manifest()
            if not manifest:
                continue
                
            state = manifest.get("final_state")
            if not state:
                # Active but not running -> interrupted
                manifest["final_state"] = OrchestratorState.HALTED.name
                manifest["halt_code"] = HaltCode.INTERNAL_ERROR.value
                manifest["failed_stage"] = "UNKNOWN"
                manifest["success"] = False
                store.save_manifest(manifest)

    def _get_lock(self, run_id: str) -> asyncio.Lock:
        if run_id not in self._run_locks:
            self._run_locks[run_id] = asyncio.Lock()
        return self._run_locks[run_id]

    def _validate_run_id(self, run_id: str) -> None:
        if not re.match(r"^[A-Za-z0-9_-]{1,64}$", run_id):
            raise ValueError("INVALID_RUN_ID")

    def _task_done_callback(self, run_id: str, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_error(f"Run task {run_id} failed with unhandled exception: {e}")
        finally:
            self._tasks.pop(run_id, None)

    async def start_run(self, request: StartResearchRequest) -> RunHandle:
        # We start by getting a new StateStore which generates a run_id
        store = StateStore(directory=str(self.data_dir))
        run_id = store.run_id
        
        async with self._get_lock(run_id):
            if run_id in self._active_runs:
                raise ValueError("RUN_ALREADY_ACTIVE")
            token = CancellationToken()
            self._active_runs.add(run_id)
            self._cancellation_tokens[run_id] = token

        manifest = store.load_manifest()
        now = datetime.datetime.now().isoformat()
        manifest.update({
            "run_id": run_id,
            "topic": request.topic,
            "provider": request.provider.value,
            "model": request.model or "",
            "strict": request.strict,
            "created_at": now,
            "cancellation_requested": False
        })
        store.save_manifest(manifest)
        
        # Submit to background
        task = asyncio.create_task(self._execute_run(
            run_id, 
            request.topic, 
            request.provider, 
            request.model, 
            request.strict,
            request.seed,
            request.max_papers,
            request.time_window_start,
            request.time_window_end
        ))
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._task_done_callback(run_id, t))
        
        return RunHandle(
            run_id=run_id,
            topic=request.topic,
            status="INIT",
            current_stage=None,
            created_at=now,
            status_resource_uri=f"research://runs/{run_id}/status",
            result_resource_uri=f"research://runs/{run_id}/result",
            plan_resource_uri=f"research://runs/{run_id}/plan"
        )

    async def resume_run(self, request: ResumeResearchRequest) -> RunHandle:
        self._validate_run_id(request.run_id)
        async with self._get_lock(request.run_id):
            if request.run_id in self._active_runs:
                raise ValueError("RUN_ALREADY_ACTIVE")
                
            store = StateStore(directory=str(self.data_dir), run_id=request.run_id)
            manifest = store.load_manifest()
            if not manifest:
                raise ValueError("UNKNOWN_RUN")
                
            if manifest.get("success"):
                raise ValueError("RESUME_NOT_ALLOWED: Cannot resume a successful run")

            token = CancellationToken()
            self._active_runs.add(request.run_id)
            self._cancellation_tokens[request.run_id] = token

            # Reset terminal states
            manifest.pop("final_state", None)
            manifest.pop("success", None)
            manifest.pop("halt_code", None)
            manifest.pop("failed_stage", None)
            manifest.pop("error_message", None)
            manifest["cancellation_requested"] = False
            
            # Clear artifacts to restart from INIT
            import shutil
            for d in ["analysis", "transcripts", "papers", "sources"]:
                target_dir = store.directory / d
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)
            if store.state_file.exists():
                store.state_file.unlink()
            
            provider_val = request.provider.value if request.provider else manifest.get("provider", LLMProvider.GROQ.value)
            model_val = request.model if request.model else manifest.get("model", "")
            
            manifest["provider"] = provider_val
            manifest["model"] = model_val
            store.save_manifest(manifest)
            
            provider_enum = LLMProvider(provider_val)
            
            task = asyncio.create_task(self._execute_run(
                request.run_id, 
                manifest.get("topic", "Unknown"), 
                provider_enum, 
                model_val, 
                manifest.get("strict", False),
                manifest.get("seed"),
                manifest.get("max_papers"),
                manifest.get("time_window_start"),
                manifest.get("time_window_end")
            ))
            self._tasks[request.run_id] = task
            task.add_done_callback(lambda t: self._task_done_callback(request.run_id, t))
            
            now = manifest.get("created_at", datetime.datetime.now().isoformat())
            
            return RunHandle(
                run_id=request.run_id,
                topic=manifest.get("topic", ""),
                status="RESUMED",
                current_stage=None,
                created_at=now,
                status_resource_uri=f"research://runs/{request.run_id}/status",
                result_resource_uri=f"research://runs/{request.run_id}/result",
                plan_resource_uri=f"research://runs/{request.run_id}/plan"
            )

    async def _execute_run(self, run_id: str, topic: str, provider: LLMProvider, model: Optional[str], strict: bool, seed: Optional[int], max_papers: Optional[int], time_window_start: Optional[int], time_window_end: Optional[int]):
        cancel_token = self._cancellation_tokens.get(run_id)

        try:
            async with self._semaphore:
                store = StateStore(directory=str(self.data_dir), run_id=run_id)
                config = load_config(
                    provider_override=provider,
                    model_override=model if model else None,
                    strict_override=strict
                )
                if seed is not None:
                    config.seed = seed
                if max_papers is not None:
                    config.max_papers = max_papers
                
                manifest = store.load_manifest()
                manifest["started_at"] = datetime.datetime.now().isoformat()
                if seed is not None: manifest["seed"] = seed
                if max_papers is not None: manifest["max_papers"] = max_papers
                if time_window_start is not None: manifest["time_window_start"] = time_window_start
                if time_window_end is not None: manifest["time_window_end"] = time_window_end
                store.save_manifest(manifest)

                orchestrator = Orchestrator(topic=topic, config=config, store=store, task_id=run_id)
                # We will inject cancel_token to orchestrator later
                orchestrator.cancel_token = cancel_token
                orchestrator.context.cancel_token = cancel_token

                loop = asyncio.get_running_loop()
                def state_cb(state_name: str):
                    async def _update_state():
                        async with self._get_lock(run_id):
                            m = store.load_manifest()
                            if m:
                                m["current_stage"] = state_name
                                m["updated_at"] = datetime.datetime.now().isoformat()
                                store.save_manifest(m)
                    asyncio.run_coroutine_threadsafe(_update_state(), loop)
                        
                orchestrator.on_state_change = state_cb
                
                result = await asyncio.to_thread(orchestrator.run)
                
                async with self._get_lock(run_id):
                    manifest = store.load_manifest()
                    manifest.update({
                        "final_state": result.final_state,
                        "success": result.success,
                        "halt_code": result.halt_code.value if result.halt_code else None,
                        "failed_stage": result.failed_stage,
                        "error_message": result.message,
                        "duration_seconds": manifest.get("duration_seconds", 0) + result.duration_seconds,
                        "artifact_paths": result.artifact_paths,
                        "finished_at": datetime.datetime.now().isoformat()
                    })
                    store.save_manifest(manifest)

        except asyncio.CancelledError:
            async with self._get_lock(run_id):
                store = StateStore(directory=str(self.data_dir), run_id=run_id)
                manifest = store.load_manifest()
                manifest.update({
                    "final_state": OrchestratorState.HALTED.name,
                    "success": False,
                    "halt_code": HaltCode.CANCELLED.value,
                    "failed_stage": "UNKNOWN", # Could be more precise if orchestrator sets it
                    "finished_at": datetime.datetime.now().isoformat()
                })
                store.save_manifest(manifest)
        except Exception as e:
            from traceback import format_exc
            log_error(f"Worker failure for run {run_id}: {format_exc()}")
            try:
                async with self._get_lock(run_id):
                    store = StateStore(directory=str(self.data_dir), run_id=run_id)
                    manifest = store.load_manifest()
                    manifest.update({
                        "final_state": OrchestratorState.HALTED.name,
                        "success": False,
                        "halt_code": HaltCode.INTERNAL_ERROR.value,
                        "failed_stage": manifest.get("current_stage", "BOOTSTRAP"),
                        "error_message": f"Worker error: {type(e).__name__}: {str(e)}",
                        "finished_at": datetime.datetime.now().isoformat()
                    })
                    store.save_manifest(manifest)
            except Exception as inner_e:
                log_error(f"Failed to persist worker failure for run {run_id}: {inner_e}")
        finally:
            async with self._get_lock(run_id):
                self._active_runs.discard(run_id)
                self._cancellation_tokens.pop(run_id, None)

    async def get_status(self, run_id: str) -> RunStatus:
        self._validate_run_id(run_id)
        store = StateStore(directory=str(self.data_dir), run_id=run_id)
        if not store.directory.exists():
            raise ValueError("UNKNOWN_RUN")
            
        manifest = store.load_manifest()
        
        success = manifest.get("success", False)
        terminal = "final_state" in manifest
        
        return RunStatus(
            run_id=run_id,
            topic=manifest.get("topic", "Unknown"),
            provider=manifest.get("provider", "Unknown"),
            model=manifest.get("model", ""),
            status=manifest.get("final_state") or "ACTIVE",
            current_stage=manifest.get("current_stage") or manifest.get("final_state") or "INIT",
            created_at=manifest.get("created_at"),
            started_at=manifest.get("started_at"),
            finished_at=manifest.get("finished_at"),
            success=success,
            terminal=terminal,
            halt_code=manifest.get("halt_code"),
            failed_stage=manifest.get("failed_stage"),
            error_message=manifest.get("error_message"),
            duration_seconds=manifest.get("duration_seconds", 0.0),
            artifact_count=len(manifest.get("artifact_paths", {})),
            cancellation_requested=manifest.get("cancellation_requested", False)
        )

    async def cancel_run(self, run_id: str) -> RunStatus:
        self._validate_run_id(run_id)
        async with self._get_lock(run_id):
            store = StateStore(directory=str(self.data_dir), run_id=run_id)
            if not store.directory.exists():
                raise ValueError("UNKNOWN_RUN")
                
            manifest = store.load_manifest()
            manifest = store.load_manifest()
            already_terminal = "final_state" in manifest
            if not already_terminal:
                manifest["cancellation_requested"] = True
                store.save_manifest(manifest)
                
                if run_id in self._cancellation_tokens:
                    self._cancellation_tokens[run_id].cancel()
                
        return await self.get_status(run_id)

    async def get_result(self, run_id: str) -> ResearchResult:
        self._validate_run_id(run_id)
        store = StateStore(directory=str(self.data_dir), run_id=run_id)
        if not store.directory.exists():
            raise ValueError("UNKNOWN_RUN")
            
        manifest = store.load_manifest()
        if "final_state" not in manifest:
            raise ValueError("RUN_NOT_READY")
            
        review = store.load_review_result()
        review_passed = review.passed if review else None
        warnings = review.issues if review and not review.passed else []
        
        return ResearchResult(
            run_id=run_id,
            final_state=manifest["final_state"],
            success=manifest.get("success", False),
            topic=manifest.get("topic", ""),
            halt_code=manifest.get("halt_code"),
            failed_stage=manifest.get("failed_stage"),
            artifact_resource_uris={
                k: f"research://runs/{run_id}/artifacts/{Path(v).name}"
                for k, v in manifest.get("artifact_paths", {}).items()
            },
            review_passed=review_passed,
            warnings=warnings
        )

    async def list_runs(
        self,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        topic: Optional[str] = None,
        created_after: Optional[str] = None,
        limit: int = 50,
    ) -> list[RunSummary]:
        runs_dir = self.data_dir / "runs"
        if not runs_dir.exists():
            return []
            
        results = []
        for run_id in os.listdir(runs_dir):
            store = StateStore(directory=str(self.data_dir), run_id=run_id)
            manifest = store.load_manifest()
            if not manifest:
                continue
                
            current_status = manifest.get("final_state", "ACTIVE")
            
            if status and current_status != status:
                continue
            if provider and manifest.get("provider") != provider:
                continue
            if topic and topic.lower() not in manifest.get("topic", "").lower():
                continue
            if created_after and manifest.get("created_at", "") < created_after:
                continue
                
            results.append(RunSummary(
                run_id=run_id,
                topic=manifest.get("topic", "Unknown"),
                provider=manifest.get("provider", "Unknown"),
                status=current_status,
                created_at=manifest.get("created_at")
            ))
            
        results.sort(key=lambda x: x.created_at or "", reverse=True)
        return results[:limit]

    async def list_artifacts(self, run_id: str) -> list[ArtifactMetadata]:
        self._validate_run_id(run_id)
        store = StateStore(directory=str(self.data_dir), run_id=run_id)
        if not store.directory.exists():
            raise ValueError("UNKNOWN_RUN")
            
        manifest = store.load_manifest()
        artifacts = []
        for name, path_str in manifest.get("artifact_paths", {}).items():
            p = Path(path_str).resolve()
            try:
                p.relative_to(store.directory.resolve())
            except ValueError:
                continue
            if p.exists():
                stat = p.stat()
                artifacts.append(ArtifactMetadata(
                    name=name,
                    path=p.name,
                    mime_type="text/markdown" if p.suffix == ".md" else "application/json",
                    size_bytes=stat.st_size,
                    modified_at=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    resource_uri=f"research://runs/{run_id}/artifacts/{p.name}"
                ))
        return artifacts

    async def read_artifact(self, run_id: str, artifact_name: str) -> bytes:
        self._validate_run_id(run_id)
        store = StateStore(directory=str(self.data_dir), run_id=run_id)
        if not store.directory.exists():
            raise ValueError("UNKNOWN_RUN")
            
        manifest = store.load_manifest()
        paths = manifest.get("artifact_paths", {})
        
        target_path = None
        for name, path_str in paths.items():
            if Path(path_str).name == artifact_name:
                target_path = Path(path_str).resolve()
                break
                
        if not target_path or not target_path.exists():
            target_path = (store.directory / artifact_name).resolve()
            
        try:
            target_path.relative_to(store.directory.resolve())
        except ValueError:
            raise ValueError("INVALID_ARGUMENT")
            
        if not target_path.exists() or not target_path.is_file():
            raise ValueError("RESOURCE_NOT_FOUND")
            
        if target_path.stat().st_size > self.max_result_bytes:
            raise ValueError("ARTIFACT_TOO_LARGE")
                
        return target_path.read_bytes()
