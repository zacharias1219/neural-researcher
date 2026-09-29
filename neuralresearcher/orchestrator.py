import asyncio
from enum import Enum, auto
import time

from neuralresearcher.config import Config
from neuralresearcher.context import AgentContext
from neuralresearcher.io.store import StateStore
from neuralresearcher.logging import (
    log_state_transition,
    log_agent_start,
    log_agent_end,
    log_error,
    log_warning,
)
from neuralresearcher.errors import NeuralResearcherError, WorkflowError, SchemaError, LLMError, ToolError, StateCorruptionError
from neuralresearcher.state import HaltCode, RunResult
from neuralresearcher.agents.topic_scope import run_topic_scope
from neuralresearcher.agents.retrieval import run_retrieval
from neuralresearcher.agents.reading import run_reading
from neuralresearcher.agents.coverage import run_coverage
from neuralresearcher.agents.gaps import run_gaps
from neuralresearcher.agents.directions import run_directions
from neuralresearcher.agents.planner import run_planner
from neuralresearcher.agents.reviewer import run_reviewer
from neuralresearcher.agents.reporting import run_reporting


class OrchestratorState(Enum):
    INIT = auto()
    SCOPED = auto()
    RETRIEVED = auto()
    READ = auto()
    MAPPED = auto()
    GAPS_IDENTIFIED = auto()
    DIRECTIONS_PROPOSED = auto()
    PLAN_DRAFTED = auto()
    PLAN_REVIEWED = auto()
    REPORT_READY = auto()
    HALTED = auto()


class Orchestrator:
    def __init__(self, topic: str, config: Config, store: StateStore, task_id: str = "default_run"):
        self.topic = topic
        self.config = config
        self.store = store
        self.task_id = task_id
        self.state = OrchestratorState.INIT
        self.cancel_token = None
        self.on_state_change = None

        self.retry_count = 0
        self.context = AgentContext(
            store=self.store,
            config=self.config,
            topic=self.topic,
            task_id=self.task_id,
        )

    def get_context(self) -> AgentContext:
        """Return the persistent AgentContext instance."""
        return self.context

    def set_state(self, new_state: OrchestratorState) -> None:
        log_state_transition(self.state.name, new_state.name)
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state.name)

    def run(self) -> RunResult:
        start_time = time.time()
        halt_code = None
        failed_stage = None
        message = None
        success = False

        try:
            while self.state not in (OrchestratorState.REPORT_READY, OrchestratorState.HALTED):
                if self.cancel_token:
                    self.cancel_token.check()

                if self.state == OrchestratorState.INIT:
                    self._to_scoped()
                elif self.state == OrchestratorState.SCOPED:
                    self._to_retrieved()
                elif self.state == OrchestratorState.RETRIEVED:
                    self._to_read()
                elif self.state == OrchestratorState.READ:
                    self._to_mapped()
                elif self.state == OrchestratorState.MAPPED:
                    self._to_gaps()
                elif self.state == OrchestratorState.GAPS_IDENTIFIED:
                    self._to_directions()
                elif self.state == OrchestratorState.DIRECTIONS_PROPOSED:
                    self._to_plan()
                elif self.state == OrchestratorState.PLAN_DRAFTED:
                    self._to_review()
                elif self.state == OrchestratorState.PLAN_REVIEWED:
                    self._to_report()
                else:
                    break
            if self.state == OrchestratorState.REPORT_READY:
                success = True
        except WorkflowError as e:
            log_warning(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = e.halt_code or HaltCode.INTERNAL_ERROR
            message = str(e)
        except SchemaError as e:
            log_error(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.INVALID_MODEL_OUTPUT
            message = str(e)
        except LLMError as e:
            log_error(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.PROVIDER_ERROR
            message = str(e)
        except ToolError as e:
            log_error(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.TOOL_ERROR
            message = str(e)
        except StateCorruptionError as e:
            log_error(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.STORAGE_FAILURE
            message = str(e)
        except asyncio.CancelledError as e:
            log_error("Run cancelled")
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.CANCELLED
            message = str(e)
        except NeuralResearcherError as e:
            log_error(str(e))
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.INTERNAL_ERROR
            message = str(e)
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            failed_stage = self.state.name
            self.set_state(OrchestratorState.HALTED)
            halt_code = HaltCode.INTERNAL_ERROR
            message = str(e)

        duration = time.time() - start_time
        
        total_tokens = 0
        try:
            import json
            for t_path in (self.store.directory / "transcripts").glob("*.jsonl"):
                with open(t_path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            total_tokens += data.get("usage", {}).get("total_tokens", 0)
        except Exception:
            pass
            
        return RunResult(
            run_id=self.task_id,
            final_state=self.state.name,
            success=success,
            halt_code=halt_code,
            failed_stage=failed_stage,
            message=message,
            artifact_paths={
                "state": str(self.store.state_file),
                "plan": str(self.store.directory / "research_plan.md")
            },
            total_tokens=total_tokens,
            duration_seconds=duration
        )

    def _validate_topic(self) -> None:
        """Layer 1: Validate topic domain."""
        topic_spec = self.store.load_topic_spec()
        if not topic_spec:
            return
            
        if topic_spec.domain.value == "other":
            raise WorkflowError("Topic is out-of-scope. Domain is not supported.", halt_code=HaltCode.OUT_OF_SCOPE)

    def _validate_retrieval(self) -> None:
        """Layer 2: Validate retrieved papers quantity and relevance."""
        papers = self.store.load_papers()
        if len(papers) < self.config.min_papers:
            raise WorkflowError(f"Found {len(papers)} papers, which is below the minimum threshold ({self.config.min_papers}). The topic might be too narrow.", halt_code=HaltCode.INSUFFICIENT_EVIDENCE)
            
        topic_spec = self.store.load_topic_spec()
        if not topic_spec:
            return
            
        keywords = [k.lower() for k in topic_spec.keywords]
        
        # Simple overlap check: count papers that contain at least one keyword in title or abstract
        relevant_papers = 0
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            if any(k in text for k in keywords):
                relevant_papers += 1
                
        ratio = relevant_papers / len(papers)
        if ratio < self.config.relevance_threshold:
            raise WorkflowError(f"Retrieval relevance too low ({ratio:.2f} < {self.config.relevance_threshold}). Retrieved papers do not match the topic keywords. Topic may be too specific or poorly phrased.", halt_code=HaltCode.LOW_RETRIEVAL_RELEVANCE)

    def _validate_coverage(self) -> None:
        """Layer 3: Validate literature coverage."""
        report = self.store.load_coverage_report()
        if not report:
            return
            
        num_clusters = len(report.clusters)
        num_warnings = len(report.warnings)
        
        if num_clusters > 0:
            warning_ratio = num_warnings / num_clusters
            if warning_ratio > self.config.coverage_warning_threshold:
                raise WorkflowError(f"Too many coverage warnings ({num_warnings} warnings for {num_clusters} clusters). Literature coverage is insufficient to form a valid plan.", halt_code=HaltCode.COVERAGE_FAILURE)

    def _to_scoped(self) -> None:
        log_agent_start("topic_scope")
        run_topic_scope(self.get_context())
        self._validate_topic()
        log_agent_end("topic_scope")
        self.set_state(OrchestratorState.SCOPED)

    def _to_retrieved(self) -> None:
        log_agent_start("retrieval")
        run_retrieval(self.get_context())
        self._validate_retrieval()
        log_agent_end("retrieval")
        self.set_state(OrchestratorState.RETRIEVED)

    def _to_read(self) -> None:
        log_agent_start("reading")
        run_reading(self.get_context())
        log_agent_end("reading")
        self.set_state(OrchestratorState.READ)

    def _to_mapped(self) -> None:
        log_agent_start("coverage")
        run_coverage(self.get_context())
        self._validate_coverage()
        log_agent_end("coverage")
        self.set_state(OrchestratorState.MAPPED)

    def _to_gaps(self) -> None:
        log_agent_start("gaps")
        run_gaps(self.get_context())
        log_agent_end("gaps")
        self.set_state(OrchestratorState.GAPS_IDENTIFIED)

    def _to_directions(self) -> None:
        log_agent_start("directions")
        run_directions(self.get_context())
        log_agent_end("directions")
        self.set_state(OrchestratorState.DIRECTIONS_PROPOSED)

    def _validate_plan(self) -> None:
        """Validate that a plan was actually generated with steps."""
        plan, steps = self.store.load_plan()
        if not plan or not steps:
            raise WorkflowError("Plan generation yielded no steps. The research pipeline failed to produce a valid plan.", halt_code=HaltCode.PLAN_VALIDATION_FAILURE)

    def _to_plan(self) -> None:
        log_agent_start("planner")
        run_planner(self.get_context())
        self._validate_plan()
        log_agent_end("planner")
        self.set_state(OrchestratorState.PLAN_DRAFTED)

    def _to_review(self) -> None:
        log_agent_start("reviewer")
        review_err = None
        try:
            run_reviewer(self.get_context())
        except WorkflowError as e:
            review_err = e
        log_agent_end("reviewer")

        review = self.store.load_review_result()
        if (review and not review.passed) or review_err:
            issues = review.issues if review else [str(review_err)]
            suggestions = review.suggestions if review else []
            issues_text = "\n".join(f"- Issue: {iss}" for iss in issues)
            suggestions_text = "\n".join(f"- Suggestion: {sug}" for sug in suggestions)
            self.context.review_feedback = f"{issues_text}\n{suggestions_text}".strip()

            if self.retry_count < self.config.plan_repair_attempts:
                self.retry_count += 1
                log_warning(
                    f"Plan review failed ({len(issues)} issue(s)). "
                    f"Retrying planner with reviewer feedback (attempt {self.retry_count}/{self.config.plan_repair_attempts})..."
                )
                self.set_state(OrchestratorState.DIRECTIONS_PROPOSED)
                return
            else:
                if self.config.strict_mode:
                    if review_err:
                        raise review_err
                    raise WorkflowError(
                        f"Plan failed review after {self.config.plan_repair_attempts} attempts: {', '.join(issues)}", halt_code=HaltCode.REVIEW_FAILURE
                    )

        self.context.review_feedback = None
        self.set_state(OrchestratorState.PLAN_REVIEWED)

    def _to_report(self) -> None:
        log_agent_start("reporting")
        run_reporting(self.get_context())
        log_agent_end("reporting")
        self.set_state(OrchestratorState.REPORT_READY)
