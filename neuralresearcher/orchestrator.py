from enum import Enum, auto
from typing import Optional

from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore
from neuralresearcher.logging import log_state_transition, log_agent_start, log_agent_end, log_error, log_warning
from neuralresearcher.errors import NeuralResearcherError, WorkflowError
from neuralresearcher.state import ALLOWED_DOMAINS


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

    def set_state(self, new_state: OrchestratorState) -> None:
        log_state_transition(self.state.name, new_state.name)
        self.state = new_state

    def run(self) -> None:
        try:
            if self.state == OrchestratorState.INIT:
                self._to_scoped()
            if self.state == OrchestratorState.SCOPED:
                self._to_retrieved()
            if self.state == OrchestratorState.RETRIEVED:
                self._to_read()
            if self.state == OrchestratorState.READ:
                self._to_mapped()
            if self.state == OrchestratorState.MAPPED:
                self._to_gaps()
            if self.state == OrchestratorState.GAPS_IDENTIFIED:
                self._to_directions()
            if self.state == OrchestratorState.DIRECTIONS_PROPOSED:
                self._to_plan()
            if self.state == OrchestratorState.PLAN_DRAFTED:
                self._to_review()
            if self.state == OrchestratorState.PLAN_REVIEWED:
                self._to_report()
        except WorkflowError as e:
            log_warning(str(e))
            self.set_state(OrchestratorState.HALTED)
        except NeuralResearcherError as e:
            log_error(str(e))
            self.set_state(OrchestratorState.HALTED)
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            self.set_state(OrchestratorState.HALTED)

    def _validate_topic(self) -> None:
        """Layer 1: Validate topic domain."""
        topic_spec = self.store.load_topic_spec()
        if not topic_spec:
            return
            
        domain = topic_spec.domain.strip().lower()
        allowed = {d.lower() for d in ALLOWED_DOMAINS}
        
        # Check if any allowed domain is in the predicted domain or vice versa
        is_allowed = any(a in domain for a in allowed) or any(domain in a for a in allowed)
        
        if not is_allowed:
            raise WorkflowError(f"Topic is out-of-scope. Predicted domain '{topic_spec.domain}' is not in allowed domains ({', '.join(ALLOWED_DOMAINS)}).")

    def _validate_retrieval(self) -> None:
        """Layer 2: Validate retrieved papers quantity and relevance."""
        papers = self.store.load_papers()
        if len(papers) < self.config.min_papers:
            raise WorkflowError(f"Found {len(papers)} papers, which is below the minimum threshold ({self.config.min_papers}). The topic might be too narrow.")
            
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
            raise WorkflowError(f"Retrieval relevance too low ({ratio:.2f} < {self.config.relevance_threshold}). Retrieved papers do not match the topic keywords. Topic may be too specific or poorly phrased.")

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
                raise WorkflowError(f"Too many coverage warnings ({num_warnings} warnings for {num_clusters} clusters). Literature coverage is insufficient to form a valid plan.")

    def _to_scoped(self) -> None:
        log_agent_start("topic_scope")
        from neuralresearcher.agents.topic_scope import run_topic_scope
        run_topic_scope(self)
        self._validate_topic()
        log_agent_end("topic_scope")
        self.set_state(OrchestratorState.SCOPED)

    def _to_retrieved(self) -> None:
        log_agent_start("retrieval")
        from neuralresearcher.agents.retrieval import run_retrieval
        run_retrieval(self)
        self._validate_retrieval()
        log_agent_end("retrieval")
        self.set_state(OrchestratorState.RETRIEVED)

    def _to_read(self) -> None:
        log_agent_start("reading")
        from neuralresearcher.agents.reading import run_reading
        run_reading(self)
        log_agent_end("reading")
        self.set_state(OrchestratorState.READ)

    def _to_mapped(self) -> None:
        log_agent_start("coverage")
        from neuralresearcher.agents.coverage import run_coverage
        run_coverage(self)
        self._validate_coverage()
        log_agent_end("coverage")
        self.set_state(OrchestratorState.MAPPED)

    def _to_gaps(self) -> None:
        log_agent_start("gaps")
        from neuralresearcher.agents.gaps import run_gaps
        run_gaps(self)
        log_agent_end("gaps")
        self.set_state(OrchestratorState.GAPS_IDENTIFIED)

    def _to_directions(self) -> None:
        log_agent_start("directions")
        from neuralresearcher.agents.directions import run_directions
        run_directions(self)
        log_agent_end("directions")
        self.set_state(OrchestratorState.DIRECTIONS_PROPOSED)

    def _to_plan(self) -> None:
        log_agent_start("planner")
        from neuralresearcher.agents.planner import run_planner
        run_planner(self)
        log_agent_end("planner")
        self.set_state(OrchestratorState.PLAN_DRAFTED)

    def _to_review(self) -> None:
        log_agent_start("reviewer")
        from neuralresearcher.agents.reviewer import run_reviewer
        run_reviewer(self)
        log_agent_end("reviewer")
        self.set_state(OrchestratorState.PLAN_REVIEWED)

    def _to_report(self) -> None:
        log_agent_start("reporting")
        from neuralresearcher.agents.reporting import run_reporting
        run_reporting(self)
        log_agent_end("reporting")
        self.set_state(OrchestratorState.REPORT_READY)
