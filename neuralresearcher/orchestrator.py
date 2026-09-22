from enum import Enum, auto
from typing import Optional

from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore
from neuralresearcher.logging import log_state_transition, log_agent_start, log_agent_end, log_error
from neuralresearcher.errors import NeuralResearcherError

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
    def __init__(self, topic: str, config: Config, store: StateStore):
        self.topic = topic
        self.config = config
        self.store = store
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
        except NeuralResearcherError as e:
            log_error(str(e))
            self.set_state(OrchestratorState.HALTED)
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            self.set_state(OrchestratorState.HALTED)

    def _to_scoped(self) -> None:
        log_agent_start("topic_scope")
        from neuralresearcher.agents.topic_scope import run_topic_scope
        run_topic_scope(self)
        log_agent_end("topic_scope")
        self.set_state(OrchestratorState.SCOPED)

    def _to_retrieved(self) -> None:
        log_agent_start("retrieval")
        from neuralresearcher.agents.retrieval import run_retrieval
        run_retrieval(self)
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
