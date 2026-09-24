from dataclasses import dataclass
from typing import Optional
from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore


@dataclass
class AgentContext:
    """Explicit context passed to each agent, replacing the orchestrator God object."""
    store: StateStore
    config: Config
    topic: str
    task_id: str = "default_run"
    review_feedback: Optional[str] = None
