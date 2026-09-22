import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from neuralresearcher.state import (
    TopicSpec, Paper, Claim, Gap, Direction, PlanStep, ResearchPlan
)
from neuralresearcher.errors import SchemaError

class StateStore:
    def __init__(self, directory: str = "research"):
        self.directory = Path(directory)
        self.state_file = self.directory / "state.json"
        self._ensure_directory()
        
    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "papers").mkdir(parents=True, exist_ok=True)
        (self.directory / "analysis").mkdir(parents=True, exist_ok=True)
        
    def _read_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
            
    def _write_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def save_topic_spec(self, topic_spec: TopicSpec) -> None:
        state = self._read_state()
        state["topic_spec"] = topic_spec.model_dump()
        self._write_state(state)

    def load_topic_spec(self) -> Optional[TopicSpec]:
        state = self._read_state()
        data = state.get("topic_spec")
        if data:
            return TopicSpec(**data)
        return None

    def save_papers(self, papers: List[Paper]) -> None:
        state = self._read_state()
        state["papers"] = [p.model_dump() for p in papers]
        self._write_state(state)

    def load_papers(self) -> List[Paper]:
        state = self._read_state()
        data = state.get("papers", [])
        return [Paper(**p) for p in data]

    def save_claims(self, claims: List[Claim]) -> None:
        state = self._read_state()
        state["claims"] = [c.model_dump() for c in claims]
        self._write_state(state)

    def load_claims(self) -> List[Claim]:
        state = self._read_state()
        data = state.get("claims", [])
        return [Claim(**c) for c in data]

    def save_gaps(self, gaps: List[Gap]) -> None:
        state = self._read_state()
        state["gaps"] = [g.model_dump() for g in gaps]
        self._write_state(state)

    def load_gaps(self) -> List[Gap]:
        state = self._read_state()
        data = state.get("gaps", [])
        return [Gap(**g) for g in data]

    def save_directions(self, directions: List[Direction]) -> None:
        state = self._read_state()
        state["directions"] = [d.model_dump() for d in directions]
        self._write_state(state)

    def load_directions(self) -> List[Direction]:
        state = self._read_state()
        data = state.get("directions", [])
        return [Direction(**d) for d in data]

    def save_plan(self, plan: ResearchPlan, steps: List[PlanStep]) -> None:
        state = self._read_state()
        state["plan"] = plan.model_dump()
        state["plan_steps"] = [s.model_dump() for s in steps]
        self._write_state(state)

    def load_plan(self) -> tuple[Optional[ResearchPlan], List[PlanStep]]:
        state = self._read_state()
        plan_data = state.get("plan")
        steps_data = state.get("plan_steps", [])
        plan = ResearchPlan(**plan_data) if plan_data else None
        steps = [PlanStep(**s) for s in steps_data]
        return plan, steps
