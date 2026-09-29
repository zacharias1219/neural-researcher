import json
import os
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from neuralresearcher.state import (
    TopicSpec, Paper, Claim, Gap, Direction, PlanStep, ResearchPlan,
    CoverageReport, ReviewResult
)
from neuralresearcher.errors import StateCorruptionError

STATE_VERSION = 1

class StateStore:
    def __init__(self, directory: str = "research", run_id: Optional[str] = None):
        self.base_directory = Path(directory)
        self.run_id = run_id or str(uuid.uuid4())
        self.directory = self.base_directory / "runs" / self.run_id
        self.state_file = self.directory / "state.json"
        self.manifest_file = self.directory / "manifest.json"
        self._ensure_directory()
        
    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "papers").mkdir(parents=True, exist_ok=True)
        (self.directory / "analysis").mkdir(parents=True, exist_ok=True)
        (self.directory / "transcripts").mkdir(parents=True, exist_ok=True)
        (self.directory / "sources").mkdir(parents=True, exist_ok=True)
        (self.directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.directory / "backups").mkdir(parents=True, exist_ok=True)
        
    def _migrate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Migration hooks for older state versions."""
        version = state.get("version", 0)
        
        if version == 0:
            # v0 -> v1 migration
            if "papers" in state:
                for p in state["papers"]:
                    p.setdefault("url", "")
                    p.setdefault("abstract", "")
            state["version"] = 1
            
        if state["version"] > STATE_VERSION:
            raise StateCorruptionError(f"Unsupported state version: {state['version']}. Max supported is {STATE_VERSION}.")
            
        return state
        
    def _read_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                return self._migrate_state(state)
        except json.JSONDecodeError as e:
            # Preserve the corrupt file
            corrupt_file = self.state_file.with_suffix(f".corrupt.{int(time.time())}.json")
            import shutil
            shutil.copy2(self.state_file, corrupt_file)
            raise StateCorruptionError(f"State file is corrupt. Saved backup to {corrupt_file}. Error: {str(e)}")
            
    def _write_state(self, state: Dict[str, Any]) -> None:
        state["version"] = STATE_VERSION
        
        if self.state_file.exists():
            import shutil
            backup_file = self.directory / "backups" / f"state.backup.{int(time.time())}.json"
            shutil.copy2(self.state_file, backup_file)
            
        tmp_file = self.state_file.with_suffix('.tmp')
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.state_file)

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        tmp_file = self.manifest_file.with_suffix('.tmp')
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.manifest_file)

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_file.exists():
            return {}
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise StateCorruptionError(f"Manifest file is corrupt. Error: {str(e)}")

    # ---- TopicSpec ----

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

    # ---- Papers ----

    def save_papers(self, papers: List[Paper]) -> None:
        state = self._read_state()
        deduped: Dict[str, Paper] = {}
        list_fields = ["methods", "datasets", "metrics", "limitations", "explicit_future_work", "citations"]
        for p in papers:
            if p.id not in deduped:
                deduped[p.id] = p.model_copy() if hasattr(p, "model_copy") else p
            else:
                ep = deduped[p.id]
                for fld in list_fields:
                    p_val = getattr(p, fld, []) or []
                    if p_val:
                        ep_val = getattr(ep, fld, []) or []
                        setattr(ep, fld, list(dict.fromkeys(ep_val + p_val)))
        sorted_papers = sorted(deduped.values(), key=lambda x: x.id)
        state["papers"] = [p.model_dump() for p in sorted_papers]
        self._write_state(state)

    def load_papers(self) -> List[Paper]:
        state = self._read_state()
        data = state.get("papers", [])
        return [Paper(**p) for p in data]

    def update_papers(self, updated_papers: List[Paper]) -> None:
        """Merge metadata into existing papers without overwriting the full list.
        
        Matches on paper.id and merges metadata fields
        (methods, datasets, metrics, limitations, explicit_future_work, citations).
        """
        existing = self.load_papers()
        existing_map = {p.id: p for p in existing}
        list_fields = ["methods", "datasets", "metrics", "limitations", "explicit_future_work", "citations"]
        
        for up in updated_papers:
            if up.id in existing_map:
                ep = existing_map[up.id]
                for fld in list_fields:
                    up_val = getattr(up, fld, []) or []
                    if up_val:
                        ep_val = getattr(ep, fld, []) or []
                        setattr(ep, fld, list(dict.fromkeys(ep_val + up_val)))
            else:
                existing.append(up)
        
        self.save_papers(existing)

    # ---- Claims ----

    def save_claims(self, claims: List[Claim]) -> None:
        state = self._read_state()
        state["claims"] = [c.model_dump() for c in claims]
        self._write_state(state)

    def load_claims(self) -> List[Claim]:
        state = self._read_state()
        data = state.get("claims", [])
        return [Claim(**c) for c in data]

    # ---- Gaps ----

    def save_gaps(self, gaps: List[Gap]) -> None:
        state = self._read_state()
        state["gaps"] = [g.model_dump() for g in gaps]
        self._write_state(state)

    def load_gaps(self) -> List[Gap]:
        state = self._read_state()
        data = state.get("gaps", [])
        return [Gap(**g) for g in data]

    # ---- Directions ----

    def save_directions(self, directions: List[Direction]) -> None:
        state = self._read_state()
        state["directions"] = [d.model_dump() for d in directions]
        self._write_state(state)

    def load_directions(self) -> List[Direction]:
        state = self._read_state()
        data = state.get("directions", [])
        return [Direction(**d) for d in data]

    # ---- Plan + Steps ----

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

    # ---- Coverage Report ----

    def save_coverage_report(self, report: CoverageReport) -> None:
        state = self._read_state()
        state["coverage_report"] = report.model_dump()
        self._write_state(state)

    def load_coverage_report(self) -> Optional[CoverageReport]:
        state = self._read_state()
        data = state.get("coverage_report")
        if data:
            return CoverageReport(**data)
        return None

    # ---- Review Result ----

    def save_review_result(self, result: ReviewResult) -> None:
        state = self._read_state()
        state["review_result"] = result.model_dump()
        self._write_state(state)

    def load_review_result(self) -> Optional[ReviewResult]:
        state = self._read_state()
        data = state.get("review_result")
        if data:
            return ReviewResult(**data)
        return None

    # ---- Transcripts ----

    def save_transcript(self, task_id: str, agent_name: str, messages: List[Dict[str, Any]], usage: Dict[str, Any], assistant_response: Optional[Dict[str, Any]] = None) -> None:
        import time
        transcript_file = self.directory / "transcripts" / f"{task_id}.jsonl"
        entry = {
            "timestamp": time.time(),
            "agent_name": agent_name,
            "messages": messages,
            "usage": usage,
        }
        if assistant_response:
            entry["assistant_response"] = assistant_response
            
        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
    def load_transcripts(self, task_id: str) -> List[Dict[str, Any]]:
        transcript_file = self.directory / "transcripts" / f"{task_id}.jsonl"
        if not transcript_file.exists():
            return []
            
        transcripts = []
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    transcripts.append(json.loads(line))
        return transcripts
