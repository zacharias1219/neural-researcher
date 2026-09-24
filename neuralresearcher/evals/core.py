from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvalTask(BaseModel):
    id: str
    topic: str
    description: str
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    negative_task: bool = False

class Outcome(BaseModel):
    score: float
    passed: bool
    details: str

class Trial(BaseModel):
    task_id: str
    run_id: str
    topic: str
    success: bool
    outcomes: Dict[str, Outcome] = Field(default_factory=dict)
    duration_sec: float
    total_tokens: int

class EvalSuite(BaseModel):
    name: str
    tasks: List[EvalTask]
