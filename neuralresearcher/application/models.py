import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict

from neuralresearcher.config import LLMProvider

class StartResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    provider: LLMProvider = Field(default=LLMProvider.GROQ)
    model: Optional[str] = None
    strict: bool = False
    seed: Optional[int] = None
    max_papers: Optional[int] = Field(default=None, ge=1, le=100)
    time_window_start: Optional[int] = Field(default=None, ge=1900)
    time_window_end: Optional[int] = Field(default=None, ge=1900)

    @model_validator(mode='after')
    def validate_time_window(self):
        max_year = datetime.datetime.now().year + 1
        if self.time_window_start is not None and self.time_window_start > max_year:
            raise ValueError(f"time_window_start cannot be > {max_year}")
        if self.time_window_end is not None and self.time_window_end > max_year:
            raise ValueError(f"time_window_end cannot be > {max_year}")
            
        if self.time_window_start is not None and self.time_window_end is not None:
            if self.time_window_start > self.time_window_end:
                raise ValueError("time_window_start cannot be greater than time_window_end")
        return self

class ResumeResearchRequest(BaseModel):
    run_id: str
    provider: Optional[LLMProvider] = None
    model: Optional[str] = None

class RunHandle(BaseModel):
    run_id: str
    topic: str
    status: str
    current_stage: Optional[str]
    created_at: str
    status_resource_uri: str
    result_resource_uri: str
    plan_resource_uri: str

class ArtifactMetadata(BaseModel):
    name: str
    path: str
    mime_type: str
    size_bytes: int
    modified_at: Optional[str]
    resource_uri: str
    description: Optional[str] = None

class RunStatus(BaseModel):
    run_id: str
    topic: str
    provider: str
    model: str
    status: str
    current_stage: Optional[str]
    created_at: Optional[str]
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    finished_at: Optional[str] = None
    success: bool
    terminal: bool
    halt_code: Optional[str]
    failed_stage: Optional[str]
    error_message: Optional[str]
    duration_seconds: float
    artifact_count: int
    cancellation_requested: bool

class ResearchResult(BaseModel):
    run_id: str
    final_state: str
    success: bool
    topic: str
    halt_code: Optional[str]
    failed_stage: Optional[str]
    artifact_resource_uris: dict[str, str]
    review_passed: Optional[bool] = None
    warnings: list[str] = Field(default_factory=list)

class RunSummary(BaseModel):
    run_id: str
    topic: str
    provider: str
    status: str
    created_at: Optional[str]
