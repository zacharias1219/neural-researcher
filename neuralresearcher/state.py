import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentLevel(str, Enum):
    METADATA_ONLY = "METADATA_ONLY"
    ABSTRACT_ONLY = "ABSTRACT_ONLY"
    FULL_TEXT = "FULL_TEXT"

class TopicDomain(str, Enum):
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    COMPUTER_SCIENCE = "computer_science"
    DEEP_LEARNING = "deep_learning"
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    SYSTEMS = "systems"
    ROBOTICS = "robotics"
    DATA_SCIENCE = "data_science"
    OTHER = "other"

ALLOWED_DOMAINS = {d.value for d in TopicDomain}

class HaltCode(str, Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LOW_RETRIEVAL_RELEVANCE = "LOW_RETRIEVAL_RELEVANCE"
    COVERAGE_FAILURE = "COVERAGE_FAILURE"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    STORAGE_FAILURE = "STORAGE_FAILURE"
    PLAN_VALIDATION_FAILURE = "PLAN_VALIDATION_FAILURE"
    REVIEW_FAILURE = "REVIEW_FAILURE"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INTERRUPTED = "INTERRUPTED"

class RunResult(BaseModel):
    run_id: str
    final_state: str
    success: bool
    halt_code: Optional[HaltCode] = None
    failed_stage: Optional[str] = None
    message: Optional[str] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    total_tokens: int = 0
    duration_seconds: float = 0.0

class TopicSpec(BaseModel):
    id: str
    raw_topic: str
    domain: TopicDomain
    subfields: List[str]
    time_window: Dict[str, int] = Field(default_factory=lambda: {"start_year": 2000, "end_year": datetime.datetime.now().year})
    scope_constraints: Dict[str, str] = Field(default_factory=dict)
    keywords: List[str]

class ResultSummary(BaseModel):
    dataset: str
    metric: str
    value: str | float
    baseline: str | float | None = None

class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    venue: str
    year: int
    url: str
    abstract: str
    content_level: ContentLevel = ContentLevel.ABSTRACT_ONLY
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    results_summary: List[ResultSummary] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    explicit_future_work: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    section_refs: Dict[str, str] = Field(default_factory=dict)

def _default_location() -> Dict[str, int | None]:
    return {"page": None, "paragraph": None}

class Claim(BaseModel):
    id: str
    paper_id: str
    type: Literal["result", "method", "assumption", "limitation", "future_work"]
    text: str
    section: str
    location: Dict[str, int | None] = Field(default_factory=_default_location)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    evidence_ref: str

class Gap(BaseModel):
    id: str
    description: str
    gap_type: Literal["unexplored_axis", "limitation", "contradiction", "missing_combination"]
    related_papers: List[str] = Field(default_factory=list)
    supporting_claims: List[str] = Field(default_factory=list)
    dimensions: Dict[str, str] = Field(default_factory=dict)
    novelty_estimate: Literal["low", "medium", "high"]
    feasibility_notes: str = ""

class Direction(BaseModel):
    id: str
    primary_gap_id: str
    hypothesis: str
    justification: str
    expected_contribution_type: str
    novelty_assessment: Literal["low", "medium", "high"]

class PlanStep(BaseModel):
    id: str
    plan_id: str = ""
    label: str
    description: str = ""
    type: Literal["data", "implementation", "experiment", "ablation", "analysis", "writing"]
    status: Literal["pending", "in_progress", "done"] = "pending"
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    estimated_cost: Dict[str, float] = Field(default_factory=lambda: {"compute_hours": 0.0, "human_hours": 0.0})
    risk_level: Literal["low", "medium", "high"]
    assumptions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

class ResearchPlan(BaseModel):
    id: str
    topic_spec_id: str
    primary_gap_ids: List[str] = Field(default_factory=list)
    hypothesis: str
    steps: List[str] = Field(default_factory=list)
    expected_contribution: str
    risk_report_ref: Optional[str] = None
    target_venue: str = ""
    timeline_weeks: int = 0
    resource_summary: Dict[str, float] = Field(default_factory=lambda: {"total_compute_hours": 0.0, "total_human_hours": 0.0})

class CoverageCluster(BaseModel):
    domain: str
    paper_ids: List[str] = Field(default_factory=list)
    claim_count: int = 0

class CoverageReport(BaseModel):
    id: str
    clusters: List[CoverageCluster] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: str = ""

class ReviewResult(BaseModel):
    id: str
    passed: bool = True
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    timestamp: str = ""
