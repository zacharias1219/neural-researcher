from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field

class TopicSpec(BaseModel):
    id: str
    raw_topic: str
    domain: str
    subfields: List[str]
    time_window: Dict[str, int] = Field(default_factory=lambda: {"start_year": 2000, "end_year": 2024})
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
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    results_summary: List[ResultSummary] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    explicit_future_work: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    section_refs: Dict[str, str] = Field(default_factory=dict)

class Claim(BaseModel):
    id: str
    paper_id: str
    type: Literal["result", "method", "assumption", "limitation", "future_work"]
    text: str
    section: str
    location: Dict[str, int | None] = Field(default_factory=lambda: {"page": None, "paragraph": None})
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
    label: str
    type: Literal["data", "implementation", "experiment", "ablation", "analysis", "writing"]
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    estimated_cost: Dict[str, float] = Field(default_factory=lambda: {"compute_hours": 0.0, "human_hours": 0.0})
    risk_level: Literal["low", "medium", "high"]
    assumptions: List[str] = Field(default_factory=list)

class ResearchPlan(BaseModel):
    id: str
    topic_spec_id: str
    primary_gap_ids: List[str] = Field(default_factory=list)
    hypothesis: str
    steps: List[str] = Field(default_factory=list)
    expected_contribution: str
    risk_report_ref: Optional[str] = None
