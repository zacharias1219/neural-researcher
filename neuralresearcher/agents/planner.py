import hashlib
import json
import re
import uuid

from neuralresearcher.context import AgentContext
from neuralresearcher.state import ResearchPlan, PlanStep
from neuralresearcher.llm import generate_structured
from neuralresearcher.errors import SchemaError
from neuralresearcher.plan_validation import normalize_plan
from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class PlanOutput(BaseModel):
    hypothesis: str
    expected_contribution: str
    target_venue: str
    timeline_weeks: int
    primary_gap_ids: List[str]

class StepOutput(BaseModel):
    label: str
    description: str
    type: Literal["data", "implementation", "experiment", "ablation", "analysis", "writing"]
    risk_level: Literal["low", "medium", "high"]
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    estimated_cost: Dict[str, float] = Field(default_factory=lambda: {"compute_hours": 0.0, "human_hours": 0.0})
    assumptions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

class PlannerResponse(BaseModel):
    plan: PlanOutput
    steps: List[StepOutput]


def run_planner(context: AgentContext) -> None:
    directions = context.store.load_directions()
    if not directions:
        return
        
    direction = directions[0]  # Pick the top-ranked direction
    
    # Gather rich context from the store
    papers = context.store.load_papers()
    abstract_only = all(p.content_level != "FULL_TEXT" for p in papers) if papers else True
    
    claims = context.store.load_claims()
    gaps = context.store.load_gaps()
    coverage_report = context.store.load_coverage_report()
    
    # Build context blocks for the prompt
    papers_block = "\n".join([
        f"- {p.title} (ID: {p.id}, Year: {p.year}) | "
        f"Methods: {', '.join(p.methods) if p.methods else 'N/A'} | "
        f"Datasets: {', '.join(p.datasets) if p.datasets else 'N/A'} | "
        f"Metrics: {', '.join(p.metrics) if p.metrics else 'N/A'}"
        for p in papers
    ])
    
    gaps_block = "\n".join([
        f"- Gap ID: {g.id} | {g.description} | Type: {g.gap_type} | "
        f"Dimensions: {json.dumps(g.dimensions)} | Novelty: {g.novelty_estimate}"
        for g in gaps
    ])
    
    coverage_warnings = ""
    if coverage_report and coverage_report.warnings:
        coverage_warnings = "\nCoverage Warnings:\n" + "\n".join(
            f"- {w}" for w in coverage_report.warnings
        )
    
    claims_block = "\n".join([
        f"- [{c.type}] {c.text} (from {c.paper_id})"
        for c in claims[:15]  # limit for context window
    ])

    feedback_block = ""
    if context.review_feedback:
        feedback_block = (
            f"\n\nIMPORTANT - PREVIOUS PLAN REVIEW FEEDBACK TO FIX:\n"
            f"{context.review_feedback}\n"
            f"You must revise the plan to address every single issue and suggestion listed above.\n"
        )

    system_prompt = (
        "You are a research planner agent. Create a DETAILED, EXECUTABLE research plan.\n\n"
        "You must return a valid JSON object matching the requested schema.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Data steps: create SEPARATE steps per domain (vision, speech, NLP, etc.)\n"
        "- Implementation steps: separate by component. If prototyping or initial training is involved, compute_hours MUST be > 0.\n"
        "- Experiment steps: specify exact datasets, model variants, scales, and baselines. Map specific metrics to evaluate in the 'metrics' list.\n"
        "- Ablation steps: specify exactly which variables are ablated. Include ALL relevant prior experiment and data steps as inputs, not just the preceding experiment.\n"
        "- Analysis steps: specify what plots/tables to produce\n"
        "- Writing steps: list specific manuscript sections. You MUST include 2-3 steps of type 'writing' (e.g., drafting intro/methods, preparing figures, submission) to make the plan end-to-end.\n"
        "- EVERY experiment step must have compute_hours > 0\n"
        "- EVERY step (except data steps) must have at least one input. EVERY step must have at least one output.\n"
        "- Use step_01, step_02, ... for cross-references in dependencies\n"
        "- Generate at least 12 steps for a thorough plan\n"
        "If the analysis is based only on paper abstracts (ABSTRACT_ONLY), add an explicit warning to the plan description or hypothesis."
    )
    
    context_note = "NOTE: Analysis is based on FULL_TEXT." if not abstract_only else "NOTE: Analysis is ABSTRACT_ONLY. Warn the user."
    
    user_prompt = (
        f"Create an executable research plan for this direction:\n\n"
        f"Hypothesis: {direction.hypothesis}\n"
        f"Justification: {direction.justification}\n"
        f"Primary Gap ID: {direction.primary_gap_id}\n\n"
        f"Available Papers:\n{papers_block}\n\n"
        f"Identified Gaps:\n{gaps_block}\n\n"
        f"Key Claims:\n{claims_block}\n"
        f"{coverage_warnings}"
        f"{feedback_block}\n\n"
        f"Context Info: {context_note}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = generate_structured(
        messages=messages,
        output_model=PlannerResponse,
        config=context.config,
        agent_name="planner",
        store=context.store,
        task_id=context.task_id
    )
    
    try:
        p_data = data.plan
        
        if context.config.seed is not None:
            plan_id = f"plan_{hashlib.sha1(f'{context.topic}_{context.config.seed}_plan'.encode()).hexdigest()[:8]}"
        else:
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        
        topic_spec = context.store.load_topic_spec()
        topic_spec_id = topic_spec.id if topic_spec else ""
        
        # Build the plan with all fields
        plan = ResearchPlan(
            id=plan_id,
            topic_spec_id=topic_spec_id,
            hypothesis=p_data.hypothesis,
            expected_contribution=p_data.expected_contribution,
            primary_gap_ids=p_data.primary_gap_ids,
            target_venue=p_data.target_venue,
            timeline_weeks=p_data.timeline_weeks,
        )
        
        # Parse steps with full metadata
        s_data = data.steps
        if not s_data:
            raise SchemaError("Planner response contained no steps.")
            
        parsed_steps = []
        for i, s_obj in enumerate(s_data):
            s = s_obj.model_dump()
            step_id = f"step_{(i + 1):02d}"
            
            # Sanitize dependencies (e.g., if LLM writes "step_01_collect_data", extract "step_01")
            raw_deps = s.get('dependencies', [])
            clean_deps = []
            for d in raw_deps:
                match = re.search(r'(step_\d+)', d)
                if match:
                    clean_deps.append(match.group(1))
                else:
                    clean_deps.append(d)
                    
            step = PlanStep(
                id=step_id,
                plan_id=plan_id,
                label=s.get('label', 'Unspecified step'),
                description=s.get('description', ''),
                type=s.get('type', 'experiment'),
                status='pending',
                risk_level=s.get('risk_level', 'medium'),
                inputs=s.get('inputs', []),
                outputs=s.get('outputs', []),
                dependencies=clean_deps,
                estimated_cost=s.get('estimated_cost', {"compute_hours": 0.0, "human_hours": 0.0}),
                assumptions=s.get('assumptions', []),
                metrics=s.get('metrics', []),
            )
            parsed_steps.append(step)
            
        # Deterministic Normalization Pass
        plan, parsed_steps = normalize_plan(plan, parsed_steps)
        
        # Store step IDs in plan
        plan.steps = [s.id for s in parsed_steps]
            
        context.store.save_plan(plan, parsed_steps)
        
    except Exception as e:
        raise SchemaError(f"Failed to parse plan: {e}")
