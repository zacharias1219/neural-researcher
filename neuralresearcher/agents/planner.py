from typing import Any
import json
import uuid
from neuralresearcher.state import ResearchPlan, PlanStep
from neuralresearcher.llm import call_llm

def run_planner(orchestrator: Any) -> None:
    directions = orchestrator.store.load_directions()
    if not directions:
        return
        
    direction = directions[0]  # Pick the top-ranked direction
    
    # Gather rich context from the store
    papers = orchestrator.store.load_papers()
    claims = orchestrator.store.load_claims()
    gaps = orchestrator.store.load_gaps()
    coverage_report = orchestrator.store.load_coverage_report()
    
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

    system_prompt = (
        "You are a research planner agent. Create a DETAILED, EXECUTABLE research plan.\n\n"
        "You must return a SINGLE valid JSON object with exactly two top-level keys: 'plan' and 'steps'.\n\n"
        "Example JSON structure:\n"
        "{\n"
        '  "plan": {\n'
        '    "hypothesis": "precise, testable hypothesis",\n'
        '    "expected_contribution": "what this work contributes to the field",\n'
        '    "target_venue": "target journal/conference (e.g., NeurIPS, ICML, JMLR)",\n'
        '    "timeline_weeks": 12,\n'
        '    "primary_gap_ids": ["gap IDs this plan addresses"]\n'
        '  },\n'
        '  "steps": [\n'
        '    {\n'
        '      "label": "concise but specific step name",\n'
        '      "description": "detailed description of what to do and how",\n'
        '      "type": "data",\n'
        '      "risk_level": "low",\n'
        '      "inputs": ["gap IDs, paper IDs, or prior step outputs this depends on (data steps can have empty inputs)"],\n'
        '      "outputs": ["concrete artifacts: dataset paths, model checkpoints, scripts, figures, sections"],\n'
        '      "dependencies": ["MUST be EXACTLY the ID string of the step (e.g. \\"step_01\\"). DO NOT use labels"],\n'
        '      "estimated_cost": {"compute_hours": 0.0, "human_hours": 10.0},\n'
        '      "assumptions": ["what must be true for this step to succeed"],\n'
        '      "metrics": ["list of exact metrics to evaluate in this step (for experiments/ablations)"]\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
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
    )
    
    user_prompt = (
        f"Create an executable research plan for this direction:\n\n"
        f"Hypothesis: {direction.hypothesis}\n"
        f"Justification: {direction.justification}\n"
        f"Primary Gap ID: {direction.primary_gap_id}\n\n"
        f"Available Papers:\n{papers_block}\n\n"
        f"Identified Gaps:\n{gaps_block}\n\n"
        f"Key Claims:\n{claims_block}\n"
        f"{coverage_warnings}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        config=orchestrator.config,
        messages=messages,
        response_format={"type": "json_object"},
        store=orchestrator.store,
        task_id=orchestrator.task_id,
        agent_name="planner"
    )
    
    try:
        data = json.loads(response.content)
        p_data = data.get("plan", {})
        
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        
        # Build the plan with all fields
        plan = ResearchPlan(
            id=plan_id,
            topic_spec_id=orchestrator.store.load_topic_spec().id,
            hypothesis=p_data.get('hypothesis', direction.hypothesis),
            expected_contribution=p_data.get('expected_contribution', direction.expected_contribution_type),
            primary_gap_ids=p_data.get('primary_gap_ids', [direction.primary_gap_id]),
            target_venue=p_data.get('target_venue', ''),
            timeline_weeks=p_data.get('timeline_weeks', 0),
        )
        
        # Parse steps with full metadata
        s_data = data.get("steps", [])
        parsed_steps = []
        for i, s in enumerate(s_data):
            step_id = f"step_{(i + 1):02d}"
            import re
            
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
        
        # Compute resource summary
        total_compute = sum(s.estimated_cost.get("compute_hours", 0.0) for s in parsed_steps)
        total_human = sum(s.estimated_cost.get("human_hours", 0.0) for s in parsed_steps)
        plan.resource_summary = {
            "total_compute_hours": total_compute,
            "total_human_hours": total_human,
        }
        
        # Store step IDs in plan
        plan.steps = [s.id for s in parsed_steps]
            
        orchestrator.store.save_plan(plan, parsed_steps)
        
    except Exception as e:
        from neuralresearcher.logging import log_error
        log_error(f"Failed to parse plan: {e}")
        return
