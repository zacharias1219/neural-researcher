import hashlib
import uuid
from datetime import datetime, timezone
from typing import List

from neuralresearcher.context import AgentContext
from neuralresearcher.state import CoverageCluster, CoverageReport
from neuralresearcher.logging import log_info, log_warning

from neuralresearcher.llm import generate_structured
from neuralresearcher.errors import SchemaError, WorkflowError
from pydantic import BaseModel, Field

class CoverageClusterOutput(BaseModel):
    domain: str
    paper_ids: List[str]

class CoverageResponse(BaseModel):
    clusters: List[CoverageClusterOutput]
    warnings: List[str] = Field(default_factory=list)


def run_coverage(context: AgentContext) -> None:
    papers = context.store.load_papers()
    claims = context.store.load_claims()
    
    if not papers:
        log_info("Coverage agent: no papers to cluster.")
        return
    
    topic_spec = context.store.load_topic_spec()
    topic_context = f"Topic: {topic_spec.raw_topic} (Domain: {topic_spec.domain.value})" if topic_spec else "Unknown topic"
    
    papers_block = "\n".join([
        f"- ID: {p.id} | Title: {p.title} | Abstract snippet: {p.abstract[:200]}..."
        for p in papers
    ])
    
    claims_block = "\n".join([
        f"- ID: {c.paper_id} | Type: {c.type} | Text: {c.text}"
        for c in claims
    ])
    
    system_prompt = (
        "You are an expert academic reviewer. Cluster the provided papers into specific sub-domains relative to the research topic.\n"
        "Return a JSON object with two keys: 'clusters' and 'warnings'.\n"
        "'clusters' is a list of objects with 'domain' (string) and 'paper_ids' (list of string IDs).\n"
        "'warnings' is a list of strings indicating any coverage gaps (e.g., missing critical sub-themes, thin coverage).\n"
    )
    
    user_prompt = (
        f"{topic_context}\n\n"
        f"Papers:\n{papers_block}\n\n"
        f"Claims:\n{claims_block}\n\n"
        "Analyze the papers and claims, identify sub-domains relative to the topic, cluster the paper IDs, and report warnings if necessary."
    )
    
    data = generate_structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        output_model=CoverageResponse,
        config=context.config,
        agent_name="coverage",
        store=context.store,
        task_id=context.task_id
    )
    
    try:
        raw_clusters = data.clusters
        warnings = data.warnings
    except Exception as e:
        raise SchemaError(f"Failed to parse coverage JSON: {e}")
        
    if not raw_clusters:
        raise WorkflowError("No coverage clusters generated.")
        
    # Build actual cluster objects
    clusters = []
    
    # Calculate claim counts per paper
    paper_claims = {}
    for c in claims:
        paper_claims[c.paper_id] = paper_claims.get(c.paper_id, 0) + 1
        
    for rc in raw_clusters:
        domain = rc.domain
        pids = rc.paper_ids
        
        # Calculate claims for this cluster
        claim_count = sum(paper_claims.get(pid, 0) for pid in pids)
        clusters.append(CoverageCluster(domain=domain, paper_ids=pids, claim_count=claim_count))
    
    # Deterministic ID & timestamp if seed is set
    if context.config.seed is not None:
        report_id = f"coverage_{hashlib.sha1(f'{context.topic}_{context.config.seed}_cov'.encode()).hexdigest()[:8]}"
        timestamp = "2024-01-01T00:00:00+00:00"
    else:
        report_id = f"coverage_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

    # --- Build and save report ---
    report = CoverageReport(
        id=report_id,
        clusters=clusters,
        warnings=warnings,
        timestamp=timestamp
    )
    
    context.store.save_coverage_report(report)
    
    log_info(f"Coverage agent: {len(clusters)} domain cluster(s), {len(warnings)} warning(s).")
    for w in warnings:
        log_warning(f"  [!] {w}")
