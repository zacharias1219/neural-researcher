from typing import Any
import uuid
from datetime import datetime, timezone

from neuralresearcher.state import CoverageCluster, CoverageReport
from neuralresearcher.logging import log_info, log_info as log_warning

# Fixed domain taxonomy for keyword-based clustering
DOMAIN_TAXONOMY = {
    "vision": [
        "image", "vision", "visual", "cnn", "vit", "convolutional",
        "object detection", "segmentation", "classification", "patch",
        "imagenet", "coco", "cifar", "resnet", "pixel",
    ],
    "nlp": [
        "language", "text", "nlp", "token", "tokenizer", "gpt",
        "bert", "transformer", "lm", "translation", "sentiment",
        "c4", "pile", "wikitext", "glue", "summarization",
    ],
    "speech": [
        "audio", "speech", "acoustic", "asr", "tts",
        "librispeech", "voxpopuli", "spectrogram", "mel",
        "keyword spotting", "speaker", "phoneme",
    ],
    "rl": [
        "reinforcement", "policy", "reward", "agent", "environment",
        "q-learning", "ppo", "sac", "mujoco", "atari", "gymnasium",
        "multi-agent", "exploration", "exploitation",
    ],
    "edge": [
        "microcontroller", "mcu", "tinyml", "edge", "embedded",
        "quantization", "pruning", "distillation", "low-power",
        "arm", "cortex", "risc-v", "iot", "on-device",
    ],
    "time_series": [
        "time series", "forecasting", "temporal", "sequence",
        "autoregressive", "recurrent", "lstm", "solar", "electricity",
        "weather", "anomaly detection",
    ],
}


def _classify_paper(paper, taxonomy: dict) -> list[str]:
    """Assign a paper to one or more domains based on keyword matching."""
    searchable = (paper.title + " " + paper.abstract).lower()
    # Also include methods and datasets if populated
    if paper.methods:
        searchable += " " + " ".join(paper.methods).lower()
    if paper.datasets:
        searchable += " " + " ".join(paper.datasets).lower()
    
    matched_domains = []
    for domain, keywords in taxonomy.items():
        for kw in keywords:
            if kw in searchable:
                matched_domains.append(domain)
                break  # one match per domain is enough
    
    return matched_domains if matched_domains else ["uncategorized"]


def run_coverage(orchestrator: Any) -> None:
    papers = orchestrator.store.load_papers()
    claims = orchestrator.store.load_claims()
    
    if not papers:
        log_info("Coverage agent: no papers to cluster.")
        return
    
    # --- Cluster papers by domain ---
    domain_to_papers: dict[str, list[str]] = {}
    for paper in papers:
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        for domain in domains:
            domain_to_papers.setdefault(domain, []).append(paper.id)
    
    # --- Count claims per domain ---
    paper_id_to_domains: dict[str, list[str]] = {}
    for domain, pids in domain_to_papers.items():
        for pid in pids:
            paper_id_to_domains.setdefault(pid, []).append(domain)
    
    domain_claim_counts: dict[str, int] = {}
    for claim in claims:
        domains_for_paper = paper_id_to_domains.get(claim.paper_id, ["uncategorized"])
        for d in domains_for_paper:
            domain_claim_counts[d] = domain_claim_counts.get(d, 0) + 1
    
    # --- Build CoverageCluster objects ---
    clusters = []
    for domain, paper_ids in sorted(domain_to_papers.items()):
        clusters.append(CoverageCluster(
            domain=domain,
            paper_ids=list(set(paper_ids)),  # deduplicate
            claim_count=domain_claim_counts.get(domain, 0)
        ))
    
    # --- Flag coverage warnings ---
    warnings = []
    
    # Check for domains with no papers at all
    covered_domains = set(domain_to_papers.keys()) - {"uncategorized"}
    all_taxonomy_domains = set(DOMAIN_TAXONOMY.keys())
    uncovered = all_taxonomy_domains - covered_domains
    for domain in sorted(uncovered):
        warnings.append(f"No papers cover the '{domain}' domain — literature coverage may be incomplete.")
    
    # Check for thin coverage
    for domain, paper_ids in domain_to_papers.items():
        if domain != "uncategorized" and len(set(paper_ids)) < 2:
            warnings.append(f"Only {len(set(paper_ids))} paper(s) in '{domain}' domain — coverage may be thin.")
    
    # Check for missing claim types
    claim_types_found = {c.type for c in claims}
    if "limitation" not in claim_types_found:
        warnings.append("No claims of type 'limitation' found — limitations may be under-represented.")
    if "future_work" not in claim_types_found:
        warnings.append("No claims of type 'future_work' found — future work directions not captured.")
    
    # Check for uncategorized papers
    uncategorized_count = len(set(domain_to_papers.get("uncategorized", [])))
    if uncategorized_count > 0:
        warnings.append(f"{uncategorized_count} paper(s) could not be assigned to any domain.")
    
    # --- Build and save report ---
    report = CoverageReport(
        id=f"coverage_{uuid.uuid4().hex[:8]}",
        clusters=clusters,
        warnings=warnings,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    orchestrator.store.save_coverage_report(report)
    
    log_info(f"Coverage agent: {len(clusters)} domain cluster(s), {len(warnings)} warning(s).")
    for w in warnings:
        log_warning(f"  [!] {w}")
