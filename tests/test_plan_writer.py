"""Tests for the plan_writer rendering all 9 sections."""
import pytest
from pathlib import Path

from neuralresearcher.state import (
    ResearchPlan, PlanStep, Paper, Gap, Direction,
    Claim, CoverageReport, CoverageCluster, ReviewResult,
)
from neuralresearcher.io.plan_writer import render_markdown_plan


@pytest.fixture
def fixture_data(tmp_path):
    """Build a complete fixture dataset for plan writer testing."""
    papers = [
        Paper(
            id="p1", title="Mamba: Linear-Time Sequence Modeling",
            authors=["A. Gu"], venue="ICLR", year=2024,
            url="http://arxiv.org/abs/mamba",
            abstract="We propose Mamba for efficient sequence modeling.",
            methods=["Mamba SSM", "Selective State Spaces"],
            datasets=["WikiText-103", "The Pile"],
            metrics=["perplexity", "throughput"],
            limitations=["Only NLP benchmarks"],
        ),
        Paper(
            id="p2", title="Vision Mamba",
            authors=["B. Liu"], venue="arXiv", year=2024,
            url="http://arxiv.org/abs/vimamba",
            abstract="Mamba adapted for image classification.",
            methods=["Mamba", "Patch Embedding"],
            datasets=["ImageNet"],
            metrics=["accuracy", "FLOPs"],
        ),
    ]
    
    claims = [
        Claim(id="c1", paper_id="p1", type="result",
              text="Mamba achieves 3x throughput vs Transformer",
              section="abstract", evidence_ref="http://test",
              datasets=["WikiText-103"], metrics=["throughput"]),
        Claim(id="c2", paper_id="p1", type="limitation",
              text="Only evaluated on NLP tasks",
              section="abstract", evidence_ref="http://test"),
    ]
    
    gaps = [
        Gap(id="gap_01", description="Mamba not evaluated on large-scale vision",
            gap_type="unexplored_axis", novelty_estimate="high",
            related_papers=["p1", "p2"], supporting_claims=["c2"],
            dimensions={"domain": "vision", "model": "Mamba", "scale": "large"}),
        Gap(id="gap_02", description="No speech benchmarks for Mamba",
            gap_type="limitation", novelty_estimate="medium"),
    ]
    
    directions = [
        Direction(id="dir_01", primary_gap_id="gap_01",
                  hypothesis="Mamba scales to large vision tasks",
                  justification="Only small-scale tests exist",
                  expected_contribution_type="empirical evaluation",
                  novelty_assessment="high"),
    ]
    
    steps = [
        PlanStep(id="step_01", plan_id="plan_01", label="Preprocess ImageNet-21k",
                 description="Download and preprocess ImageNet-21k into 224x224 patches",
                 type="data", risk_level="medium",
                 inputs=["gap_01"], outputs=["data/imagenet21k/"],
                 estimated_cost={"compute_hours": 8.0, "human_hours": 4.0},
                 assumptions=["2TB storage available"]),
        PlanStep(id="step_02", plan_id="plan_01", label="Implement Vision Adapter",
                 description="Build patch embedding and positional encoding for Mamba",
                 type="implementation", risk_level="medium",
                 inputs=["p1"], outputs=["src/vision_adapter.py"],
                 dependencies=["step_01"],
                 estimated_cost={"compute_hours": 0.0, "human_hours": 16.0}),
        PlanStep(id="step_03", plan_id="plan_01", label="Train M3 vs ViT baseline on ImageNet",
                 description="Compare M3 against ViT-Base at comparable parameter count",
                 type="experiment", risk_level="high",
                 inputs=["step_01", "step_02"], outputs=["checkpoints/m3_imagenet/"],
                 dependencies=["step_01", "step_02"],
                 estimated_cost={"compute_hours": 96.0, "human_hours": 8.0},
                 assumptions=["4x A100 GPUs available"]),
        PlanStep(id="step_04", plan_id="plan_01", label="Ablate patch sizes",
                 description="Test 8x8, 16x16, 32x32 patch sizes",
                 type="ablation", risk_level="medium",
                 inputs=["step_03"], outputs=["results/patch_ablation.csv"],
                 dependencies=["step_03"],
                 estimated_cost={"compute_hours": 48.0, "human_hours": 4.0}),
        PlanStep(id="step_05", plan_id="plan_01", label="Analyze scaling curves",
                 description="Fit power-law scaling curves for M3 vs ViT",
                 type="analysis", risk_level="low",
                 inputs=["step_03", "step_04"], outputs=["figures/scaling_curves.pdf"],
                 dependencies=["step_03", "step_04"],
                 estimated_cost={"compute_hours": 1.0, "human_hours": 8.0}),
        PlanStep(id="step_06", plan_id="plan_01", label="Draft Experiments Section",
                 description="Write the experiments section describing datasets, setup, and results",
                 type="writing", risk_level="low",
                 inputs=["step_05"], outputs=["manuscript/experiments.tex"],
                 dependencies=["step_05"],
                 estimated_cost={"compute_hours": 0.0, "human_hours": 16.0}),
    ]
    
    plan = ResearchPlan(
        id="plan_01", topic_spec_id="topic_01",
        primary_gap_ids=["gap_01"],
        hypothesis="Mamba-based M3 achieves competitive accuracy on large-scale vision",
        expected_contribution="First large-scale Mamba evaluation across vision benchmarks",
        target_venue="NeurIPS",
        timeline_weeks=24,
        resource_summary={"total_compute_hours": 153.0, "total_human_hours": 56.0},
        steps=[s.id for s in steps],
    )
    
    coverage_report = CoverageReport(
        id="cov_01",
        clusters=[
            CoverageCluster(domain="nlp", paper_ids=["p1"], claim_count=2),
            CoverageCluster(domain="vision", paper_ids=["p2"], claim_count=0),
        ],
        warnings=["No papers cover the 'speech' domain — literature coverage may be incomplete."],
        timestamp="2024-01-01T00:00:00Z",
    )
    
    review_result = ReviewResult(
        id="rev_01", passed=True,
        issues=[], suggestions=["Add more baselines"],
        timestamp="2024-01-01T00:00:00Z",
    )
    
    output_path = str(tmp_path / "research_plan.md")
    
    return {
        "plan": plan, "steps": steps, "papers": papers,
        "gaps": gaps, "directions": directions, "claims": claims,
        "coverage_report": coverage_report, "review_result": review_result,
        "output_path": output_path,
    }


class TestPlanWriter:
    def _render_and_read(self, fixture_data):
        render_markdown_plan(**fixture_data)
        return Path(fixture_data["output_path"]).read_text(encoding="utf-8")

    def test_section_1_overview(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 1. Overview" in content
        assert "Mamba-based M3" in content
        assert "NeurIPS" in content
        assert "24 weeks" in content

    def test_section_2_field_summary(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 2. Field Summary" in content
        assert "Mamba: Linear-Time" in content

    def test_section_3_gap_analysis(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 3. Gap Analysis" in content
        assert "Primary Gaps Addressed" in content
        assert "large-scale vision" in content
        assert "Dimensions" in content
        assert "vision" in content

    def test_section_3_coverage_warnings(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "Coverage Warnings" in content
        assert "speech" in content

    def test_section_4_experiment_design(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 4. Experiment Design" in content
        assert "Datasets" in content
        assert "ImageNet-21k" in content
        assert "Experiments & Baselines" in content

    def test_section_4_metrics(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "Evaluation Metrics" in content
        assert "perplexity" in content

    def test_section_4_hardware(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "Hardware Assumptions" in content
        assert "A100" in content

    def test_section_5_sop(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 5. Standard Operating Procedure" in content
        assert "Data Phase" in content
        assert "Experiment Phase" in content
        assert "`step_01`" in content
        assert "96.0" in content  # compute hours

    def test_section_6_timeline(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 6. Timeline & Resources" in content
        assert "153.0" in content  # total compute
        assert "56.0" in content   # total human
        assert "Per-Phase Cost Breakdown" in content

    def test_section_7_manuscript_outline(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 7. Manuscript Outline" in content
        assert "Experiments Section" in content

    def test_section_8_review_summary(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 8. Review Summary" in content
        assert "PASSED" in content
        assert "Add more baselines" in content

    def test_section_9_references(self, fixture_data):
        content = self._render_and_read(fixture_data)
        assert "## 9. Key Literature References" in content
        assert "Mamba: Linear-Time" in content
        assert "Vision Mamba" in content

    def test_all_9_sections_present(self, fixture_data):
        content = self._render_and_read(fixture_data)
        for i in range(1, 10):
            assert f"## {i}." in content, f"Section {i} is missing"

    def test_graceful_with_no_optional_data(self, fixture_data):
        """Should render without crashing when optional data is None."""
        fixture_data["claims"] = None
        fixture_data["coverage_report"] = None
        fixture_data["review_result"] = None
        content = self._render_and_read(fixture_data)
        assert "## 1. Overview" in content
        assert "No review has been performed" in content

    def test_failed_review_renders_issues(self, fixture_data):
        fixture_data["review_result"] = ReviewResult(
            id="rev_fail", passed=False,
            issues=["No experiment steps", "Empty gap IDs"],
        )
        content = self._render_and_read(fixture_data)
        assert "FAILED" in content
        assert "No experiment steps" in content
