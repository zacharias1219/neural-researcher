"""Tests for Pydantic model serialization/deserialization with new fields."""
import json
import pytest
from neuralresearcher.state import (
    PlanStep, ResearchPlan, CoverageCluster, CoverageReport,
    ReviewResult, Paper, Claim, Gap, Direction, TopicSpec
)


class TestPlanStep:
    def test_new_fields_have_defaults(self):
        """PlanStep should work with just the required fields."""
        step = PlanStep(id="s1", label="Test step", type="experiment", risk_level="low")
        assert step.plan_id == ""
        assert step.description == ""
        assert step.status == "pending"
        assert step.inputs == []
        assert step.outputs == []
        assert step.dependencies == []
        assert step.estimated_cost == {"compute_hours": 0.0, "human_hours": 0.0}
        assert step.assumptions == []

    def test_full_fields_roundtrip(self):
        """PlanStep with all fields should serialize and deserialize cleanly."""
        step = PlanStep(
            id="step_01",
            plan_id="plan_abc",
            label="Train M3 on ImageNet",
            description="Train the M3 model on ImageNet-21k at base scale",
            type="experiment",
            status="in_progress",
            inputs=["gap_x1", "data/imagenet21k_preprocessed"],
            outputs=["checkpoints/m3_imagenet_base/"],
            dependencies=["step_00"],
            estimated_cost={"compute_hours": 48.0, "human_hours": 2.0},
            risk_level="high",
            assumptions=["4x A100 GPUs available", "ImageNet-21k license obtained"],
        )
        data = step.model_dump()
        restored = PlanStep(**data)
        assert restored.plan_id == "plan_abc"
        assert restored.status == "in_progress"
        assert restored.description == "Train the M3 model on ImageNet-21k at base scale"
        assert restored.estimated_cost["compute_hours"] == 48.0

    def test_json_roundtrip(self):
        step = PlanStep(id="s1", label="Test", type="data", risk_level="low", status="done")
        json_str = step.model_dump_json()
        restored = PlanStep.model_validate_json(json_str)
        assert restored.status == "done"


class TestResearchPlan:
    def test_new_fields_have_defaults(self):
        plan = ResearchPlan(
            id="p1", topic_spec_id="t1",
            hypothesis="test", expected_contribution="test"
        )
        assert plan.target_venue == ""
        assert plan.timeline_weeks == 0
        assert plan.resource_summary == {"total_compute_hours": 0.0, "total_human_hours": 0.0}

    def test_full_plan_roundtrip(self):
        plan = ResearchPlan(
            id="plan_01",
            topic_spec_id="topic_abc",
            primary_gap_ids=["gap_01", "gap_02"],
            hypothesis="M3 scales to large vision tasks",
            steps=["step_01", "step_02"],
            expected_contribution="First large-scale Mamba evaluation",
            target_venue="NeurIPS",
            timeline_weeks=24,
            resource_summary={"total_compute_hours": 500.0, "total_human_hours": 120.0},
        )
        data = plan.model_dump()
        restored = ResearchPlan(**data)
        assert restored.target_venue == "NeurIPS"
        assert restored.timeline_weeks == 24
        assert restored.resource_summary["total_compute_hours"] == 500.0


class TestCoverageReport:
    def test_empty_report(self):
        report = CoverageReport(id="cov_01")
        assert report.clusters == []
        assert report.warnings == []

    def test_full_report_roundtrip(self):
        report = CoverageReport(
            id="cov_01",
            clusters=[
                CoverageCluster(domain="vision", paper_ids=["p1", "p2"], claim_count=5),
                CoverageCluster(domain="nlp", paper_ids=["p3"], claim_count=2),
            ],
            warnings=["No papers cover 'speech' domain"],
            timestamp="2024-01-01T00:00:00Z",
        )
        data = report.model_dump()
        restored = CoverageReport(**data)
        assert len(restored.clusters) == 2
        assert restored.clusters[0].domain == "vision"
        assert restored.clusters[0].claim_count == 5
        assert len(restored.warnings) == 1


class TestReviewResult:
    def test_defaults(self):
        result = ReviewResult(id="rev_01")
        assert result.passed is True
        assert result.issues == []
        assert result.suggestions == []

    def test_failed_review_roundtrip(self):
        result = ReviewResult(
            id="rev_01",
            passed=False,
            issues=["No experiment steps", "Empty primary_gap_ids"],
            suggestions=["Add baselines"],
            timestamp="2024-01-01T00:00:00Z",
        )
        data = result.model_dump()
        restored = ReviewResult(**data)
        assert restored.passed is False
        assert len(restored.issues) == 2


class TestPaperClaimGap:
    """Ensure existing models still work after imports changed."""
    
    def test_paper_metadata_fields(self):
        paper = Paper(
            id="p1", title="Test", authors=["A"], venue="arXiv",
            year=2024, url="http://test", abstract="test",
            methods=["Mamba"], datasets=["ImageNet"],
            metrics=["accuracy"], limitations=["small scale"],
            explicit_future_work=["large scale evaluation"],
        )
        assert paper.methods == ["Mamba"]
        data = paper.model_dump()
        assert data["methods"] == ["Mamba"]

    def test_claim_with_datasets_metrics(self):
        claim = Claim(
            id="c1", paper_id="p1", type="result",
            text="Test claim", section="abstract",
            evidence_ref="http://test",
            datasets=["COCO"], metrics=["mAP"],
        )
        assert claim.datasets == ["COCO"]

    def test_gap_with_evidence(self):
        gap = Gap(
            id="g1", description="Test gap",
            gap_type="unexplored_axis", novelty_estimate="high",
            related_papers=["p1", "p2"],
            supporting_claims=["c1"],
            dimensions={"domain": "vision", "scale": "large"},
        )
        assert gap.related_papers == ["p1", "p2"]
        assert gap.dimensions["domain"] == "vision"
