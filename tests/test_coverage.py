import pytest
from unittest.mock import patch

from neuralresearcher.state import Paper, Claim, TopicSpec, TopicDomain
from neuralresearcher.context import AgentContext
from neuralresearcher.io.store import StateStore
from neuralresearcher.config import Config
from neuralresearcher.errors import WorkflowError, SchemaError
from neuralresearcher.agents.coverage import run_coverage, CoverageResponse, CoverageClusterOutput

@pytest.fixture
def tmp_store(tmp_path):
    store = StateStore(directory=str(tmp_path / "research"))
    store.save_topic_spec(TopicSpec(id="1", raw_topic="test", domain=TopicDomain.MACHINE_LEARNING, subfields=[], keywords=[]))
    store.save_papers([
        Paper(id="p1", title="title1", authors=[], venue="", year=2024, url="", abstract="abs1"),
        Paper(id="p2", title="title2", authors=[], venue="", year=2024, url="", abstract="abs2")
    ])
    store.save_claims([
        Claim(id="c1", paper_id="p1", type="result", text="t1", section="abstract", evidence_ref="text"),
        Claim(id="c2", paper_id="p2", type="result", text="t2", section="abstract", evidence_ref="text")
    ])
    return store

@patch("neuralresearcher.agents.coverage.generate_structured")
def test_coverage_success(mock_gen, tmp_store):
    mock_gen.return_value = CoverageResponse(
        clusters=[CoverageClusterOutput(domain="sub1", paper_ids=["p1", "p2"])],
        warnings=["warning 1"]
    )
    
    ctx = AgentContext(topic="test", config=Config(), store=tmp_store, task_id="test")
    run_coverage(ctx)
    
    report = tmp_store.load_coverage_report()
    assert report is not None
    assert len(report.clusters) == 1
    assert report.clusters[0].domain == "sub1"
    assert report.clusters[0].claim_count == 2
    assert len(report.warnings) == 1
    assert report.warnings[0] == "warning 1"

@patch("neuralresearcher.agents.coverage.generate_structured")
def test_coverage_empty_clusters(mock_gen, tmp_store):
    mock_gen.return_value = CoverageResponse(
        clusters=[],
        warnings=[]
    )
    
    ctx = AgentContext(topic="test", config=Config(), store=tmp_store, task_id="test")
    with pytest.raises(WorkflowError, match="No coverage clusters generated"):
        run_coverage(ctx)

@patch("neuralresearcher.agents.coverage.generate_structured")
def test_coverage_schema_error(mock_gen, tmp_store):
    mock_gen.side_effect = SchemaError("Failed to parse coverage JSON: ...")
    
    ctx = AgentContext(topic="test", config=Config(), store=tmp_store, task_id="test")
    with pytest.raises(SchemaError, match="Failed to parse coverage JSON"):
        run_coverage(ctx)
