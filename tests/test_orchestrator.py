import pytest
from unittest.mock import patch, MagicMock
from neuralresearcher.orchestrator import Orchestrator, OrchestratorState
from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore
from neuralresearcher.state import TopicSpec, Paper, CoverageReport, CoverageCluster
from neuralresearcher.errors import WorkflowError

@patch("neuralresearcher.agents.topic_scope.run_topic_scope")
@patch("neuralresearcher.agents.retrieval.run_retrieval")
@patch("neuralresearcher.agents.reading.run_reading")
@patch("neuralresearcher.agents.coverage.run_coverage")
@patch("neuralresearcher.agents.gaps.run_gaps")
@patch("neuralresearcher.agents.directions.run_directions")
@patch("neuralresearcher.agents.planner.run_planner")
@patch("neuralresearcher.agents.reviewer.run_reviewer")
@patch("neuralresearcher.agents.reporting.run_reporting")
def test_orchestrator_state_transitions(
    mock_reporting, mock_reviewer, mock_planner, mock_directions, mock_gaps, mock_coverage, mock_reading, mock_retrieval, mock_topic_scope, tmp_path
):
    store = StateStore(directory=str(tmp_path))
    config = Config()
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    
    # Assert initial state
    assert orchestrator.state == OrchestratorState.INIT
    
    # Mock happy path for validations
    store.save_topic_spec(TopicSpec(id="1", raw_topic="t", domain="ML", subfields=[], keywords=["mamba"]))
    store.save_papers([
        Paper(id="p1", title="mamba paper", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p2", title="mamba paper 2", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p3", title="mamba paper 3", authors=[], venue="v", year=2024, url="", abstract="")
    ])
    store.save_coverage_report(CoverageReport(id="r1", clusters=[CoverageCluster(domain="ML")], warnings=[]))

    # Run the orchestrator
    orchestrator.run()
    
    # Assert all agents were called in order
    mock_topic_scope.assert_called_once_with(orchestrator)
    mock_retrieval.assert_called_once_with(orchestrator)
    mock_reading.assert_called_once_with(orchestrator)
    mock_coverage.assert_called_once_with(orchestrator)
    mock_gaps.assert_called_once_with(orchestrator)
    mock_directions.assert_called_once_with(orchestrator)
    mock_planner.assert_called_once_with(orchestrator)
    mock_reviewer.assert_called_once_with(orchestrator)
    mock_reporting.assert_called_once_with(orchestrator)
    
    # Assert final state
    assert orchestrator.state == OrchestratorState.REPORT_READY


def test_guardrail_layer_1_topic_validation(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config()
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    
    # Store out-of-scope topic spec
    store.save_topic_spec(TopicSpec(id="1", raw_topic="t", domain="Literature", subfields=[], keywords=[]))
    
    with pytest.raises(WorkflowError, match="Topic is out-of-scope"):
        orchestrator._validate_topic()
        
    # Store in-scope topic spec
    store.save_topic_spec(TopicSpec(id="1", raw_topic="t", domain="Deep Learning", subfields=[], keywords=[]))
    orchestrator._validate_topic() # Should not raise


def test_guardrail_layer_2_retrieval_validation(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config(min_papers=3, relevance_threshold=0.5)
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    
    store.save_topic_spec(TopicSpec(id="1", raw_topic="t", domain="ML", subfields=[], keywords=["mamba", "optimization"]))
    
    # Case 1: Too few papers
    store.save_papers([Paper(id="p1", title="mamba", authors=[], venue="v", year=2024, url="", abstract="")])
    with pytest.raises(WorkflowError, match="below the minimum threshold"):
        orchestrator._validate_retrieval()

    # Case 2: Enough papers, but low relevance
    store.save_papers([
        Paper(id="p1", title="mamba", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p2", title="irrelevant", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p3", title="irrelevant", authors=[], venue="v", year=2024, url="", abstract="")
    ])
    with pytest.raises(WorkflowError, match="Retrieval relevance too low"):
        orchestrator._validate_retrieval()
        
    # Case 3: Enough papers and high relevance
    store.save_papers([
        Paper(id="p1", title="mamba", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p2", title="optimization techniques", authors=[], venue="v", year=2024, url="", abstract=""),
        Paper(id="p3", title="irrelevant", authors=[], venue="v", year=2024, url="", abstract="")
    ])
    orchestrator._validate_retrieval() # 2/3 = 0.66 > 0.5, should not raise


def test_guardrail_layer_3_coverage_validation(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config(coverage_warning_threshold=0.6)
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    
    # Case 1: Too many warnings
    store.save_coverage_report(CoverageReport(
        id="r1",
        clusters=[CoverageCluster(domain="D1")],
        warnings=["w1", "w2"] # 2/1 = 2.0 > 0.6
    ))
    with pytest.raises(WorkflowError, match="Too many coverage warnings"):
        orchestrator._validate_coverage()
        
    # Case 2: Acceptable warnings
    store.save_coverage_report(CoverageReport(
        id="r1",
        clusters=[CoverageCluster(domain="D1"), CoverageCluster(domain="D2")],
        warnings=["w1"] # 1/2 = 0.5 <= 0.6
    ))
    orchestrator._validate_coverage() # Should not raise
