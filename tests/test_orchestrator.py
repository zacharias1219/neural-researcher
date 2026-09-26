import pytest
from unittest.mock import patch, MagicMock
from neuralresearcher.orchestrator import Orchestrator, OrchestratorState
from neuralresearcher.context import AgentContext
from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore
from neuralresearcher.state import TopicSpec, Paper, CoverageReport, CoverageCluster, ReviewResult
from neuralresearcher.errors import WorkflowError

@patch("neuralresearcher.orchestrator.run_topic_scope")
@patch("neuralresearcher.orchestrator.run_retrieval")
@patch("neuralresearcher.orchestrator.run_reading")
@patch("neuralresearcher.orchestrator.run_coverage")
@patch("neuralresearcher.orchestrator.run_gaps")
@patch("neuralresearcher.orchestrator.run_directions")
@patch("neuralresearcher.orchestrator.run_planner")
@patch("neuralresearcher.orchestrator.run_reviewer")
@patch("neuralresearcher.orchestrator.run_reporting")
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

    # Reviewer passes by default
    store.save_review_result(ReviewResult(id="rev1", passed=True, issues=[], suggestions=[]))

    def fake_planner(ctx: AgentContext):
        from neuralresearcher.state import ResearchPlan, PlanStep
        plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
        step = PlanStep(id="s1", label="l", type="experiment", risk_level="low")
        ctx.store.save_plan(plan, [step])

    mock_planner.side_effect = fake_planner

    # Run the orchestrator
    orchestrator.run()
    
    # Assert all agents were called with AgentContext
    for mock_agent in [
        mock_topic_scope, mock_retrieval, mock_reading, mock_coverage,
        mock_gaps, mock_directions, mock_planner, mock_reviewer, mock_reporting
    ]:
        mock_agent.assert_called_once()
        ctx = mock_agent.call_args[0][0]
        assert isinstance(ctx, AgentContext)
        assert ctx.topic == "test"
        assert ctx.task_id == "test_task"
    
    # Assert final state
    assert orchestrator.state == OrchestratorState.REPORT_READY


def test_orchestrator_reviewer_retry_loop(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config(max_retries=2, strict_mode=True)
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    orchestrator.set_state(OrchestratorState.DIRECTIONS_PROPOSED)

    call_count = 0
    feedback_seen = []

    def fake_planner(ctx: AgentContext):
        nonlocal feedback_seen
        feedback_seen.append(ctx.review_feedback)
        from neuralresearcher.state import ResearchPlan, PlanStep
        plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
        step = PlanStep(id="s1", label="l", type="experiment", risk_level="low")
        ctx.store.save_plan(plan, [step])

    def fake_reviewer(ctx: AgentContext):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt fails
            store.save_review_result(ReviewResult(id="rev1", passed=False, issues=["Missing ablation steps"], suggestions=[]))
        else:
            # Second attempt passes
            store.save_review_result(ReviewResult(id="rev2", passed=True, issues=[], suggestions=[]))

    with patch("neuralresearcher.orchestrator.run_planner", side_effect=fake_planner), \
         patch("neuralresearcher.orchestrator.run_reviewer", side_effect=fake_reviewer), \
         patch("neuralresearcher.orchestrator.run_directions") as mock_run_directions, \
         patch("neuralresearcher.orchestrator.run_reporting"):
        orchestrator.run()

    assert call_count == 2
    assert feedback_seen[0] is None  # Initial attempt has no feedback
    assert "Missing ablation steps" in feedback_seen[1]  # Second attempt got feedback
    assert orchestrator.state == OrchestratorState.REPORT_READY
    mock_run_directions.assert_not_called()


def test_orchestrator_reviewer_retry_exhausted_strict_mode(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config(max_retries=2, strict_mode=True)
    orchestrator = Orchestrator(topic="test", config=config, store=store, task_id="test_task")
    orchestrator.set_state(OrchestratorState.DIRECTIONS_PROPOSED)

    def fake_reviewer(ctx: AgentContext):
        store.save_review_result(ReviewResult(id="rev_fail", passed=False, issues=["Persistent issue"], suggestions=[]))

    def fake_planner(ctx: AgentContext):
        from neuralresearcher.state import ResearchPlan, PlanStep
        plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
        step = PlanStep(id="s1", label="l", type="experiment", risk_level="low")
        ctx.store.save_plan(plan, [step])

    with patch("neuralresearcher.orchestrator.run_planner", side_effect=fake_planner), \
         patch("neuralresearcher.orchestrator.run_reviewer", side_effect=fake_reviewer):
        orchestrator.run()

    # In strict mode, exhausting retries should transition to HALTED
    assert orchestrator.state == OrchestratorState.HALTED


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
def test_empty_plan_validation_silent_success(tmp_path):
    store = StateStore(directory=str(tmp_path))
    config = Config()
    orchestrator = Orchestrator(topic='test', config=config, store=store, task_id='test_task')
    
    with pytest.raises(WorkflowError, match='Plan generation yielded no steps'):
        orchestrator._to_plan()

