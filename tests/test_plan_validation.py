from neuralresearcher.state import ResearchPlan, PlanStep
from neuralresearcher.plan_validation import validate_plan

def test_validate_happy_path():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1", "s2", "s3"])
    steps = [
        PlanStep(id="s1", label="experiment", type="experiment", outputs=["exp_out"], risk_level="low"),
        PlanStep(id="s2", label="analysis", type="analysis", inputs=["exp_out"], outputs=["analysis_out"], dependencies=["s1"], risk_level="low"),
        PlanStep(id="s3", label="writing", type="writing", inputs=["analysis_out"], outputs=["paper.pdf"], dependencies=["s2"], risk_level="low")
    ]
    issues = validate_plan(plan, steps)
    assert not issues

def test_missing_experiment():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
    steps = [PlanStep(id="s1", label="writing", type="writing", outputs=["paper.pdf"], risk_level="low")]
    issues = validate_plan(plan, steps)
    assert any(i.code == "MISSING_EXPERIMENT" for i in issues)

def test_missing_input_producer():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
    steps = [PlanStep(id="s1", label="experiment", type="experiment", inputs=["nonexistent_input"], outputs=["out"], risk_level="low")]
    issues = validate_plan(plan, steps)
    assert any(i.code == "MISSING_INPUT_PRODUCER" for i in issues)

def test_dependency_cycle():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1", "s2"])
    steps = [
        PlanStep(id="s1", label="experiment", type="experiment", outputs=["out1"], dependencies=["s2"], risk_level="low"),
        PlanStep(id="s2", label="analysis", type="analysis", inputs=["out1"], outputs=["out2"], dependencies=["s1"], risk_level="low")
    ]
    issues = validate_plan(plan, steps)
    assert any(i.code == "DEPENDENCY_CYCLE" for i in issues)

def test_disconnected_analysis():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1", "s2"])
    steps = [
        PlanStep(id="s1", label="experiment", type="experiment", outputs=["exp_out"], risk_level="low"),
        PlanStep(id="s2", label="analysis", type="analysis", inputs=["external:data"], outputs=["analysis_out"], risk_level="low")
    ]
    issues = validate_plan(plan, steps)
    assert any(i.code == "DISCONNECTED_ANALYSIS" for i in issues)

def test_orphaned_artifact():
    plan = ResearchPlan(id="1", topic_spec_id="1", hypothesis="h", expected_contribution="c", steps=["s1"])
    steps = [PlanStep(id="s1", label="experiment", type="experiment", outputs=["orphaned_out"], risk_level="low")]
    issues = validate_plan(plan, steps)
    assert any(i.code == "ORPHANED_ARTIFACT" for i in issues)
