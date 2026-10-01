from pathlib import Path

from neuralresearcher.agents.reading import run_reading
from neuralresearcher.config import Config
from neuralresearcher.context import AgentContext
from neuralresearcher.io.plan_writer import render_markdown_plan
from neuralresearcher.io.store import StateStore
from neuralresearcher.state import Claim, ContentLevel, Paper, ResearchPlan
from neuralresearcher.tools import IMPLEMENTATIONS, TOOL_SCHEMAS


def test_extract_sections_impl_absent_from_production():
    """Prove extract_sections_impl is not in production modules or tool registry."""
    for schema in TOOL_SCHEMAS:
        assert schema["function"]["name"] != "extract_sections_impl"
        assert "extract_sections_impl" not in str(schema)

    assert "extract_sections_impl" not in IMPLEMENTATIONS

    # Check production code paths for the string
    import neuralresearcher
    base_path = Path(neuralresearcher.__file__).parent
    found = False
    for py_file in base_path.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if "extract_sections_impl" in content:
            found = True
            break
    assert not found, "extract_sections_impl found in production code!"


def test_abstract_retrieval_produces_abstract_only():
    """Prove retrieved papers have ContentLevel.ABSTRACT_ONLY by default."""
    p = Paper(
        id="test", title="Test", authors=[], venue="V", year=2024,
        url="http", abstract="abstract text"
    )
    assert p.content_level == ContentLevel.ABSTRACT_ONLY


def test_no_abstract_source_produces_page_level_provenance():
    """Prove no abstract-only source can produce page-level provenance."""
    # This is enforced in reading.py line 98: if paper.content_level !=
    # "FULL_TEXT": c_dict['section'] = 'abstract'
    Claim(
        id="c", paper_id="p", evidence_ref="url", type="result",
        text="text", section="Page 4"
    )
    # The agent explicitly overrides this. We mock the LLM returning Page 4.
    p = Paper(
        id="test",
        title="Test",
        authors=[],
        venue="V",
        year=2024,
        url="http",
        abstract="abstract text",
        content_level=ContentLevel.ABSTRACT_ONLY)
    from unittest.mock import patch

    from neuralresearcher.agents.reading import PaperMetadata, ReadingClaim, ReadingResponse

    with patch("neuralresearcher.agents.reading.generate_structured") as mock_gen:
        mock_gen.return_value = ReadingResponse(
            paper_metadata=PaperMetadata(),
            claims=[ReadingClaim(type="result", text="t", section="Page 4")]
        )

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            store.save_papers([p])
            ctx = AgentContext(
                topic="topic",
                config=Config(),
                store=store,
                task_id="task")
            run_reading(ctx)

            claims = store.load_claims()
            assert len(claims) == 1
            assert claims[0].section == "abstract", "Abstract-only paper cannot have page-level provenance"


def test_report_warns_when_findings_rely_only_on_abstracts(tmp_path):
    """Prove reports warn when findings rely only on abstracts."""
    plan = ResearchPlan(
        id="1",
        topic_spec_id="1",
        hypothesis="h",
        expected_contribution="c",
        steps=["s1"])
    papers = [
        Paper(
            id="test",
            title="Test",
            authors=[],
            venue="V",
            year=2024,
            url="http",
            abstract="abstract text",
            content_level=ContentLevel.ABSTRACT_ONLY)]

    out_path = tmp_path / "plan.md"
    render_markdown_plan(plan, [], papers, [], [], output_path=str(out_path))

    content = out_path.read_text(encoding="utf-8")
    assert "WARNING" in content
    assert "rely exclusively on paper abstracts" in content
