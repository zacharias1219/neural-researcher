import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from neuralresearcher.tools import (
    execute_tool_call,
    search_papers_impl,
    fetch_paper_impl,
    ToolError,
)
from neuralresearcher.llm import ToolCall


@pytest.fixture
def sample_arxiv_xml() -> bytes:
    fixture_path = Path(__file__).parent / "fixtures" / "arxiv_sample.xml"
    with open(fixture_path, "rb") as f:
        return f.read()


def test_execute_tool_call_invalid_json():
    tc = ToolCall(id="1", function={"name": "search_papers", "arguments": "invalid json"})
    with pytest.raises(ToolError, match="Invalid JSON"):
        execute_tool_call(tc)


def test_execute_tool_call_unknown_tool():
    tc = ToolCall(id="1", function={"name": "unknown_tool", "arguments": "{}"})
    with pytest.raises(ToolError, match="not implemented"):
        execute_tool_call(tc)


@patch("neuralresearcher.tools.requests.get")
def test_search_papers_parsing(mock_get, sample_arxiv_xml):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = sample_arxiv_xml
    mock_get.return_value = mock_resp

    with patch.dict(os.environ, {}, clear=True):
        papers = search_papers_impl(keywords=["mamba", "optimization"], max_results=2)

    assert len(papers) == 2
    p1 = papers[0]
    assert p1["id"] == "2312.00752v1"
    assert "Mamba" in p1["title"]
    assert "Albert Gu" in p1["authors"]
    assert "Tri Dao" in p1["authors"]
    assert p1["year"] == 2023
    assert "linear-time" in p1["abstract"].lower()


@patch("neuralresearcher.tools.requests.get")
def test_search_papers_empty_keywords(mock_get):
    papers = search_papers_impl(keywords=[])
    assert papers == []
    mock_get.assert_not_called()


@patch("neuralresearcher.tools.requests.get")
def test_search_papers_api_error(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_get.return_value = mock_resp

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ToolError, match="Failed to fetch from arXiv API: 503"):
            search_papers_impl(keywords=["mamba"])


def test_search_papers_cassette_replay():
    fixture_path = str(Path(__file__).parent / "fixtures" / "arxiv_sample.xml")
    with patch.dict(os.environ, {"ARXIV_CASSETTE_PATH": fixture_path}):
        # requests.get should not be called when cassette is present
        with patch("neuralresearcher.tools.requests.get") as mock_get:
            papers = search_papers_impl(keywords=["mamba"])
            mock_get.assert_not_called()

    assert len(papers) == 3
    assert papers[0]["id"] == "2312.00752v1"
    assert papers[1]["id"] == "2405.21060v1"


@patch("neuralresearcher.tools.requests.get")
def test_fetch_paper_parsing(mock_get, sample_arxiv_xml):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = sample_arxiv_xml
    mock_get.return_value = mock_resp

    with patch.dict(os.environ, {}, clear=True):
        paper = fetch_paper_impl(paper_id="2312.00752v1")

    assert paper["id"] == "2312.00752v1"
    assert "Mamba" in paper["title"]
    assert paper["year"] == 2023


def test_execute_tool_call_successful(sample_arxiv_xml):
    fixture_path = str(Path(__file__).parent / "fixtures" / "arxiv_sample.xml")
    with patch.dict(os.environ, {"ARXIV_CASSETTE_PATH": fixture_path}):
        tc = ToolCall(
            id="call_1",
            function={
                "name": "search_papers",
                "arguments": json.dumps({"keywords": ["mamba"], "max_results": 2}),
            },
        )
        args, result_json = execute_tool_call(tc)
        assert args["keywords"] == ["mamba"]
        data = json.loads(result_json)
        assert len(data) == 2
        assert data[0]["id"] == "2312.00752v1"
