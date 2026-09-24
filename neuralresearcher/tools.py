import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
import urllib.parse
import xml.etree.ElementTree as ET
import requests

from neuralresearcher.llm import ToolCall
from neuralresearcher.errors import ToolError

# Schemas
SEARCH_PAPERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search for academic papers matching a list of MUST-HAVE keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 2 to 3 core keywords that MUST all be present in the paper (e.g. ['mamba', 'optimization']). Keep the list short."
                },
                "max_results": {"type": "integer", "description": "Max number of papers to return."}
            },
            "required": ["keywords"]
        }
    }
}

FETCH_PAPER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_paper",
        "description": "Fetch the abstract and metadata for a specific paper by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "The ID of the paper (e.g. arXiv ID)."}
            },
            "required": ["paper_id"]
        }
    }
}

TOOL_SCHEMAS = [SEARCH_PAPERS_SCHEMA, FETCH_PAPER_SCHEMA]


def _fetch_arxiv_xml(url: str) -> bytes:
    """Fetch XML from arXiv or replay from a local cassette fixture for determinism."""
    cassette_path = os.environ.get("ARXIV_CASSETTE_PATH")
    if not cassette_path and os.environ.get("NEURAL_RESEARCHER_OFFLINE") == "1":
        default_fixture = Path("tests/fixtures/arxiv_sample.xml")
        if default_fixture.exists():
            cassette_path = str(default_fixture)

    if cassette_path and Path(cassette_path).exists():
        with open(cassette_path, "rb") as f:
            return f.read()

    response = requests.get(url)
    if response.status_code != 200:
        raise ToolError(f"Failed to fetch from arXiv API: {response.status_code}")

    record_path = os.environ.get("ARXIV_RECORD_CASSETTE_PATH")
    if record_path:
        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        with open(record_path, "wb") as f:
            f.write(response.content)

    return response.content


def search_papers_impl(keywords: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
    """Search papers via arXiv API or replay cassette."""
    terms = [urllib.parse.quote(term.strip()) for term in keywords if term.strip()]
    if not terms:
        return []
        
    formatted_query = "+AND+all:".join(terms)
    url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results={max_results}"
    xml_content = _fetch_arxiv_xml(url)
        
    root = ET.fromstring(xml_content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    
    for entry in root.findall('atom:entry', ns):
        paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        published = entry.find('atom:published', ns).text
        year = int(published[:4])
        url_link = entry.find('atom:id', ns).text
        
        papers.append({
            "id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "url": url_link,
            "abstract": summary
        })
        
    return papers[:max_results]


def fetch_paper_impl(paper_id: str) -> Dict[str, Any]:
    """Fetch specific paper from arXiv by ID or replay cassette."""
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    xml_content = _fetch_arxiv_xml(url)
        
    root = ET.fromstring(xml_content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entry = root.find('atom:entry', ns)
    
    if not entry:
        raise ToolError(f"Paper {paper_id} not found.")
        
    title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
    summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
    authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
    published = entry.find('atom:published', ns).text
    year = int(published[:4])
    url_link = entry.find('atom:id', ns).text
    
    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "url": url_link,
        "abstract": summary
    }


IMPLEMENTATIONS = {
    "search_papers": search_papers_impl,
    "fetch_paper": fetch_paper_impl
}


def execute_tool_call(tool_call: ToolCall) -> Tuple[Dict[str, Any], str]:
    func_name = tool_call.function.get("name")
    if func_name not in IMPLEMENTATIONS:
        raise ToolError(f"Tool {func_name} is not implemented.")
        
    try:
        args = json.loads(tool_call.function.get("arguments", "{}"))
    except json.JSONDecodeError:
        raise ToolError(f"Invalid JSON arguments for tool {func_name}.")
        
    impl = IMPLEMENTATIONS[func_name]
    try:
        result = impl(**args)
        return args, json.dumps(result)
    except Exception as e:
        raise ToolError(f"Error executing tool {func_name}: {str(e)}")
