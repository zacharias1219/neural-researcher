import pytest
import json
from neuralresearcher.tools import execute_tool_call, ToolError
from neuralresearcher.llm import ToolCall

def test_execute_tool_call_invalid_json():
    tc = ToolCall(id="1", function={"name": "search_papers", "arguments": "invalid json"})
    with pytest.raises(ToolError, match="Invalid JSON"):
        execute_tool_call(tc)

def test_execute_tool_call_unknown_tool():
    tc = ToolCall(id="1", function={"name": "unknown_tool", "arguments": "{}"})
    with pytest.raises(ToolError, match="not implemented"):
        execute_tool_call(tc)

# Note: Valid tool calls would require mocking arxiv, which is beyond this basic test.
