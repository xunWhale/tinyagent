import importlib
import inspect
import sys
import types
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from tinyagent.config import MCPStdio
from tinyagent.tools.mcp.mcp_client import MCPClient


@pytest.mark.asyncio
async def test_create_tool_function_with_complex_schema() -> None:
    """Test the core tool function creation with comprehensive parameter handling."""
    from mcp.types import Tool as MCPTool

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock()]
    mock_result.content[0].text = "Tool executed successfully"
    mock_session.call_tool.return_value = mock_result

    config = MCPStdio(command="test", args=[])
    client = MCPClient(config=config)
    client._session = mock_session

    complex_tool = MCPTool(
        name="complex_search",
        description="Search with multiple parameter types",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include result metadata",
                },
                "filters": {"type": "object", "description": "Search filters"},
                "tags": {"type": "array", "description": "Filter tags"},
                "threshold": {"type": "number", "description": "Similarity threshold"},
                "optional_param": {
                    "type": "string",
                    "description": "Optional parameter",
                },
            },
            "required": ["query", "max_results", "include_metadata"],
        },
    )

    tool_func = client._create_tool_function(complex_tool)

    assert tool_func.__name__ == "complex_search"
    assert tool_func.__doc__ is not None
    assert "Search with multiple parameter types" in tool_func.__doc__
    assert "query: Search query string" in tool_func.__doc__
    assert "max_results: Maximum number of results" in tool_func.__doc__

    sig = inspect.signature(tool_func)
    params = sig.parameters

    assert "query" in params
    assert params["query"].annotation is str
    assert params["query"].default is inspect.Parameter.empty

    assert "max_results" in params
    assert params["max_results"].annotation is int
    assert params["max_results"].default is inspect.Parameter.empty

    assert "include_metadata" in params
    assert params["include_metadata"].annotation is bool
    assert params["include_metadata"].default is inspect.Parameter.empty

    assert "optional_param" in params
    assert params["optional_param"].annotation == Optional[str]  # noqa: UP045
    assert params["optional_param"].default is None

    assert params["filters"].annotation == Optional[dict]  # noqa: UP045
    assert params["tags"].annotation == Optional[list]  # noqa: UP045
    assert params["threshold"].annotation == Optional[float]  # noqa: UP045

    assert sig.return_annotation is str

    result = await tool_func(query="test query", max_results=10, include_metadata=True)
    assert result == "Tool executed successfully"
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {"query": "test query", "max_results": 10, "include_metadata": True},
    )

    await tool_func(
        query="another query",
        max_results=5,
        include_metadata=False,
        optional_param="optional_value",
        threshold=0.8,
    )
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {
            "query": "another query",
            "max_results": 5,
            "include_metadata": False,
            "optional_param": "optional_value",
            "threshold": 0.8,
        },
    )

    client._session = None
    error_result = await tool_func(query="test", max_results=1, include_metadata=True)
    assert "Error: MCP session not available" in error_result


def test_mcp_client_import_without_mcp_resolves_name_error(monkeypatch):
    """Verify that importing or reloading mcp_client without mcp does not raise NameError."""
    import tinyagent.tools.mcp.mcp_client as mcp_module

    # Mask mcp modules to simulate environment without mcp package
    masked_modules = {
        name: None
        for name in list(sys.modules.keys())
        if name == "mcp" or name.startswith("mcp.")
    }
    with monkeypatch.context() as m:
        for mod, val in masked_modules.items():
            m.setitem(sys.modules, mod, val)
        # Reloading must not crash with NameError (e.g. MCPTool not defined)
        reloaded = importlib.reload(mcp_module)
        assert reloaded.missing_mcp_error is not None
        assert reloaded.MCPTool is Any
        assert reloaded.ClientSession is Any

        # Calling MCPClient must raise clean ImportError, not NameError
        config = MCPStdio(command="test", args=[])
        with pytest.raises(ImportError) as exc_info:
            reloaded.MCPClient(config=config)
        assert "MCP support requires the `mcp` package" in str(exc_info.value)

    # Restore clean module state
    importlib.reload(mcp_module)


def test_streamable_http_client_import_fallback(monkeypatch):
    """Verify fallback to legacy streamablehttp_client when mcp < 2.0."""
    import tinyagent.tools.mcp.mcp_client as mcp_module

    legacy_client_mock = object()
    fake_streamable_mod = types.ModuleType("mcp.client.streamable_http")
    fake_streamable_mod.streamablehttp_client = legacy_client_mock

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mcp.client.streamable_http", fake_streamable_mod)
        reloaded = importlib.reload(mcp_module)
        assert reloaded.streamable_http_client is legacy_client_mock

    importlib.reload(mcp_module)


def test_streamable_http_client_import_v2_mainline():
    """Verify primary streamable_http_client import when mcp >= 2.0."""
    import tinyagent.tools.mcp.mcp_client as mcp_module

    # In modern environment, streamable_http_client is available
    assert callable(mcp_module.streamable_http_client)


def test_create_tool_function_legacy_camel_case_schema():
    """Verify compatibility with legacy MCP 1.x camelCase inputSchema."""
    class LegacyTool:
        name = "legacy_tool"
        description = "Legacy tool description"
        inputSchema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        }

    config = MCPStdio(command="test", args=[])
    client = MCPClient(config=config)
    tool_func = client._create_tool_function(LegacyTool())
    assert tool_func.__name__ == "legacy_tool"
