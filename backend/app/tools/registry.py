"""JARVIS Tool Registry — register, discover, and execute tools by name."""
from __future__ import annotations

import logging
from typing import Any

from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_registry: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """Register a tool instance in the global registry."""
    if not tool.name:
        raise ValueError(f"Tool {type(tool).__name__} has no name.")
    _registry[tool.name] = tool
    logger.debug("Tool registered: %s", tool.name)


def get_tool(name: str) -> BaseTool | None:
    """Return a registered tool by name, or None."""
    return _registry.get(name)


def list_tools() -> list[dict[str, str]]:
    """Return description of all registered tools."""
    return [{"name": t.name, "description": t.description} for t in _registry.values()]


async def execute_tool(name: str, **kwargs: Any) -> ToolResult:
    """
    Execute a registered tool by name.
    Never raises — returns ToolResult with success=False on failure.
    """
    tool = get_tool(name)
    if tool is None:
        return ToolResult(
            success=False, output=None,
            error=f"Tool '{name}' not found.", tool_name=name,
        )
    try:
        logger.info("Executing tool '%s' with %d arg(s).", name, len(kwargs))
        result = await tool.execute(**kwargs)
        if not isinstance(result, ToolResult):
            result = ToolResult(success=True, output=result, tool_name=name)
        return result
    except Exception as exc:
        logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, str(exc)[:200])
        return ToolResult(success=False, output=None, error=str(exc), tool_name=name)


def get_tools_description() -> str:
    """Return a formatted tool list for injection into LLM system prompt."""
    if not _registry:
        return ""
    lines = ["Available tools (use JSON format to call them):"]
    for tool in _registry.values():
        lines.append(f"- {tool.name}: {tool.description}")
    lines.append(
        "\nTo use a tool, respond with: "
        '{"tool": "tool_name", "args": {"param": "value"}}'
    )
    return "\n".join(lines)
