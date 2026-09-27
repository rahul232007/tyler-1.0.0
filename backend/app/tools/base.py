"""JARVIS Tool System - Base class for all tools."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    output: Any  # The actual result data
    error: str | None = None
    tool_name: str = ""


class BaseTool(ABC):
    """Abstract base for all JARVIS tools."""

    name: str = ""
    description: str = ""
    requires_auth: bool = True

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters. Never raises — returns ToolResult."""
        ...

    def _ok(self, output: Any) -> ToolResult:
        return ToolResult(success=True, output=output, tool_name=self.name)

    def _err(self, message: str) -> ToolResult:
        return ToolResult(success=False, output=None, error=message, tool_name=self.name)
