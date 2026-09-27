"""JARVIS Tools — All tool implementations."""
from __future__ import annotations

import ast
import logging
import math
import operator
from datetime import datetime, timezone
from typing import Any

from app.tools.base import BaseTool, ToolResult
from app.tools.registry import register_tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Web Search Tool
# ─────────────────────────────────────────────
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information. Args: query (str), max_results (int, optional)."

    async def execute(self, query: str = "", max_results: int = 5, **_: Any) -> ToolResult:
        if not query.strip():
            return self._err("Query cannot be empty.")
        from app.services.search_service import web_search, format_search_results_for_context
        results = await web_search(query, max_results=max_results)
        if not results:
            return self._ok({"results": [], "formatted": "No results found."})
        formatted = format_search_results_for_context(results)
        return self._ok({
            "results": [r.to_dict() for r in results],
            "formatted": formatted,
        })


# ─────────────────────────────────────────────
# Memory Search Tool
# ─────────────────────────────────────────────
class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = "Search the user's personal memories. Args: query (str), limit (int, optional)."

    def __init__(self, session=None, user_id=None):
        self._session = session
        self._user_id = user_id

    async def execute(self, query: str = "", limit: int = 5, **_: Any) -> ToolResult:
        if not self._session or not self._user_id:
            return self._err("Memory search requires an authenticated session.")
        if not query.strip():
            return self._err("Query cannot be empty.")
        from app.services.memory_service import get_relevant_memories, format_memories_for_context
        memories = await get_relevant_memories(
            self._session, self._user_id, query, limit=limit
        )
        formatted = format_memories_for_context(memories)
        return self._ok({"memories": formatted, "count": len(formatted)})


# ─────────────────────────────────────────────
# DateTime Tool
# ─────────────────────────────────────────────
class DateTimeTool(BaseTool):
    name = "datetime_tool"
    description = "Get current date and time. No args required."
    requires_auth = False

    async def execute(self, **_: Any) -> ToolResult:
        now = datetime.now(timezone.utc)
        return self._ok({
            "utc": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timezone": "UTC",
        })


# ─────────────────────────────────────────────
# Calculator Tool (safe — no exec/eval)
# ─────────────────────────────────────────────
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "pow": math.pow,
    "int": int, "float": float,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a safe AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        if func_name in _SAFE_NAMES and callable(_SAFE_NAMES[func_name]):
            args = [_safe_eval(a) for a in node.args]
            return _SAFE_NAMES[func_name](*args)
    raise ValueError(f"Unsafe expression element: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a math expression safely. Args: expression (str). E.g. '2+2', 'sqrt(144)', 'sin(pi/2)'."
    requires_auth = False

    async def execute(self, expression: str = "", **_: Any) -> ToolResult:
        if not expression.strip():
            return self._err("Expression cannot be empty.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree)
            return self._ok({"expression": expression, "result": result})
        except ZeroDivisionError:
            return self._err("Division by zero.")
        except (ValueError, TypeError) as exc:
            return self._err(f"Invalid expression: {exc}")
        except SyntaxError as exc:
            return self._err(f"Syntax error: {exc}")


# ─────────────────────────────────────────────
# Conversation Search Tool
# ─────────────────────────────────────────────
class ConversationSearchTool(BaseTool):
    name = "conversation_search"
    description = "Search past conversations. Args: query (str), limit (int, optional)."

    def __init__(self, session=None, user_id=None):
        self._session = session
        self._user_id = user_id

    async def execute(self, query: str = "", limit: int = 5, **_: Any) -> ToolResult:
        if not self._session or not self._user_id:
            return self._err("Conversation search requires an authenticated session.")
        if not query.strip():
            return self._err("Query cannot be empty.")
        from sqlalchemy import select
        from app.models import Message, Conversation
        # Simple ILIKE text search across message content
        stmt = (
            select(Message.content, Conversation.title, Message.created_at)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == self._user_id,
                Message.content.ilike(f"%{query}%"),
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        results = [
            {"title": r.title, "snippet": r.content[:200], "date": str(r.created_at)}
            for r in rows
        ]
        return self._ok({"results": results, "count": len(results)})


# ─────────────────────────────────────────────
# Document Search Tool
# ─────────────────────────────────────────────
class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search uploaded documents. Args: query (str)."

    def __init__(self, session=None, user_id=None):
        self._session = session
        self._user_id = user_id

    async def execute(self, query: str = "", **_: Any) -> ToolResult:
        if not self._session or not self._user_id:
            return self._err("Document search requires an authenticated session.")
        if not query.strip():
            return self._err("Query cannot be empty.")
        from sqlalchemy import select
        from app.models import Document
        stmt = (
            select(Document)
            .where(
                Document.user_id == self._user_id,
                Document.extracted_text.ilike(f"%{query}%"),
            )
            .limit(3)
        )
        docs = list(await self._session.scalars(stmt))
        results = [
            {
                "document_id": str(d.id),
                "filename": d.filename,
                "snippet": (d.extracted_text or "")[:300],
            }
            for d in docs
        ]
        return self._ok({"results": results, "count": len(results)})


# ─────────────────────────────────────────────
# Learning Tool
# ─────────────────────────────────────────────
class LearningTool(BaseTool):
    name = "learning_tool"
    description = "Get the user's current learning topics and progress. No args required."

    def __init__(self, session=None, user_id=None):
        self._session = session
        self._user_id = user_id

    async def execute(self, **_: Any) -> ToolResult:
        if not self._session or not self._user_id:
            return self._err("Learning tool requires an authenticated session.")
        from sqlalchemy import select
        from app.models import LearningTopic, LearningProgress
        stmt = (
            select(LearningTopic)
            .where(LearningTopic.user_id == self._user_id)
            .limit(10)
        )
        topics = list(await self._session.scalars(stmt))
        result = [
            {
                "title": t.title,
                "status": t.status,
                "difficulty": t.difficulty_level,
            }
            for t in topics
        ]
        return self._ok({"topics": result, "count": len(result)})


# ─────────────────────────────────────────────
# Register all stateless tools at module load
# ─────────────────────────────────────────────
def register_stateless_tools() -> None:
    """Register tools that don't need DB session (stateless tools)."""
    register_tool(WebSearchTool())
    register_tool(DateTimeTool())
    register_tool(CalculatorTool())


def get_session_tools(session, user_id) -> list[BaseTool]:
    """Return session-bound tool instances (need DB access)."""
    return [
        MemorySearchTool(session=session, user_id=user_id),
        ConversationSearchTool(session=session, user_id=user_id),
        DocumentSearchTool(session=session, user_id=user_id),
        LearningTool(session=session, user_id=user_id),
    ]
