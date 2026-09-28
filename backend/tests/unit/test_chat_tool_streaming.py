"""Regression tests for validated streaming and non-streaming tool calls."""

import pytest
from app.api.v1.chat import _handle_tool_calling, _stream_chat_response
from app.tools import registry as tool_registry
from app.tools.base import BaseTool

TOOL_REQUEST = '{"tool": "test_stream_tool", "args": {"value": "input"}}'


class StubLLM:
    def __init__(self, stream_chunks, generated_response="Final answer"):
        self.stream_chunks = list(stream_chunks)
        self.generated_response = generated_response
        self.stream_prompts = []
        self.generate_prompts = []

    async def stream(self, prompt, system=""):
        self.stream_prompts.append(prompt)
        chunks = self.stream_chunks.pop(0)
        for chunk in chunks:
            yield chunk

    async def generate(self, prompt, system=""):
        self.generate_prompts.append(prompt)
        return self.generated_response


class SuccessfulTool(BaseTool):
    name = "test_stream_tool"
    description = "Test-only tool."

    def __init__(self):
        self.calls = 0

    async def execute(self, value="", **kwargs):
        self.calls += 1
        return self._ok({"received": value})


class FailingTool(BaseTool):
    name = "test_stream_tool"
    description = "Test-only failing tool."

    async def execute(self, **kwargs):
        raise RuntimeError("private-tool-exception-sentinel")


@pytest.mark.asyncio
async def test_streaming_tools_execute_and_stream_the_final_answer(monkeypatch):
    tool = SuccessfulTool()
    monkeypatch.setitem(tool_registry._registry, tool.name, tool)
    llm = StubLLM([[TOOL_REQUEST, ""], ["Final ", "answer"]])

    chunks = [
        chunk
        async for chunk in _stream_chat_response(
            llm, "User: calculate", "Use tools", use_tools=True
        )
    ]

    assert "".join(chunks) == "Final answer"
    assert tool.calls == 1
    assert len(llm.stream_prompts) == 2
    assert '"received": "input"' in llm.stream_prompts[1]


@pytest.mark.asyncio
async def test_streaming_with_tools_disabled_never_executes_tool(monkeypatch):
    tool = SuccessfulTool()
    monkeypatch.setitem(tool_registry._registry, tool.name, tool)
    llm = StubLLM([[TOOL_REQUEST]])

    chunks = [
        chunk
        async for chunk in _stream_chat_response(
            llm, "User: calculate", "No tools", use_tools=False
        )
    ]

    assert "".join(chunks) == TOOL_REQUEST
    assert tool.calls == 0
    assert len(llm.stream_prompts) == 1


@pytest.mark.asyncio
async def test_non_streaming_tool_behavior_remains_available(monkeypatch):
    tool = SuccessfulTool()
    monkeypatch.setitem(tool_registry._registry, tool.name, tool)
    llm = StubLLM([], generated_response="Result is ready")

    response, tools_used = await _handle_tool_calling(
        TOOL_REQUEST, "User: calculate", "Use tools", llm
    )

    assert response == "Result is ready"
    assert tools_used == [tool.name]
    assert tool.calls == 1
    assert '"received": "input"' in llm.generate_prompts[0]


@pytest.mark.asyncio
async def test_unregistered_tool_is_not_executed_and_stream_continues():
    request = '{"tool": "unregistered_private_function", "args": {}}'
    llm = StubLLM([[request], ["Safe ", "answer"]])

    chunks = [
        chunk
        async for chunk in _stream_chat_response(
            llm, "User: do something", "Use tools", use_tools=True
        )
    ]

    assert "".join(chunks) == "Safe answer"
    assert len(llm.stream_prompts) == 2
    assert "The requested tool is not available." in llm.stream_prompts[1]


@pytest.mark.asyncio
async def test_tool_failure_is_safe_and_does_not_abort_stream(monkeypatch, caplog):
    tool = FailingTool()
    monkeypatch.setitem(tool_registry._registry, tool.name, tool)
    llm = StubLLM([[TOOL_REQUEST], ["Fallback ", "answer"]])

    chunks = [
        chunk
        async for chunk in _stream_chat_response(
            llm, "User: calculate", "Use tools", use_tools=True
        )
    ]

    assert "".join(chunks) == "Fallback answer"
    assert "The requested tool could not complete." in llm.stream_prompts[1]
    assert "private-tool-exception-sentinel" not in caplog.text