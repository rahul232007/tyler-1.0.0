"""Unit tests for JARVIS Tool System (registry, calculator, datetime, error handling)."""

import pytest
from app.tools.registry import execute_tool
from app.tools.tools import CalculatorTool, DateTimeTool, register_stateless_tools


@pytest.fixture(autouse=True)
def init_tools():
    register_stateless_tools()


@pytest.mark.asyncio
async def test_calculator_basic_operations():
    calc = CalculatorTool()

    res = await calc.execute(expression="2 + 2 * 10")
    assert res.success is True
    assert res.output["result"] == 22

    res_div = await calc.execute(expression="100 / 4")
    assert res_div.success is True
    assert res_div.output["result"] == 25.0

    res_math = await calc.execute(expression="sqrt(144) + 8")
    assert res_math.success is True
    assert res_math.output["result"] == 20.0


@pytest.mark.asyncio
async def test_calculator_safety_against_arbitrary_code():
    calc = CalculatorTool()

    # Attempt code injection / unsafe builtins
    res = await calc.execute(expression="__import__('os').system('echo hacked')")
    assert res.success is False
    assert "Invalid" in res.error or "Syntax" in res.error or "Unsafe" in res.error

    res_zero = await calc.execute(expression="10 / 0")
    assert res.success is False
    assert "zero" in res_zero.error.lower()


@pytest.mark.asyncio
async def test_datetime_tool():
    dt_tool = DateTimeTool()
    res = await dt_tool.execute()

    assert res.success is True
    assert "utc" in res.output
    assert "date" in res.output
    assert "time" in res.output
    assert "day_of_week" in res.output


@pytest.mark.asyncio
async def test_tool_registry_execution():
    res = await execute_tool("calculator", expression="5 * 5")
    assert res.success is True
    assert res.output["result"] == 25

    res_not_found = await execute_tool("non_existent_tool_xyz")
    assert res_not_found.success is False
    assert "not found" in res_not_found.error
