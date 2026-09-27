"""Unit tests for Learning System (grading, recommendations, schema validation)."""

import pytest
from app.services.learning import (
    _generate_recommendation,
    _parse_questions,
    grade_mcq,
    grade_open_answer,
)


def test_mcq_grading_correct_and_incorrect():
    is_corr, score, fb = grade_mcq("A", "A")
    assert is_corr is True
    assert score == 100.0
    assert "Correct" in fb

    is_corr, score, fb = grade_mcq("b", "B")  # case-insensitive
    assert is_corr is True
    assert score == 100.0

    is_corr, score, fb = grade_mcq("C", "A")
    assert is_corr is False
    assert score == 0.0
    assert "A" in fb


@pytest.mark.asyncio
async def test_open_answer_grading_fallback():
    class DummyLLM:
        async def generate(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    # Tests fallback keyword matching
    is_corr, score, fb = await grade_open_answer(
        question="What is a Python generator?",
        user_answer="A function that yields values using the yield keyword instead of return",
        expected_answer="A function that yields values using the yield keyword",
        llm_service=DummyLLM(),
    )
    assert score >= 50.0
    assert is_corr is True

    # Empty answer
    is_corr, score, fb = await grade_open_answer(
        question="What is recursion?",
        user_answer="",
        expected_answer="A function calling itself",
        llm_service=DummyLLM(),
    )
    assert is_corr is False
    assert score == 0.0


def test_parse_questions_mcq_validation():
    raw_json = """
    [
        {
            "question": "What does CPU stand for?",
            "options": [
                {"label": "A", "text": "Central Processing Unit"},
                {"label": "B", "text": "Computer Personal Unit"},
                {"label": "C", "text": "Central Power Unit"},
                {"label": "D", "text": "Control Processing Unit"}
            ],
            "correct_answer": "A",
            "explanation": "Standard abbreviation."
        },
        {
            "question": "Invalid item without options",
            "correct_answer": "B"
        }
    ]
    """
    parsed = _parse_questions(raw_json, "mcq")
    assert len(parsed) == 1
    assert parsed[0]["correct_answer"] == "A"
    assert len(parsed[0]["options"]) == 4


def test_weak_topic_recommendation_logic():
    rec_low = _generate_recommendation("Recursion", 20.0)
    assert "fundamentals" in rec_low.lower()

    rec_mid = _generate_recommendation("Dynamic Programming", 45.0)
    assert "practice more" in rec_mid.lower()

    rec_high = _generate_recommendation("Arrays", 75.0)
    assert "improving" in rec_high.lower()
