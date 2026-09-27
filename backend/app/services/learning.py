"""
JARVIS - Learning System Service
Handles AI-powered practice generation, grading, progress tracking,
weak-topic detection, and recommendations.

Uses the existing LLM router — no duplicate provider logic.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LearningProgress, LearningTopic, PracticeResult
from app.schemas.learning import (
    WeakTopicInfo,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Practice Generation Prompts
# ─────────────────────────────────────────────

MCQ_PROMPT = """Generate {num} multiple-choice questions about "{topic}" at {difficulty} level.
Return ONLY a valid JSON array. Each question object must have:
  - "question": the question text
  - "question_type": "mcq"
  - "options": [{{"label": "A", "text": "..."}}, {{"label": "B", "text": "..."}}, ...]  (4 options)
  - "correct_answer": "A" or "B" or "C" or "D"
  - "explanation": brief explanation of correct answer
  - "difficulty": "{difficulty}"

Focus on practical understanding, not rote memorization.
Return ONLY the JSON array with no other text."""

SHORT_ANSWER_PROMPT = """Generate {num} short-answer questions about "{topic}" at {difficulty} level.
Return ONLY a valid JSON array. Each question object must have:
  - "question": the question text
  - "question_type": "short_answer"
  - "correct_answer": the expected answer (1-3 sentences)
  - "explanation": key points the answer should cover
  - "difficulty": "{difficulty}"

Return ONLY the JSON array with no other text."""

CODING_PROMPT = """Generate {num} coding exercises about "{topic}" at {difficulty} level.
Return ONLY a valid JSON array. Each question object must have:
  - "question": the coding problem description
  - "question_type": "coding"
  - "correct_answer": the expected solution or key concepts (code + explanation)
  - "explanation": what concepts this tests
  - "difficulty": "{difficulty}"

Return ONLY the JSON array with no other text."""

GRADE_PROMPT = """Grade the following short-answer or coding question.

Question: {question}
Expected Answer: {expected}
User's Answer: {user_answer}

Return ONLY a JSON object with:
  - "is_correct": boolean (true if substantially correct)
  - "score": float 0-100 (percentage credit)
  - "feedback": helpful 1-2 sentence feedback

Return ONLY the JSON object."""


# ─────────────────────────────────────────────
# Practice Generation
# ─────────────────────────────────────────────


async def generate_practice_questions(
    topic_title: str,
    question_type: str,
    difficulty: str,
    num_questions: int,
    llm_service,
) -> list[dict[str, Any]]:
    """
    Use the LLM router to generate practice questions.
    Returns validated question list or raises RuntimeError.
    """
    if question_type == "mcq":
        prompt = MCQ_PROMPT.format(
            num=num_questions, topic=topic_title, difficulty=difficulty
        )
    elif question_type == "coding":
        prompt = CODING_PROMPT.format(
            num=num_questions, topic=topic_title, difficulty=difficulty
        )
    else:
        prompt = SHORT_ANSWER_PROMPT.format(
            num=num_questions, topic=topic_title, difficulty=difficulty
        )

    system = "You are a strict JSON-only question generator. Never add explanatory text outside the JSON array."
    raw = await llm_service.generate(prompt=prompt, system=system)

    questions = _parse_questions(raw, question_type)
    if not questions:
        raise RuntimeError(
            "LLM did not return valid practice questions. Please try again."
        )

    logger.info(
        "Generated %d %s questions for topic '%s' (%s)",
        len(questions),
        question_type,
        topic_title,
        difficulty,
    )
    return questions


def _parse_questions(raw: str, question_type: str) -> list[dict[str, Any]]:
    """Parse and validate LLM-returned question JSON."""
    # Extract JSON array from response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        questions = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(questions, list):
        return []

    validated = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if not q.get("question") or not q.get("correct_answer"):
            continue
        q["question_type"] = question_type
        # Validate MCQ options
        if question_type == "mcq":
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) < 2:
                continue
            correct = str(q.get("correct_answer", "")).strip().upper()
            if correct not in {"A", "B", "C", "D"}:
                continue
            q["correct_answer"] = correct
        validated.append(q)

    return validated[:20]  # Hard cap


# ─────────────────────────────────────────────
# Answer Grading
# ─────────────────────────────────────────────


def grade_mcq(user_answer: str, correct_answer: str) -> tuple[bool, float, str]:
    """Grade an MCQ answer. Returns (is_correct, score, feedback)."""
    user = user_answer.strip().upper()
    correct = correct_answer.strip().upper()
    is_correct = user == correct
    score = 100.0 if is_correct else 0.0
    feedback = (
        "Correct!" if is_correct else f"Incorrect. The correct answer is {correct}."
    )
    return is_correct, score, feedback


async def grade_open_answer(
    question: str,
    user_answer: str,
    expected_answer: str,
    llm_service,
) -> tuple[bool, float, str]:
    """
    Grade a short-answer or coding question using the LLM.
    Returns (is_correct, score, feedback).
    Falls back to keyword matching if LLM fails.
    """
    if not user_answer.strip():
        return False, 0.0, "No answer provided."

    prompt = GRADE_PROMPT.format(
        question=question[:500],
        expected=expected_answer[:500],
        user_answer=user_answer[:1000],
    )
    try:
        raw = await llm_service.generate(
            prompt=prompt,
            system="You are a strict JSON-only grader. Return only the JSON object.",
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            is_correct = bool(result.get("is_correct", False))
            score = float(result.get("score", 0.0))
            score = max(0.0, min(100.0, score))
            feedback = str(result.get("feedback", ""))
            return is_correct, score, feedback
    except Exception as exc:
        logger.warning(
            "LLM grading failed: %s — falling back to keyword match.",
            type(exc).__name__,
        )

    # Fallback: keyword-based scoring
    user_lower = user_answer.lower()
    expected_keywords = re.findall(r"\b\w{4,}\b", expected_answer.lower())
    if not expected_keywords:
        return True, 50.0, "Answer accepted (keyword check not possible)."
    matches = sum(1 for kw in expected_keywords if kw in user_lower)
    score = min(100.0, (matches / len(expected_keywords)) * 100)
    is_correct = score >= 60.0
    feedback = f"Score based on keyword matching: {score:.0f}%."
    return is_correct, score, feedback


# ─────────────────────────────────────────────
# Weak Topic Detection
# ─────────────────────────────────────────────


async def detect_weak_topics(
    session: AsyncSession,
    user_id: UUID,
    threshold: float = 60.0,
) -> list[WeakTopicInfo]:
    """
    Find topics where the user's average practice score is below threshold.
    """
    # Aggregate scores per topic
    stmt = (
        select(
            PracticeResult.topic_id,
            func.avg(PracticeResult.score).label("avg_score"),
            func.count(PracticeResult.id).label("attempt_count"),
        )
        .where(
            PracticeResult.user_id == user_id,
            PracticeResult.score.isnot(None),
            PracticeResult.topic_id.isnot(None),
        )
        .group_by(PracticeResult.topic_id)
        .having(func.avg(PracticeResult.score) < threshold)
    )
    rows = (await session.execute(stmt)).all()

    weak = []
    for row in rows:
        topic = await session.get(LearningTopic, row.topic_id)
        if topic is None:
            continue
        avg_score = float(row.avg_score or 0)
        recommendation = _generate_recommendation(topic.title, avg_score)
        weak.append(
            WeakTopicInfo(
                topic_id=topic.id,
                topic_title=topic.title,
                average_score=round(avg_score, 1),
                attempt_count=int(row.attempt_count),
                recommendation=recommendation,
            )
        )
    return weak


def _generate_recommendation(topic_title: str, avg_score: float) -> str:
    if avg_score < 30:
        return f"You need to revisit the fundamentals of '{topic_title}'. Start from the basics."
    if avg_score < 50:
        return f"Practice more '{topic_title}' exercises. Try easier difficulty first."
    return f"You're improving in '{topic_title}'! Keep practicing to reach mastery."


# ─────────────────────────────────────────────
# Progress Update
# ─────────────────────────────────────────────


async def update_progress_after_practice(
    session: AsyncSession,
    user_id: UUID,
    topic_id: UUID,
    score_percent: float,
) -> bool:
    """
    Update LearningProgress after a practice session.
    Returns True if progress was updated.
    """
    progress = await session.scalar(
        select(LearningProgress).where(
            LearningProgress.user_id == user_id,
            LearningProgress.topic_id == topic_id,
        )
    )

    if progress is None:
        progress = LearningProgress(
            user_id=user_id,
            topic_id=topic_id,
            status="in_progress",
            percent_complete=min(score_percent, 100.0),
            last_studied_at=datetime.now(timezone.utc),
        )
        session.add(progress)
    else:
        # Move progress forward if new score is better
        progress.percent_complete = min(
            100.0,
            max(progress.percent_complete, score_percent),
        )
        progress.last_studied_at = datetime.now(timezone.utc)
        if score_percent >= 80 and progress.status != "completed":
            progress.status = "in_progress"
        if score_percent >= 95:
            progress.status = "completed"

    await session.flush()
    return True
