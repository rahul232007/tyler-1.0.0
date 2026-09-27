"""Pydantic schemas for the Learning System."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─────────────────────────────────────────────
# Enums / Literals
# ─────────────────────────────────────────────
TopicStatus = Literal["planned", "in_progress", "completed", "paused"]
ProgressStatus = Literal["not_started", "in_progress", "completed", "needs_review"]
DifficultyLevel = Literal["beginner", "intermediate", "advanced"]
QuestionType = Literal["mcq", "short_answer", "coding"]
SessionStatus = Literal["pending", "in_progress", "completed"]


# ─────────────────────────────────────────────
# Learning Topics
# ─────────────────────────────────────────────
class TopicCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    difficulty_level: DifficultyLevel | None = None
    status: TopicStatus = "planned"


class TopicUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    difficulty_level: DifficultyLevel | None = None
    status: TopicStatus | None = None

    @model_validator(mode="after")
    def require_change(self) -> "TopicUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")
        return self


class TopicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    description: str | None
    difficulty_level: str | None
    status: str
    created_at: datetime
    updated_at: datetime


# ─────────────────────────────────────────────
# Learning Progress
# ─────────────────────────────────────────────
class ProgressCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: UUID
    status: ProgressStatus = "not_started"
    percent_complete: float = Field(default=0.0, ge=0.0, le=100.0)
    notes: str | None = Field(default=None, max_length=5000)


class ProgressUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ProgressStatus | None = None
    percent_complete: float | None = Field(default=None, ge=0.0, le=100.0)
    notes: str | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def require_change(self) -> "ProgressUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")
        return self


class ProgressRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    topic_id: UUID
    status: str
    percent_complete: float
    notes: str | None
    last_studied_at: datetime | None
    created_at: datetime
    updated_at: datetime


# ─────────────────────────────────────────────
# Practice Generation
# ─────────────────────────────────────────────
class PracticeGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: UUID
    question_type: QuestionType = "mcq"
    difficulty: DifficultyLevel = "intermediate"
    num_questions: int = Field(default=5, ge=1, le=20)


class MCQOption(BaseModel):
    label: str  # "A", "B", "C", "D"
    text: str


class GeneratedQuestion(BaseModel):
    question: str
    question_type: QuestionType
    options: list[MCQOption] | None = None  # For MCQ
    correct_answer: str  # For MCQ: "A"/"B"/"C"/"D"; for others: expected answer text
    explanation: str | None = None
    difficulty: str | None = None


class PracticeSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    topic_id: UUID | None
    questions: list[Any]
    question_type: str
    difficulty: str | None
    total_questions: int
    answered_count: int
    score: float | None
    status: str
    created_at: datetime
    completed_at: datetime | None


# ─────────────────────────────────────────────
# Practice Submission
# ─────────────────────────────────────────────
class QuestionAnswer(BaseModel):
    """Answer to one question in a practice session."""
    question_index: int = Field(ge=0)
    answer: str = Field(min_length=0, max_length=5000)


class PracticeSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[QuestionAnswer] = Field(min_length=1)


class QuestionResult(BaseModel):
    question_index: int
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    score: float  # 0-100
    feedback: str | None = None


class PracticeSubmitResponse(BaseModel):
    session_id: UUID
    total_questions: int
    answered_count: int
    correct_count: int
    score_percent: float  # 0-100
    results: list[QuestionResult]
    weak_topics: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    progress_updated: bool = False


# ─────────────────────────────────────────────
# Practice History
# ─────────────────────────────────────────────
class PracticeHistoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    topic_id: UUID | None
    question_type: str
    difficulty: str | None
    total_questions: int
    score: float | None
    status: str
    created_at: datetime
    completed_at: datetime | None


# ─────────────────────────────────────────────
# Weak Topics & Recommendations
# ─────────────────────────────────────────────
class WeakTopicInfo(BaseModel):
    topic_id: UUID
    topic_title: str
    average_score: float
    attempt_count: int
    recommendation: str
