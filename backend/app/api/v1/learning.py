"""
JARVIS - Learning System API Endpoints
Provides:
  - Topics CRUD
  - Progress tracking
  - AI-powered practice generation & submission
  - Weak-topic detection and recommendations
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentUser, DatabaseSession
from app.models import LearningProgress, LearningTopic, PracticeResult, PracticeSession
from app.schemas.learning import (
    PracticeGenerateRequest,
    PracticeHistoryRead,
    PracticeSessionRead,
    PracticeSubmitRequest,
    PracticeSubmitResponse,
    ProgressCreate,
    ProgressRead,
    ProgressUpdate,
    QuestionResult,
    TopicCreate,
    TopicRead,
    TopicUpdate,
    WeakTopicInfo,
)
from app.services.learning import (
    detect_weak_topics,
    generate_practice_questions,
    grade_mcq,
    grade_open_answer,
    update_progress_after_practice,
)
from app.services.llm import get_llm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/learning", tags=["Learning System"])


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
async def _get_owned_topic(topic_id: UUID, user_id: UUID, session: DatabaseSession) -> LearningTopic:
    topic = await session.scalar(
        select(LearningTopic).where(
            LearningTopic.id == topic_id,
            LearningTopic.user_id == user_id,
        )
    )
    if topic is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning topic not found.")
    return topic


# ─────────────────────────────────────────────
# Topics Endpoints
# ─────────────────────────────────────────────
@router.post("/topics", response_model=TopicRead, status_code=status.HTTP_201_CREATED)
async def create_topic(
    payload: TopicCreate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningTopic:
    """Create a new learning topic for the authenticated user."""
    topic = LearningTopic(
        user_id=current_user.id,
        title=payload.title.strip(),
        description=payload.description.strip() if payload.description else None,
        difficulty_level=payload.difficulty_level,
        status=payload.status,
    )
    session.add(topic)
    await session.flush()
    await session.refresh(topic)
    return topic


@router.get("/topics", response_model=list[TopicRead])
async def list_topics(
    current_user: CurrentUser,
    session: DatabaseSession,
    status_filter: str | None = Query(default=None, alias="status"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[LearningTopic]:
    """List learning topics owned by current user."""
    stmt = select(LearningTopic).where(LearningTopic.user_id == current_user.id)
    if status_filter:
        stmt = stmt.where(LearningTopic.status == status_filter)
    stmt = stmt.order_by(LearningTopic.updated_at.desc()).offset(offset).limit(limit)
    result = await session.scalars(stmt)
    return list(result)


@router.get("/topics/{topic_id}", response_model=TopicRead)
async def get_topic(
    topic_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningTopic:
    """Get single learning topic by ID."""
    return await _get_owned_topic(topic_id, current_user.id, session)


@router.put("/topics/{topic_id}", response_model=TopicRead)
async def update_topic(
    topic_id: UUID,
    payload: TopicUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningTopic:
    """Update a learning topic."""
    topic = await _get_owned_topic(topic_id, current_user.id, session)
    updates = payload.model_dump(exclude_unset=True)
    for field_name, val in updates.items():
        setattr(topic, field_name, val)
    await session.flush()
    await session.refresh(topic)
    return topic


@router.delete("/topics/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_topic(
    topic_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Response:
    """Delete a learning topic and associated progress/practice."""
    topic = await _get_owned_topic(topic_id, current_user.id, session)
    await session.delete(topic)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ─────────────────────────────────────────────
# Progress Endpoints
# ─────────────────────────────────────────────
@router.post("/progress", response_model=ProgressRead, status_code=status.HTTP_201_CREATED)
async def create_progress(
    payload: ProgressCreate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningProgress:
    """Create or initialize progress for a topic."""
    await _get_owned_topic(payload.topic_id, current_user.id, session)

    existing = await session.scalar(
        select(LearningProgress).where(
            LearningProgress.user_id == current_user.id,
            LearningProgress.topic_id == payload.topic_id,
        )
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Progress record for this topic already exists. Use PUT to update it.",
        )

    progress = LearningProgress(
        user_id=current_user.id,
        topic_id=payload.topic_id,
        status=payload.status,
        percent_complete=payload.percent_complete,
        notes=payload.notes,
        last_studied_at=datetime.now(timezone.utc),
    )
    session.add(progress)
    await session.flush()
    await session.refresh(progress)
    return progress


@router.get("/progress", response_model=list[ProgressRead])
async def list_progress(
    current_user: CurrentUser,
    session: DatabaseSession,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[LearningProgress]:
    """List all topic progress entries for caller."""
    stmt = (
        select(LearningProgress)
        .where(LearningProgress.user_id == current_user.id)
        .order_by(LearningProgress.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.scalars(stmt)
    return list(result)


@router.get("/progress/{topic_id}", response_model=ProgressRead)
async def get_progress(
    topic_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningProgress:
    """Get progress entry for a specific topic."""
    progress = await session.scalar(
        select(LearningProgress).where(
            LearningProgress.user_id == current_user.id,
            LearningProgress.topic_id == topic_id,
        )
    )
    if progress is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Progress record not found for this topic.")
    return progress


@router.put("/progress/{topic_id}", response_model=ProgressRead)
async def update_progress(
    topic_id: UUID,
    payload: ProgressUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> LearningProgress:
    """Update progress for a specific topic."""
    progress = await session.scalar(
        select(LearningProgress).where(
            LearningProgress.user_id == current_user.id,
            LearningProgress.topic_id == topic_id,
        )
    )
    if progress is None:
        # Auto-create if not exists
        await _get_owned_topic(topic_id, current_user.id, session)
        progress = LearningProgress(
            user_id=current_user.id,
            topic_id=topic_id,
            status=payload.status or "in_progress",
            percent_complete=payload.percent_complete or 0.0,
            notes=payload.notes,
            last_studied_at=datetime.now(timezone.utc),
        )
        session.add(progress)
    else:
        updates = payload.model_dump(exclude_unset=True)
        for field_name, val in updates.items():
            setattr(progress, field_name, val)
        progress.last_studied_at = datetime.now(timezone.utc)

    await session.flush()
    await session.refresh(progress)
    return progress


# ─────────────────────────────────────────────
# Practice Endpoints
# ─────────────────────────────────────────────
@router.post("/practice/generate", response_model=PracticeSessionRead, status_code=status.HTTP_201_CREATED)
async def generate_practice(
    payload: PracticeGenerateRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> PracticeSession:
    """
    Generate an AI practice session (MCQ, short answer, or coding questions)
    using the LLM router.
    """
    topic = await _get_owned_topic(payload.topic_id, current_user.id, session)
    llm = get_llm_service()

    try:
        questions = await generate_practice_questions(
            topic_title=topic.title,
            question_type=payload.question_type,
            difficulty=payload.difficulty,
            num_questions=payload.num_questions,
            llm_service=llm,
        )
    except Exception as exc:
        logger.error("Practice question generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate practice session: {exc}",
        ) from exc

    practice_session = PracticeSession(
        user_id=current_user.id,
        topic_id=topic.id,
        questions=questions,
        question_type=payload.question_type,
        difficulty=payload.difficulty,
        total_questions=len(questions),
        answered_count=0,
        status="pending",
    )
    session.add(practice_session)
    await session.flush()
    await session.refresh(practice_session)
    return practice_session


@router.post("/practice/{practice_id}/submit", response_model=PracticeSubmitResponse)
async def submit_practice(
    practice_id: UUID,
    payload: PracticeSubmitRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> PracticeSubmitResponse:
    """
    Submit answers to a practice session.
    Grades the quiz, updates score, stores PracticeResult rows,
    updates topic learning progress, and detects weak topics.
    """
    practice_session = await session.scalar(
        select(PracticeSession)
        .where(
            PracticeSession.id == practice_id,
            PracticeSession.user_id == current_user.id,
        )
        .options(selectinload(PracticeSession.results))
    )
    if practice_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Practice session not found.")

    questions = practice_session.questions or []
    if not questions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Practice session has no questions.")

    answers_by_idx = {a.question_index: a.answer for a in payload.answers}
    llm = get_llm_service()

    graded_results: list[QuestionResult] = []
    total_score = 0.0
    correct_count = 0

    for idx, q_data in enumerate(questions):
        user_ans = answers_by_idx.get(idx, "").strip()
        q_text = q_data.get("question", "")
        correct_ans = q_data.get("correct_answer", "")
        q_type = q_data.get("question_type", practice_session.question_type)

        if q_type == "mcq":
            is_corr, score, feedback = grade_mcq(user_ans, correct_ans)
        else:
            is_corr, score, feedback = await grade_open_answer(
                question=q_text,
                user_answer=user_ans,
                expected_answer=correct_ans,
                llm_service=llm,
            )

        if is_corr:
            correct_count += 1
        total_score += score

        # Store individual result
        result_row = PracticeResult(
            user_id=current_user.id,
            topic_id=practice_session.topic_id,
            session_id=practice_session.id,
            question=q_text,
            question_type=q_type,
            options=q_data.get("options"),
            user_answer=user_ans,
            expected_answer=correct_ans,
            is_correct=is_corr,
            score=score,
            feedback=feedback,
        )
        session.add(result_row)

        graded_results.append(
            QuestionResult(
                question_index=idx,
                question=q_text,
                user_answer=user_ans,
                correct_answer=correct_ans,
                is_correct=is_corr,
                score=score,
                feedback=feedback,
            )
        )

    # Compute overall percentage score
    avg_score = round(total_score / len(questions), 2)
    practice_session.answered_count = len(answers_by_idx)
    practice_session.score = avg_score
    practice_session.status = "completed"
    practice_session.completed_at = datetime.now(timezone.utc)

    # Update LearningProgress if linked to topic
    progress_updated = False
    if practice_session.topic_id:
        progress_updated = await update_progress_after_practice(
            session=session,
            user_id=current_user.id,
            topic_id=practice_session.topic_id,
            score_percent=avg_score,
        )

    # Detect weak topics
    weak_info = await detect_weak_topics(session, current_user.id)
    weak_titles = [w.topic_title for w in weak_info]
    recommendations = [w.recommendation for w in weak_info]

    if not recommendations and avg_score >= 80:
        recommendations.append("Outstanding work! You have strong mastery of this topic.")
    elif not recommendations:
        recommendations.append("Keep reviewing and practicing to solidify your knowledge.")

    await session.flush()

    return PracticeSubmitResponse(
        session_id=practice_session.id,
        total_questions=len(questions),
        answered_count=len(answers_by_idx),
        correct_count=correct_count,
        score_percent=avg_score,
        results=graded_results,
        weak_topics=weak_titles,
        recommendations=recommendations,
        progress_updated=progress_updated,
    )


@router.get("/practice/history", response_model=list[PracticeHistoryRead])
async def practice_history(
    current_user: CurrentUser,
    session: DatabaseSession,
    topic_id: UUID | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[PracticeSession]:
    """Get history of completed or pending practice sessions."""
    stmt = select(PracticeSession).where(PracticeSession.user_id == current_user.id)
    if topic_id:
        stmt = stmt.where(PracticeSession.topic_id == topic_id)
    stmt = stmt.order_by(PracticeSession.created_at.desc()).offset(offset).limit(limit)
    result = await session.scalars(stmt)
    return list(result)


@router.get("/practice/weak-topics", response_model=list[WeakTopicInfo])
async def get_weak_topics_endpoint(
    current_user: CurrentUser,
    session: DatabaseSession,
    threshold: float = Query(default=60.0, ge=0.0, le=100.0),
) -> list[WeakTopicInfo]:
    """Get topics where user average score is below the given threshold."""
    return await detect_weak_topics(session, current_user.id, threshold=threshold)


@router.get("/practice/{practice_id}", response_model=PracticeSessionRead)
async def get_practice_session(
    practice_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> PracticeSession:
    """Get a specific practice session by ID."""
    practice = await session.scalar(
        select(PracticeSession).where(
            PracticeSession.id == practice_id,
            PracticeSession.user_id == current_user.id,
        )
    )
    if practice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Practice session not found.")
    return practice
