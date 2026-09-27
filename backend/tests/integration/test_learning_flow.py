"""Integration tests for Learning System flow (topics, progress, practice session submit, weak topics)."""
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models import PracticeSession


@pytest.mark.asyncio
async def test_learning_topics_and_progress_lifecycle(client: AsyncClient, auth_headers: dict[str, str]):
    # 1. Create learning topic
    topic_res = await client.post(
        "/api/v1/learning/topics",
        headers=auth_headers,
        json={
            "title": "Machine Learning Fundamentals",
            "description": "Supervised, unsupervised, and deep learning basics",
            "difficulty_level": "intermediate",
            "status": "planned",
        },
    )
    assert topic_res.status_code == 201
    topic_id = topic_res.json()["id"]

    # 2. List topics
    list_res = await client.get("/api/v1/learning/topics", headers=auth_headers)
    assert list_res.status_code == 200
    assert any(t["id"] == topic_id for t in list_res.json())

    # 3. Create progress for topic
    prog_res = await client.post(
        "/api/v1/learning/progress",
        headers=auth_headers,
        json={
            "topic_id": topic_id,
            "status": "in_progress",
            "percent_complete": 25.0,
            "notes": "Finished linear regression chapter",
        },
    )
    assert prog_res.status_code == 201
    assert prog_res.json()["percent_complete"] == 25.0

    # 4. Update progress
    update_prog_res = await client.put(
        f"/api/v1/learning/progress/{topic_id}",
        headers=auth_headers,
        json={"percent_complete": 60.0, "status": "in_progress"},
    )
    assert update_prog_res.status_code == 200
    assert update_prog_res.json()["percent_complete"] == 60.0

    # 5. Get progress
    get_prog_res = await client.get(f"/api/v1/learning/progress/{topic_id}", headers=auth_headers)
    assert get_prog_res.status_code == 200
    assert get_prog_res.json()["percent_complete"] == 60.0


@pytest.mark.asyncio
async def test_practice_session_submission_and_weak_topics(client: AsyncClient, auth_headers: dict[str, str], db_session):
    # Create topic
    t_res = await client.post(
        "/api/v1/learning/topics",
        headers=auth_headers,
        json={"title": "Data Structures & Algorithms", "status": "in_progress"},
    )
    topic_id = t_res.json()["id"]

    # Seed a practice session directly in DB with known questions for reliable testing
    me_res = await client.get("/api/v1/auth/me", headers=auth_headers)
    user_id = me_res.json()["id"]

    test_session = PracticeSession(
        user_id=user_id,
        topic_id=topic_id,
        question_type="mcq",
        difficulty="intermediate",
        total_questions=2,
        questions=[
            {
                "question": "What is the time complexity of binary search?",
                "question_type": "mcq",
                "options": [
                    {"label": "A", "text": "O(log n)"},
                    {"label": "B", "text": "O(n)"},
                    {"label": "C", "text": "O(n^2)"},
                    {"label": "D", "text": "O(1)"},
                ],
                "correct_answer": "A",
            },
            {
                "question": "Which data structure uses FIFO ordering?",
                "question_type": "mcq",
                "options": [
                    {"label": "A", "text": "Stack"},
                    {"label": "B", "text": "Queue"},
                    {"label": "C", "text": "Tree"},
                    {"label": "D", "text": "Graph"},
                ],
                "correct_answer": "B",
            },
        ],
        status="pending",
    )
    db_session.add(test_session)
    await db_session.commit()
    await db_session.refresh(test_session)
    session_id = test_session.id

    # Submit practice answers: 1 correct (A), 1 incorrect (A instead of B)
    submit_res = await client.post(
        f"/api/v1/learning/practice/{session_id}/submit",
        headers=auth_headers,
        json={
            "answers": [
                {"question_index": 0, "answer": "A"},
                {"question_index": 1, "answer": "A"},
            ]
        },
    )
    assert submit_res.status_code == 200
    res_data = submit_res.json()
    assert res_data["total_questions"] == 2
    assert res_data["correct_count"] == 1
    assert res_data["score_percent"] == 50.0  # 1 out of 2 correct = 50%
    assert res_data["progress_updated"] is True

    # Practice history
    history_res = await client.get("/api/v1/learning/practice/history", headers=auth_headers)
    assert history_res.status_code == 200
    assert any(h["id"] == str(session_id) for h in history_res.json())

    # Check weak topics detection (50% < 60% threshold)
    weak_res = await client.get("/api/v1/learning/practice/weak-topics?threshold=60", headers=auth_headers)
    assert weak_res.status_code == 200
    weak_titles = [w["topic_title"] for w in weak_res.json()]
    assert "Data Structures & Algorithms" in weak_titles
