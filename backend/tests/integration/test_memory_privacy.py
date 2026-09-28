"""Database persistence and API privacy regressions for sensitive memories."""

import json

import pytest
from app.api.v1 import memories as memories_api
from app.models import PersonalMemory
from app.services.memory_service import decode_memory_value, encrypt_value
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_manual_sensitive_memory_is_encrypted_in_database(
    client, auth_headers, db_session: AsyncSession, test_user
):
    private_value = {"note": "manual-sensitive-sentinel-7f2c"}
    response = await client.post(
        "/api/v1/memories/",
        headers=auth_headers,
        json={
            "category": "context",
            "memory_key": "private_note",
            "value": private_value,
            "is_sensitive": True,
        },
    )

    assert response.status_code == 201
    assert response.json()["value"] == {"redacted": True}
    raw_value, encrypted_value, embedding = (
        await db_session.execute(
            select(
                PersonalMemory.value,
                PersonalMemory.encrypted_value,
                PersonalMemory.embedding,
            ).where(
                PersonalMemory.user_id == test_user.id,
                PersonalMemory.memory_key == "private_note",
            )
        )
    ).one()

    assert "manual-sensitive-sentinel-7f2c" not in json.dumps(
        {"value": raw_value, "embedding": embedding}
    )
    assert encrypted_value is not None
    assert decode_memory_value(raw_value) == private_value


@pytest.mark.asyncio
async def test_extracted_sensitive_memory_is_encrypted_in_database(
    client, auth_headers, db_session: AsyncSession, test_user, monkeypatch
):
    private_value = {"health_note": "extracted-sensitive-sentinel-3a91"}

    class ExtractionLLM:
        async def generate(self, prompt, system=""):
            return json.dumps(
                [
                    {
                        "category": "context",
                        "memory_key": "health_note",
                        "value": private_value,
                        "importance": 0.8,
                        "is_sensitive": True,
                    }
                ]
            )

    monkeypatch.setattr(
        memories_api, "get_llm_service", lambda: ExtractionLLM()
    )
    response = await client.post(
        "/api/v1/memories/extract",
        headers=auth_headers,
        json={"conversation_text": "Keep this private health note."},
    )

    assert response.status_code == 200
    assert response.json()["extracted_count"] == 1
    assert response.json()["memories"][0]["value"] == {"redacted": True}
    assert "extracted-sensitive-sentinel-3a91" not in response.text
    raw_value, encrypted_value, embedding = (
        await db_session.execute(
            select(
                PersonalMemory.value,
                PersonalMemory.encrypted_value,
                PersonalMemory.embedding,
            ).where(
                PersonalMemory.user_id == test_user.id,
                PersonalMemory.memory_key == "health_note",
            )
        )
    ).one()

    assert "extracted-sensitive-sentinel-3a91" not in json.dumps(
        {"value": raw_value, "embedding": embedding}
    )
    assert encrypted_value is not None
    assert decode_memory_value(raw_value) == private_value


def test_encryption_helper_round_trip_remains_compatible():
    private_value = {"secret": "helper-round-trip-sentinel"}

    assert decode_memory_value(
        {"__jarvis_encrypted_value__": encrypt_value(private_value)}
    ) == private_value