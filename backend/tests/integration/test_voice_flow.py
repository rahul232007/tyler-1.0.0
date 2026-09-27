"""Integration tests for Voice system (status, Faster-Whisper real audio transcription, and TTS)."""
import logging
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.stt import stt_service
from app.services.tts import tts_service

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENGLISH_WAV = BACKEND_DIR / "sample_english_speech.wav"
TAMIL_WAV = BACKEND_DIR / "sample_tamil_speech.wav"


@pytest.mark.asyncio
async def test_voice_status_endpoint(client: AsyncClient):
    res = await client.get("/api/v1/voice/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "en" in data["supported_languages"]
    assert "ta" in data["supported_languages"]
    assert data["local_processing"] is True


@pytest.mark.asyncio
async def test_real_faster_whisper_english_transcription():
    """Directly test Faster-Whisper STT with real human-spoken English WAV file."""
    assert ENGLISH_WAV.exists(), f"Missing test audio file at {ENGLISH_WAV}"

    result = await stt_service.transcribe_file_async(str(ENGLISH_WAV), language="en")
    assert "text" in result
    assert len(result["text"]) > 0, "Transcribed text should not be empty"
    assert result["language"] == "en"
    logger.info("Real English STT Result (len=%d): %s", len(result["text"]), result["text"][:60])


@pytest.mark.asyncio
async def test_real_faster_whisper_tamil_transcription():
    """Directly test Faster-Whisper STT with real human-spoken Tamil WAV file."""
    assert TAMIL_WAV.exists(), f"Missing test audio file at {TAMIL_WAV}"

    result = await stt_service.transcribe_file_async(str(TAMIL_WAV), language="ta")
    assert "text" in result
    assert len(result["text"]) > 0, "Transcribed text should not be empty"
    # Safely log Tamil text length to prevent Windows console cp1252 print errors
    logger.info("Real Tamil STT Result (len=%d, lang=%s)", len(result["text"]), result.get("language"))


@pytest.mark.asyncio
async def test_transcribe_audio_upload_endpoint(client: AsyncClient):
    """Test POST /voice/transcribe with real audio file upload."""
    assert ENGLISH_WAV.exists()

    with open(ENGLISH_WAV, "rb") as f:
        file_bytes = f.read()

    files = {"audio_file": ("sample_english_speech.wav", file_bytes, "audio/wav")}
    res = await client.post("/api/v1/voice/transcribe", files=files, data={"language": "en"})
    assert res.status_code == 200
    data = res.json()
    assert "text" in data
    assert len(data["text"]) > 0


def test_tts_configuration():
    """Check ElevenLabs TTS configuration status."""
    is_conf = tts_service.is_configured()
    assert isinstance(is_conf, bool)
