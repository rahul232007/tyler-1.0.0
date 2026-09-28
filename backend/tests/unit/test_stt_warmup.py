"""Regression coverage for STT executor warmup and transcription wrappers."""

from concurrent.futures import ThreadPoolExecutor

import app.services.stt as stt_module
import pytest
from app.services.stt import SpeechToTextService


class RecordingExecutor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.submit_count = 0

    def submit(self, function, *args):
        self.submit_count += 1
        return self.delegate.submit(function, *args)


@pytest.mark.asyncio
async def test_warmup_uses_shared_executor_without_loading_whisper(monkeypatch):
    service = SpeechToTextService()
    with ThreadPoolExecutor(max_workers=1) as delegate:
        executor = RecordingExecutor(delegate)
        monkeypatch.setattr(stt_module, "_executor", executor)
        monkeypatch.setattr(service, "_get_model", lambda: "mock-model")

        await service.warmup_async()

    assert executor.submit_count == 1


@pytest.mark.asyncio
async def test_async_transcription_wrapper_still_runs_on_shared_executor(monkeypatch):
    service = SpeechToTextService()
    with ThreadPoolExecutor(max_workers=1) as delegate:
        executor = RecordingExecutor(delegate)
        monkeypatch.setattr(stt_module, "_executor", executor)
        monkeypatch.setattr(
            service,
            "_transcribe_path_sync",
            lambda path, language: {"text": "transcribed", "language": language},
        )

        result = await service.transcribe_file_async("audio.test", language="en")

    assert result == {"text": "transcribed", "language": "en"}
    assert executor.submit_count == 1