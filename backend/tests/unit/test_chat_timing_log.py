"""Tests for configurable, non-fatal chat timing telemetry."""

import json

from app.api.v1 import chat
from app.core.config import get_settings


def test_timing_log_uses_configured_path(monkeypatch, tmp_path):
    timing_path = tmp_path / "diagnostics" / "timings.jsonl"
    monkeypatch.setattr(chat.settings, "timing_log_path", timing_path)

    chat._write_timing_record({"first_token": 1.25})

    assert json.loads(timing_path.read_text(encoding="utf-8")) == {
        "first_token": 1.25
    }


def test_timing_log_path_reads_environment_override(monkeypatch, tmp_path):
    configured_path = tmp_path / "custom" / "jarvis-timing.jsonl"
    monkeypatch.setenv("JARVIS_TIMING_LOG_PATH", str(configured_path))
    get_settings.cache_clear()
    try:
        assert get_settings().timing_log_path == configured_path
    finally:
        get_settings.cache_clear()


def test_timing_log_write_failure_is_logged_and_non_fatal(
    monkeypatch, tmp_path, caplog
):
    parent_file = tmp_path / "not_a_directory"
    parent_file.write_text("blocker", encoding="utf-8")
    monkeypatch.setattr(
        chat.settings, "timing_log_path", parent_file / "timings.jsonl"
    )

    chat._write_timing_record({"first_token": None})

    assert "Could not write chat timing log" in caplog.text