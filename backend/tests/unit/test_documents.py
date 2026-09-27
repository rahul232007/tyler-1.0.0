"""Unit tests for document validation, extraction, and context summarization."""
import pytest

from app.services.document_service import (
    extract_text_from_bytes,
    summarize_for_context,
    validate_document,
)


def test_validate_document_success():
    validate_document("notes.txt", "text/plain", 1024)
    validate_document("guide.md", "text/markdown", 2048)
    validate_document("sample.pdf", "application/pdf", 4096)


def test_validate_document_failures():
    # Empty file
    with pytest.raises(ValueError, match="empty"):
        validate_document("empty.txt", "text/plain", 0)

    # Unsupported extension
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_document("script.exe", "application/x-msdownload", 500)

    # Oversized file
    with pytest.raises(ValueError, match="too large"):
        validate_document("huge.pdf", "application/pdf", 100 * 1024 * 1024)


def test_extract_text_from_plain_and_markdown():
    text_content = "Hello, this is JARVIS tutor notes for Python basics."
    extracted = extract_text_from_bytes(text_content.encode("utf-8"), "notes.txt")
    assert extracted == text_content


def test_summarize_for_context():
    short_text = "Brief summary note."
    assert summarize_for_context(short_text, max_chars=100) == short_text

    long_text = "A" * 500
    truncated = summarize_for_context(long_text, max_chars=50)
    assert len(truncated) > 50
    assert "truncated" in truncated
