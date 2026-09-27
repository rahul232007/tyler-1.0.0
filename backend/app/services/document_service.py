"""
JARVIS - Document Service
Handles file upload, text extraction, validation, and safe temp storage.
Supports PDF, plain text, and markdown files.

Security:
- File size limits enforced
- Path traversal protection (never use user-provided filenames directly)
- Content-type validation
- Safe temp directory management
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_BYTES = settings.max_document_size_mb * 1024 * 1024

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/octet-stream",
}

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def validate_document(filename: str, content_type: str | None, size_bytes: int) -> None:
    """Raise ValueError for invalid/unsafe documents."""
    # Size check
    if size_bytes > MAX_BYTES:
        raise ValueError(
            f"File too large ({size_bytes / 1024 / 1024:.1f} MB). "
            f"Maximum is {settings.max_document_size_mb} MB."
        )
    if size_bytes == 0:
        raise ValueError("File is empty.")

    # Extension check (path traversal protection — use stem only)
    safe_name = Path(filename).name  # strips any directory components
    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Content-type check (soft — browsers may send generic types)
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        logger.warning(
            "Unexpected content-type '%s' for document upload.", content_type
        )


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError as exc:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf") from exc
    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {type(exc).__name__}: {exc}") from exc


def extract_text_from_bytes(
    data: bytes,
    filename: str,
    content_type: str | None = None,
) -> str:
    """
    Extract text from document bytes based on file extension.
    Returns extracted text string.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" or (content_type and "pdf" in content_type):
        return extract_text_from_pdf(data)
    # Plain text / markdown
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1").strip()
        except Exception as exc:
            raise ValueError("Cannot decode file as text.") from exc


def summarize_for_context(text: str, max_chars: int = 4000) -> str:
    """Truncate document text to fit in chat context."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[Document truncated at {max_chars} characters...]"
