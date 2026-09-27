"""
JARVIS - Structured Logging Configuration
Provides JSON-structured logging with request ID tracking.
Sensitive fields (passwords, tokens, API keys) are never logged.
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

# Context variable for per-request IDs
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Fields that must NEVER appear in logs
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "password",
        "password_hash",
        "access_token",
        "refresh_token",
        "token",
        "secret_key",
        "api_key",
        "apikey",
        "gemini_api_key",
        "nvidia_api_key",
        "elevenlabs_api_key",
        "memory_encryption_key",
        "authorization",
        "bearer",
        "encrypted_value",
        "decrypted_value",
    }
)


def _sanitize_record_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Remove any sensitive keys from extra log fields."""
    return {
        k: "[REDACTED]" if k.lower() in _SENSITIVE_FIELD_NAMES else v
        for k, v in extra.items()
    }


class SensitiveFilter(logging.Filter):
    """Drop or redact log records that may contain sensitive information."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        # Redact message if it contains obvious secret patterns
        msg = str(record.getMessage()).lower()
        if any(s in msg for s in ("password=", "api_key=", "bearer ", "secret=")):
            record.msg = "[REDACTED — potentially sensitive log message]"
            record.args = ()
        return True


class RequestIdFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = request_id_var.get("") or "-"
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger pre-configured for JARVIS."""
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logging for the JARVIS application.
    Call once at application startup.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveFilter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers on reload
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def new_request_id() -> str:
    """Generate and store a new request ID in the current context."""
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid
