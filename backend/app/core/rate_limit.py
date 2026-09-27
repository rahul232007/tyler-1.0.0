"""
JARVIS - In-memory sliding-window rate limiter.
Applied as FastAPI middleware. Uses slowapi when available, otherwise a
custom implementation based on per-IP request counts.

Never logs sensitive data.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Per-route rate limit rules: (requests, window_seconds)
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "/api/v1/auth/login": (10, 60),
    "/api/v1/auth/register": (5, 60),
    "/api/v1/chat": (60, 60),
    "/api/v1/voice": (20, 60),
    "/api/v1/learning/practice/generate": (30, 60),
    "/api/v1/learning": (60, 60),
    "/api/v1/memories": (60, 60),
    "/api/v1/conversations": (60, 60),
    "/api/v1/search": (20, 60),
    "/api/v1/documents": (20, 60),
}

DEFAULT_LIMIT = (120, 60)  # 120 requests per 60 seconds for unmatched routes


@dataclass
class _Window:
    requests: deque = field(default_factory=deque)
    lock: Lock = field(default_factory=Lock)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window per-IP rate limiter.
    Checks route prefix against RATE_LIMITS table.
    Returns 429 with Retry-After header when limit exceeded.
    """

    def __init__(self, app, enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled
        self._windows: dict[str, _Window] = defaultdict(_Window)

    def _get_limit(self, path: str) -> tuple[int, int]:
        """Return (max_requests, window_seconds) for the given path."""
        for prefix, limit in RATE_LIMITS.items():
            if path.startswith(prefix):
                return limit
        return DEFAULT_LIMIT

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For for proxy setups."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        # Health/docs endpoints exempt from rate limiting
        if path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        key = f"{client_ip}:{path}"
        max_requests, window_secs = self._get_limit(path)

        window = self._windows[key]
        now = time.monotonic()

        with window.lock:
            # Remove expired timestamps
            cutoff = now - window_secs
            while window.requests and window.requests[0] < cutoff:
                window.requests.popleft()

            if len(window.requests) >= max_requests:
                retry_after = int(window_secs - (now - window.requests[0]))
                logger.warning(
                    "Rate limit exceeded | ip=%s | path=%s | limit=%d/%ds",
                    client_ip[:15],  # truncate IP for log safety
                    path,
                    max_requests,
                    window_secs,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": True,
                        "message": "Too many requests. Please slow down.",
                        "status_code": 429,
                        "retry_after_seconds": max(retry_after, 1),
                    },
                    headers={"Retry-After": str(max(retry_after, 1))},
                )

            window.requests.append(now)

        return await call_next(request)
