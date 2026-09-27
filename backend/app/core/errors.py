"""
JARVIS - Consistent API Error Responses and Global Exception Handlers.
All API errors return a uniform JSON envelope so clients parse one format.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Standard error envelope
# ─────────────────────────────────────────────
def error_response(
    status_code: int,
    message: str,
    detail: str | list | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    body: dict = {"error": True, "message": message, "status_code": status_code}
    if detail is not None:
        body["detail"] = detail
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=body)


# ─────────────────────────────────────────────
# Register handlers on the FastAPI app
# ─────────────────────────────────────────────
def register_error_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        from app.core.logging_config import request_id_var

        rid = request_id_var.get("") or "-"
        # Don't log 401/404 at error level — these are expected
        log_fn = logger.warning if exc.status_code < 500 else logger.error
        log_fn(
            "HTTP %s — %s %s — %s",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
            extra={"request_id": rid},
        )
        return error_response(
            status_code=exc.status_code,
            message=exc.detail if isinstance(exc.detail, str) else "Request error",
            request_id=rid,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        from app.core.logging_config import request_id_var

        rid = request_id_var.get("") or "-"
        errors = exc.errors()
        # Sanitize: remove any 'input' fields that may contain user secrets
        safe_errors = [
            {k: v for k, v in e.items() if k not in ("input", "ctx")} for e in errors
        ]
        logger.warning(
            "Validation error — %s %s — %d field(s) invalid",
            request.method,
            request.url.path,
            len(errors),
            extra={"request_id": rid},
        )
        return error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed.",
            detail=safe_errors,
            request_id=rid,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        from app.core.logging_config import request_id_var

        rid = request_id_var.get("") or "-"
        logger.error(
            "Unhandled exception — %s %s — %s: %s",
            request.method,
            request.url.path,
            type(exc).__name__,
            str(exc)[:200],  # Truncate to avoid leaking large payloads
            exc_info=True,
            extra={"request_id": rid},
        )
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="An internal server error occurred. Please try again later.",
            request_id=rid,
        )
