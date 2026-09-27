"""
JARVIS - FastAPI Main Application Entry Point
Production-hardened, structured logging, rate limiting, error handling,
tool registry initialization, and comprehensive router configuration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1 import auth, chat, conversations, documents, learning, memories, voice
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging_config import configure_logging, new_request_id
from app.core.rate_limit import RateLimitMiddleware
from app.tools.tools import register_stateless_tools

settings = get_settings()

# Initialize structured logging
configure_logging(level=settings.log_level)


import asyncio
import ctypes
import logging
import sys

logger = logging.getLogger(__name__)


def _prevent_windows_sleep():
    """Prevent Windows Modern Standby / Sleep from suspending Jarvis during active runtime."""
    if sys.platform == "win32":
        try:
            # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            logger.info("Windows active execution state set (prevent idle suspend)")
        except Exception as e:
            logger.debug("Failed to set execution state: %s", e)


async def _background_warmup():
    """Warm up Faster-Whisper and Ollama Qwen models in background upon startup."""
    _prevent_windows_sleep()
    try:
        from app.services.stt import stt_service

        await stt_service.warmup_async()
        logger.info("Faster-Whisper STT model pre-warmed successfully")
    except Exception as exc:
        logger.debug("STT warmup skipped: %s", exc)

    try:
        from app.services.llm import get_llm_service

        llm = get_llm_service()
        await llm.ollama.generate(prompt="hi", num_predict=1)
        logger.info("Ollama Qwen model pre-warmed successfully")
    except Exception as exc:
        logger.debug("Ollama warmup skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    register_stateless_tools()
    asyncio.create_task(_background_warmup())
    yield


# ─────────────────────────────────────────────
# Create FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="JARVIS - Personal AI Assistant & Tutor",
    description="Private voice-enabled AI tutor with long-term memory, tools, and learning systems.",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Middleware 1: Request ID Injection & Header
# ─────────────────────────────────────────────
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = new_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestIdMiddleware)


# ─────────────────────────────────────────────
# Middleware 2: Rate Limiting
# ─────────────────────────────────────────────
app.add_middleware(RateLimitMiddleware, enabled=settings.rate_limit_enabled)


# ─────────────────────────────────────────────
# Middleware 3: CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────
register_error_handlers(app)


# ─────────────────────────────────────────────
# Register API Routers
# ─────────────────────────────────────────────
app.include_router(chat.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(conversations.router, prefix="/api/v1")
app.include_router(memories.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(learning.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


# ─────────────────────────────────────────────
# Health & Root Check
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "message": "JARVIS is online. Vanakkam! 🙏",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }
