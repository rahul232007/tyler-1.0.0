"""
JARVIS - Core Configuration
Reads all settings from .env file using Pydantic Settings.
Never hard-code secrets here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    app_name: str = Field(default="Jarvis", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    secret_key: str = Field(alias="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=60, ge=1, le=1440, alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # --- LLM ---
    default_ai_provider: str = Field(default="auto", alias="DEFAULT_AI_PROVIDER")
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")

    # Gemini
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # NVIDIA
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    nvidia_model: str = Field(
        default="meta/llama-3.1-70b-instruct", alias="NVIDIA_MODEL"
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen2.5:3b", alias="OLLAMA_MODEL")

    @property
    def active_default_provider(self) -> str:
        if self.default_ai_provider and self.default_ai_provider != "auto":
            return self.default_ai_provider
        return self.llm_provider or self.default_ai_provider or "auto"

    # --- Database ---
    database_url: str = Field(alias="DATABASE_URL")

    # --- Voice / STT ---
    whisper_model_size: str = Field(default="tiny", alias="WHISPER_MODEL_SIZE")

    # --- Voice / TTS ---
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model_id: str = Field(
        default="eleven_multilingual_v2", alias="ELEVENLABS_MODEL_ID"
    )
    elevenlabs_stability: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="ELEVENLABS_STABILITY"
    )
    elevenlabs_similarity_boost: float = Field(
        default=0.75, ge=0.0, le=1.0, alias="ELEVENLABS_SIMILARITY_BOOST"
    )
    elevenlabs_style: float = Field(
        default=0.0, ge=0.0, le=1.0, alias="ELEVENLABS_STYLE"
    )

    # --- Memory ---
    memory_encryption_key: str = Field(alias="MEMORY_ENCRYPTION_KEY")
    max_conversation_history: int = Field(default=50, alias="MAX_CONVERSATION_HISTORY")

    # --- Search ---
    web_search_enabled: bool = Field(default=True, alias="WEB_SEARCH_ENABLED")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")

    # --- Documents ---
    max_document_size_mb: int = Field(default=10, alias="MAX_DOCUMENT_SIZE_MB")

    # --- Rate Limiting ---
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timing_log_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "scratch"
        / "server_timings.jsonl",
        alias="JARVIS_TIMING_LOG_PATH",
    )

    # --- CORS ---
    frontend_url: str = Field(default="http://localhost:5173", alias="FRONTEND_URL")

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """
    Returns cached settings instance.
    lru_cache ensures .env is read only once on startup.
    """
    return Settings()
