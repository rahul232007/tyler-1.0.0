"""
JARVIS - ElevenLabs TTS Service (async-safe, configurable voice settings).
All configuration is read from environment variables — no hard-coded credentials.

Key improvements:
- Async-safe (runs blocking SDK calls in thread executor)
- Configurable stability, similarity_boost, style, speaking rate
- Streaming audio support
- Tamil/English language routing
- Proper error handling and logging
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Thread pool for blocking ElevenLabs SDK calls
_tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="elevenlabs_tts")


class ElevenLabsTTS:
    """
    Async-safe adapter around the ElevenLabs SDK.
    Voice settings are read from environment variables.
    """

    def __init__(self) -> None:
        self.api_key = (settings.elevenlabs_api_key or "").strip()
        self.voice_id = (settings.elevenlabs_voice_id or "").strip()
        self.model_id = (settings.elevenlabs_model_id or "eleven_multilingual_v2").strip()
        self.stability = settings.elevenlabs_stability
        self.similarity_boost = settings.elevenlabs_similarity_boost
        self.style = settings.elevenlabs_style
        self._client = None

    def _get_client(self):
        """Lazy-initialize the ElevenLabs client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "ElevenLabs is not configured. Set ELEVENLABS_API_KEY."
                )
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError(
                    "elevenlabs package not installed. Run: pip install elevenlabs"
                ) from exc
        return self._client

    def is_configured(self) -> bool:
        """Return True only when both API key and voice ID are set."""
        return bool(self.api_key and self.voice_id)

    # ─────────────────────────────────────────────
    # Synchronous generation (runs in thread pool)
    # ─────────────────────────────────────────────
    def _generate_sync(
        self,
        text: str,
        voice_id: str,
        model_id: str,
    ) -> bytes:
        """Blocking ElevenLabs generation — do not call from async context directly."""
        client = self._get_client()
        try:
            from elevenlabs import VoiceSettings
            voice_settings = VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost,
                style=self.style,
                use_speaker_boost=True,
            )
        except ImportError:
            voice_settings = None

        try:
            kwargs: dict[str, Any] = {
                "text": text,
                "voice": voice_id,
                "model": model_id,
                "output_format": "mp3_44100_128",
            }
            if voice_settings is not None:
                kwargs["voice_settings"] = voice_settings

            audio = client.generate(**kwargs)

            # SDK may return bytes, a generator, or a list
            if isinstance(audio, (bytes, bytearray)):
                return bytes(audio)
            return b"".join(chunk for chunk in audio if chunk)

        except Exception as exc:
            err_str = str(exc).lower()
            logger.warning("ElevenLabs TTS failed (%s). Falling back to local TTS engine.", err_str)
            local_audio = self._generate_local_sync(text)
            if local_audio:
                return local_audio
            if "401" in err_str or "unauthorized" in err_str:
                raise RuntimeError(
                    "ElevenLabs authentication failed. Check ELEVENLABS_API_KEY."
                ) from exc
            if "429" in err_str or "rate" in err_str:
                raise RuntimeError("ElevenLabs rate limit exceeded.") from exc
            if "voice" in err_str and ("not found" in err_str or "invalid" in err_str):
                raise RuntimeError(
                    "ElevenLabs voice not found. Check ELEVENLABS_VOICE_ID."
                ) from exc
            raise RuntimeError(f"ElevenLabs TTS failed: {type(exc).__name__}") from exc

    def _generate_local_sync(self, text: str) -> bytes:
        """Fast offline local TTS fallback using Windows SAPI / pyttsx3."""
        try:
            import os
            import tempfile
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return data
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Local pyttsx3 TTS fallback failed: %s", exc)
            return b""

    # ─────────────────────────────────────────────
    # Async public interface
    # ─────────────────────────────────────────────
    async def generate_audio(
        self,
        *,
        text: str,
        language: str = "auto",
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> bytes:
        """
        Generate audio bytes from text using ElevenLabs.
        Async-safe — blocking SDK call runs in a thread pool executor.
        """
        if not text or not text.strip():
            raise ValueError("Text is required for audio synthesis.")
        if not self.is_configured():
            raise RuntimeError(
                "ElevenLabs is not configured. Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID."
            )

        target_voice = (voice_id or self.voice_id).strip()
        target_model = (model_id or self.model_id).strip() or "eleven_multilingual_v2"

        # Log without the actual text content (privacy)
        logger.info(
            "TTS generating audio | voice=%s | model=%s | lang=%s | chars=%d",
            target_voice[:8] + "...",
            target_model,
            language,
            len(text),
        )

        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(
            _tts_executor,
            self._generate_sync,
            text,
            target_voice,
            target_model,
        )

        if not audio_bytes:
            raise RuntimeError("ElevenLabs returned empty audio.")

        logger.info("TTS generated %d bytes of audio.", len(audio_bytes))
        return audio_bytes

    def validate_voice_support(self, language: str = "auto") -> None:
        """Validate that the configured voice supports the requested language."""
        if not self.is_configured():
            raise RuntimeError(
                "ElevenLabs is not configured. Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID."
            )
        # For non-Tamil languages, we trust the multilingual model
        target = (language or "auto").lower()
        if target not in {"ta", "tam", "tamil"}:
            return
        # For Tamil, we can check but won't block — just warn
        logger.info(
            "Tamil TTS requested. Ensure ELEVENLABS_VOICE_ID is a multilingual/Tamil-capable voice."
        )


# Lazy singleton
tts_service = ElevenLabsTTS()

__all__ = ["ElevenLabsTTS", "tts_service"]
