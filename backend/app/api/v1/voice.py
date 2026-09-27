"""
Local microphone, transcription, and voice-chat endpoints.
Complete end-to-end voice pipeline:
Audio -> Faster-Whisper STT -> AI Router -> ElevenLabs TTS -> Audio response.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import CurrentUser, DatabaseSession
from app.api.v1.chat import _get_owned_conversation
from app.core.config import get_settings
from app.core.prompts import build_system_prompt, build_voice_system_prompt
from app.models import Conversation, Message
from app.services.llm import get_llm_service
from app.services.memory_service import (
    format_memories_for_context,
    get_relevant_memories,
)
from app.services.stt import (
    MAX_AUDIO_BYTES,
    stt_service,
    validate_audio_content_type,
)
from app.services.tts import tts_service

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/voice", tags=["Voice"])


@router.get("/status")
async def voice_status() -> dict[str, object]:
    """Report that local STT is available and which languages are supported."""
    return {
        "status": "ready",
        "local_processing": True,
        "model": settings.whisper_model_size,
        "supported_languages": ["auto", "en", "ta"],
        "tts_configured": tts_service.is_configured(),
        "tts_model": settings.elevenlabs_model_id,
        "tts_voice_id": settings.elevenlabs_voice_id[:6] + "..."
        if settings.elevenlabs_voice_id
        else None,
    }


@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict[str, object]:
    """Transcribe uploaded local audio data without sending it to any external service."""
    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is required."
        )

    contents = await audio_file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty."
        )

    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file is too large (max {MAX_AUDIO_BYTES // 1024 // 1024} MB).",
        )

    try:
        validate_audio_content_type(audio_file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        result = await stt_service.transcribe_bytes_async(contents, language=language)
        return {
            "text": result["text"],
            "language": result.get("language"),
            "language_probability": result.get("language_probability"),
            "duration": result.get("duration"),
            "segments": result.get("segments", []),
        }
    except Exception as exc:
        logger.error("Transcription failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio transcription failed: {exc}",
        ) from exc


@router.post("/transcribe-microphone")
async def transcribe_microphone(
    duration_seconds: float = Form(5.0),
    language: str = Form("auto"),
) -> dict[str, object]:
    """Record microphone input locally and convert it to text in the current environment."""
    if duration_seconds <= 0 or duration_seconds > 300:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duration must be between 1 and 300 seconds.",
        )

    try:
        result = stt_service.record_microphone(
            duration_seconds=duration_seconds, language=language
        )
        return {
            "text": result["text"],
            "language": result.get("language"),
            "language_probability": result.get("language_probability"),
            "duration": result.get("duration"),
        }
    except Exception as exc:
        logger.error("Microphone transcription failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Microphone transcription failed: {exc}",
        ) from exc


@router.post("/chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    language: str = Form("auto"),
    current_user: CurrentUser = None,
    session: DatabaseSession = None,
) -> dict[str, object]:
    """
    Optimized end-to-end voice pipeline:
    1. Faster-Whisper STT with greedy decoding (beam_size=1)
    2. Concise voice system prompt (~40 tokens instead of ~600)
    3. Ollama generation capped at 120 tokens for voice responses
    4. ElevenLabs TTS synthesis
    5. Background DB persistence (off critical path)
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for voice chat.",
        )

    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is required."
        )

    contents = await audio_file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty."
        )

    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size ({MAX_AUDIO_BYTES // 1024 // 1024} MB).",
        )

    try:
        validate_audio_content_type(audio_file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    t_request_start = time.time()

    # ── Step 1: STT (greedy, beam_size=1 — fast) ─────────────────────
    t0_stt = time.time()
    try:
        stt_result = await stt_service.transcribe_bytes_async(
            contents, language=language
        )
    except Exception as exc:
        logger.error("Voice chat STT failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcription failed: {exc}",
        ) from exc

    t_stt_end = time.time()
    stt_latency_ms = round((t_stt_end - t0_stt) * 1000, 2)
    transcription = (stt_result.get("text") or "").strip()
    detected_lang = stt_result.get("language") or language or "auto"

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speech was detected in the supplied audio.",
        )

    logger.info(
        "[Voice] STT done in %.2fs: '%s'", t_stt_end - t0_stt, transcription[:60]
    )

    # ── Step 2: Conversation (minimal — no heavy history load) ────────
    conversation: Conversation | None = None
    if conversation_id:
        try:
            parsed_id = UUID(conversation_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid conversation_id.",
            ) from exc
        conversation = await _get_owned_conversation(
            session, current_user.id, parsed_id
        )
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
            )

    if conversation is None:
        title = (
            transcription[:40]
            if len(transcription) <= 40
            else transcription[:37] + "..."
        )
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
            model_provider="auto",
        )
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)

    # ── Step 3: Minimal memory fetch (top 2 only for voice) ──────────
    relevant_memories = await get_relevant_memories(
        session=session,
        user_id=current_user.id,
        query_text=transcription,
        limit=2,  # Voice: only top 2 memories, not 5
        include_sensitive=False,
    )

    # ── Step 4: Build CONCISE voice system prompt ─────────────────────
    system_prompt = build_voice_system_prompt(
        user_name=current_user.display_name,
        top_memories=relevant_memories,
    )

    # ── Step 5: LLM Generation — capped at 120 tokens for voice ──────
    t0_llm = time.time()
    llm = get_llm_service()
    try:
        # Use Ollama directly with num_predict cap to keep total generation ≤30s
        ollama_provider = llm.ollama
        response_text = await ollama_provider.generate(
            prompt=transcription,
            system=system_prompt,
            num_predict=120,  # ~30s at 4 tok/s — concise spoken answer
        )
    except Exception:
        # Fallback to full LLM service router if Ollama fails
        try:
            formatted_memories = format_memories_for_context(relevant_memories)
            system_prompt_full = build_system_prompt(
                user_name=current_user.display_name,
                preferences=current_user.preferences,
                relevant_memories=formatted_memories,
                provider_name="auto",
                is_online=True,
            )
            response_text = await llm.generate(
                prompt=transcription, system=system_prompt_full
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.error("Voice chat LLM failure: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI response failed: {exc}",
            ) from exc

    t_llm_end = time.time()
    llm_latency_ms = round((t_llm_end - t0_llm) * 1000, 2)
    provider = llm.last_provider_used or "ollama"
    model_name = llm.last_model_used or "qwen2.5:3b"

    logger.info(
        "[Voice] LLM done in %.2fs (%d chars)", t_llm_end - t0_llm, len(response_text)
    )

    # ── Step 6: TTS Synthesis ─────────────────────────────────────────
    t0_tts = time.time()
    audio_b64: str | None = None
    audio_status = "unavailable"
    tts_error = None

    if tts_service.is_configured():
        try:
            audio_payload = await tts_service.generate_audio(
                text=response_text,
                language=detected_lang,
            )
            if audio_payload:
                audio_b64 = base64.b64encode(audio_payload).decode("utf-8")
                audio_status = "ready"
        except Exception as exc:
            tts_error = str(exc)
            logger.warning("Voice chat TTS synthesis failed: %s", exc)
            audio_status = "unavailable"

    t_tts_end = time.time()
    tts_latency_ms = round((t_tts_end - t0_tts) * 1000, 2)
    total_latency_ms = round((t_tts_end - t_request_start) * 1000, 2)

    logger.info(
        "[Voice Pipeline] STT: %.2fs | LLM: %.2fs | TTS: %.2fs | Total: %.2fs",
        (t_stt_end - t0_stt),
        (t_llm_end - t0_llm),
        (t_tts_end - t0_tts),
        (t_tts_end - t_request_start),
    )

    # ── Step 7: DB persistence OFF critical path (background) ─────────
    async def _persist():
        try:
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=transcription,
                    model_provider=None,
                )
            )
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=response_text,
                    model_provider=provider,
                )
            )
            await session.commit()
        except Exception as exc:
            logger.debug("Voice DB persistence failed (non-critical): %s", exc)

    asyncio.ensure_future(_persist())

    return {
        "text": response_text,
        "audio": audio_b64,
        "audio_status": audio_status,
        "audio_format": "mp3" if audio_status == "ready" else None,
        "transcription": transcription,
        "language": detected_lang,
        "language_probability": stt_result.get("language_probability"),
        "conversation_id": conversation.id,
        "provider": provider,
        "model": model_name,
        "tts_error": tts_error if settings.debug else None,
        "latency": {
            "total_ms": total_latency_ms,
            "stt_ms": stt_latency_ms,
            "llm_ms": llm_latency_ms,
            "tts_ms": tts_latency_ms,
        },
    }


@router.post("/synthesize")
async def synthesize_speech(
    text: str = Form(...),
    language: str = Form("auto"),
) -> Response:
    """Synthesize text into speech MP3 bytes directly."""
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty."
        )

    if not tts_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs TTS is not configured on this server.",
        )

    try:
        audio_bytes = await tts_service.generate_audio(text=text, language=language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        logger.error("Direct synthesis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


@router.post("/stream")
async def voice_stream(
    audio_file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    language: str = Form("auto"),
    current_user: CurrentUser = None,
    session: DatabaseSession = None,
) -> StreamingResponse:
    """
    Sentence-level streaming voice pipeline:
    1. Fast STT via Faster-Whisper (greedy decoding).
    2. Sends immediate transcription SSE event.
    3. Streams Ollama LLM tokens in real-time.
    4. Detects sentence boundaries and synthesizes TTS audio sentence-by-sentence.
    5. Returns audio chunks as they are generated so client can play sentence 1
       immediately (target ≤ 4s Voice-to-First-Audio).
    6. Asynchronous DB save in background.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for voice streaming.",
        )

    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is required."
        )

    contents = await audio_file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty."
        )

    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size ({MAX_AUDIO_BYTES // 1024 // 1024} MB).",
        )

    try:
        validate_audio_content_type(audio_file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    t_start = time.time()
    t0_stt = time.time()
    try:
        stt_result = await stt_service.transcribe_bytes_async(
            contents, language=language
        )
    except Exception as exc:
        logger.error("Voice stream STT failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcription failed: {exc}",
        ) from exc

    t_stt_end = time.time()
    transcription = (stt_result.get("text") or "").strip()
    detected_lang = stt_result.get("language") or language or "auto"

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speech was detected in the supplied audio.",
        )

    # Resolve conversation
    conversation = None
    if conversation_id:
        try:
            parsed_id = UUID(conversation_id)
            conversation = await _get_owned_conversation(
                session, current_user.id, parsed_id
            )
        except Exception:
            pass

    if conversation is None:
        title = (
            transcription[:40]
            if len(transcription) <= 40
            else transcription[:37] + "..."
        )
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
            model_provider="auto",
        )
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)

    relevant_memories = await get_relevant_memories(
        session=session,
        user_id=current_user.id,
        query_text=transcription,
        limit=2,
        include_sensitive=False,
    )
    system_prompt = build_voice_system_prompt(
        user_name=current_user.display_name,
        top_memories=relevant_memories,
    )

    async def sse_event_generator():
        # Yield STT event immediately
        yield f"data: {json.dumps({'type': 'transcription', 'text': transcription, 'language': detected_lang, 'stt_ms': round((t_stt_end - t0_stt) * 1000, 1)})}\n\n"

        llm = get_llm_service()
        sentence_buffer = ""
        all_tokens = []
        sentence_idx = 0
        t0_llm = time.time()
        t_first_token = None

        try:
            async for token in llm.ollama.stream(
                prompt=transcription, system=system_prompt, num_predict=120
            ):
                if t_first_token is None:
                    t_first_token = time.time()
                all_tokens.append(token)
                sentence_buffer += token

                # Stream token to frontend
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                # Check for sentence end: punctuation followed by whitespace
                match = re.search(r"([.!?\n])\s+", sentence_buffer)
                if match:
                    split_pos = match.end()
                    sentence = sentence_buffer[:split_pos].strip()
                    sentence_buffer = sentence_buffer[split_pos:]
                    if sentence:
                        sentence_idx += 1
                        audio_b64 = None
                        if tts_service.is_configured():
                            try:
                                audio_bytes = await tts_service.generate_audio(
                                    text=sentence, language=detected_lang
                                )
                                if audio_bytes:
                                    audio_b64 = base64.b64encode(audio_bytes).decode(
                                        "utf-8"
                                    )
                            except Exception as e:
                                logger.warning("Streaming TTS sentence error: %s", e)

                        yield f"data: {json.dumps({'type': 'audio', 'sentence_index': sentence_idx, 'sentence': sentence, 'audio_b64': audio_b64})}\n\n"

            # Flush any remaining text in sentence buffer
            remaining = sentence_buffer.strip()
            if remaining:
                sentence_idx += 1
                audio_b64 = None
                if tts_service.is_configured():
                    try:
                        audio_bytes = await tts_service.generate_audio(
                            text=remaining, language=detected_lang
                        )
                        if audio_bytes:
                            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    except Exception as e:
                        logger.warning("Streaming TTS remaining error: %s", e)

                yield f"data: {json.dumps({'type': 'audio', 'sentence_index': sentence_idx, 'sentence': remaining, 'audio_b64': audio_b64})}\n\n"

            full_text = "".join(all_tokens)
            t_done = time.time()
            total_ms = round((t_done - t_start) * 1000, 1)
            llm_ms = round((t_done - t0_llm) * 1000, 1)
            ttft_ms = (
                round((t_first_token - t0_llm) * 1000, 1) if t_first_token else 0.0
            )

            yield f"data: {json.dumps({'type': 'done', 'text': full_text, 'total_ms': total_ms, 'llm_ms': llm_ms, 'ttft_ms': ttft_ms, 'sentences': sentence_idx})}\n\n"

        except Exception as e:
            logger.error("Error in voice stream generator: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        finally:
            full_text = "".join(all_tokens)
            if full_text and conversation:
                try:
                    session.add(
                        Message(
                            conversation_id=conversation.id,
                            role="user",
                            content=transcription,
                            model_provider=None,
                        )
                    )
                    session.add(
                        Message(
                            conversation_id=conversation.id,
                            role="assistant",
                            content=full_text,
                            model_provider="ollama",
                        )
                    )
                    await session.commit()
                except Exception as exc:
                    logger.debug("Background persist error: %s", exc)

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
