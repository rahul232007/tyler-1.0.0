"""
JARVIS - Advanced Chat API Endpoint
Handles:
  - Text chat with intelligent AI router (Gemini / NVIDIA / Ollama)
  - Streaming and non-streaming responses
  - Context optimization (token/history limit, recent message priority, summarization)
  - Memory intelligence (semantic retrieval & injection into prompt)
  - Document context injection
  - Tool execution system (web search, calculator, datetime, memory, etc.)
  - Background memory extraction from user conversation
"""

from __future__ import annotations

import json
import logging
import re
import time
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.dependencies import CurrentUser, DatabaseSession
from app.core.config import get_settings
from app.core.prompts import build_system_prompt
from app.models import Conversation, Document, Message
from app.services.document_service import summarize_for_context
from app.services.llm import get_llm_service
from app.services.memory_service import (
    extract_memories_from_conversation,
    format_memories_for_context,
    get_relevant_memories,
    store_extracted_memories,
)
from app.tools.registry import execute_tool, get_tools_description
from app.tools.tools import get_session_tools

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/chat", tags=["Chat"])


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    stream: bool = False
    conversation_id: UUID | None = None
    document_id: UUID | None = None
    use_tools: bool = True


class ChatResponse(BaseModel):
    response: str
    provider_used: str = "unknown"
    conversation_id: UUID | None = None
    model_used: str | None = None
    fallback_used: bool = False
    tools_used: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Database Helpers
# ─────────────────────────────────────────────
async def _get_owned_conversation(
    session: DatabaseSession,
    user_id: UUID,
    conversation_id: UUID | None,
) -> Conversation | None:
    if conversation_id is None:
        return None
    return await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )


async def _build_optimized_context(
    session: DatabaseSession,
    conversation: Conversation | None,
    new_message: str,
    max_history: int,
) -> str:
    """
    Select recent conversation history up to max_history.
    If older messages exist, summarize them to prevent token explosion.
    """
    if conversation is None:
        return f"User: {new_message}"

    history = list(
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
        )
    )
    if not history:
        return f"User: {new_message}"

    # Context window optimization: take last N messages
    if len(history) > max_history:
        older = history[:-max_history]
        recent = history[-max_history:]
        summary_snippets = [f"{m.role}: {m.content[:60]}..." for m in older[-4:]]
        summary_note = f"[Previous conversation context: {'; '.join(summary_snippets)}]"
        lines = [summary_note] + [f"{m.role.capitalize()}: {m.content}" for m in recent]
    else:
        lines = [f"{m.role.capitalize()}: {m.content}" for m in history]

    lines.append(f"User: {new_message}")
    return "\n".join(lines)


async def _build_chat_prompt(
    session: DatabaseSession,
    conversation: Conversation | None,
    new_message: str,
) -> str:
    """Build chat prompt with conversation context."""
    return await _build_optimized_context(
        session, conversation, new_message, settings.max_conversation_history
    )


# ─────────────────────────────────────────────
# Tool Execution Helper
# ─────────────────────────────────────────────
async def _handle_tool_calling(
    raw_response: str,
    prompt_text: str,
    system_prompt: str,
    llm,
) -> tuple[str, list[str]]:
    """
    Check if the LLM output requested a tool call (e.g. {"tool": "...", "args": {...}}).
    If so, execute the tool, feed the output back to the LLM, and return final answer.
    """
    tools_used = []
    current_response = raw_response

    # Look for tool call JSON patterns
    tool_match = re.search(
        r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}',
        current_response,
    )
    if not tool_match:
        # Also check for markdown code block json
        tool_match = re.search(
            r'```(?:json)?\s*\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}\s*```',
            current_response,
        )

    if tool_match:
        tool_name = tool_match.group(1).strip()
        try:
            tool_args = json.loads(tool_match.group(2))
        except Exception:
            tool_args = {}

        logger.info(
            "LLM triggered tool: %s with args: %s", tool_name, list(tool_args.keys())
        )
        result = await execute_tool(tool_name, **tool_args)
        tools_used.append(tool_name)

        tool_output_str = json.dumps(
            result.output or {"error": result.error}, default=str
        )
        followup_prompt = (
            f"{prompt_text}\n\n"
            f"[Tool Result for '{tool_name}']:\n{tool_output_str}\n\n"
            f"Now provide the final natural language answer to the user based on this tool result."
        )
        try:
            current_response = await llm.generate(
                prompt=followup_prompt, system=system_prompt
            )
        except Exception as exc:
            logger.warning("Followup generation after tool failed: %s", exc)
            current_response = (
                f"I retrieved the information ({tool_name}): {tool_output_str}"
            )

    return current_response, tools_used


# ─────────────────────────────────────────────
# Chat Endpoints
# ─────────────────────────────────────────────
@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
):
    """
    Send a message to JARVIS with:
    - Context optimization & conversation persistence
    - Semantic memory retrieval & prompt injection
    - Optional document context integration
    - Tool system reasoning & execution
    - Automatic memory extraction
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty"
        )

    llm = get_llm_service()

    # Step 1: Conversation lookup or creation
    conversation = await _get_owned_conversation(
        session, current_user.id, request.conversation_id
    )
    if request.conversation_id is not None and conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
        )

    if conversation is None:
        title = user_msg[:35] if len(user_msg) <= 35 else user_msg[:32] + "..."
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
            model_provider="auto",
        )
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)

    # Step 2: Document context injection
    doc_context = ""
    if request.document_id:
        doc = await session.scalar(
            select(Document).where(
                Document.id == request.document_id, Document.user_id == current_user.id
            )
        )
        if doc and doc.extracted_text:
            doc_context = f"\n\n[DOCUMENT CONTEXT: {doc.filename}]\n{summarize_for_context(doc.extracted_text, 3000)}\n"

    # Step 3: Semantic memory retrieval
    relevant_mems = await get_relevant_memories(
        session=session,
        user_id=current_user.id,
        query_text=user_msg,
        limit=5,
        include_sensitive=False,
    )
    formatted_memories = format_memories_for_context(relevant_mems)

    # Step 4: Register session tools
    get_session_tools(session, current_user.id)

    # Step 5: System prompt construction
    tools_desc = get_tools_description() if request.use_tools else ""
    system_prompt = build_system_prompt(
        user_name=current_user.display_name,
        preferences=current_user.preferences,
        relevant_memories=formatted_memories,
        provider_name="auto",
        is_online=True,
    )
    if doc_context:
        system_prompt += doc_context
    if tools_desc:
        system_prompt += f"\n\n{tools_desc}"

    # Step 6: Build optimized chat history prompt
    prompt_text = await _build_optimized_context(
        session=session,
        conversation=conversation,
        new_message=user_msg,
        max_history=settings.max_conversation_history,
    )

    # Persist user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=user_msg,
        model_provider=None,
    )
    session.add(user_message)
    await session.flush()

    t_fastapi_receive = time.time()
    try:
        # Streaming response
        if request.stream:

            async def _persist_and_extract(
                assistant_text: str,
                provider: str,
                conversation_id,
                user_msg_text: str,
            ):
                """Background task: DB commit + memory extraction — NOT on the streaming path."""
                import asyncio
                try:
                    from app.db.base import async_session_factory
                    async with async_session_factory() as bg_session:
                        bg_msg = Message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=assistant_text,
                            model_provider=provider,
                        )
                        bg_session.add(bg_msg)
                        await bg_session.commit()
                        logger.info("[TIMING_STAGE] BG DB Commit done for conv %s", conversation_id)
                        try:
                            from app.services.llm import get_llm_service as _get_llm
                            from app.services.memory_service import (
                                extract_memories_from_conversation,
                                store_extracted_memories,
                            )
                            snippet = f"User: {user_msg_text}\nAssistant: {assistant_text[:300]}"
                            llm2 = _get_llm()
                            extracted = await extract_memories_from_conversation(snippet, llm2)
                            if extracted:
                                await store_extracted_memories(bg_session, current_user.id, extracted)
                        except Exception as mem_exc:
                            logger.debug("BG memory extraction skipped: %s", mem_exc)
                except Exception as exc:
                    logger.debug("BG DB persist failed (non-critical): %s", exc)

            async def token_generator():
                tokens = []
                t_first_token = None
                t_ollama_start = time.time()
                logger.info(
                    "[TIMING_STAGE] FastAPI Receive: %.4f | Ollama Start: %.4f",
                    t_fastapi_receive,
                    t_ollama_start,
                )
                try:
                    async for token in llm.stream(
                        prompt=prompt_text, system=system_prompt
                    ):
                        if t_first_token is None:
                            t_first_token = time.time()
                            ttft = round(t_first_token - t_fastapi_receive, 3)
                            logger.info(
                                "[TIMING_STAGE] First Token: %.4f | TTFT: %.3fs",
                                t_first_token,
                                ttft,
                            )
                        tokens.append(token)
                        yield token
                finally:
                    t_final_token = time.time()
                    logger.info("[TIMING_STAGE] Final Token: %.4f", t_final_token)

                    assistant_text = "".join(tokens)
                    if assistant_text and conversation:
                        provider = llm.last_provider_used or "ollama"

                        # Timing log (file-write only, non-blocking)
                        try:
                            import json, os as _os
                            timing_dir = r"d:\project\jarvis model 1\scratch"
                            _os.makedirs(timing_dir, exist_ok=True)
                            timing_file = _os.path.join(timing_dir, "server_timings.jsonl")
                            record = {
                                "fastapi_receive": t_fastapi_receive,
                                "ollama_start": t_ollama_start,
                                "first_token": t_first_token,
                                "final_token": t_final_token,
                            }
                            with open(timing_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record) + "\n")
                        except Exception:
                            pass

                        # ✅ Fire-and-forget DB + memory — does NOT block streaming
                        asyncio.create_task(_persist_and_extract(
                            assistant_text=assistant_text,
                            provider=provider,
                            conversation_id=conversation.id,
                            user_msg_text=user_msg,
                        ))

            headers = {
                "X-Provider-Used": llm.last_provider_used or "ollama",
                "Access-Control-Expose-Headers": "X-Provider-Used",
            }
            return StreamingResponse(
                token_generator(), media_type="text/plain", headers=headers
            )

        # Non-streaming response
        raw_response = await llm.generate(prompt=prompt_text, system=system_prompt)
        final_response, tools_used = await _handle_tool_calling(
            raw_response, prompt_text, system_prompt, llm
        )

        provider = llm.last_provider_used or "unknown"
        model_used = llm.last_model_used
        fallback_used = bool(
            llm.last_fallback_info and llm.last_fallback_info.get("fallback_used")
        )

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_response,
            model_provider=provider,
        )
        session.add(assistant_message)
        await session.flush()
        await session.refresh(conversation)

        # Automatic memory extraction from recent interaction
        try:
            interaction_snippet = f"User: {user_msg}\nAssistant: {final_response[:300]}"
            extracted = await extract_memories_from_conversation(
                interaction_snippet, llm
            )
            if extracted:
                await store_extracted_memories(session, current_user.id, extracted)
        except Exception as exc:
            logger.debug("Background memory extraction skipped: %s", exc)

        return ChatResponse(
            response=final_response,
            provider_used=provider,
            conversation_id=conversation.id,
            model_used=model_used,
            fallback_used=fallback_used,
            tools_used=tools_used,
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        logger.error("Chat generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM error: {e!s}",
        )


@router.get("/status")
async def llm_status():
    """Check which AI providers are currently available."""
    llm = get_llm_service()
    return await llm.status()
