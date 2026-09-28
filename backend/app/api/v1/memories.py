"""Authenticated personal-memory CRUD, semantic search, and intelligence API."""

import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from app.api.dependencies import CurrentUser, DatabaseSession
from app.models import PersonalMemory
from app.schemas.memory import (
    ClearMemoryResponse,
    MemoryCategory,
    MemoryCreate,
    MemoryRead,
    MemoryUpdate,
)
from app.services.llm import get_llm_service
from app.services.memory_service import (
    compute_embedding,
    decode_memory_value,
    encode_memory_value,
    extract_memories_from_conversation,
    format_memories_for_context,
    get_relevant_memories,
    store_extracted_memories,
)

router = APIRouter(prefix="/memories", tags=["Personal Memory"])


class MemoryExtractRequest(BaseModel):
    conversation_text: str = Field(min_length=1)


class MemoryExtractResponse(BaseModel):
    extracted_count: int
    memories: list[dict]


async def get_owned_memory(
    memory_id: UUID, user_id: UUID, session: DatabaseSession
) -> PersonalMemory:
    """Fetch one memory only when it belongs to the authenticated user."""
    memory = await session.scalar(
        select(PersonalMemory).where(
            PersonalMemory.id == memory_id,
            PersonalMemory.user_id == user_id,
        )
    )
    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found."
        )
    return memory


def _memory_response(memory: PersonalMemory) -> MemoryRead:
    """Serialize an owned memory without exposing sensitive plaintext or ciphertext."""
    result = MemoryRead.model_validate(memory)
    if memory.is_sensitive:
        result = result.model_copy(update={"value": {"redacted": True}})
    return result


async def ensure_unique_memory_key(
    *,
    user_id: UUID,
    category: str,
    memory_key: str,
    session: DatabaseSession,
    exclude_id: UUID | None = None,
) -> None:
    """Prevent duplicate memory identities for an individual user."""
    query = select(PersonalMemory.id).where(
        PersonalMemory.user_id == user_id,
        PersonalMemory.category == category,
        PersonalMemory.memory_key == memory_key,
    )
    if exclude_id is not None:
        query = query.where(PersonalMemory.id != exclude_id)
    if await session.scalar(query) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A memory with this category and key already exists.",
        )


@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate, current_user: CurrentUser, session: DatabaseSession
) -> PersonalMemory:
    """Create one memory owned by the authenticated user with encryption & embeddings."""
    await ensure_unique_memory_key(
        user_id=current_user.id,
        category=payload.category,
        memory_key=payload.memory_key,
        session=session,
    )

    stored_value, encrypted_val = encode_memory_value(
        payload.value, payload.is_sensitive
    )

    embedding = None
    if not payload.is_sensitive:
        mem_text = f"{payload.memory_key} {json.dumps(payload.value)}"
        embedding = {"vector": compute_embedding(mem_text), "model": "tfidf"}

    memory = PersonalMemory(
        user_id=current_user.id,
        category=payload.category,
        memory_key=payload.memory_key,
        value=stored_value,
        is_sensitive=payload.is_sensitive,
        source=payload.source,
        importance=0.6,
        encrypted_value=encrypted_val,
        embedding=embedding,
    )
    session.add(memory)
    await session.flush()
    await session.refresh(memory)
    return _memory_response(memory)


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    current_user: CurrentUser,
    session: DatabaseSession,
    category: MemoryCategory | None = None,
    include_sensitive: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[PersonalMemory]:
    """List the caller's memories; sensitive values stay hidden by default."""
    query = select(PersonalMemory).where(PersonalMemory.user_id == current_user.id)
    if category is not None:
        query = query.where(PersonalMemory.category == category)
    if not include_sensitive:
        query = query.where(PersonalMemory.is_sensitive.is_(False))
    query = query.order_by(PersonalMemory.updated_at.desc()).offset(offset).limit(limit)
    result = list(await session.scalars(query))
    return [_memory_response(memory) for memory in result]


@router.get("/search")
async def search_memories(
    current_user: CurrentUser,
    session: DatabaseSession,
    q: str = Query(..., min_length=1, description="Semantic search query"),
    limit: int = Query(default=5, ge=1, le=20),
    include_sensitive: bool = False,
) -> dict[str, object]:
    """Semantic/vector similarity search over user's personal memories."""
    memories = await get_relevant_memories(
        session=session,
        user_id=current_user.id,
        query_text=q,
        limit=limit,
        include_sensitive=include_sensitive,
    )
    formatted = format_memories_for_context(memories)
    return {
        "query": q,
        "count": len(formatted),
        "results": formatted,
    }


@router.post("/extract", response_model=MemoryExtractResponse)
async def extract_memories(
    payload: MemoryExtractRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> MemoryExtractResponse:
    """Trigger AI memory extraction from conversation text and store validated memories."""
    llm = get_llm_service()
    candidates = await extract_memories_from_conversation(
        payload.conversation_text, llm
    )
    stored_count = 0
    if candidates:
        stored_count = await store_extracted_memories(
            session, current_user.id, candidates
        )
    response_memories = [
        {
            **candidate,
            "value": {"redacted": True}
            if candidate.get("is_sensitive")
            else candidate["value"],
        }
        for candidate in candidates
    ]
    return MemoryExtractResponse(
        extracted_count=stored_count, memories=response_memories
    )


@router.get("/{memory_id}", response_model=MemoryRead)
async def read_memory(
    memory_id: UUID, current_user: CurrentUser, session: DatabaseSession
) -> PersonalMemory:
    """Read one owned memory, including a sensitive one when explicitly requested."""
    memory = await get_owned_memory(memory_id, current_user.id, session)
    return _memory_response(memory)


@router.patch("/{memory_id}", response_model=MemoryRead)
async def update_memory(
    memory_id: UUID,
    payload: MemoryUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> PersonalMemory:
    """Update one user-owned memory after checking for key conflicts."""
    memory = await get_owned_memory(memory_id, current_user.id, session)
    updates = payload.model_dump(exclude_unset=True)
    new_category = updates.get("category", memory.category)
    new_memory_key = updates.get("memory_key", memory.memory_key)
    if new_category != memory.category or new_memory_key != memory.memory_key:
        await ensure_unique_memory_key(
            user_id=current_user.id,
            category=new_category,
            memory_key=new_memory_key,
            session=session,
            exclude_id=memory.id,
        )

    current_value = (
        decode_memory_value(memory.value, memory.encrypted_value)
        if memory.is_sensitive
        else memory.value
    )
    new_value = updates.pop("value", current_value)
    new_is_sensitive = updates.pop("is_sensitive", memory.is_sensitive)
    for field_name, value in updates.items():
        setattr(memory, field_name, value)

    memory.is_sensitive = new_is_sensitive
    memory.value, memory.encrypted_value = encode_memory_value(
        new_value, new_is_sensitive
    )
    memory.embedding = None
    if not new_is_sensitive:
        mem_text = f"{memory.memory_key} {json.dumps(new_value)}"
        memory.embedding = {"vector": compute_embedding(mem_text), "model": "tfidf"}

    await session.flush()
    await session.refresh(memory)
    return _memory_response(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID, current_user: CurrentUser, session: DatabaseSession
) -> Response:
    """Delete one user-owned memory."""
    memory = await get_owned_memory(memory_id, current_user.id, session)
    await session.delete(memory)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/", response_model=ClearMemoryResponse)
async def clear_memories(
    current_user: CurrentUser,
    session: DatabaseSession,
    confirm: bool = Query(..., description="Must be true to clear every memory."),
) -> ClearMemoryResponse:
    """Clear all memories for the authenticated user after explicit confirmation."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm=true to clear all memories.",
        )
    result = await session.execute(
        delete(PersonalMemory).where(PersonalMemory.user_id == current_user.id)
    )
    return ClearMemoryResponse(deleted_count=result.rowcount or 0)
