"""Authenticated conversation management, message history, search, and export endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentUser, DatabaseSession
from app.models import Conversation, Message
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDeleteHistoryResponse,
    ConversationDetailRead,
    ConversationRead,
    ConversationUpdate,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


async def get_owned_conversation(
    conversation_id: UUID,
    user_id: UUID,
    session: DatabaseSession,
) -> Conversation:
    """Fetch a conversation only when it belongs to the authenticated user."""
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .options(selectinload(Conversation.messages))
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
        )
    return conversation


@router.post("/", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Conversation:
    """Create a new conversation owned by the authenticated user."""
    conversation = Conversation(
        user_id=current_user.id,
        title=payload.title or "New conversation",
        model_provider=payload.model_provider,
    )
    session.add(conversation)
    await session.flush()
    await session.refresh(conversation)
    return conversation


@router.get("/", response_model=list[ConversationRead])
async def list_conversations(
    current_user: CurrentUser,
    session: DatabaseSession,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[Conversation]:
    """List the caller's conversations ordered by most recent update."""
    result = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result)


@router.get("/search")
async def search_conversations(
    current_user: CurrentUser,
    session: DatabaseSession,
    q: str = Query(
        ..., min_length=1, description="Search term across conversations and messages"
    ),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, object]:
    """Search conversation titles and message content for the authenticated user."""
    search_term = f"%{q.strip()}%"

    # Search in conversation titles
    conv_matches = list(
        await session.scalars(
            select(Conversation)
            .where(
                Conversation.user_id == current_user.id,
                Conversation.title.ilike(search_term),
            )
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
    )

    # Search in message content
    msg_stmt = (
        select(Message, Conversation.title.label("conversation_title"))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Conversation.user_id == current_user.id,
            Message.content.ilike(search_term),
        )
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    msg_rows = (await session.execute(msg_stmt)).all()

    return {
        "query": q,
        "conversations": [
            {"id": str(c.id), "title": c.title, "updated_at": c.updated_at.isoformat()}
            for c in conv_matches
        ],
        "messages": [
            {
                "id": str(row[0].id),
                "conversation_id": str(row[0].conversation_id),
                "conversation_title": row[1],
                "role": row[0].role,
                "content": row[0].content,
                "created_at": row[0].created_at.isoformat(),
            }
            for row in msg_rows
        ],
    }


@router.get("/{conversation_id}", response_model=ConversationDetailRead)
async def get_conversation(
    conversation_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Conversation:
    """Fetch one conversation and its message history for the authenticated user."""
    return await get_owned_conversation(conversation_id, current_user.id, session)


@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> dict[str, object]:
    """Export a complete conversation transcript in JSON format."""
    conv = await get_owned_conversation(conversation_id, current_user.id, session)
    return {
        "id": str(conv.id),
        "title": conv.title,
        "model_provider": conv.model_provider,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "model_provider": m.model_provider,
                "created_at": m.created_at.isoformat(),
            }
            for m in conv.messages
        ],
    }


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Conversation:
    """Rename or update a conversation owned by the authenticated user."""
    conversation = await get_owned_conversation(
        conversation_id, current_user.id, session
    )
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one field to update.",
        )
    for field_name, value in updates.items():
        setattr(conversation, field_name, value)
    await session.flush()
    await session.refresh(conversation)
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Response:
    """Delete a conversation and all nested messages for the authenticated user."""
    conversation = await get_owned_conversation(
        conversation_id, current_user.id, session
    )
    await session.delete(conversation)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/{conversation_id}/history", response_model=ConversationDeleteHistoryResponse
)
async def delete_conversation_history(
    conversation_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> ConversationDeleteHistoryResponse:
    """Delete all messages inside a conversation but keep the conversation itself."""
    conversation = await get_owned_conversation(
        conversation_id, current_user.id, session
    )
    result = await session.execute(
        delete(Message).where(Message.conversation_id == conversation.id)
    )
    return ConversationDeleteHistoryResponse(
        conversation_id=conversation.id,
        deleted_count=result.rowcount or 0,
    )
