"""Schemas for conversation and chat-message persistence."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    model_provider: str | None = Field(default=None, max_length=50)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    model_provider: str | None = Field(default=None, max_length=50)


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    model_provider: str | None
    created_at: datetime


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str | None
    model_provider: str | None
    created_at: datetime
    updated_at: datetime


class ConversationDetailRead(ConversationRead):
    messages: list[MessageRead] = Field(default_factory=list)


class ConversationDeleteHistoryResponse(BaseModel):
    conversation_id: UUID
    deleted_count: int
