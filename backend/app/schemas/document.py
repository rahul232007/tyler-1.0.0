"""Document upload and retrieval schemas."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    filename: str
    content_type: str | None
    file_size_bytes: int | None
    created_at: datetime


class DocumentDetailRead(DocumentRead):
    extracted_text: str | None


class DocumentDeleteResponse(BaseModel):
    document_id: UUID
    deleted: bool = True
