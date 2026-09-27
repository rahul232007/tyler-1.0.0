"""
JARVIS - Document Management API Endpoints
Handles file upload, text extraction (PDF / TXT / MD), document retrieval,
and document context integration.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select

from app.api.dependencies import CurrentUser, DatabaseSession
from app.core.config import get_settings
from app.models import Document
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailRead,
    DocumentRead,
)
from app.services.document_service import extract_text_from_bytes, validate_document

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload", response_model=DocumentDetailRead, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
    session: DatabaseSession = None,
) -> Document:
    """
    Upload a document (PDF, TXT, MD), validate safety & size limits,
    extract text content, and store it for JARVIS context.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="A file is required."
        )

    data = await file.read()
    try:
        validate_document(file.filename, file.content_type, len(data))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        extracted_text = extract_text_from_bytes(data, file.filename, file.content_type)
    except Exception as exc:
        logger.error(
            "Failed to extract text from document '%s': %s", file.filename, exc
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract readable text from document: {exc}",
        ) from exc

    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        content_type=file.content_type,
        extracted_text=extracted_text,
        file_size_bytes=len(data),
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


@router.get("/", response_model=list[DocumentRead])
async def list_documents(
    current_user: CurrentUser,
    session: DatabaseSession,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[Document]:
    """List uploaded documents owned by caller."""
    stmt = (
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.scalars(stmt)
    return list(result)


@router.get("/{document_id}", response_model=DocumentDetailRead)
async def get_document(
    document_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Document:
    """Get full document with extracted text."""
    doc = await session.scalar(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )
    return doc


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> DocumentDeleteResponse:
    """Delete an uploaded document."""
    doc = await session.scalar(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )
    await session.delete(doc)
    return DocumentDeleteResponse(document_id=document_id, deleted=True)
