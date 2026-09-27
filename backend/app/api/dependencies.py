"""Reusable FastAPI dependencies for authenticated endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db_session
from app.models import User

bearer_scheme = HTTPBearer(auto_error=False)
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


def authentication_error() -> HTTPException:
    """Return the same response for absent, invalid, and expired credentials."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: DatabaseSession,
) -> User:
    """Resolve a valid JWT to its active user and enforce token logout state."""
    if credentials is None:
        raise authentication_error()

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = UUID(payload["sub"])
        token_version = int(payload["tv"])
    except (JWTError, KeyError, TypeError, ValueError):
        raise authentication_error()

    user = await session.scalar(select(User).where(User.id == user_id))
    if user is None or not user.is_active or user.token_version != token_version:
        raise authentication_error()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
