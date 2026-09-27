"""Registration, login, logout, and current-user endpoints."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import CurrentUser, DatabaseSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.schemas.auth import (
    MessageResponse,
    TokenResponse,
    UserLogin,
    UserPublic,
    UserRegistration,
)
from app.schemas.user import UserPublicFull, UserUpdate

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED
)
async def register_user(payload: UserRegistration, session: DatabaseSession) -> User:
    """Register a user with a bcrypt password hash, never a plaintext password."""
    existing_user = await session.scalar(
        select(User).where(User.email == payload.email)
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=payload.email,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login_user(payload: UserLogin, session: DatabaseSession) -> TokenResponse:
    """Authenticate a user and issue a short-lived JWT access token."""
    user = await session.scalar(select(User).where(User.email == payload.email))
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    return TokenResponse(access_token=create_access_token(user.id, user.token_version))


@router.get("/me", response_model=UserPublicFull)
async def read_current_user(current_user: CurrentUser) -> User:
    """Return the user and preferences represented by the supplied access token."""
    return current_user


@router.patch("/me", response_model=UserPublicFull)
async def update_current_user(
    payload: UserUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> User:
    """Update display name or preferences for the authenticated user."""
    if payload.display_name is not None:
        current_user.display_name = (
            payload.display_name.strip() if payload.display_name else None
        )

    if payload.preferences is not None:
        # Merge updated preferences with existing preferences
        existing_prefs = dict(current_user.preferences or {})
        existing_prefs.update(payload.preferences)
        current_user.preferences = existing_prefs

    await session.flush()
    await session.refresh(current_user)
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout_user(current_user: CurrentUser) -> MessageResponse:
    """Invalidate all current access tokens for this user."""
    current_user.token_version += 1
    return MessageResponse(message="Logged out successfully.")
