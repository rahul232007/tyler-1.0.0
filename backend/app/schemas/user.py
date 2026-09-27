"""User profile schemas for personalization."""
from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

ResponseStyle = Literal["concise", "detailed", "bullet_points", "friendly", "formal"]
LanguagePreference = Literal["tanglish", "tamil", "english", "auto"]


class UserPreferences(BaseModel):
    """Structured user preferences stored in users.preferences JSONB."""
    model_config = ConfigDict(extra="allow")

    response_style: ResponseStyle | None = None
    language: LanguagePreference | None = None
    provider_preference: Literal["gemini", "nvidia", "ollama", "auto"] | None = None
    voice_enabled: bool | None = None
    memory_enabled: bool | None = None
    web_search_enabled: bool | None = None


class UserUpdate(BaseModel):
    """Update user display name and/or preferences."""
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    preferences: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_change(self) -> "UserUpdate":
        if self.display_name is None and self.preferences is None:
            raise ValueError("Provide display_name or preferences to update.")
        return self


class UserPublicFull(BaseModel):
    """Extended user info including preferences."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    display_name: str | None
    is_active: bool
    preferences: dict[str, Any]
