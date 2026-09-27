"""Validated schemas for user-owned personal memory records."""

import json
import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MemoryCategory = Literal[
    "preference", "learning", "goal", "project", "context", "other"
]
MEMORY_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class MemoryBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    memory_key: str = Field(min_length=1, max_length=120)
    value: dict[str, Any]
    is_sensitive: bool = False
    source: str | None = Field(default=None, max_length=100)

    @field_validator("memory_key")
    @classmethod
    def normalize_memory_key(cls, value: str) -> str:
        key = value.strip().lower()
        if not MEMORY_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                "Memory keys may use lowercase letters, numbers, dots, dashes, and underscores."
            )
        return key

    @field_validator("value")
    @classmethod
    def validate_memory_value(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("Memory value cannot be empty.")
        try:
            serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("Memory value must contain valid JSON data.")
        if len(serialized.encode("utf-8")) > 16_000:
            raise ValueError("Memory value must be 16 KB or smaller.")
        return value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str | None) -> str | None:
        return value.strip() if value else None


class MemoryCreate(MemoryBase):
    pass


class MemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory | None = None
    memory_key: str | None = Field(default=None, min_length=1, max_length=120)
    value: dict[str, Any] | None = None
    is_sensitive: bool | None = None
    source: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def require_change(self) -> "MemoryUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")
        return self

    @field_validator("memory_key")
    @classmethod
    def normalize_memory_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return MemoryBase.normalize_memory_key(value)

    @field_validator("value")
    @classmethod
    def validate_memory_value(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return MemoryBase.validate_memory_value(value)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str | None) -> str | None:
        return value.strip() if value else None


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    category: MemoryCategory
    memory_key: str
    value: dict[str, Any]
    is_sensitive: bool
    source: str | None
    created_at: datetime
    updated_at: datetime


class ClearMemoryResponse(BaseModel):
    deleted_count: int
