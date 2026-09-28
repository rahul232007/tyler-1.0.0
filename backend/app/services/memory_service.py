"""
JARVIS - Memory Intelligence Service
Handles:
- Encryption/decryption of sensitive memories using Fernet
- Automatic memory extraction from conversation text
- Memory importance scoring
- Simple embedding-based semantic similarity search
- Memory injection into chat context
- Duplicate prevention

Privacy rules:
- Encrypted values are never logged
- Decrypted values are never exposed in logs
- is_sensitive=True memories are only included in context when explicitly requested
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import PersonalMemory

logger = logging.getLogger(__name__)
settings = get_settings()
ENCRYPTED_VALUE_MARKER = "__jarvis_encrypted_value__"


# ─────────────────────────────────────────────
# Encryption helpers (Fernet symmetric encryption)
# ─────────────────────────────────────────────


def _get_fernet():
    """Lazy-load Fernet with the configured memory encryption key."""
    try:
        import base64

        from cryptography.fernet import Fernet

        # Fernet key must be 32 url-safe base64-encoded bytes
        raw_key = settings.memory_encryption_key.encode("utf-8")
        # Derive a valid 32-byte key from the config value
        key_bytes = hashlib.sha256(raw_key).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        return Fernet(fernet_key)
    except ImportError as exc:
        raise RuntimeError(
            "cryptography package required for memory encryption."
        ) from exc


def encrypt_value(value: dict[str, Any]) -> str:
    """Encrypt a dict value to a Fernet token (URL-safe base64 string)."""
    f = _get_fernet()
    plaintext = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return f.encrypt(plaintext).decode("utf-8")


def decrypt_value(token: str) -> dict[str, Any]:
    """Decrypt a Fernet token back to a dict. Raises ValueError on failure."""
    try:
        f = _get_fernet()
        plaintext = f.decrypt(token.encode("utf-8"))
        return json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            "Memory decryption failed — invalid or corrupted token."
        ) from exc


def encode_memory_value(
    value: dict[str, Any], is_sensitive: bool
) -> tuple[dict[str, Any], str | None]:
    """Return database-safe value fields, encrypting sensitive values once."""
    if not is_sensitive:
        return value, None

    if set(value) == {ENCRYPTED_VALUE_MARKER}:
        token = value[ENCRYPTED_VALUE_MARKER]
        if isinstance(token, str):
            decrypt_value(token)
            return value, token

    token = encrypt_value(value)
    return {ENCRYPTED_VALUE_MARKER: token}, token


def decode_memory_value(
    value: dict[str, Any], encrypted_value: str | None = None
) -> dict[str, Any]:
    """Decode a sensitive value from current or legacy database representation."""
    token = encrypted_value
    if token is None and set(value) == {ENCRYPTED_VALUE_MARKER}:
        token = value.get(ENCRYPTED_VALUE_MARKER)
    if token is not None:
        if not isinstance(token, str):
            raise ValueError("Memory decryption failed — invalid or corrupted token.")
        return decrypt_value(token)
    return value


# ─────────────────────────────────────────────
# Simple TF-IDF embedding (no GPU/model required)
# ─────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_embedding(text: str) -> dict[str, float]:
    """
    Compute a TF-based term frequency vector.
    Maps tokens to L2-normalized frequency weights.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    freq: dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0

    norm = math.sqrt(sum(v * v for v in freq.values()))
    if norm == 0:
        return freq
    return {k: v / norm for k, v in freq.items()}


def _cosine_similarity(
    a: dict[str, float] | list[float], b: dict[str, float] | list[float]
) -> float:
    """Compute cosine similarity between two term frequency dictionaries or vectors."""
    if not a or not b:
        return 0.0

    # If dictionaries mapping word -> weight
    if isinstance(a, dict) and isinstance(b, dict):
        common_keys = set(a.keys()) & set(b.keys())
        if not common_keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in common_keys)
        return max(0.0, min(1.0, dot))

    # Fallback for list vectors
    if isinstance(a, list) and isinstance(b, list):
        min_len = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(min_len))
        return max(0.0, min(1.0, dot))

    return 0.0


def text_similarity(a: str, b: str) -> float:
    """Return cosine similarity between the TF embeddings of two texts."""
    return _cosine_similarity(compute_embedding(a), compute_embedding(b))


# ─────────────────────────────────────────────
# Automatic memory extraction from conversation
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """Analyze the following conversation and extract important facts about the user.
Return ONLY a JSON array of memory objects. Each object must have:
  - "category": one of ["preference", "learning", "goal", "project", "context", "other"]
  - "memory_key": short snake_case identifier (e.g., "preferred_language", "current_project")
  - "value": an object with the extracted information
  - "importance": float 0.0 to 1.0 (how important/permanent this fact is)
  - "is_sensitive": boolean (true only for private personal data like health, finances, passwords)

Return [] if no meaningful facts are found.
Only extract facts that are explicitly stated or strongly implied.
Do NOT invent or assume facts.

Conversation:
{conversation}

JSON array:"""


async def extract_memories_from_conversation(
    conversation_text: str,
    llm_service,
) -> list[dict[str, Any]]:
    """
    Use the LLM to extract memory candidates from conversation text.
    Returns a list of validated memory dicts.
    """
    if not conversation_text.strip():
        return []

    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text[:3000])
    try:
        raw = await llm_service.generate(
            prompt=prompt,
            system="You are a memory extraction assistant. Return only valid JSON arrays.",
        )
        # Find JSON array in response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        candidates = json.loads(match.group())
        return _validate_memory_candidates(candidates)
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", type(exc).__name__)
        return []


def _validate_memory_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Validate LLM-returned memory candidates strictly."""
    if not isinstance(candidates, list):
        return []

    valid_categories = {"preference", "learning", "goal", "project", "context", "other"}
    key_pattern = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")
    results = []

    for item in candidates:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other")).strip().lower()
        memory_key = str(item.get("memory_key", "")).strip().lower()
        value = item.get("value")
        importance = float(item.get("importance", 0.5))
        is_sensitive = bool(item.get("is_sensitive", False))

        # Validate
        if category not in valid_categories:
            continue
        if not memory_key or not key_pattern.match(memory_key):
            continue
        if not isinstance(value, dict) or not value:
            continue
        importance = max(0.0, min(1.0, importance))

        results.append(
            {
                "category": category,
                "memory_key": memory_key,
                "value": value,
                "importance": importance,
                "is_sensitive": is_sensitive,
                "source": "auto_extraction",
            }
        )

    return results


# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────


async def get_relevant_memories(
    session: AsyncSession,
    user_id,
    query_text: str,
    limit: int = 5,
    include_sensitive: bool = False,
) -> list[PersonalMemory]:
    """
    Fetch memories semantically similar to query_text.
    Uses TF embedding comparison on memory_key + stringified value.
    """
    stmt = select(PersonalMemory).where(PersonalMemory.user_id == user_id)
    if not include_sensitive:
        stmt = stmt.where(PersonalMemory.is_sensitive.is_(False))
    stmt = stmt.order_by(PersonalMemory.importance.desc()).limit(100)

    all_memories = list(await session.scalars(stmt))
    if not all_memories:
        return []

    query_emb = compute_embedding(query_text)

    scored: list[tuple[float, PersonalMemory]] = []
    for mem in all_memories:
        mem_text = f"{mem.memory_key} {json.dumps(mem.value)}"
        if mem.embedding and isinstance(mem.embedding, dict):
            mem_emb = mem.embedding.get("vector", [])
        else:
            mem_emb = compute_embedding(mem_text)
        sim = _cosine_similarity(query_emb, mem_emb)
        scored.append((sim, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


async def store_extracted_memories(
    session: AsyncSession,
    user_id,
    candidates: list[dict[str, Any]],
) -> int:
    """
    Store extracted memory candidates, preventing duplicates.
    Returns the number of memories actually stored.
    """
    stored = 0
    for candidate in candidates:
        # Check for existing memory with same category+key
        existing = await session.scalar(
            select(PersonalMemory).where(
                PersonalMemory.user_id == user_id,
                PersonalMemory.category == candidate["category"],
                PersonalMemory.memory_key == candidate["memory_key"],
            )
        )
        if existing:
            # Update value and importance if significance has increased
            if candidate.get("importance", 0.5) >= existing.importance:
                is_sensitive = candidate.get("is_sensitive", False)
                stored_value, encrypted_value = encode_memory_value(
                    candidate["value"], is_sensitive
                )
                existing.value = stored_value
                existing.encrypted_value = encrypted_value
                existing.is_sensitive = is_sensitive
                existing.embedding = (
                    None
                    if is_sensitive
                    else {
                        "vector": compute_embedding(
                            f"{candidate['memory_key']} {json.dumps(candidate['value'])}"
                        ),
                        "model": "tfidf",
                    }
                )
                existing.importance = candidate.get("importance", 0.5)
                await session.flush()
            continue

        is_sensitive = candidate.get("is_sensitive", False)
        stored_value, encrypted_value = encode_memory_value(
            candidate["value"], is_sensitive
        )
        embedding = None
        if not is_sensitive:
            mem_text = f"{candidate['memory_key']} {json.dumps(candidate['value'])}"
            embedding = {"vector": compute_embedding(mem_text), "model": "tfidf"}

        mem = PersonalMemory(
            user_id=user_id,
            category=candidate["category"],
            memory_key=candidate["memory_key"],
            value=stored_value,
            is_sensitive=is_sensitive,
            source=candidate.get("source", "auto_extraction"),
            importance=candidate.get("importance", 0.5),
            encrypted_value=encrypted_value,
            embedding=embedding,
        )
        session.add(mem)
        await session.flush()
        stored += 1

    return stored


def format_memories_for_context(
    memories: list[PersonalMemory], reveal_sensitive: bool = False
) -> list[dict[str, Any]]:
    """
    Format memories for injection into chat context.
    Never includes encrypted raw values.
    """
    result = []
    for mem in memories:
        result.append(
            {
                "memory_key": mem.memory_key,
                "category": mem.category,
                "value": (
                    decode_memory_value(mem.value, mem.encrypted_value)
                    if mem.is_sensitive and reveal_sensitive
                    else {"redacted": True}
                    if mem.is_sensitive
                    else mem.value
                ),
                "importance": mem.importance,
            }
        )
    return result
