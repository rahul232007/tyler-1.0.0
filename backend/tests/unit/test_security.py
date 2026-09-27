"""Unit tests for security utilities (password hashing, JWT verification, token versioning)."""
import time
from uuid import uuid4

import pytest
from jose import JWTError

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification():
    raw_password = "SecurePassword123!"
    hashed = hash_password(raw_password)

    assert hashed != raw_password
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(raw_password, None) is False


def test_password_max_length_enforcement():
    too_long = "a" * 73
    with pytest.raises(ValueError, match="at most 72 bytes"):
        hash_password(too_long)


def test_jwt_token_roundtrip():
    user_id = uuid4()
    token_version = 2
    token = create_access_token(user_id, token_version)

    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_access_token(token)
    assert payload["sub"] == str(user_id)
    assert payload["tv"] == token_version
    assert "exp" in payload


def test_jwt_invalid_token():
    with pytest.raises(JWTError):
        decode_access_token("invalid.jwt.token.string")
