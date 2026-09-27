"""Integration tests for Authentication flow and User Isolation."""
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_full_auth_lifecycle(client: AsyncClient):
    email = f"user_{uuid4().hex[:8]}@jarvis.ai"
    password = "StrongPassword987!"

    # 1. Register
    reg_res = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "display_name": "New Explorer"},
    )
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["email"] == email
    assert user_data["display_name"] == "New Explorer"

    # Duplicate registration should be 409
    dup_res = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert dup_res.status_code == 409

    # 2. Login
    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Read current user (/me)
    me_res = await client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == email

    # 4. Update user preferences (/me PATCH)
    patch_res = await client.patch(
        "/api/v1/auth/me",
        headers=headers,
        json={"display_name": "Updated Name", "preferences": {"language": "tamil"}},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["display_name"] == "Updated Name"
    assert patch_res.json()["preferences"]["language"] == "tamil"

    # 5. Logout
    logout_res = await client.post("/api/v1/auth/logout", headers=headers)
    assert logout_res.status_code == 200

    # 6. Accessing /me after logout should now fail with 401
    post_logout_res = await client.get("/api/v1/auth/me", headers=headers)
    assert post_logout_res.status_code == 401


@pytest.mark.asyncio
async def test_user_isolation_memories(client: AsyncClient, auth_headers: dict[str, str]):
    # User 1 creates a memory
    mem_res = await client.post(
        "/api/v1/memories/",
        headers=auth_headers,
        json={
            "category": "preference",
            "memory_key": "favorite_food",
            "value": {"item": "Dosa"},
            "is_sensitive": False,
        },
    )
    assert mem_res.status_code == 201
    memory_id = mem_res.json()["id"]

    # Register User 2
    user2_email = f"user2_{uuid4().hex[:8]}@jarvis.ai"
    await client.post(
        "/api/v1/auth/register",
        json={"email": user2_email, "password": "Password123!", "display_name": "User Two"},
    )
    login2 = await client.post(
        "/api/v1/auth/login",
        json={"email": user2_email, "password": "Password123!"},
    )
    user2_headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}

    # User 2 tries to access User 1's memory -> 404
    forbidden_res = await client.get(f"/api/v1/memories/{memory_id}", headers=user2_headers)
    assert forbidden_res.status_code == 404

    # User 2's memories list should not contain User 1's memory
    list_res = await client.get("/api/v1/memories/", headers=user2_headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) == 0
