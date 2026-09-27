"""Integration tests for Conversations, Messages, Search, Export, and Documents."""

import io

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_conversation_crud_and_search_export(
    client: AsyncClient, auth_headers: dict[str, str]
):
    # 1. Create conversation
    create_res = await client.post(
        "/api/v1/conversations/",
        headers=auth_headers,
        json={"title": "Python Async Programming", "model_provider": "gemini"},
    )
    assert create_res.status_code == 201
    conv = create_res.json()
    conv_id = conv["id"]
    assert conv["title"] == "Python Async Programming"

    # 2. List conversations
    list_res = await client.get("/api/v1/conversations/", headers=auth_headers)
    assert list_res.status_code == 200
    ids = [c["id"] for c in list_res.json()]
    assert conv_id in ids

    # 3. Update conversation
    patch_res = await client.patch(
        f"/api/v1/conversations/{conv_id}",
        headers=auth_headers,
        json={"title": "Advanced Python Asyncio"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Advanced Python Asyncio"

    # 4. Search conversations
    search_res = await client.get(
        "/api/v1/conversations/search?q=Asyncio", headers=auth_headers
    )
    assert search_res.status_code == 200
    results = search_res.json()["conversations"]
    assert any(c["id"] == conv_id for c in results)

    # 5. Export conversation
    export_res = await client.get(
        f"/api/v1/conversations/{conv_id}/export", headers=auth_headers
    )
    assert export_res.status_code == 200
    export_data = export_res.json()
    assert export_data["id"] == conv_id
    assert "messages" in export_data

    # 6. Delete conversation
    del_res = await client.delete(
        f"/api/v1/conversations/{conv_id}", headers=auth_headers
    )
    assert del_res.status_code == 204

    # 7. Get after delete -> 404
    get_res = await client.get(f"/api/v1/conversations/{conv_id}", headers=auth_headers)
    assert get_res.status_code == 404


@pytest.mark.asyncio
async def test_document_upload_and_extraction(
    client: AsyncClient, auth_headers: dict[str, str]
):
    # Upload text document
    file_bytes = (
        b"JARVIS is a private AI tutor designed to teach Python, AI, and Mathematics."
    )
    files = {"file": ("tutor_curriculum.txt", io.BytesIO(file_bytes), "text/plain")}

    upload_res = await client.post(
        "/api/v1/documents/upload", headers=auth_headers, files=files
    )
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["id"]
    assert doc_data["filename"] == "tutor_curriculum.txt"
    assert "tutor designed" in doc_data["extracted_text"]

    # List documents
    list_res = await client.get("/api/v1/documents/", headers=auth_headers)
    assert list_res.status_code == 200
    assert any(d["id"] == doc_id for d in list_res.json())

    # Get single document
    get_res = await client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["extracted_text"] == file_bytes.decode()

    # Delete document
    del_res = await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True
