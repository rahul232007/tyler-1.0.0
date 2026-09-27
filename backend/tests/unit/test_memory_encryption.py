"""Unit tests for memory encryption/decryption, embeddings, and similarity."""
import pytest

from app.services.memory_service import (
    _cosine_similarity,
    compute_embedding,
    decrypt_value,
    encrypt_value,
    text_similarity,
)


def test_memory_encryption_and_decryption():
    secret_data = {
        "medical_info": "Mild pollen allergy",
        "family_note": "Sister's birthday on Oct 14",
        "nested": {"pin": 1234, "secure": True},
    }

    encrypted_token = encrypt_value(secret_data)
    assert isinstance(encrypted_token, str)
    # Ensure sensitive plaintext is NOT in encrypted token
    assert "pollen" not in encrypted_token
    assert "1234" not in encrypted_token

    decrypted_data = decrypt_value(encrypted_token)
    assert decrypted_data == secret_data


def test_memory_decryption_failure():
    with pytest.raises(ValueError, match="Memory decryption failed"):
        decrypt_value("corrupted-token-not-valid-fernet")


def test_compute_embedding_unit_normalized():
    text = "Machine learning algorithms with neural networks in Python"
    emb = compute_embedding(text)

    assert isinstance(emb, dict)
    assert len(emb) > 0
    # Vector should have unit norm (approx 1.0)
    norm = sum(x * x for x in emb.values())
    assert abs(norm - 1.0) < 0.001


def test_text_similarity_semantic_ranking():
    query = "Python programming and coding tutorial"
    match_high = "Python code and function tutorial"
    match_low = "Baking strawberry cheesecake dessert recipe"

    sim_high = text_similarity(query, match_high)
    sim_low = text_similarity(query, match_low)

    assert sim_high > sim_low
    assert sim_high > 0.0
