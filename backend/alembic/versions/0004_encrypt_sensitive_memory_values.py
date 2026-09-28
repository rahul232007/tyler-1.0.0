"""Remove legacy sensitive memory plaintext from JSON values and embeddings."""

from alembic import op

revision = "0004_encrypt_sensitive_memory_values"
down_revision = "0003_learning_memory_docs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE personal_memories
        SET value = jsonb_build_object('__jarvis_encrypted_value__', encrypted_value),
            embedding = NULL
        WHERE is_sensitive AND encrypted_value IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE personal_memories
        SET value = '{"__jarvis_encrypted_value__": null}'::jsonb,
            embedding = NULL
        WHERE is_sensitive AND encrypted_value IS NULL
        """
    )


def downgrade() -> None:
    # Plaintext cannot be restored without reintroducing the privacy vulnerability.
    pass