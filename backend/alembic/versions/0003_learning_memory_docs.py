"""Add practice_sessions, documents, memory intelligence columns, and performance indexes."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_learning_memory_docs"
down_revision = "0002_add_token_version_to_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Update personal_memories with intelligence columns
    op.add_column("personal_memories", sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"))
    op.add_column("personal_memories", sa.Column("encrypted_value", sa.Text(), nullable=True))
    op.add_column("personal_memories", sa.Column("embedding", postgresql.JSONB(), nullable=True))
    op.create_index("ix_personal_memories_user_category", "personal_memories", ["user_id", "category"])

    # 2. Create practice_sessions table
    op.create_table(
        "practice_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("learning_topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("questions", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("question_type", sa.String(length=30), nullable=False, server_default="mcq"),
        sa.Column("difficulty", sa.String(length=30), nullable=True),
        sa.Column("total_questions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("answered_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'in_progress', 'completed')", name="ck_practice_sessions_status"),
    )
    op.create_index("ix_practice_sessions_user_id", "practice_sessions", ["user_id"])
    op.create_index("ix_practice_sessions_topic_id", "practice_sessions", ["topic_id"])
    op.create_index("ix_practice_sessions_created_at", "practice_sessions", ["created_at"])

    # 3. Extend practice_results table
    op.add_column(
        "practice_results",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("practice_sessions.id", ondelete="CASCADE"), nullable=True),
    )
    op.add_column("practice_results", sa.Column("question_type", sa.String(length=30), nullable=True))
    op.add_column("practice_results", sa.Column("options", postgresql.JSONB(), nullable=True))
    op.create_index("ix_practice_results_session_id", "practice_results", ["session_id"])
    op.create_index("ix_practice_results_created_at", "practice_results", ["created_at"])

    # 4. Create documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    # 5. Add index on messages(created_at)
    op.create_index("ix_messages_created_at", "messages", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_table("documents")
    op.drop_index("ix_practice_results_created_at", table_name="practice_results")
    op.drop_index("ix_practice_results_session_id", table_name="practice_results")
    op.drop_column("practice_results", "options")
    op.drop_column("practice_results", "question_type")
    op.drop_column("practice_results", "session_id")
    op.drop_table("practice_sessions")
    op.drop_index("ix_personal_memories_user_category", table_name="personal_memories")
    op.drop_column("personal_memories", "embedding")
    op.drop_column("personal_memories", "encrypted_value")
    op.drop_column("personal_memories", "importance")
