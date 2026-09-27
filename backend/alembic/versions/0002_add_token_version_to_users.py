"""Add token_version to users for auth logout/session invalidation."""

import sqlalchemy as sa
from alembic import op

revision = "0002_add_token_version_to_users"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("users", "token_version")
