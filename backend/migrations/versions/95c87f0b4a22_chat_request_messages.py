"""Associate visible messages with idempotent chat requests.

Revision ID: 95c87f0b4a22
Revises: 681eda53217c
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "95c87f0b4a22"
down_revision = "681eda53217c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_requests", sa.Column("request_content", sa.Text(), nullable=True))
    op.add_column("chat_requests", sa.Column("result_type", sa.String(32), nullable=True))
    op.add_column("messages", sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_messages_request_id", "messages", "chat_requests", ["request_id"], ["id"])
    op.create_index("ix_messages_request_id", "messages", ["request_id"])
    op.create_unique_constraint("uq_messages_request_sender", "messages", ["request_id", "sender_type"])


def downgrade() -> None:
    op.drop_constraint("uq_messages_request_sender", "messages", type_="unique")
    op.drop_index("ix_messages_request_id", table_name="messages")
    op.drop_constraint("fk_messages_request_id", "messages", type_="foreignkey")
    op.drop_column("messages", "request_id")
    op.drop_column("chat_requests", "result_type")
    op.drop_column("chat_requests", "request_content")
