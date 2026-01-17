"""create table_upload table

Revision ID: 8b79f44be5c2
Revises: 40fb2cd83300
Create Date: 2026-01-10 14:58:28.234365

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from basejump.alembic.utils import refresh_views

# revision identifiers, used by Alembic.
revision: str = "8b79f44be5c2"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "table_upload",
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("upload_id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("upload_uuid", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("table_name", sa.String(), nullable=False),
        sa.Column("table_location", sa.String(), nullable=False),
        sa.Column("db_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["client_id"], ["account.client.client_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["db_id"], ["connect.database.db_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("client_id", "upload_id"),
        schema="connect",
    )
    refresh_views(tables=["connect.table_upload"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("table_upload", schema="connect")
    refresh_views(tables=["connect.table_upload"])
