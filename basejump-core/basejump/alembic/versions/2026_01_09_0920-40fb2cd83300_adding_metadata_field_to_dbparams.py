"""Adding metadata field to DBParams

Revision ID: 40fb2cd83300
Revises: 660fa00ad149
Create Date: 2026-01-09 09:20:45.420998

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from basejump.alembic.utils import refresh_views

# revision identifiers, used by Alembic.
revision: str = "40fb2cd83300"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("database", sa.Column("database_metadata", sa.LargeBinary(), nullable=True), schema="connect")
    refresh_views(tables=["database.tables"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("database", "database_metadata", schema="connect")
    refresh_views(tables=["database.tables"])
