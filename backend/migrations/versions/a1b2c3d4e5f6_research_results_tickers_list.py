"""research_results: query+ticker → tickers list

Revision ID: a1b2c3d4e5f6
Revises: cf6294f3f61a
Create Date: 2026-05-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'cf6294f3f61a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('research_results', sa.Column('tickers', sa.JSON(), nullable=True))
    op.drop_column('research_results', 'query')
    op.drop_column('research_results', 'ticker')


def downgrade() -> None:
    op.add_column('research_results', sa.Column('ticker', sa.String(length=20), nullable=True))
    op.add_column('research_results', sa.Column('query', sa.Text(), nullable=False, server_default=''))
    op.drop_column('research_results', 'tickers')
