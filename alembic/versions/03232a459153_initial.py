"""initial

Revision ID: 03232a459153
Revises:
Create Date: 2024-07-14 07:19:02.084380

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "03232a459153"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "predicted_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=True),
        sa.Column("open_price", sa.Float(), nullable=True),
        sa.Column("take_price", sa.Float(), nullable=True),
        sa.Column("percentage_change", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_predicted_trades_id"), "predicted_trades", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_predicted_trades_symbol"), "predicted_trades", ["symbol"], unique=False
    )
    op.create_table(
        "tickers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("symbol", sa.String(), nullable=True),
        sa.Column("image", sa.String(), nullable=True),
        sa.Column("market_cap", sa.Integer(), nullable=True),
        sa.Column("gross_margin_pct", sa.Float(), nullable=True),
        sa.Column("net_margin_pct", sa.Float(), nullable=True),
        sa.Column("trailing_pe", sa.Float(), nullable=True),
        sa.Column("piotroski_score", sa.Integer(), nullable=True),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("sector", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tickers_id"), "tickers", ["id"], unique=False)
    op.create_index(op.f("ix_tickers_name"), "tickers", ["name"], unique=False)
    op.create_index(op.f("ix_tickers_symbol"), "tickers", ["symbol"], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_tickers_symbol"), table_name="tickers")
    op.drop_index(op.f("ix_tickers_name"), table_name="tickers")
    op.drop_index(op.f("ix_tickers_id"), table_name="tickers")
    op.drop_table("tickers")
    op.drop_index(op.f("ix_predicted_trades_symbol"), table_name="predicted_trades")
    op.drop_index(op.f("ix_predicted_trades_id"), table_name="predicted_trades")
    op.drop_table("predicted_trades")
    # ### end Alembic commands ###
