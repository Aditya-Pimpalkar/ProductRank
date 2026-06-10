"""initial schema: documents (sparse+dense), queries, qrels, ingest_state

Revision ID: 0001
Revises:
Create Date: 2026-06-08
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from productrank.config import settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector extension must exist before the vector column is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "doc_metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(settings.embedding_dim), nullable=True),
    )

    op.create_table(
        "queries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("split", sa.String(), nullable=False, server_default="test"),
    )

    op.create_table(
        "qrels",
        sa.Column("pk", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "query_id",
            sa.String(),
            sa.ForeignKey("queries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "doc_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relevance", sa.Integer(), nullable=False),
    )
    op.create_index("ix_qrels_query_id", "qrels", ["query_id"])
    op.create_index("uq_qrels_query_doc", "qrels", ["query_id", "doc_id"], unique=True)

    op.create_table(
        "ingest_state",
        sa.Column("phase", sa.String(), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("ingest_state")
    op.drop_index("uq_qrels_query_doc", table_name="qrels")
    op.drop_index("ix_qrels_query_id", table_name="qrels")
    op.drop_table("qrels")
    op.drop_table("queries")
    op.drop_table("documents")
