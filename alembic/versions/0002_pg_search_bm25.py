"""pg_search BM25 index over documents(title, text)

Switches the sparse stage from stock Postgres FTS (ts_rank, no IDF) to ParadeDB's
pg_search BM25 index (Tantivy-backed, real IDF term weighting). This keeps the
"single Postgres" design while giving a baseline that lands in range of the published
BEIR FiQA BM25 number (NFR-3) — stock FTS measured ~0.06 NDCG@10 vs the published ~0.236
because it cannot weight rare, decisive terms.

The BM25 index is maintained incrementally on insert, so creating it before seeding is
fine; rows added by seed.py populate it automatically.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    # key_field='id' ties index rows to the document PK; title + text are the searchable
    # fields. paradedb.score(id) reads BM25 scores from this index at query time.
    op.execute(
        "CREATE INDEX documents_bm25 ON documents "
        "USING bm25 (id, title, text) WITH (key_field='id')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_bm25")
