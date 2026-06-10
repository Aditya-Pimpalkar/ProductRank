"""ORM models for the corpus, queries, and relevance judgments.

Schema mirrors BEIR's three artifacts:
  - documents : the corpus (FiQA), with both retrieval representations on one row —
                a tsvector (sparse) and a pgvector embedding (dense).
  - queries   : the evaluation queries.
  - qrels     : ground-truth relevance judgments (query_id, doc_id, relevance).

Keeping both retrieval representations co-located on the document row is what lets a
single Postgres instance serve sparse and dense retrieval (ARCHITECTURE §1).

Sparse retrieval uses ParadeDB's `pg_search` (a BM25 index over title+text, created in
migration 0002) rather than a stored tsvector — pg_search does real BM25 with IDF
weighting, where stock FTS (`ts_rank`) does not. The BM25 index is declared in SQL
(migration), not as an ORM column, since it indexes existing columns directly.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from productrank.config import settings
from productrank.db import Base


class Document(Base):
    __tablename__ = "documents"

    # External (BEIR) id is the natural key; we keep it as the PK so qrels join directly.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Sparse representation: a pg_search BM25 index over (title, text), created in
    # migration 0002. Not an ORM column — the index reads the title/text columns above.

    # Dense representation: filled in by the embedding pipeline (PR-04). Nullable so a
    # freshly-ingested corpus is valid before embeddings exist; the embed job is resumable
    # precisely by selecting WHERE embedding IS NULL.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim), nullable=True
    )

    # The BM25 (pg_search) index and the IVFFlat (cosine) dense index are both created
    # in SQL: BM25 in migration 0002; IVFFlat by the embed pipeline AFTER vectors are
    # loaded, since IVFFlat must learn its centroids from populated data.


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    split: Mapped[str] = mapped_column(String, default="test", nullable=False)


class Qrel(Base):
    __tablename__ = "qrels"

    pk: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("queries.id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    relevance: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_qrels_query_id", "query_id"),
        Index("uq_qrels_query_doc", "query_id", "doc_id", unique=True),
    )


class IngestState(Base):
    """Tracks completion of one-shot ingest/embed phases so seed.py is idempotent."""

    __tablename__ = "ingest_state"

    phase: Mapped[str] = mapped_column(String, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
