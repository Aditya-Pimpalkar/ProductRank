"""Idempotent bulk loading of corpus / queries / qrels into Postgres.

Idempotency strategy (NFR-4, PR-03 DoD "re-run is a no-op"):
  - documents/queries: PK is the BEIR id → INSERT ... ON CONFLICT (id) DO NOTHING.
  - qrels: unique (query_id, doc_id) → ON CONFLICT DO NOTHING.
A second `seed.py` run therefore inserts zero rows and returns fast.

Loading is batched so a 57K-document corpus streams in constant memory rather than
materializing the whole list.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from productrank.models import Document, IngestState, Qrel, Query

T = TypeVar("T")
BATCH = 1000


def _chunks(it: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(it)
    while batch := list(islice(iterator, size)):
        yield batch


def load_documents(session: Session, rows: Iterable[dict]) -> int:
    inserted = 0
    for batch in _chunks(rows, BATCH):
        stmt = insert(Document).values(batch).on_conflict_do_nothing(index_elements=["id"])
        inserted += session.execute(stmt).rowcount or 0
        session.commit()
    return inserted


def load_queries(session: Session, rows: Iterable[dict]) -> int:
    inserted = 0
    for batch in _chunks(rows, BATCH):
        stmt = insert(Query).values(batch).on_conflict_do_nothing(index_elements=["id"])
        inserted += session.execute(stmt).rowcount or 0
        session.commit()
    return inserted


def load_qrels(session: Session, rows: Iterable[tuple[str, str, int]]) -> int:
    inserted = 0
    for batch in _chunks(rows, BATCH):
        values = [{"query_id": q, "doc_id": d, "relevance": r} for (q, d, r) in batch]
        stmt = (
            insert(Qrel)
            .values(values)
            .on_conflict_do_nothing(index_elements=["query_id", "doc_id"])
        )
        inserted += session.execute(stmt).rowcount or 0
        session.commit()
    return inserted


def record_phase(session: Session, phase: str, count: int) -> None:
    stmt = (
        insert(IngestState)
        .values(phase=phase, count=count)
        .on_conflict_do_update(index_elements=["phase"], set_={"count": count})
    )
    session.execute(stmt)
    session.commit()


def wipe(session: Session) -> None:
    """Clear corpus/queries/qrels/ingest_state so a different dataset can be loaded into
    the same schema. Used when switching datasets (e.g. FiQA → MS MARCO); doc ids are not
    namespaced across datasets, so coexistence would collide — a clean swap is the safe
    contract. CASCADE handles the qrels FKs; RESTART IDENTITY resets the qrels PK."""
    session.execute(
        text("TRUNCATE documents, queries, qrels, ingest_state RESTART IDENTITY CASCADE")
    )
    session.commit()


def counts(session: Session) -> dict[str, int]:
    return {
        "documents": session.scalar(select(func.count()).select_from(Document)) or 0,
        "queries": session.scalar(select(func.count()).select_from(Query)) or 0,
        "qrels": session.scalar(select(func.count()).select_from(Qrel)) or 0,
        "embedded": session.scalar(
            select(func.count()).select_from(Document).where(Document.embedding.isnot(None))
        )
        or 0,
    }
