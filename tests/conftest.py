"""Shared pytest fixtures.

The pure unit tests (fusion, metrics, significance) need nothing external. The
integration / E2E tests need the live Postgres with a seeded + embedded corpus; they are
skipped automatically when the DB is unreachable or empty, so `pytest` stays green on a
machine without the stack up (CI runs only the unit tests).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def db_session():
    from sqlalchemy import func, select

    from productrank.db import SessionLocal
    from productrank.models import Document

    try:
        session = SessionLocal()
        n = session.scalar(select(func.count()).select_from(Document)) or 0
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {e}")

    if n == 0:
        session.close()
        pytest.skip("Corpus not seeded — run seed.py to enable integration tests.")

    yield session
    session.close()


@pytest.fixture(scope="session")
def has_embeddings(db_session) -> bool:
    from sqlalchemy import func, select

    from productrank.models import Document

    n = (
        db_session.scalar(
            select(func.count()).select_from(Document).where(Document.embedding.isnot(None))
        )
        or 0
    )
    return n > 0
