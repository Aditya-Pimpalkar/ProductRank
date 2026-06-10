"""Database engine and session factory.

A synchronous SQLAlchemy engine is deliberate. The retrieval request path does a
small number of fast, indexed queries; psycopg3's sync path is simple and avoids the
foot-guns of mixing async DB I/O with the in-process (blocking) cross-encoder. FastAPI
runs sync route handlers in a threadpool, so the event loop is not blocked.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from productrank.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
