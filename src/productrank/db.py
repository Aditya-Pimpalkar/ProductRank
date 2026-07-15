"""Database engines and session factories — one per dataset (Path 2).

A synchronous SQLAlchemy engine is deliberate. The retrieval request path does a small
number of fast, indexed queries; psycopg3's sync path is simple and avoids the foot-guns
of mixing async DB I/O with the in-process (blocking) cross-encoder. FastAPI runs sync
route handlers in a threadpool, so the event loop is not blocked.

Multi-dataset: each dataset is a separate database inside one ParadeDB instance. We keep a
registry of engines/sessionmakers keyed by dataset and hand the right session to the
otherwise-unchanged retrieval code. Engines are created lazily so importing this module
(e.g. in unit tests) never opens a connection.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from productrank.config import DATASETS, DEFAULT_DATASET, settings


class Base(DeclarativeBase):
    pass


_ENGINES: dict[str, Engine] = {}
_SESSIONMAKERS: dict[str, sessionmaker] = {}


def _validate(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; allowed: {DATASETS}")
    return dataset


def engine_for(dataset: str) -> Engine:
    """Return (creating on first use) the engine for a dataset's database."""
    _validate(dataset)
    if dataset not in _ENGINES:
        _ENGINES[dataset] = create_engine(
            settings.database_url_for(dataset),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
    return _ENGINES[dataset]


def sessionmaker_for(dataset: str) -> sessionmaker:
    _validate(dataset)
    if dataset not in _SESSIONMAKERS:
        _SESSIONMAKERS[dataset] = sessionmaker(
            bind=engine_for(dataset), autoflush=False, expire_on_commit=False
        )
    return _SESSIONMAKERS[dataset]


def session_for(dataset: str) -> Session:
    """Open a Session bound to the given dataset's database. Caller closes it
    (use as a context manager: `with session_for(ds) as s:`)."""
    return sessionmaker_for(dataset)()


# --- Backward-compatible default-dataset handles (used by single-dataset call sites) ---


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request on the DEFAULT dataset. Routes that are
    dataset-aware resolve their own session via session_for(<validated dataset>)."""
    session = session_for(DEFAULT_DATASET)
    try:
        yield session
    finally:
        session.close()


def __getattr__(name: str):
    # Lazy module-level `engine` / `SessionLocal` for the default dataset, so importing
    # this module never connects but legacy references keep working.
    if name == "engine":
        return engine_for(DEFAULT_DATASET)
    if name == "SessionLocal":
        return sessionmaker_for(DEFAULT_DATASET)
    raise AttributeError(name)
