"""Structured JSON logging with per-request correlation ids (PR-15 / FR-9).

structlog emits machine-parseable JSON logs (the format real log pipelines ingest), and
a contextvar-bound correlation id ties every log line from one request together — the
thing you actually need when debugging a latency spike in production.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def new_correlation_id() -> str:
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def _add_correlation_id(_logger, _method, event_dict):
    event_dict["correlation_id"] = _correlation_id.get()
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "productrank"):
    return structlog.get_logger(name)
