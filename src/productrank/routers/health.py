"""Liveness/readiness endpoint (ARCHITECTURE §4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from productrank import cache
from productrank.db import get_session
from productrank.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        pg_ok = True
    except Exception:  # noqa: BLE001
        pg_ok = False
    redis_ok = cache.get_client() is not None
    status = "ok" if pg_ok else "degraded"
    return HealthResponse(status=status, postgres=pg_ok, redis=redis_ok)
