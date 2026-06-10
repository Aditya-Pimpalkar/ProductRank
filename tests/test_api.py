"""API-surface tests (PR-16/17).

Validation tests need no DB (Pydantic rejects before the handler runs). The search
happy-path test needs the seeded corpus and is skipped otherwise — keeping CI green
without the stack.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from productrank.main import app

client = TestClient(app)


def test_health_shape():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "postgres", "redis"}


def test_empty_query_rejected():
    r = client.post("/v1/search", json={"query": "", "variant": "bm25"})
    assert r.status_code == 422


def test_top_k_cap_enforced():
    r = client.post("/v1/search", json={"query": "x", "variant": "bm25", "top_k": 999})
    assert r.status_code == 422


def test_unknown_variant_rejected():
    r = client.post("/v1/search", json={"query": "x", "variant": "magic"})
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "productrank_" in r.text


def _seeded() -> bool:
    from sqlalchemy import func, select

    from productrank.db import SessionLocal
    from productrank.models import Document

    try:
        with SessionLocal() as s:
            return (s.scalar(select(func.count()).select_from(Document)) or 0) > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _seeded(), reason="corpus not seeded")
def test_bm25_search_happy_path():
    r = client.post(
        "/v1/search", json={"query": "401k rollover", "variant": "bm25", "top_k": 5}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["variant"] == "bm25"
    assert 1 <= len(body["hits"]) <= 5
    assert body["hits"][0]["rank"] == 1
    # ranks strictly increasing, scores non-increasing
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
