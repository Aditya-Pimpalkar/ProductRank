"""Redis cache tests (PR-13).

Skips when Redis is unreachable (the cache is fail-soft, so the app works without it).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def cache_mod():
    from productrank import cache

    if cache.get_client() is None:
        pytest.skip("Redis not reachable")
    return cache


def test_query_embedding_round_trip(cache_mod):
    q = "test query for cache round trip"
    cache_mod.set_query_embedding(q, [0.1, 0.2, 0.3])
    assert cache_mod.get_query_embedding(q) == [0.1, 0.2, 0.3]


def test_missing_key_returns_none(cache_mod):
    assert cache_mod.get_query_embedding("a query never cached zzz-unique") is None


def test_result_round_trip(cache_mod):
    payload = {"hits": [["d1", 1.0], ["d2", 0.5]]}
    cache_mod.set_result("hybrid", "some query", 10, payload)
    assert cache_mod.get_result("hybrid", "some query", 10) == payload


def test_fail_soft_when_redis_down(monkeypatch):
    """With no client, cache ops must no-op rather than raise."""
    from productrank import cache

    monkeypatch.setattr(cache, "get_client", lambda: None)
    cache.set_query_embedding("x", [1.0])  # must not raise
    assert cache.get_query_embedding("x") is None
    assert cache.get_result("v", "x", 10) is None
