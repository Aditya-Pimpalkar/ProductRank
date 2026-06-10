"""Redis caching layer (PR-13 / FR-7).

Three independently-justified caches (ARCHITECTURE §6):

  emb:{hash}            query embedding      long TTL  — embeddings are deterministic for
                                                         a fixed model; re-embedding a
                                                         repeated query wastes an API call
                                                         and ~200ms.
  res:{variant}:{hash}  result set           short TTL — hot-query latency, but the index
                                                         can change, so staleness is bounded.
  rr:{hash}             rerank result        (demo)    — pre-warmed for sub-second demos.

Keys hash the query so arbitrary text is safe in the key. Values are JSON. The cache is
fail-soft: if Redis is down, callers fall through to recomputation rather than erroring —
caching is a performance optimization, never a correctness dependency.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from productrank.config import settings

# TTLs (seconds). Embeddings effectively never change for a fixed model → long.
EMBED_TTL = 60 * 60 * 24 * 30  # 30 days
RESULT_TTL = 60 * 5  # 5 minutes
RERANK_TTL = 60 * 60 * 24  # 1 day (demo pre-warm)

_client: redis.Redis | None = None


def get_client() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.from_url(settings.redis_url, decode_responses=True)
            _client.ping()
        except Exception:  # noqa: BLE001 — fail-soft: run without cache
            _client = None
    return _client


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- query embedding cache -------------------------------------------------

def get_query_embedding(query: str) -> list[float] | None:
    client = get_client()
    if client is None:
        return None
    raw = client.get(f"emb:{_hash(query)}")
    return json.loads(raw) if raw else None


def set_query_embedding(query: str, vector: list[float]) -> None:
    client = get_client()
    if client is None:
        return
    client.set(f"emb:{_hash(query)}", json.dumps(vector), ex=EMBED_TTL)


# --- result-set cache ------------------------------------------------------

def get_result(variant: str, query: str, top_k: int) -> Any | None:
    client = get_client()
    if client is None:
        return None
    raw = client.get(f"res:{variant}:{top_k}:{_hash(query)}")
    return json.loads(raw) if raw else None


def set_result(variant: str, query: str, top_k: int, payload: Any) -> None:
    client = get_client()
    if client is None:
        return
    client.set(f"res:{variant}:{top_k}:{_hash(query)}", json.dumps(payload), ex=RESULT_TTL)


def cache_stats() -> dict[str, Any]:
    """Lightweight visibility for the /metrics and dashboard surfaces."""
    client = get_client()
    if client is None:
        return {"connected": False}
    info = client.info(section="stats")
    hits = int(info.get("keyspace_hits", 0))
    misses = int(info.get("keyspace_misses", 0))
    total = hits + misses
    return {
        "connected": True,
        "keyspace_hits": hits,
        "keyspace_misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
    }
