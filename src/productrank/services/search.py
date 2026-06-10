"""Retrieval orchestration: the four variants behind one interface.

Variants (PRD FR-1, ARCHITECTURE §2):
  BM25          : sparse only.
  DENSE         : dense only.
  HYBRID        : sparse + dense → RRF fusion.
  HYBRID_RERANK : sparse + dense → RRF → top-N → cross-encoder rerank.

Every call records per-stage latency and candidate counts, so the API and the eval
harness can show *where* time goes and *how many* candidates each stage handled — the
latency-per-stage story that the target ad-retrieval domain cares about (§8). This is
the single composition point; nothing else stitches stages together.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from productrank.config import settings
from productrank.retrieval.dense import search_dense
from productrank.retrieval.fusion import reciprocal_rank_fusion
from productrank.retrieval.rerank import rerank
from productrank.retrieval.sparse import search_sparse
from productrank.retrieval.types import RankedList


class Variant(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class SearchResult:
    variant: Variant
    query: str
    hits: RankedList
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(self.stage_latency_ms.values()), 2)


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - t0) * 1000, 2)


def _fetch_texts(session: Session, doc_ids: list[str]) -> dict[str, str]:
    """Fetch (title+text) for rerank candidates in one query, preserving nothing about
    order (the caller already has the order)."""
    if not doc_ids:
        return {}
    stmt = text(
        "SELECT id, coalesce(title,'') || ' ' || text AS body FROM documents WHERE id = ANY(:ids)"
    ).bindparams(bindparam("ids"))
    rows = session.execute(stmt, {"ids": doc_ids}).all()
    return {r[0]: r[1] for r in rows}


def search(
    session: Session,
    query: str,
    variant: Variant = Variant.HYBRID_RERANK,
    top_k: int = 10,
    *,
    candidate_k: int = 100,
    query_vector: list[float] | None = None,
) -> SearchResult:
    """Run one variant end to end, returning ranked hits + per-stage instrumentation.

    candidate_k is the per-retriever depth for the hybrid/rerank stages; rerank operates
    on the top `settings.rerank_candidates` of the fused list.
    """
    timer = _Timer()
    counts: dict[str, int] = {}

    if variant == Variant.BM25:
        with timer.stage("sparse"):
            hits = search_sparse(session, query, top_k=top_k)
        counts["sparse"] = len(hits)

    elif variant == Variant.DENSE:
        with timer.stage("dense"):
            hits = search_dense(session, query, top_k=top_k, query_vector=query_vector)
        counts["dense"] = len(hits)

    else:  # HYBRID or HYBRID_RERANK both need both retrievers + fusion
        with timer.stage("sparse"):
            sparse_hits = search_sparse(session, query, top_k=candidate_k)
        with timer.stage("dense"):
            dense_hits = search_dense(session, query, top_k=candidate_k, query_vector=query_vector)
        counts["sparse"] = len(sparse_hits)
        counts["dense"] = len(dense_hits)

        with timer.stage("fusion"):
            fused = reciprocal_rank_fusion([sparse_hits, dense_hits], k=settings.rrf_k)
        counts["fused"] = len(fused)

        if variant == Variant.HYBRID:
            hits = fused[:top_k]
        else:  # HYBRID_RERANK
            candidates = fused[: settings.rerank_candidates]
            counts["rerank_in"] = len(candidates)
            with timer.stage("rerank"):
                texts = _fetch_texts(session, [h.doc_id for h in candidates])
                pairs = [(h.doc_id, texts.get(h.doc_id, "")) for h in candidates]
                hits = rerank(query, pairs, top_k=top_k)

    return SearchResult(
        variant=variant,
        query=query,
        hits=hits,
        stage_latency_ms=timer.timings,
        candidate_counts=counts,
    )
