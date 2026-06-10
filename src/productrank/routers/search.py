"""POST /v1/search — the request-path entry to the retrieval pipeline (PR-17 / FR-1).

Flow: validate → result-cache lookup → orchestrate variant → hydrate hit cards →
record metrics → cache. Caching is fail-soft and keyed by (variant, top_k, query);
the short result-set TTL bounds staleness if the index changes (ARCHITECTURE §6).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from productrank import cache
from productrank.db import get_session
from productrank.observability import metrics
from productrank.observability.logging import get_logger
from productrank.ratelimit import limiter
from productrank.schemas import Hit, SearchRequest, SearchResponse
from productrank.services.search import search

router = APIRouter(prefix="/v1", tags=["search"])
log = get_logger("search")

SNIPPET_CHARS = 240


def _hydrate(session: Session, hits) -> list[Hit]:
    """Attach title + a short snippet to each ranked hit (one query for all ids)."""
    if not hits:
        return []
    ids = [h.doc_id for h in hits]
    stmt = text("SELECT id, title, text FROM documents WHERE id = ANY(:ids)").bindparams(
        bindparam("ids")
    )
    rows = {r[0]: (r[1], r[2]) for r in session.execute(stmt, {"ids": ids}).all()}
    out: list[Hit] = []
    for rank, h in enumerate(hits, 1):
        title, body = rows.get(h.doc_id, ("", ""))
        snippet = (body or "")[:SNIPPET_CHARS]
        out.append(
            Hit(rank=rank, doc_id=h.doc_id, score=round(h.score, 6), title=title, snippet=snippet)
        )
    return out


@router.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search_endpoint(
    request: Request,  # required by slowapi for per-IP keying
    body: SearchRequest,
    session: Session = Depends(get_session),
) -> SearchResponse:
    variant = body.variant

    cached = cache.get_result(variant.value, body.query, body.top_k)
    if cached is not None:
        metrics.record_search(variant.value, cache_hit=True)
        return SearchResponse(**cached, cache_hit=True)

    result = search(
        session,
        body.query,
        variant,
        top_k=body.top_k,
        candidate_k=body.candidate_k,
    )
    payload = SearchResponse(
        variant=variant,
        query=body.query,
        total_latency_ms=result.total_latency_ms,
        stage_latency_ms=result.stage_latency_ms,
        candidate_counts=result.candidate_counts,
        hits=_hydrate(session, result.hits),
        cache_hit=False,
    )

    metrics.record_stage_latencies(variant.value, result.stage_latency_ms)
    metrics.record_search(variant.value, cache_hit=False)
    cache.set_result(variant.value, body.query, body.top_k, payload.model_dump(exclude={"cache_hit"}))
    log.info(
        "search",
        variant=variant.value,
        top_k=body.top_k,
        latency_ms=result.total_latency_ms,
        counts=result.candidate_counts,
    )
    return payload
