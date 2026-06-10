"""Pydantic request/response models for the API (PR-16 validation surface).

Bounds live here, not scattered in handlers: query length is capped and top_k is
capped so a public demo can't be used to hammer the embedding/rerank path
(ARCHITECTURE §9.1 cost-control). Validation errors become 422s automatically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from productrank.services.search import Variant

MAX_QUERY_CHARS = 512
MAX_TOP_K = 50


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    variant: Variant = Variant.HYBRID_RERANK
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    candidate_k: int = Field(default=100, ge=10, le=200)


class Hit(BaseModel):
    rank: int
    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""


class SearchResponse(BaseModel):
    variant: Variant
    query: str
    total_latency_ms: float
    stage_latency_ms: dict[str, float]
    candidate_counts: dict[str, int]
    cache_hit: bool = False
    hits: list[Hit]


class ProductResponse(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    redis: bool


# --- experiments (A/B) -----------------------------------------------------

MAX_QUERY_SET = 1000


class ExperimentRequest(BaseModel):
    variant_a: Variant = Variant.BM25
    variant_b: Variant = Variant.HYBRID_RERANK
    query_set_size: int = Field(default=100, ge=2, le=MAX_QUERY_SET)
    split: str = "dev"  # MS MARCO's eval split (the currently-loaded dataset)


class ExperimentResponse(BaseModel):
    """Job handle + (once complete) the side-by-side metrics table and significance."""

    id: str
    status: str  # pending | running | completed | error
    progress: float = 0.0  # 0..1
    variant_a: Variant | None = None
    variant_b: Variant | None = None
    query_set_size: int | None = None
    metrics_a: dict[str, float] | None = None
    metrics_b: dict[str, float] | None = None
    significance: list[dict] | None = None  # one entry per metric
    error: str | None = None
