"""Pydantic request/response models for the API (PR-16 validation surface).

Bounds live here, not scattered in handlers: query length is capped and top_k is
capped so a public demo can't be used to hammer the embedding/rerank path
(ARCHITECTURE §9.1 cost-control). Validation errors become 422s automatically.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from productrank.services.search import Variant

MAX_QUERY_CHARS = 512
MAX_TOP_K = 50


class Dataset(StrEnum):
    """The dataset allowlist. Used as a request-model field type, so Pydantic rejects any
    unknown value with a 422 *before* it can reach connection/dbname construction
    (hard requirement: the raw param never touches the engine registry)."""

    MSMARCO = "msmarco"
    FIQA = "fiqa"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    variant: Variant = Variant.HYBRID_RERANK
    dataset: Dataset = Dataset.MSMARCO
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

# Public-deploy cap: an A/B run embeds + reranks query_set_size queries, so this bounds
# both OpenAI spend and CPU per job (deploy hardening).
MAX_QUERY_SET = 100


class ExperimentRequest(BaseModel):
    variant_a: Variant = Variant.BM25
    variant_b: Variant = Variant.HYBRID_RERANK
    dataset: Dataset = Dataset.MSMARCO  # split is derived from the dataset
    query_set_size: int = Field(default=50, ge=2, le=MAX_QUERY_SET)


class ExperimentResponse(BaseModel):
    """Job handle + (once complete) the side-by-side metrics table and significance."""

    id: str
    status: str  # pending | running | completed | error
    progress: float = 0.0  # 0..1
    dataset: Dataset | None = None
    variant_a: Variant | None = None
    variant_b: Variant | None = None
    query_set_size: int | None = None
    metrics_a: dict[str, float] | None = None
    metrics_b: dict[str, float] | None = None
    significance: list[dict] | None = None  # one entry per metric
    error: str | None = None
