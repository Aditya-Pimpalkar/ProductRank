"""Prometheus metrics: per-stage retrieval latency + request counters (PR-14 / FR-9).

Histograms (not just averages) are used for latency because the story that matters for
the target ad-retrieval domain is the *tail* — p95/p99 under a latency budget, not the
mean (ARCHITECTURE §8). Prometheus histograms let p50/p95/p99 be computed at query time
from the bucket counts.

Stage labels: sparse / dense / fusion / rerank / end_to_end. The `/metrics` endpoint
(wired in the app) exposes these in Prometheus exposition format; in production a
Prometheus server scrapes it and Grafana renders the percentiles. Here we expose the
endpoint and read percentiles directly (the deliberate scoping decision in §8).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Buckets tuned for a retrieval path: sub-ms fusion up to multi-second cold rerank.
_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

STAGE_LATENCY = Histogram(
    "productrank_stage_latency_seconds",
    "Per-stage retrieval latency",
    labelnames=("variant", "stage"),
    buckets=_LATENCY_BUCKETS,
)

SEARCH_REQUESTS = Counter(
    "productrank_search_requests_total",
    "Search requests handled",
    labelnames=("variant", "cache"),
)

SEARCH_ERRORS = Counter(
    "productrank_search_errors_total",
    "Search requests that raised",
    labelnames=("variant",),
)


def record_stage_latencies(variant: str, stage_latency_ms: dict[str, float]) -> None:
    for stage, ms in stage_latency_ms.items():
        STAGE_LATENCY.labels(variant=variant, stage=stage).observe(ms / 1000.0)


def record_search(variant: str, cache_hit: bool) -> None:
    SEARCH_REQUESTS.labels(variant=variant, cache="hit" if cache_hit else "miss").inc()
