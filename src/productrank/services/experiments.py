"""A/B experiment jobs (PR-23 / FR-5, FR-8).

An A/B eval over hundreds of queries is a batch workload, not a request-path operation,
so it runs as a background job with state in Redis and is polled via GET. Async is scoped
*strictly* to eval runs — the search path stays synchronous (ARCHITECTURE §7). At this
scale FastAPI BackgroundTasks is sufficient; a Celery queue is the documented upgrade
path for concurrent, long-running eval.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select

from productrank import cache
from productrank.db import SessionLocal
from productrank.evaluation.metrics import METRIC_KEYS, evaluate
from productrank.evaluation.run import _load_qrels, _load_queries
from productrank.evaluation.significance import paired_significance
from productrank.models import Document
from productrank.observability.logging import get_logger
from productrank.retrieval.embeddings import embed_texts
from productrank.services.search import Variant, search

log = get_logger("experiments")

JOB_TTL = 60 * 60  # 1 hour — results are cheap to recompute
# Metrics we report significance on (the headline ranking metrics).
SIG_METRICS = ["ndcg_cut_10", "ndcg_cut_100", "recip_rank", "map"]


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def create_job(variant_a: str, variant_b: str, query_set_size: int, split: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    _write(
        job_id,
        {
            "id": job_id,
            "status": "pending",
            "progress": 0.0,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "query_set_size": query_set_size,
            "split": split,
        },
    )
    return job_id


def get_job(job_id: str) -> dict | None:
    client = cache.get_client()
    if client is None:
        return None
    raw = client.get(_job_key(job_id))
    return json.loads(raw) if raw else None


def _write(job_id: str, state: dict) -> None:
    client = cache.get_client()
    if client is None:
        return
    client.set(_job_key(job_id), json.dumps(state), ex=JOB_TTL)


def _build_run(session, variant: Variant, queries, query_vectors, top_k):
    run: dict[str, dict[str, float]] = {}
    for qid, qtext in queries.items():
        res = search(
            session,
            qtext,
            variant,
            top_k=top_k,
            candidate_k=top_k,
            query_vector=query_vectors.get(qid),
        )
        run[qid] = {h.doc_id: h.score for h in res.hits}
    return run


def run_experiment(job_id: str) -> None:
    """Execute the A/B run end to end and persist results to Redis. Runs in the
    background; never raises into the request path — failures land in the job state."""
    state = get_job(job_id)
    if state is None:
        return
    try:
        state["status"] = "running"
        _write(job_id, state)

        variant_a = Variant(state["variant_a"])
        variant_b = Variant(state["variant_b"])
        size = int(state["query_set_size"])
        split = state["split"]

        with SessionLocal() as session:
            queries = _load_queries(session, split, limit=size)
            qrels = _load_qrels(session, list(queries))

            # Embed shared query set once, reused across both variants.
            query_vectors: dict[str, list[float]] = {}
            if variant_a != Variant.BM25 or variant_b != Variant.BM25:
                qids = list(queries)
                if session.scalar(
                    select(Document.id).where(Document.embedding.isnot(None)).limit(1)
                ):
                    vectors = embed_texts([queries[q] for q in qids])
                    query_vectors = dict(zip(qids, vectors, strict=True))

            run_a = _build_run(session, variant_a, queries, query_vectors, top_k=100)
            state["progress"] = 0.5
            _write(job_id, state)
            run_b = _build_run(session, variant_b, queries, query_vectors, top_k=100)

        eval_a = evaluate(qrels, run_a)
        eval_b = evaluate(qrels, run_b)

        sig = []
        for metric in SIG_METRICS:
            a_pq = {q: v[metric] for q, v in eval_a.per_query.items()}
            b_pq = {q: v[metric] for q, v in eval_b.per_query.items()}
            r = paired_significance(a_pq, b_pq, metric=metric)
            sig.append(
                {
                    "metric": metric,
                    "mean_a": r.mean_a,
                    "mean_b": r.mean_b,
                    "mean_diff": r.mean_diff,
                    "p_value": r.p_value,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "significant": r.significant,
                }
            )

        state.update(
            {
                "status": "completed",
                "progress": 1.0,
                "metrics_a": {k: eval_a.aggregate[k] for k in METRIC_KEYS},
                "metrics_b": {k: eval_b.aggregate[k] for k in METRIC_KEYS},
                "significance": sig,
            }
        )
        _write(job_id, state)
        log.info("experiment_done", job_id=job_id, a=variant_a.value, b=variant_b.value)
    except Exception as exc:  # noqa: BLE001 — surface failure in job state, never crash
        state["status"] = "error"
        state["error"] = str(exc)
        _write(job_id, state)
        log.error("experiment_failed", job_id=job_id, error=str(exc))
