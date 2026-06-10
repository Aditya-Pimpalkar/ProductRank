"""Run the evaluation: all four variants over a query set → metrics table.

This is the script that produces the project's first **real** NDCG numbers (PR-10).
It builds a trec_eval-style `run` per variant by executing the orchestration service
over every evaluation query, scores each against the FiQA qrels, and prints a
side-by-side table plus saves per-query metrics (the input significance testing needs
in PR-22).

Variants that need dense vectors are skipped with a clear message when the corpus is
not yet embedded, so `eval` still yields the BM25 baseline without an API key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from productrank.db import SessionLocal
from productrank.evaluation.metrics import METRIC_KEYS, METRIC_LABELS, Qrels, evaluate
from productrank.models import Document, Qrel, Query
from productrank.retrieval.embeddings import embed_texts
from productrank.services.search import SearchResult, Variant, search

RESULTS_DIR = Path("results")


def _load_queries(session: Session, split: str, limit: int | None) -> dict[str, str]:
    stmt = select(Query.id, Query.text).where(Query.split == split).order_by(Query.id)
    if limit:
        stmt = stmt.limit(limit)
    return dict(session.execute(stmt).all())


def _load_qrels(session: Session, qids: list[str]) -> Qrels:
    rows = session.execute(
        select(Qrel.query_id, Qrel.doc_id, Qrel.relevance).where(Qrel.query_id.in_(qids))
    ).all()
    qrels: Qrels = {}
    for qid, did, rel in rows:
        qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


def _has_embeddings(session: Session) -> bool:
    return (
        session.scalar(
            select(func.count()).select_from(Document).where(Document.embedding.isnot(None))
        )
        or 0
    ) > 0


def _print_table(agg: dict[str, dict[str, float]]) -> None:
    variants = list(agg.keys())
    col_w = 14
    header = "Metric".ljust(12) + "".join(v.ljust(col_w) for v in variants)
    print("\n" + header)
    print("-" * len(header))
    for key in METRIC_KEYS:
        row = METRIC_LABELS[key].ljust(12)
        row += "".join(f"{agg[v][key]:.4f}".ljust(col_w) for v in variants)
        print(row)


def _load_existing(out_path: Path) -> dict:
    """Load a prior results file so a partial re-run (e.g. just the reranker) merges
    into the existing table instead of discarding completed variants."""
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def run_evaluation(
    limit: int | None = None,
    top_k: int = 100,
    split: str = "test",
    only: list[str] | None = None,
    tag: str | None = None,
) -> dict:
    with SessionLocal() as session:
        queries = _load_queries(session, split, limit)
        if not queries:
            raise RuntimeError(f"No queries for split={split}. Did you run seed.py?")
        qrels = _load_qrels(session, list(queries))

        variants = [Variant.BM25]
        if _has_embeddings(session):
            variants += [Variant.DENSE, Variant.HYBRID, Variant.HYBRID_RERANK]
        else:
            print(
                "⚠ No embeddings found — evaluating BM25 only. Run "
                "`productrank embed` (needs OPENAI_API_KEY) for the dense/hybrid/rerank "
                "variants."
            )

        if only:
            wanted = {v.lower() for v in only}
            variants = [v for v in variants if v.value in wanted]
            if not variants:
                raise RuntimeError(f"No matching variants for --variants {only}")

        # Embed every query ONCE up front (batched) and reuse the vector across the
        # dense/hybrid/rerank variants. Embedding per-query inside the loop would re-embed
        # the same 648 queries three times — ~2000 sequential API calls vs. a handful of
        # batch calls here.
        query_vectors: dict[str, list[float]] = {}
        needs_dense = any(v != Variant.BM25 for v in variants)
        if needs_dense:
            print("Pre-embedding queries (batched) …")
            qids = list(queries)
            vectors = embed_texts([queries[q] for q in qids])
            query_vectors = dict(zip(qids, vectors, strict=True))

        print(f"Evaluating {len(variants)} variant(s) over {len(queries)} queries …")

        RESULTS_DIR.mkdir(exist_ok=True)
        # `tag` lets a sample run (e.g. a 100-query rerank comparison) write to its own
        # file rather than clobbering the full-corpus base table.
        fname = f"eval_{split}_{tag}.json" if tag else f"eval_{split}.json"
        out_path = RESULTS_DIR / fname
        existing = _load_existing(out_path)

        # Seed with prior results so a partial re-run merges rather than overwrites.
        agg: dict[str, dict[str, float]] = dict(existing.get("aggregate", {}))
        per_query: dict[str, dict[str, dict[str, float]]] = dict(existing.get("per_query", {}))
        latency: dict[str, float] = dict(existing.get("wall_seconds", {}))

        def _save() -> None:
            out_path.write_text(
                json.dumps(
                    {
                        "split": split,
                        "num_queries": len(queries),
                        "top_k": top_k,
                        "aggregate": agg,
                        "per_query": per_query,
                        "wall_seconds": latency,
                    },
                    indent=2,
                )
            )

        for variant in variants:
            t0 = time.perf_counter()
            run: dict[str, dict[str, float]] = {}
            for i, (qid, qtext) in enumerate(queries.items(), 1):
                res: SearchResult = search(
                    session,
                    qtext,
                    variant,
                    top_k=top_k,
                    candidate_k=top_k,
                    query_vector=query_vectors.get(qid),
                )
                # trec_eval needs strictly-ranked scores; ranks are already correct.
                run[qid] = {h.doc_id: h.score for h in res.hits}
                if i % 50 == 0:
                    print(f"  {variant.value}: {i}/{len(queries)}", end="\r", flush=True)

            result = evaluate(qrels, run)
            agg[variant.value] = result.aggregate
            per_query[variant.value] = result.per_query
            latency[variant.value] = round(time.perf_counter() - t0, 1)
            _save()  # incremental: a crash in a later variant can't lose this one
            print(
                f"  {variant.value}: done in {latency[variant.value]}s — "
                f"NDCG@10={result.aggregate['ndcg_cut_10']:.4f}"
            )

    # Print in canonical variant order, including any merged-in prior variants.
    order = [v.value for v in Variant]
    ordered_agg = {v: agg[v] for v in order if v in agg}
    _print_table(ordered_agg)
    print(f"\nSaved → {out_path}")
    return {"aggregate": agg, "per_query": per_query, "wall_seconds": latency}
