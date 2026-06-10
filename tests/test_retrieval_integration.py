"""Integration + E2E tests against the live seeded corpus (NFR-5, PR-26).

These run only when Postgres is up with a seeded (and, for some, embedded) corpus —
otherwise they skip (see conftest.py). The E2E test asserts BM25 NDCG@10 clears a
threshold, guarding against silent ranking regressions in the retrieval path.
"""

from __future__ import annotations

import pytest


def test_sparse_returns_relevant_doc(db_session):
    """A pointed lexical query should retrieve a topically-relevant document."""
    from productrank.retrieval.sparse import search_sparse

    hits = search_sparse(db_session, "capital gains tax on stocks", top_k=10)
    assert hits, "sparse retrieval returned nothing"
    assert all(h.score >= 0 for h in hits)
    # scores must be in non-increasing rank order
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_e2e_bm25_ndcg_above_threshold(db_session):
    """E2E: BM25 over a query sample must exceed a known NDCG@10 floor.

    Measured full-set BM25 NDCG@10 ≈ 0.238; we assert > 0.15 over a sample so the guard
    is robust to sampling while still catching a real regression.
    """
    from sqlalchemy import select

    from productrank.evaluation.metrics import evaluate
    from productrank.models import Qrel, Query
    from productrank.services.search import Variant, search

    queries = dict(
        db_session.execute(select(Query.id, Query.text).order_by(Query.id).limit(50)).all()
    )
    rows = db_session.execute(
        select(Qrel.query_id, Qrel.doc_id, Qrel.relevance).where(Qrel.query_id.in_(list(queries)))
    ).all()
    qrels: dict[str, dict[str, int]] = {}
    for qid, did, rel in rows:
        qrels.setdefault(qid, {})[did] = int(rel)

    run = {
        qid: {h.doc_id: h.score for h in search(db_session, qtext, Variant.BM25, top_k=100).hits}
        for qid, qtext in queries.items()
    }
    result = evaluate(qrels, run)
    assert result.aggregate["ndcg_cut_10"] > 0.15, result.aggregate


def test_dense_retrieval_runs(db_session, has_embeddings):
    """Dense retrieval returns cosine-ranked hits (needs embeddings)."""
    if not has_embeddings:
        pytest.skip("corpus not embedded")
    from productrank.retrieval.dense import search_dense

    hits = search_dense(db_session, "how do bonds work", top_k=10)
    assert hits
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)  # cosine similarity range
