"""Diagnostic: why doesn't the reranker beat dense on FiQA?

Compares, over a query sample, NDCG@10 for:
  - dense                 : pure dense top-k (the current winner)
  - rerank(fused pool)    : current pipeline — CE over RRF top-100
  - rerank(dense pool)    : CE over DENSE top-100 (isolates "diluted pool" vs "weak CE")

If rerank(dense pool) >> dense  → the RRF pool dilution is the problem (fix: rerank dense).
If rerank(dense pool) <= dense  → the cross-encoder is the limit (fix: stronger reranker).

Run: HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run python scripts/diagnose_rerank.py [N]
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from productrank.config import settings
from productrank.db import SessionLocal
from productrank.evaluation.metrics import evaluate
from productrank.evaluation.run import _load_qrels, _load_queries
from productrank.models import Document
from productrank.retrieval.dense import search_dense
from productrank.retrieval.embeddings import embed_texts
from productrank.retrieval.fusion import reciprocal_rank_fusion
from productrank.retrieval.rerank import rerank
from productrank.retrieval.sparse import search_sparse


def _texts(session, doc_ids):
    if not doc_ids:
        return {}
    rows = session.execute(
        select(Document.id, Document.title, Document.text).where(Document.id.in_(doc_ids))
    ).all()
    return {r[0]: f"{r[1] or ''} {r[2]}" for r in rows}


def main(n: int = 60) -> None:
    with SessionLocal() as s:
        queries = _load_queries(s, "test", limit=n)
        qrels = _load_qrels(s, list(queries))
        qids = list(queries)
        vecs = dict(zip(qids, embed_texts([queries[q] for q in qids]), strict=True))

        runs: dict[str, dict] = {"dense": {}, "rerank_fused": {}, "rerank_dense": {}}
        for qid, qtext in queries.items():
            qv = vecs[qid]
            dense_hits = search_dense(s, qtext, top_k=100, query_vector=qv)
            sparse_hits = search_sparse(s, qtext, top_k=100)
            fused = reciprocal_rank_fusion([sparse_hits, dense_hits], k=settings.rrf_k)

            runs["dense"][qid] = {h.doc_id: h.score for h in dense_hits[:10]}

            # rerank over fused top-100 (current pipeline)
            fpool = fused[:100]
            ftext = _texts(s, [h.doc_id for h in fpool])
            fr = rerank(qtext, [(h.doc_id, ftext.get(h.doc_id, "")) for h in fpool], top_k=10)
            runs["rerank_fused"][qid] = {h.doc_id: h.score for h in fr}

            # rerank over dense top-100 (isolates pool quality)
            dpool = dense_hits[:100]
            dtext = _texts(s, [h.doc_id for h in dpool])
            dr = rerank(qtext, [(h.doc_id, dtext.get(h.doc_id, "")) for h in dpool], top_k=10)
            runs["rerank_dense"][qid] = {h.doc_id: h.score for h in dr}

        print(f"\nReranker: {settings.rerank_model}   (n={len(queries)} queries)\n")
        for name, run in runs.items():
            res = evaluate(qrels, run)
            print(f"  {name:16s} NDCG@10={res.aggregate['ndcg_cut_10']:.4f}  "
                  f"MRR={res.aggregate['recip_rank']:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
