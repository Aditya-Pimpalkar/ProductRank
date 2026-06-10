"""Reciprocal Rank Fusion (RRF).

Why rank-based fusion and not score addition: BM25/ts_rank scores are unbounded and
corpus-dependent; cosine similarities live in [-1, 1]. Adding them is meaningless —
incompatible scales (ARCHITECTURE §2.3). RRF fuses on *rank position* instead:

    score(d) = Σ_i  1 / (k + rank_i(d))

summed over every ranking i that contains d, where rank is 1-based and k is a smoothing
constant (default 60) that damps the influence of the very top ranks. No score
normalization needed, robust, and the standard hybrid-fusion choice.
"""

from __future__ import annotations

from collections.abc import Sequence

from productrank.retrieval.types import Hit, RankedList


def reciprocal_rank_fusion(
    rankings: Sequence[RankedList],
    k: int = 60,
    top_k: int | None = None,
) -> RankedList:
    """Fuse multiple ranked lists into one by reciprocal rank.

    Only rank position is used; the per-list scores are ignored by design. Ties in the
    fused score are broken by doc_id for determinism (important for reproducible eval).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [Hit(doc_id=doc_id, score=score) for doc_id, score in ordered]
