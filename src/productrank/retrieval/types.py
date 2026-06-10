"""Shared retrieval types.

A `Hit` is one ranked result: a document id and the score from whatever stage
produced it (ts_rank_cd for sparse, cosine similarity for dense, RRF score for fusion,
cross-encoder logit for rerank). Ranked lists are plain `list[Hit]` in descending
score order — simple, ordered, and trivially fusable.
"""

from __future__ import annotations

from typing import NamedTuple


class Hit(NamedTuple):
    doc_id: str
    score: float


RankedList = list[Hit]
