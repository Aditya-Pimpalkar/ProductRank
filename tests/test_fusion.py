"""RRF unit tests with hand-computed expected fused order (PR-07 DoD).

With k=60 and two rankings:
    A = [d1, d2, d3]   (ranks 1,2,3)
    B = [d2, d3, d4]   (ranks 1,2,3)

    score(d1) = 1/61                  = 0.016393
    score(d2) = 1/62 + 1/61          = 0.032522   ← appears high in both → wins
    score(d3) = 1/63 + 1/62          = 0.032002
    score(d4) = 1/63                  = 0.015873

Expected fused order: [d2, d3, d1, d4].
"""

from productrank.retrieval.fusion import reciprocal_rank_fusion
from productrank.retrieval.types import Hit


def _ranked(ids):
    # scores are intentionally arbitrary — RRF must ignore them and use rank only.
    return [Hit(doc_id=i, score=999.0) for i in ids]


def test_rrf_hand_computed_order():
    a = _ranked(["d1", "d2", "d3"])
    b = _ranked(["d2", "d3", "d4"])
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert [h.doc_id for h in fused] == ["d2", "d3", "d1", "d4"]


def test_rrf_scores_match_formula():
    a = _ranked(["d1", "d2", "d3"])
    b = _ranked(["d2", "d3", "d4"])
    fused = {h.doc_id: h.score for h in reciprocal_rank_fusion([a, b], k=60)}
    assert fused["d2"] == 1 / 62 + 1 / 61
    assert fused["d3"] == 1 / 63 + 1 / 62
    assert fused["d1"] == 1 / 61
    assert fused["d4"] == 1 / 63


def test_rrf_ignores_per_list_scores():
    # Flipping the (ignored) scores must not change the rank-based fusion result.
    a = [Hit("d1", 0.001), Hit("d2", 0.002)]
    b = [Hit("d2", 1000.0), Hit("d1", 0.0)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    # d1: 1/61 + 1/62 ; d2: 1/62 + 1/61  → equal scores, tie broken by doc_id asc.
    assert [h.doc_id for h in fused] == ["d1", "d2"]


def test_rrf_top_k_truncation():
    a = _ranked(["d1", "d2", "d3", "d4"])
    fused = reciprocal_rank_fusion([a], k=60, top_k=2)
    assert [h.doc_id for h in fused] == ["d1", "d2"]


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
