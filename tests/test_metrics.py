"""Evaluation metric wrapper tests against known qrel/run pairs (PR-10 / NFR-5).

These pin the pytrec_eval wrapper's output to values computable by hand, guarding both
the measure-name mapping and the aggregation.
"""

import math

from productrank.evaluation.metrics import evaluate


def test_perfect_ranking():
    # The single relevant doc is ranked first → every top-heavy metric is maximal.
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d1": 2.0, "d2": 1.0, "d3": 0.5}}
    res = evaluate(qrels, run)
    agg = res.aggregate
    assert math.isclose(agg["ndcg_cut_10"], 1.0)
    assert math.isclose(agg["recip_rank"], 1.0)
    assert math.isclose(agg["map"], 1.0)
    assert math.isclose(agg["recall_10"], 1.0)
    assert math.isclose(agg["P_10"], 0.1)  # 1 relevant in top 10 / 10


def test_relevant_at_rank_two():
    # Relevant doc at position 2 → MRR = 1/2; NDCG discounted by log2(2+1)=log2(3).
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d2": 2.0, "d1": 1.0}}
    res = evaluate(qrels, run)
    assert math.isclose(res.aggregate["recip_rank"], 0.5)
    # NDCG@10 = (1/log2(3)) / (1/log2(2)) = 1/log2(3)
    assert math.isclose(res.aggregate["ndcg_cut_10"], 1.0 / math.log2(3), rel_tol=1e-6)


def test_aggregate_is_mean_over_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}
    run = {
        "q1": {"d1": 2.0},  # MRR 1.0
        "q2": {"dx": 2.0, "d9": 1.0},  # MRR 0.5
    }
    res = evaluate(qrels, run)
    assert res.num_queries == 2
    assert math.isclose(res.aggregate["recip_rank"], (1.0 + 0.5) / 2)


def test_labeled_aggregate_keys():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d1": 1.0}}
    labeled = evaluate(qrels, run).labeled_aggregate()
    assert "NDCG@10" in labeled and "MRR" in labeled and "Recall@100" in labeled
