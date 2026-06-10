"""IR metrics via pytrec_eval (the standard trec_eval binding).

Using the field-standard tool rather than hand-rolled metrics is itself a credibility
signal (ARCHITECTURE §3) — and avoids the subtle bugs that DIY NDCG/MAP implementations
are notorious for (gain formula, tie handling, ideal-DCG normalization).

NDCG is the headline metric: it rewards relevant results *and* rewards them appearing
near the top via a logarithmic positional discount. MRR is reported alongside (not
instead), because MRR only sees the first relevant result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytrec_eval

# trec_eval measure spec → the per-query keys it emits.
_MEASURES = {"ndcg_cut.10,100", "recip_rank", "map", "recall.10,100", "P.10"}

# Ordered, human-friendly view of the metrics we surface.
METRIC_KEYS: list[str] = [
    "ndcg_cut_10",
    "ndcg_cut_100",
    "map",
    "recip_rank",
    "recall_10",
    "recall_100",
    "P_10",
]

METRIC_LABELS: dict[str, str] = {
    "ndcg_cut_10": "NDCG@10",
    "ndcg_cut_100": "NDCG@100",
    "map": "MAP",
    "recip_rank": "MRR",
    "recall_10": "Recall@10",
    "recall_100": "Recall@100",
    "P_10": "P@10",
}

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


@dataclass
class EvalResult:
    per_query: dict[str, dict[str, float]]  # qid → {metric_key: value}
    aggregate: dict[str, float]  # metric_key → mean over queries
    num_queries: int

    def labeled_aggregate(self) -> dict[str, float]:
        return {METRIC_LABELS[k]: self.aggregate.get(k, 0.0) for k in METRIC_KEYS}


def evaluate(qrels: Qrels, run: Run) -> EvalResult:
    """Compute metrics for a single run against qrels.

    qrels: {query_id: {doc_id: relevance_int}}
    run:   {query_id: {doc_id: score_float}}  (only ranked docs need appear)
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, _MEASURES)
    per_query_raw = evaluator.evaluate(run)

    per_query: dict[str, dict[str, float]] = {
        qid: {k: float(metrics.get(k, 0.0)) for k in METRIC_KEYS}
        for qid, metrics in per_query_raw.items()
    }

    n = len(per_query) or 1
    aggregate = {k: sum(q[k] for q in per_query.values()) / n for k in METRIC_KEYS}
    return EvalResult(per_query=per_query, aggregate=aggregate, num_queries=len(per_query))
