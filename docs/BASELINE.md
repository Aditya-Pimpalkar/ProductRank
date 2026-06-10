# BM25 Baseline Sanity Check

**Dataset:** BEIR / FiQA-2018, `test` split (648 queries, 57,638 documents).
**Eval tool:** `pytrec_eval` (trec_eval binding).

## Why this check exists

The BM25 baseline should land within range of the published BEIR FiQA baseline. A baseline
that matches the literature confirms the retrieval and evaluation harness are wired
correctly; a baseline that doesn't indicates a bug in one of them.

## Published reference

The BEIR paper / leaderboard reports **BM25 (Anserini/Lucene) NDCG@10 ≈ 0.236** on FiQA.

## Measured (ProductRank, pg_search BM25)

| Metric | Measured | Published BM25 | Note |
|---|---|---|---|
| NDCG@10 | **0.2391** | ~0.236 | within range ✓ |
| NDCG@100 | 0.2932 | — | |
| MAP | 0.1900 | — | |
| MRR | 0.3072 | — | |
| Recall@10 | 0.3002 | — | |
| Recall@100 | 0.5100 | ~0.539 | within range ✓ |
| P@10 | 0.0665 | — | |

## How the baseline was corrected

The first implementation used **stock Postgres FTS** (`ts_rank_cd` over a `tsvector`).
It measured **NDCG@10 ≈ 0.06** — a 4× miss. Root cause: stock FTS has **no IDF term
weighting**, so a document matching the common word "tax" scores as highly as one
matching the rare, decisive term ("EWU"). Two compounding bugs were found along the way:

1. `websearch_to_tsquery` defaults to **AND** semantics, so a relevant doc missing one
   query term was never retrieved — fixed by rewriting the tsquery to OR.
2. Even with OR, the **missing IDF** left rare-term docs buried under common-term docs.

The fix that actually closed the gap was switching the sparse stage to **ParadeDB's
`pg_search`** (Tantivy-backed BM25 with real IDF), staying inside a single Postgres
instance. The baseline then matched the published number almost exactly. See the
[README](../README.md#high-level-design) for the full decision record.

## Variant comparison

See the results table in the [README](../README.md#key-results) and
`results/eval_test.json` for the full four-variant comparison
(BM25 / Dense / Hybrid / Hybrid+Rerank).
