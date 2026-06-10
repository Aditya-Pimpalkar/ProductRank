# ProductRank — Product Requirements Document

**Status:** Draft v1
**Owner:** Aditya Pimpalkar
**Type:** Portfolio / learning project (single-engineer, single-operator)
**Last updated:** June 2026

---

## 1. Purpose & Honest Framing

ProductRank is an **evaluation-first retrieval and ranking platform**. It implements the multi-stage retrieval pattern that powers production search, ads, and recommendation systems — candidate generation followed by reranking — and, critically, it *measures* the quality of each stage against ground-truth relevance labels using standard information-retrieval (IR) metrics.

This is a portfolio and learning project, not a production service. There are no real users, no SLAs, and no on-call rotation. Where this document uses the word "production," it means **production-grade architecture** — built the way real systems are built (separation of concerns, evaluation, caching, observability hooks, tests, deployment) — not literal production traffic. This distinction is deliberate and is restated in interviews so that claims survive scrutiny.

### Why this project exists
1. **Deep understanding** of how retrieval and ranking actually work, beyond "wrap an LLM and call it RAG."
2. **Resume and interview signal** for search, recommendations, ads-ranking, GenAI-retrieval, and backend/distributed-systems roles.
3. **Portfolio / LinkedIn asset** — a live demo and a teaching narrative with measurable results.

The project is explicitly *not* built for a single job application. It is the flagship systems project intended to remain relevant across an entire category of roles.

---

## 2. Problem Statement

When a user issues a query against a large corpus, the system must decide which results to show and in what order. Ranking quality is the difference between a useful result at position 1 and a useless one — and most portfolio "AI search" projects never measure whether their ranking is actually good.

ProductRank solves two problems at once:
- **Retrieval quality:** Demonstrate that a multi-stage hybrid pipeline beats naive single-method retrieval.
- **Provable quality:** Show *with numbers* that each stage earns its place, rather than asserting it.

---

## 3. Goals & Non-Goals

### 3.1 Goals
- Implement four retrieval variants: BM25 (sparse), dense vector, hybrid via Reciprocal Rank Fusion (RRF), and hybrid + cross-encoder rerank.
- Evaluate all variants against a labeled IR dataset using NDCG@10, NDCG@100, MRR, Recall@10, Recall@100, Precision@10.
- Provide an A/B experiment runner that compares two variants across a query set and reports metrics with statistical significance.
- Expose a side-by-side comparison UI that makes ranking differences visible at a glance.
- Track per-stage latency (p50/p95/p99) and expose a metrics endpoint.
- Deploy a live, pre-warmed demo with example queries.
- Produce an architecture document explaining every design decision, including what was deliberately not built.

### 3.2 Non-Goals (deliberately out of scope)
- **Ad allocation / auction mechanics.** ProductRank ranks by *relevance*. A real ad system multiplies relevance by bid and predicted click-through-rate under an allocation constraint. This boundary is named explicitly; it is not built, because it requires auction data the project does not have.
- **Personalization from real user behavior.** Reranking by synthetic click history would create an evaluation question that cannot be answered honestly (how do you measure success on invented users?). Personalization is named as a future extension, not built.
- **Training custom embedding or ranking models.** The project uses pretrained models (OpenAI embeddings, a pretrained cross-encoder). Training two-tower models is weeks of work and not the point.
- **Hyper-scale.** Runs on a ~57K-document corpus. The architecture is correct; the scale is honestly stated as demo-scale.
- **LLM-as-judge for relevance.** The chosen dataset ships with real relevance labels, so judging is unnecessary. Named as a fallback for label-free domains, not built in the MVP.

---

## 4. Target Users (of the demo, not a product)

| Audience | What they need from it | Primary artifact |
|---|---|---|
| Recruiters / hiring screens | Instant proof the candidate built ranking systems | Live demo + LinkedIn screenshot |
| Interviewers (technical loop) | Defensible design decisions and IR fluency | Architecture doc + live teaching queries |
| The builder (Aditya) | Genuine understanding of retrieval internals | The codebase itself + eval numbers |

---

## 5. Dataset Decision

**Chosen: BEIR / FiQA-2018** (financial-domain question-answering retrieval).

- ~57K documents, ~6.6K queries, with relevance judgments (qrels) shipped with the dataset.
- Small enough to embed affordably on a personal budget.
- Real qrels → NDCG and friends are computed honestly, not invented.
- Published BEIR baselines exist → the BM25 baseline can be sanity-checked against the literature ("my baseline matches the published number"), which is a credibility signal in interviews.

**Rejected: Amazon Reviews 2023.** Thematically closer to "Amazon Ads / product retrieval," but it ships **no relevance qrels**. Since the evaluation engine is the entire differentiator, a dataset with no ground truth would gut the project's core claim. Thematic fit is worth nothing if nothing can be measured.

**Alternative if FiQA proves too small for a compelling lift story:** a sampled subset of MS MARCO passage ranking. Larger, the academic standard, well-documented eval harness — at the cost of heavier embedding spend.

---

## 6. Functional Requirements

### FR-1 — Multi-variant search
Given a query, the system returns ranked results for any of the four variants. Each response includes: ranked document list, per-result scores, per-stage latency breakdown, and candidate counts at each stage.

### FR-2 — Reciprocal Rank Fusion
Hybrid variant fuses BM25 and dense rankings by rank position (not raw score), with a tunable `k` parameter. Rank-based fusion is required because BM25 scores (unbounded) and cosine similarities (0–1) live on incompatible scales.

### FR-3 — Cross-encoder reranking
The rerank variant takes the top-N (default 100) RRF candidates and rescores them with a pretrained cross-encoder, producing the final order. Reranking runs only on the narrowed candidate set, never the full corpus.

### FR-4 — Evaluation engine
Computes NDCG@{10,100}, MRR, MAP, Recall@{10,100}, Precision@10 via `pytrec_eval` against dataset qrels, for any variant over any query subset.

### FR-5 — A/B experiment runner
User selects two variants and a query set size; system evaluates both, returns a side-by-side metrics table with a paired significance test (paired t-test or bootstrap CI) so differences aren't mistaken for noise. Runs as a background job (see FR-8).

### FR-6 — Side-by-side comparison UI
A query returns four columns (one per variant). Where a document's rank differs sharply between variants, the UI highlights it ("#3 in Hybrid, #47 in BM25"). The divergence *is* the story for both LinkedIn and interviews.

### FR-7 — Caching
Redis caches query embeddings (avoid re-embedding repeats), hot query result sets (short TTL), and rerank results for demo queries. Each cache is independently justifiable; caching is a deliberate performance decision, not decoration.

### FR-8 — Background evaluation jobs
A/B eval over hundreds–thousands of queries runs asynchronously with status updates and stored results, so the API is not blocked. Async is scoped to eval runs only and does not infect the request path.

### FR-9 — Observability hooks
Per-stage latency (p50/p95/p99) tracked and exposed via a `/metrics` endpoint in Prometheus format; structured logs with correlation IDs. Full Prometheus+Grafana wiring is documented as the production posture but not stood up locally.

### FR-10 — Live demo affordances
Deployed URL; a row of pre-filled example queries; pre-warmed / pre-computed rerank results for demo queries so first interaction is sub-second.

---

## 7. Non-Functional Requirements

| # | Requirement | Target | Rationale |
|---|---|---|---|
| NFR-1 | Demo query latency (cached/warm) | < 1 s end-to-end | A cold visitor gives ~8 seconds |
| NFR-2 | Cold local boot | < 1 min, one command | Must come up instantly when an interviewer says "pull it up" |
| NFR-3 | BM25 baseline fidelity | Within range of published BEIR FiQA baseline | Credibility signal |
| NFR-4 | Reproducibility | One-command data seed + index | Demonstrates data-engineering discipline |
| NFR-5 | Test coverage of core logic | RRF math, metrics, retrieval paths unit-tested; one E2E asserting NDCG > threshold | Defensible correctness |
| NFR-6 | Every dependency justifiable | No component without a stated reason | Bloat actively hurts in interviews |

---

## 8. Success Metrics

The project succeeds if:
1. The eval engine produces **real, defensible** NDCG numbers showing a monotonic lift across variants (BM25 < dense or hybrid < hybrid+rerank — exact values to be measured, never invented).
2. The live demo loads a comparison in under a second on a pre-filled query.
3. The architecture doc lets the builder defend every decision, including the deliberate non-goals, without hedging.
4. The resume bullet is written **from measured results**, naming the dataset.
5. At least one LinkedIn post ships with the comparison-table screenshot and the teaching narrative.

> **Resume bullet is authored last, from real numbers.** A defensible measured lift beats an impressive invented one. An interviewer who spots a gap between a claimed figure and the live demo has found a credibility problem where none needed to exist.

---

## 9. Release Plan

- **Milestone 1 — Functional core:** Four variants + eval engine producing first real NDCG numbers over FiQA. Resume-ready.
- **Milestone 2 — Comparison UI + caching:** Side-by-side view with divergence highlighting; Redis caching; latency tracking.
- **Milestone 3 — Experiments + polish:** A/B runner with significance test; background jobs; live deploy with pre-warmed demo; architecture doc; LinkedIn post + Medium draft.

Each milestone is independently shippable. Milestone 1 alone is a portfolio-worthy project.

---

## 10. Open Questions / Risks

- **FiQA lift magnitude:** If the rerank lift on FiQA is undramatic, switch to an MS MARCO sample for a more compelling story. Decide after Milestone 1's first eval.
- **Cross-encoder latency on free-tier deploy:** The reranker is the heavy component. Mitigate by using a small model (`ms-marco-MiniLM-L-6-v2`) and pre-computing rerank for demo queries.
- **Java requirement vs. Python stack:** Target roles (e.g., Amazon Ads SDE) require Java/C++/C# in basic qualifications. ProductRank is Python (best IR ecosystem). The Java requirement is covered by prior Cognizant Spring Boot experience and the langchain4j Java OSS contribution — stated explicitly, not glossed.
