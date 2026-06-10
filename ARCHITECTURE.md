# ProductRank — Technical Architecture Document

**Status:** Draft v1
**Owner:** Aditya Pimpalkar
**Companion to:** PRD.md, TICKETS.md
**Last updated:** June 2026

> This is the single most important interview artifact. Interviewers spend more time on *why you built it this way* than on reading code. Every decision below carries its reasoning and the alternatives considered. The "Deliberately Not Built" section (§10) is as important as everything else.

---

## 1. System Overview

ProductRank is a multi-stage retrieval and ranking service with an evaluation engine, exposed through a FastAPI backend and a Next.js frontend, backed by a single PostgreSQL instance (sparse + dense retrieval) and Redis (caching + job results).

```
                    ┌─────────────────────────────┐
                    │      NEXT.JS FRONTEND       │
                    │  Side-by-side comparison    │
                    │  Experiment runner          │
                    │  Metrics view               │
                    └──────────────┬──────────────┘
                                   │ HTTP / JSON
                    ┌──────────────▼──────────────┐
                    │      FASTAPI BACKEND        │
                    │                             │
                    │  /v1/search   (4 variants)  │
                    │  /v1/experiments (A/B)      │
                    │  /v1/products/:id           │
                    │  /metrics  (Prometheus)     │
                    │  /health                    │
                    └───┬───────────┬──────────┬──┘
                        │           │          │
              ┌─────────▼──┐  ┌─────▼─────┐  ┌─▼────────┐
              │ ParadeDB   │  │   Redis   │  │  OpenAI  │
              │ (Postgres) │  │           │  │   API    │
              │ pgvector   │  │ embed +   │  │ (embed   │
              │  (dense)   │  │ result +  │  │  only)   │
              │ pg_search  │  │ rerank    │  └──────────┘
              │ BM25(sparse)│ │ cache +   │
              │ + qrels    │  │ job state │
              └────────────┘  └─────┬─────┘
                                    │
                            ┌───────▼────────┐
                            │ Eval worker    │
                            │ (async A/B runs)│
                            └────────────────┘

         Cross-encoder rerank: in-process (sentence-transformers)
         Deployment: Postgres + Redis + API + frontend
```

**Single database is a deliberate choice.** PostgreSQL handles both retrieval modes: `pgvector` for dense vector similarity, and a **`pg_search` BM25 index (ParadeDB)** for the sparse baseline. This removes Elasticsearch entirely — one fewer service to run, no separate hosting headache, and a local stack that boots in under a minute.

> **Implementation note (revised from the original plan).** The first cut used stock Postgres full-text search (`ts_rank_cd` over a `tsvector`). Measured on FiQA it scored **NDCG@10 ≈ 0.06 vs. the published BEIR BM25 ≈ 0.236** — a 4× gap — because stock FTS has **no IDF term weighting**: a document matching a common word ("tax") ranks as highly as one matching the rare, decisive term ("EWU"). That gap failed NFR-3 (baseline fidelity) *and* capped hybrid recall. The fix was to adopt ParadeDB's `pg_search`, a Tantivy-backed BM25 index that does real IDF weighting, while staying inside a single Postgres instance (ParadeDB *is* Postgres + `pg_search` + `pgvector`). The measured BM25 baseline is now **NDCG@10 = 0.239** on FiQA, essentially matching the published number. This is the "Lucene/Tantivy upgrade path" the original doc named — taken early, because credible baselines are the project's whole point. A Lucene/Elasticsearch deployment remains the horizontal-scale path beyond a single node.
>
> The image is `paradedb/paradedb:0.15.26-pg17`. Confirmed in the running container: `SELECT extname FROM pg_extension` returns **both `pg_search` and `vector`**, and the `documents` table carries a `bm25` index *and* an `ivfflat` cosine index simultaneously — so the "single Postgres, two retrieval modes" claim is verified, not aspirational.

---

## 2. The Retrieval Pipeline (the core)

```
Query
  │
  ├─► Sparse retrieval (pg_search BM25)      ─┐
  │                                          ├─► RRF fusion ─► top-100 ─► Cross-encoder rerank ─► final
  └─► Dense retrieval (pgvector + embedding) ─┘
```

### 2.1 Why two stages at all
Candidate generation → reranking exists because of **cost asymmetry**. The accurate model (a cross-encoder that reads query and document *together*) is too slow to run over the whole corpus. So stage 1 cheaply narrows millions → ~100 using fast methods; stage 2 spends the expensive compute only on the survivors. This is the canonical pattern in production search, ads, and recsys.

### 2.2 Why hybrid (sparse + dense)
Lexical and semantic retrieval have **complementary failure modes**. Sparse (keyword) retrieval is excellent for exact-match and rare terms (model numbers, names) but blind to synonyms. Dense (embedding) retrieval captures meaning ("couch" ≈ "sofa") but adds noise on exact-match/identifier queries. Running both and fusing recovers each method's strengths. Hybrid wins overall precisely because where one fails, the other tends to succeed.

### 2.3 Why RRF (and not score addition)
BM25/`ts_rank` scores are unbounded and corpus-dependent; cosine similarities are 0–1. **Adding them directly is meaningless** — incompatible scales. Reciprocal Rank Fusion sidesteps this by fusing on *rank position*: `score(d) = Σ 1/(k + rank_i(d))` across rankings, with `k` a smoothing constant (default 60). It needs no score normalization, is robust, and is the standard hybrid-fusion choice. Alternative considered: min-max or z-score normalization then linear interpolation — more tunable but brittle and dataset-sensitive.

### 2.4 Why a cross-encoder for rerank (not the bi-encoder again)
Stage 1 dense retrieval uses a **bi-encoder**: query and document are embedded *independently*, so vectors can be precomputed and searched with an ANN index — fast, scalable, less accurate. A **cross-encoder** feeds query+document *together* through the model, so it models their interaction directly — much more accurate, but quadratically expensive and impossible to precompute. That cost is exactly why it's confined to the top-100. Right tool, right stage.

### 2.5 Why IVFFlat (pgvector index)
At ~57K vectors, `IVFFlat` gives fast approximate cosine search with simple tuning (`lists`, `probes`) and cheap build time. HNSW offers better recall-latency at scale but with heavier memory and build cost — overkill here and worth naming as the scale-up path. The index choice is a recall/latency/memory tradeoff, stated as such.

---

## 3. Evaluation Engine

The differentiator. Search → results → **measurement**.

- **Library:** `pytrec_eval` (Python binding to `trec_eval`, the IR-standard evaluation tool). Using the standard tool — not hand-rolled metrics — is itself a credibility signal.
- **Metrics:** NDCG@10, NDCG@100, MRR, MAP, Recall@10, Recall@100, Precision@10.
- **Why NDCG is the headline:** it rewards relevant results *and* rewards them appearing near the top, applying a logarithmic positional discount. Accuracy is the wrong frame for ranking; NDCG is the right one.
- **Why MRR alone is insufficient:** MRR only cares about the first relevant result's position — misleading when you care about the whole ranked list. Reported alongside, not instead of, NDCG.
- **Ground truth:** real qrels shipped with BEIR — FiQA and (sampled) MS MARCO. No invented labels, no LLM judging in the MVP. See §3.1 for how each dataset is used.
- **Significance:** paired t-test (or bootstrap confidence interval) on per-query metrics between two variants, so a lift isn't reported when it's within noise.
- **Failure analysis:** surface the queries where each variant underperforms — diagnostic value and great interview material ("here's a query where dense loses to BM25, and why").

### 3.1 Datasets & measured results

Two datasets, used for **different jobs**. Read the labels carefully — one is an honest absolute baseline, the other is reported for *relative deltas only*.

**FiQA (`test`, full corpus) — the literature-comparable absolute baseline.**
57,638 documents, 648 queries, full BEIR corpus. BM25 here lands at the published number, so absolute scores are trustworthy and comparable to the literature.

| FiQA `test` — full 57,638-doc corpus · **literature-comparable** | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank |
|---|---|---|---|---|
| **NDCG@10** | **0.2391** (pub. BM25 ≈ 0.236 ✓) | **0.4483** | 0.3677 | 0.3754 |
| NDCG@100 | 0.2938 | 0.5184 | 0.4499 | 0.4510 |
| MRR | 0.3083 | 0.5295 | 0.4516 | 0.4547 |

**MS MARCO (`dev`, sampled) — PRIMARY for variant-to-variant deltas, NOT leaderboard-comparable.**

> ⚠️ **Sampled MS MARCO (51,070 docs: every judged answer present + ~50K file-order distractors, no hard negatives) — report DELTAS between variants only. Absolute values are inflated by an easy corpus and are NOT comparable to full-benchmark (8.8M-passage) MS MARCO leaderboards.** ~1.07 relevant docs/query (binary). Built by `data/ingest/msmarco.py`: it keeps all relevant passages and adds distractors in file order — so the answer is always present and no per-query hard negatives compete.

| MS MARCO `dev` — sampled 51K · ⚠️ **deltas only, not leaderboard-comparable** | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank |
|---|---|---|---|---|
| **NDCG@10** | 0.7460 | 0.8999 | 0.8420 | **0.9413** |
| MRR | 0.7275 | 0.8891 | 0.8245 | **0.9310** |
| Recall@10 | 0.8357 | 0.9462 | 0.9218 | **0.9790** |
| Recall@100 | 0.9338 | 0.9605 | **0.9930** | 0.9930 |

All MS MARCO deltas are statistically significant (paired t-test + bootstrap CI, n=1000): Hybrid→Rerank +0.099 (p=9e-30) and Dense→Rerank +0.041 (p=3e-9).

### 3.2 Named result — reranker domain transfer

The headline empirical finding, and the reason both datasets exist:

- **In-domain (sampled MS MARCO):** the `ms-marco-MiniLM-L-6-v2` cross-encoder lifts NDCG@10 from the hybrid candidate set **0.842 → 0.941** — the textbook candidate-generation → rerank gain, beating even pure dense (0.900).
- **Out-of-domain (FiQA, financial):** the *same* reranker **degrades** results, **0.448 (dense) → 0.375 (rerank)**.

**Why:** the cross-encoder was trained on MS MARCO web-search relevance, so it transfers well to in-domain queries and underperforms a strong general embedding (`text-embedding-3-small`) on out-of-domain financial text. This is empirical evidence that **reranker domain match matters** — a finding about transfer, *not* a tooling mistake. (Confirmed not a candidate-pool artifact: `scripts/diagnose_rerank.py` shows that even reranking dense's *own* top-100 on FiQA, the cross-encoder still loses to dense.) On MS MARCO the dense > hybrid gap is the known RRF dilution effect — fusion lifts Recall@100 (0.961 → 0.993) but an equal-weight blend with the ~20-point-weaker BM25 nudges the single relevant doc just below rank 1; the reranker then reorders that richer pool back to the top.

---

## 4. Backend Design

- **Framework:** FastAPI. Async-native, Pydantic validation, auto-generated OpenAPI/Swagger.
- **Language choice (Python):** chosen for the IR ecosystem — `sentence-transformers`, `pytrec_eval`, BEIR loaders, cross-encoders are all first-class in Python and poor or absent elsewhere. See §11 for how the Java requirement on target roles is addressed.
- **Module layout:**
  - `retrieval/` — `sparse.py`, `dense.py`, `fusion.py` (RRF), `rerank.py` (cross-encoder)
  - `evaluation/` — `metrics.py` (pytrec_eval wrapper), `significance.py`
  - `services/` — orchestration that composes variants and records per-stage timing
  - `cache.py` — Redis layer
  - `observability/` — Prometheus instrumentation, structlog config
  - `routers/` — `search`, `experiments`, `products`, `health`
- **API surface (versioned, `/v1`):**
  - `POST /v1/search` — body: query, variant, top_k → ranked results + per-stage latency + candidate counts
  - `POST /v1/experiments` — body: variant_a, variant_b, query_set_size → job id
  - `GET /v1/experiments/{id}` — job status + metrics table + significance
  - `GET /v1/products/{id}` — document detail
  - `GET /metrics` — Prometheus scrape
  - `GET /health` — liveness
- **Validation & limits:** Pydantic request models; `slowapi` rate limiting; correct HTTP status codes; CORS scoped to the known frontend origin.

---

## 5. Data & Indexing Pipeline

- **Ingestion:** one-command seed loads the FiQA corpus, queries, and qrels into Postgres.
- **Sparse index:** `pg_search` BM25 index (ParadeDB) over `documents(title, text)`, maintained incrementally on insert.
- **Dense index:** batch-embed documents via OpenAI `text-embedding-3-small` (1536d); store in `pgvector` column; build `IVFFlat` cosine index.
- **Batching & idempotency:** embedding runs in batches to respect API rate limits; re-runs are idempotent/resumable (skip already-embedded docs) so a failure mid-ingest doesn't force a restart.
- **Reproducibility:** `seed.py` brings a clean environment to a fully indexed, queryable state in one command (NFR-4).

---

## 6. Caching Layer (Redis)

| Cached item | Key shape | TTL | Why |
|---|---|---|---|
| Query embedding | `emb:{hash(query)}` | long | Re-embedding a repeated query is a wasted API call + ~200ms |
| Result set (hot queries) | `res:{variant}:{hash(query)}` | short | Hot-query latency; short TTL bounds staleness |
| Rerank results (demo queries) | `rr:{hash(query)}` | pre-warmed | Sub-second demo (NFR-1) |

Interview talking points this enables: cache invalidation strategy, TTL tradeoffs, memory vs. latency, why embeddings get a long TTL (deterministic for a fixed model) while result sets get a short one (index can change).

---

## 7. Asynchronous Jobs

A/B evaluation over hundreds–thousands of queries is a batch workload, not a request-path operation. It runs as a **background job** with status polling and results stored in Redis. Async is **scoped strictly to eval runs** — the search request path stays synchronous and simple.

**Honest scoping note for interviews:** for this scale a background task (FastAPI `BackgroundTasks` or a single worker process) is sufficient; a full Celery + broker topology is unnecessary. In production with concurrent users and long eval runs, this is exactly where a Celery queue belongs — *here is the boundary where I would introduce it.* Demonstrating *when* a queue is warranted is stronger signal than building one that isn't.

---

## 8. Observability (posture, not full stack)

- **In code:** per-stage timers (retrieval / fusion / rerank / end-to-end) recording p50/p95/p99; structured JSON logs with correlation IDs (`structlog`); a `/metrics` endpoint in Prometheus exposition format.
- **Not stood up locally:** Prometheus server + Grafana dashboards. Documented as the production posture: "`/metrics` is scraped by Prometheus and rendered in Grafana; in a portfolio context I record and surface percentiles directly because standing up the full stack adds operational surface without changing what the project demonstrates."
- **Why this is the senior move:** it shows knowledge of the production pattern plus a reasoned scoping decision — stronger than wiring Grafana to prove tutorial-following. For the target ad-retrieval domain, latency percentiles per stage are directly on-topic (ad pages assemble under tight latency budgets), so latency is treated as a first-class story, not an afterthought.

---

## 9. Security & Access (section, not separate doc)

ProductRank is a single-operator demo with **no real user data, no authentication of end users, and a public read-only demo surface**. A standalone security document describing auth flows the system doesn't have would misrepresent it. Instead, the honest security posture:

### 9.1 What the project actually does
- **No PII / no user accounts.** The corpus is a public academic dataset (FiQA). There is no personal data to protect, no login, no user-generated content stored.
- **Secrets management.** The only secret is the OpenAI API key, supplied via environment variable (`.env`, git-ignored; `.env.example` committed). Never hard-coded, never logged. On the deploy host it lives in the platform's secret store.
- **Input validation.** All request bodies validated via Pydantic; query length bounded; pagination/`top_k` capped to prevent abuse of the embedding/rerank path.
- **Rate limiting.** `slowapi` per-IP limits on `/v1/search` and `/v1/experiments` to protect the OpenAI budget and the reranker from a public demo being hammered.
- **CORS.** Restricted to the known frontend origin; not wildcarded in the deployed config.
- **Cost-control as a security concern.** Because a public demo can incur real OpenAI spend, the embedding path is cache-first and rate-limited; demo queries are pre-computed so the common path costs nothing.
- **Dependency hygiene.** Pinned dependencies; CI runs a vulnerability scan (e.g. `pip-audit`).
- **No secrets in client.** The frontend never holds the OpenAI key; all model calls are server-side only.

### 9.2 What production would add (named, not built)
- **AuthN/AuthZ:** API keys or OAuth for write/experiment endpoints; per-tenant isolation if multi-user.
- **Network:** private subnets for Postgres/Redis, security groups, TLS everywhere (the deploy platform terminates TLS by default).
- **Data:** encryption at rest for the database, encrypted connections (already standard on managed Postgres).
- **Audit & abuse:** request audit logs, anomaly detection on the experiment endpoint, WAF in front of the public surface.
- **Least privilege:** scoped IAM/service-account roles for the deploy, separate read vs. migrate database roles.

Stating this boundary — "here's the real posture, here's what production adds and why" — is itself the senior security answer.

---

## 10. Deliberately Not Built (read this in the interview)

| Not built | Why | One-line interview answer |
|---|---|---|
| Ad allocation / auction | Needs bid + pCTR + auction data | "My reranker scores relevance; a real ad system multiplies by bid × pCTR and allocates under a latency budget — here's where that fits." |
| Personalization | Synthetic users can't be honestly evaluated | "Real personalization needs real interaction data and online eval; I built the retrieval foundation it sits on, which I *can* measure." |
| Custom-trained models | Weeks of work, not the point | "Pretrained cross-encoder gives ~90% of the signal; training a two-tower model is a different project." |
| Elasticsearch | One DB suffices at this scale | "ParadeDB (`pg_search` BM25 + `pgvector`) covers both retrieval modes in one Postgres; ES is the upgrade path if I need horizontal scale across nodes." |
| Celery/full queue | Overkill for batch eval at this scale | "Background task is enough now; Celery belongs here when eval runs are concurrent and long." |
| Grafana stack | Adds ops surface, not understanding | "`/metrics` is Prometheus-ready; I surface percentiles directly for a portfolio build." |
| LLM-as-judge | Dataset has real labels | "FiQA ships qrels; LLM judging is the fallback for label-free domains." |

---

## 11. The Java Question (target-role alignment)

Several target roles (e.g., Amazon Ads SDE) list **Java/C++/C# and distributed systems** in *basic* qualifications, while listing **information retrieval / ML** in *preferred* qualifications. ProductRank is Python because that's where the IR ecosystem lives.

**Coverage plan, stated openly:** ProductRank demonstrates the IR/ranking and systems-design muscle (preferred quals + the role's actual day-to-day). The Java/distributed-systems requirement is covered by (a) ~3 years building Java/Spring Boot microservices at Cognizant and (b) the langchain4j Java OSS contribution. The narrative is: *Python project proves the retrieval depth; existing Java experience proves the language requirement.* The alternative — rebuilding the retrieval service in Java/Spring Boot (Lucene + ONNX cross-encoder) — is a precise JD match but fights the tooling and risks an unfinished project; deliberately not chosen.

---

## 12. Frontend Specification (section, not separate doc)

The frontend is small — three pages — so it lives here rather than as its own document.

### 12.1 Stack
Next.js (App Router) + Tailwind + shadcn/ui + Recharts. Server components for data fetching where possible; all model/secret access stays server-side.

### 12.2 Page 1 — Search Comparison (the LinkedIn screenshot)
- **Query bar** with a row of **pre-filled example-query buttons** (a cold visitor won't invent a good query; NFR-1/FR-10).
- **Four result columns:** BM25, Dense, Hybrid, Hybrid+Rerank. Each column shows ranked product/document cards with a **score badge** and the column's **latency tag**.
- **Divergence highlighting (FR-6):** when the same document ranks very differently across columns, it's visually linked/highlighted — "#3 here, #47 there." This divergence is the demo's whole point; it must be impossible to miss.
- **Per-variant metric chips** above each column (NDCG@10 for the current query if labeled).

### 12.3 Page 2 — Experiment Runner
- Pick **Variant A** and **Variant B**, choose query-set size, hit Run.
- **Progress indicator** while the background job runs (FR-8).
- **Side-by-side metrics table** on completion: NDCG@10/@100, MRR, Recall@K, with **significance markers** (e.g. ✱ when the paired test clears the threshold). This table is the second key screenshot.

### 12.4 Page 3 — Analytics / Dashboard
- All-time metrics summary across variants; latency histograms (p50/p95/p99 per stage); top query categories. A bar chart of NDCG-by-variant is the at-a-glance "this person built ranking systems" visual.

### 12.5 Design constraints driven by the demo
- **Sub-second warm path** (NFR-1): pre-warmed/pre-computed demo queries.
- **Example queries chosen to show divergence** — including at least one where dense *loses* to BM25, for the interview teaching moment.
- **No client-side secrets.** All retrieval calls proxy through the backend.

### 12.6 Out of scope (frontend)
User accounts, saved searches, theming, mobile-first polish. The UI exists to make ranking quality *visible*, not to be a product.

---

## 13. Testing Strategy

- **Unit:** RRF math (fixed-input expected output), metric wrappers (against known qrel/run pairs), sparse/dense retrieval paths.
- **Integration:** full pipeline against a tiny fixture corpus.
- **E2E:** run the pipeline on a small labeled set and **assert NDCG exceeds a known threshold** — guards against silent ranking regressions.
- **CI:** GitHub Actions — lint, type-check, test on push.

---

## 14. Deployment

- **Database: self-hosted ParadeDB container, not a managed-Postgres add-on.** `pg_search` is a ParadeDB-specific extension and is **not available on managed Postgres offerings** (Railway/Fly Postgres, AWS RDS, Supabase, Neon) — they only expose a fixed extension allowlist (pgvector is common; pg_search is not). Because the sparse stage depends on `pg_search`, the database must run as the `paradedb/paradedb` Docker image. Fly.io and Railway can both run arbitrary container images, so this is straightforward: deploy ParadeDB as a container with a persistent volume, rather than provisioning their managed-Postgres product. (If a managed add-on were a hard requirement, the fallback is moving the sparse stage to a separate Lucene/Elasticsearch service — the horizontal-scale path in §10 — but that reintroduces the second service ParadeDB was chosen to avoid.)
- Managed Redis + API container on a PaaS (Railway/Fly); frontend on Vercel.
- TLS terminated by the platform; secrets in the platform secret store.
- Cross-encoder runs in-process with a small model (`ms-marco-MiniLM-L-6-v2`); demo-query rerank results pre-computed so the public path is instant and cheap (NFR-1, §9.1 cost-control).
- One-command local stack via Docker Compose (ParadeDB + Redis + API + frontend) booting in under a minute (NFR-2).

### 14.1 Operational note — Apple Silicon (local dev)

On macOS/arm64, `sentence-transformers` auto-selects the **MPS (Metal) backend for the cross-encoder, which deadlocks**: the rerank process hangs in an uninterruptible `metal gpu stream` wait with no progress (diagnosed via `sample`; it is *not* an out-of-memory or model-download issue). The rerank path therefore **pins to CPU** (`RERANK_DEVICE=cpu`, plus `torch.set_num_threads(1)` to dodge the Accelerate/OpenMP fork deadlock). Set `HF_HUB_OFFLINE=1` once the model is cached so a long batch eval doesn't stall on an HF Hub network check. Warm CPU rerank is ~1.1s/query — well within the demo budget. On Linux/CUDA deploy targets this constraint does not apply.
