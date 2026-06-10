# ProductRank

**An evaluation-first, multi-stage retrieval and ranking platform.** It implements the
candidate-generation → reranking pipeline used in production search, ads, and
recommendation systems, and measures the quality of every stage against ground-truth
relevance labels using IR-standard metrics.

Four retrieval strategies run over the same query, and the platform reports NDCG, MRR,
MAP, Recall, and Precision for each, with statistical-significance testing on the
differences.

The project runs at demonstration scale (≤57K documents, pretrained models, single node).
The architecture is built the way production systems are; the scale is stated honestly.

<!-- LIVE DEMO: <url> -->
<!-- DEMO GIF: assets/demo.gif -->

---

## Table of contents

- [Key results](#key-results)
- [Reranker domain transfer](#reranker-domain-transfer)
- [What I learned](#what-i-learned)
- [High-level design](#high-level-design)
- [Low-level design](#low-level-design)
- [Evaluation methodology](#evaluation-methodology)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Usage](#usage)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Deployment](#deployment)
- [Design decisions and scope](#design-decisions-and-scope)

---

## Key results

Measured with [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval) (the standard
`trec_eval` binding), with paired t-test + bootstrap confidence intervals on per-query
NDCG@10. Two datasets serve two distinct purposes.

> **Counterintuitive finding:** the *same* cross-encoder dominates in-domain on MS MARCO
> (+0.099 NDCG@10, p=9e-30) yet *degrades* results out-of-domain on FiQA. Domain match
> matters more than reranker quality — see
> [Reranker domain transfer](#reranker-domain-transfer).

### FiQA `test` — full corpus, literature-comparable baseline

57,638 documents, 648 queries (full BEIR FiQA). Absolute scores are trustworthy: BM25
matches the published BEIR baseline.

| Metric | BM25 | Dense | Hybrid (RRF) | Hybrid + Rerank |
|---|---|---|---|---|
| **NDCG@10** | 0.2391 | **0.4483** | 0.3677 | 0.3754 |
| NDCG@100 | 0.2938 | **0.5184** | 0.4499 | 0.4510 |
| MRR | 0.3083 | **0.5295** | 0.4516 | 0.4547 |
| MAP | 0.1908 | **0.3892** | 0.3137 | 0.3153 |
| Recall@100 | 0.5100 | **0.7797** | 0.7425 | 0.7425 |

BM25 = **0.2391** vs. the published BEIR FiQA BM25 of **≈0.236** ([details](docs/BASELINE.md)).

### MS MARCO `dev` — sampled corpus, report deltas only

> ⚠️ **Sampled MS MARCO: 51,070 passages (every judged answer present + ~50K file-order
> distractors, no hard negatives), ~1.07 relevant docs/query.** The task is intentionally
> easier than full-corpus (8.8M-passage) MS MARCO, so these **absolute values are inflated
> and are not comparable to published leaderboards.** Use them for variant-to-variant
> deltas and the in-domain reranker lift only.

| Metric (sampled — deltas only) | BM25 | Dense | Hybrid (RRF) | Hybrid + Rerank |
|---|---|---|---|---|
| **NDCG@10** | 0.7460 | 0.8999 | 0.8420 | **0.9413** |
| MRR | 0.7275 | 0.8891 | 0.8245 | **0.9310** |
| Recall@10 | 0.8357 | 0.9462 | 0.9218 | **0.9790** |
| Recall@100 | 0.9338 | 0.9605 | **0.9930** | 0.9930 |

Every reranker lift is statistically significant (n=1000): Hybrid→Rerank **+0.099**
(p=9e-30), Dense→Rerank **+0.041** (p=3e-9).

---

## Reranker domain transfer

The same `ms-marco-MiniLM-L-6-v2` cross-encoder behaves oppositely on the two datasets:

| | First stage (best) | + Rerank | Effect |
|---|---|---|---|
| **MS MARCO** (in-domain) | 0.842 (hybrid) | **0.941** | reranking dominates |
| **FiQA** (out-of-domain, financial) | 0.448 (dense) | 0.375 | reranking degrades |

The cross-encoder was trained on MS MARCO web-search relevance. It transfers well to
in-domain queries and underperforms a strong general embedding (`text-embedding-3-small`)
on out-of-domain financial text — empirical evidence that reranker domain match matters.

This is not a candidate-pool artifact: `scripts/diagnose_rerank.py` shows that even when
the cross-encoder reranks dense's own top-100 on FiQA, it still loses to dense (0.42 vs
0.49). On MS MARCO, dense edging out hybrid is the expected RRF dilution effect — fusion
raises Recall@100 (0.961 → 0.993), but an equal-weight blend with the ~20-point weaker
BM25 pushes the single relevant doc just below rank 1; the reranker then reorders that
richer pool back to the top.

---

## What I learned

- **IDF weighting is non-negotiable for sparse retrieval** — stock Postgres FTS scored
  NDCG@10 ≈ 0.06 on FiQA, versus 0.239 once the sparse stage used real BM25 (`pg_search`).
- **Reranker domain match matters more than reranker quality** — a strong general embedding
  beat an out-of-domain cross-encoder on FiQA, while the same reranker dominated in-domain
  on MS MARCO.
- **RRF dilutes a dominant signal** — equal-weight fusion with a ~20-point-weaker BM25
  pushed the relevant document below rank 1 until the reranker recovered it.

---

## High-level design

```
                ┌─────────────────────────────┐
                │       NEXT.JS FRONTEND       │
                │  Search comparison (4-way)   │
                │  A/B experiment runner       │
                │  Analytics dashboard         │
                └───────────────┬──────────────┘
                                │ HTTP / JSON  (server-side proxy — no client secrets)
                ┌───────────────▼──────────────┐
                │        FASTAPI BACKEND        │
                │  routers → services → retrieval / evaluation
                │  middleware: CORS · rate limit · correlation-id logging
                └───┬───────────────┬───────┬───┘
                    │               │       │
          ┌─────────▼────────┐  ┌───▼───┐  ┌▼─────────┐
          │     ParadeDB     │  │ Redis │  │  OpenAI  │
          │   (PostgreSQL)   │  │ cache │  │  API     │
          │  pg_search BM25  │  │ + job │  │ (embed   │
          │  pgvector (IVF)  │  │ state │  │  only)   │
          └──────────────────┘  └───────┘  └──────────┘

   Cross-encoder rerank runs in-process (sentence-transformers, CPU).
   Async A/B evaluation runs as a FastAPI BackgroundTask with state in Redis.
```

### Components

| Component | Responsibility |
|---|---|
| **Next.js frontend** | Three pages (comparison, experiments, analytics). Calls the API server-side so no secret or model call reaches the browser. |
| **FastAPI backend** | Request validation, rate limiting, correlation-id logging; composes the retrieval variants; serves search, experiments, results, health, and metrics. |
| **ParadeDB (PostgreSQL)** | Single datastore for both retrieval modes: `pg_search` (BM25) and `pgvector` (dense, IVFFlat). Stores corpus, queries, and qrels. |
| **Redis** | Query-embedding cache, result-set cache, and async A/B job state. Fail-soft — the app runs without it. |
| **OpenAI API** | Document and query embeddings (`text-embedding-3-small`), server-side only. |

### Why a single database

One ParadeDB image (`paradedb/paradedb:0.15.26-pg17`) provides both `pg_search`
(Tantivy-backed BM25 with real IDF weighting) and `pgvector` (cosine search via IVFFlat).
This removes Elasticsearch from the stack — one fewer service, and a local environment
that boots in under a minute. Verified in the running container: the `pg_search` and
`vector` extensions coexist, and the `documents` table carries a BM25 index and an IVFFlat
index at the same time.

> The first implementation used stock Postgres FTS (`ts_rank`), which lacks IDF weighting
> and scored NDCG@10 ≈ 0.06 on FiQA. Switching the sparse stage to `pg_search` brought it
> to 0.239, matching the published baseline, while keeping the single-Postgres design.

### Search request flow

```
client → Next.js proxy → POST /v1/search
       → rate limit + Pydantic validation
       → result-cache lookup (hit → return)
       → orchestration: sparse ∥ dense → RRF → [rerank]
       → hydrate hits (title + snippet) → record stage latencies → cache → response
```

### Evaluation flow

```
CLI / experiment job
  → load queries + qrels for the split
  → batch-embed all queries once (reused across variants)
  → for each variant: build a run via the same search() path
  → pytrec_eval → per-query + aggregate metrics
  → paired significance → write results/eval_<split>.json (or Redis job state)
```

---

## Low-level design

### Data model

A single `documents` row carries both retrieval representations, which is what lets one
Postgres serve sparse and dense retrieval.

| Table | Columns | Indexes |
|---|---|---|
| `documents` | `id` (PK, str), `title`, `text`, `doc_metadata` (JSONB), `embedding` (`vector(1536)`, nullable) | `bm25` (pg_search over `title,text`), `ivfflat` cosine on `embedding` |
| `queries` | `id` (PK), `text`, `split` | — |
| `qrels` | `query_id` (FK), `doc_id` (FK), `relevance` (int) | unique `(query_id, doc_id)`, index on `query_id` |
| `ingest_state` | `phase` (PK), `count`, `updated_at` | — used for idempotent seed/embed |

`embedding` is nullable so a freshly ingested corpus is valid before embeddings exist; the
embed job is resumable by selecting `WHERE embedding IS NULL`. The BM25 index is created in
migration `0002`; the IVFFlat index is built after vectors are populated (it learns its
centroids from data).

### Retrieval modules (`src/productrank/retrieval/`)

| Module | Entry point | Implementation |
|---|---|---|
| `sparse.py` | `search_sparse(session, query, top_k)` | `paradedb.boolean(should=[match(title), match(text)])`, ordered by `paradedb.score(id)` |
| `dense.py` | `search_dense(session, query, top_k, query_vector?)` | embed query (cache-aware) → pgvector cosine (`<=>`) top-k, IVFFlat `probes` configurable |
| `fusion.py` | `reciprocal_rank_fusion(lists, k=60)` | `score(d) = Σ 1/(k + rankᵢ(d))` over input rankings |
| `rerank.py` | `rerank(query, candidates, top_k)` | `CrossEncoder.predict` over (query, doc) pairs; CPU-pinned, `torch.set_num_threads(1)` |
| `embeddings.py` | `embed_query` / `embed_texts` | OpenAI embeddings; `embed_query` is Redis-cached, `embed_texts` batches |

### Orchestration and variants (`services/search.py`)

A single `search(session, query, variant, top_k, candidate_k, query_vector)` composes the
four variants behind one interface and records per-stage latency and candidate counts via
a stage timer:

- `BM25` — sparse only.
- `DENSE` — dense only.
- `HYBRID` — sparse + dense (each to `candidate_k`) → RRF → top-k.
- `HYBRID_RERANK` — RRF → top `RERANK_CANDIDATES` → cross-encoder → top-k.

Returns a `SearchResult` with ranked hits, `stage_latency_ms`, and `candidate_counts`. The
evaluation harness and the API call the exact same path, so reported numbers and live
results cannot diverge.

### Caching (`cache.py`)

| Cache | Key | TTL | Rationale |
|---|---|---|---|
| Query embedding | `emb:{sha256(query)[:16]}` | 30 days | Embeddings are deterministic for a fixed model; avoids a repeat API call (~200 ms). |
| Result set | `res:{variant}:{top_k}:{hash}` | 5 minutes | Hot-query latency; short TTL bounds staleness if the index changes. |

All cache access is fail-soft: if Redis is unreachable, callers recompute rather than
error. Cache hit/miss is observable via `/metrics`.

### Async A/B evaluation (`services/experiments.py`)

`POST /v1/experiments` creates a job (`job:{id}` in Redis, status `pending`) and schedules
a FastAPI `BackgroundTask`; the request returns the job id immediately. The worker embeds
the shared query set once, builds a run per variant through `search()`, scores with
`pytrec_eval`, computes paired significance, and transitions the job `running → completed`
(or `error`, captured in state). `GET /v1/experiments/{id}` polls. Async is scoped to
evaluation only; the search request path stays synchronous.

### Observability (`observability/`)

- **Metrics:** Prometheus histogram `productrank_stage_latency_seconds{variant,stage}`
  (per-stage p50/p95/p99) and request counters, exposed at `/metrics`.
- **Logging:** `structlog` JSON logs with a per-request correlation id bound via a
  contextvar and surfaced in the `x-correlation-id` response header.

### Configuration (`config.py`)

Env-driven via `pydantic-settings`: database/Redis URLs, embedding/rerank model names,
`RRF_K`, `RERANK_CANDIDATES`, `IVFFLAT_LISTS`/`PROBES`, `top_k` defaults, CORS origin. The
OpenAI key is read once and never logged.

---

## Evaluation methodology

- **Tool:** `pytrec_eval` — the standard `trec_eval` binding, not hand-rolled metrics.
- **Metrics:** NDCG@{10,100}, MRR, MAP, Recall@{10,100}, Precision@10. NDCG is the headline
  metric because it rewards relevant results *and* rewards them appearing near the top via
  a logarithmic positional discount.
- **Significance:** paired t-test plus a bootstrap confidence interval on per-query metrics,
  so a lift is reported only when it clears the noise.
- **Ground truth:** real qrels shipped with BEIR (FiQA and sampled MS MARCO). No invented
  labels, no LLM-as-judge.
- **Datasets:** FiQA is the full-corpus, literature-comparable baseline; sampled MS MARCO
  is used for variant deltas and the in-domain reranker lift (see the caveat above).

---

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI · Pydantic · SQLAlchemy 2.0 (sync, psycopg3) |
| Database | ParadeDB (PostgreSQL 17 + `pg_search` BM25 + `pgvector`) |
| Cache / jobs | Redis |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, in-process) |
| Evaluation | `pytrec_eval` · SciPy (paired t-test, bootstrap CI) |
| Observability | Prometheus `/metrics` · `structlog` correlation ids |
| Hardening | `slowapi` rate limits · CORS · Pydantic validation |
| Frontend | Next.js (App Router) · Tailwind · Recharts |
| Tooling | uv · ruff · black · mypy · pytest · Alembic · Docker Compose · GitHub Actions |

---

## Quick start

**Prerequisites:** Docker, Python 3.11/3.12, [uv](https://docs.astral.sh/uv/), Node ≥18,
and an OpenAI API key (for embeddings).

```bash
# 1. Bring up ParadeDB + Redis
cp .env.example .env          # then set OPENAI_API_KEY
docker compose up -d

# 2. Python environment
uv sync --extra dev

# 3. Migrate, then seed a dataset (downloads corpus/queries/qrels)
uv run alembic upgrade head
uv run python seed.py                 # FiQA      (full 57K corpus, split=test)
# or
uv run python seed_msmarco.py         # MS MARCO  (sampled 51K, split=dev) — wipes first

# 4. Embed the corpus and build the dense index (uses OPENAI_API_KEY)
uv run python -m productrank.cli embed

# 5. Evaluate all four variants
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run python -m productrank.cli eval --split test
```

Run the services:

```bash
# API
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run uvicorn productrank.main:app --port 8000

# Frontend (separate shell)
cd frontend && npm install && BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

> **Apple Silicon:** sentence-transformers auto-selects the MPS (Metal) backend, which
> deadlocks for this cross-encoder. The rerank path pins to CPU (`RERANK_DEVICE=cpu`); set
> `HF_HUB_OFFLINE=1` once the model is cached so batch eval doesn't stall on an HF Hub
> network check. Linux/CUDA targets are unaffected.

---

## Usage

**CLI** (`python -m productrank.cli ...`):

| Command | Purpose |
|---|---|
| `embed` | Embed the corpus (resumable, batched) and build the IVFFlat index |
| `build-index` | (Re)build the dense index only |
| `eval --split {test,dev} [--variants ...]` | Run the four-variant evaluation, write `results/eval_<split>.json` |

**Search via the API:**

```bash
curl -X POST localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "where did olives originate from", "variant": "hybrid_rerank", "top_k": 5}'
```

The frontend exposes a four-column search comparison with divergence highlighting, an A/B
experiment runner with significance markers, and an analytics dashboard.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/search` | Run one variant; returns ranked hits, per-stage latency, candidate counts |
| `POST` | `/v1/experiments` | Start an async A/B run over a query set → job id |
| `GET` | `/v1/experiments/{id}` | Poll job status; returns metrics table + significance |
| `GET` | `/v1/products/{id}` | Document detail |
| `GET` | `/v1/results?split=` | Recorded evaluation results for a split |
| `GET` | `/health` | Liveness (Postgres + Redis) |
| `GET` | `/metrics` | Prometheus exposition (per-stage latency histograms) |

Interactive OpenAPI docs at `/docs` when the API is running.

---

## Project structure

```
src/productrank/
├── config.py            # env-driven settings (pydantic-settings)
├── db.py · models.py    # SQLAlchemy engine + ORM (documents, queries, qrels)
├── ingest.py · embed.py # corpus loading + batched embedding
├── cli.py               # embed / build-index / eval
├── retrieval/           # sparse, dense, fusion (RRF), rerank, types
├── services/            # search orchestration + async A/B experiments
├── evaluation/          # pytrec_eval metrics, significance, runner
├── routers/             # search, experiments, products, results, health
├── observability/       # structlog config + Prometheus metrics
└── cache.py · ratelimit.py · schemas.py · main.py
data/ingest/             # FiQA + sampled MS MARCO downloaders
alembic/                 # migrations (0001 schema, 0002 pg_search BM25 index)
frontend/                # Next.js app (comparison, experiments, analytics)
tests/                   # unit, integration, e2e
scripts/diagnose_rerank.py
```

---

## Testing

```bash
uv run pytest                                       # unit tests, no stack needed
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run pytest    # incl. integration/e2e (needs seeded DB)
```

Unit tests cover RRF math, the `pytrec_eval` wrappers, significance, the cache, and API
validation. Integration/e2e tests run the pipeline against the seeded corpus and assert
NDCG clears a threshold, guarding against silent ranking regressions (they self-skip when
no seeded Postgres is present). CI (GitHub Actions), on every push and pull request, runs
`ruff` lint, `ruff format --check`, the unit suite, and a `pip-audit` dependency scan
(report-only).

---

## Deployment

`pg_search` is ParadeDB-specific and is not available on managed Postgres add-ons (RDS,
Supabase, Neon, Railway/Fly Postgres). The database must run as the `paradedb/paradedb`
container, so deploy targets that run arbitrary images (Fly.io, Railway, a VPS) are
required.

A practical setup: API + ParadeDB + Redis as containers on Fly.io or Railway (with a
persistent volume for the embeddings and indexes), and the frontend on Vercel pointing at
the API via `BACKEND_URL`. Secrets live in the platform secret store. Cost on the public
path is bounded by the Redis cache, `slowapi` rate limits, and pre-warmed demo queries.

---

## Design decisions and scope

Components intentionally left out, and why:

| Not built | Rationale |
|---|---|
| Ad allocation / auction | Needs bid + pCTR + auction data. The reranker scores relevance; a real ad system multiplies by bid × pCTR under a latency budget. |
| Personalization | Honest evaluation needs real interaction data and online experiments; synthetic users can't be measured. |
| Custom-trained models | A pretrained cross-encoder provides most of the signal; training a two-tower model is a separate effort. |
| Elasticsearch | ParadeDB covers both retrieval modes in one Postgres. ES is the horizontal-scale path beyond a single node. |
| Celery / task queue | A background task suffices for batch eval at this scale; a broker belongs when eval runs are concurrent and long. |
| Grafana stack | `/metrics` is Prometheus-ready; percentiles are surfaced directly rather than standing up the full stack. |

**Scope:** demo-scale corpora (≤57K docs), pretrained models only, no authentication
(public read-only, no PII).

---

## License

MIT. Built by Aditya Pimpalkar.
