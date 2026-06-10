# ProductRank

**An evaluation-first, multi-stage retrieval and ranking platform.** It implements the
candidate-generation → reranking pipeline that powers production search, ads, and
recommendation systems, and — the part most "AI search" demos skip — it *measures* the
quality of every stage against ground-truth relevance labels using IR-standard metrics.

Four retrieval strategies run side by side over the same query, and the platform reports
NDCG, MRR, MAP, Recall, and Precision for each, with statistical significance testing on
the differences.

> Portfolio / learning project. "Production" here means *production-grade architecture*
> (separation of concerns, evaluation, caching, observability, tests), not production
> traffic. There are no real users and no SLAs.

---

## Table of contents

- [Key results](#key-results)
- [The standout finding: reranker domain transfer](#the-standout-finding-reranker-domain-transfer)
- [Architecture](#architecture)
- [The retrieval pipeline](#the-retrieval-pipeline)
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

All numbers are measured with [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval)
(the standard `trec_eval` binding), with paired t-test + bootstrap confidence intervals
on per-query NDCG@10. Two datasets are used for two different purposes — read the labels.

### FiQA `test` — full corpus, literature-comparable baseline

57,638 documents, 648 queries (full BEIR FiQA). Absolute scores are trustworthy here:
BM25 matches the published BEIR baseline.

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
> and are *not* comparable to published leaderboards.** Use them for variant-to-variant
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

## The standout finding: reranker domain transfer

The same `ms-marco-MiniLM-L-6-v2` cross-encoder behaves oppositely on the two datasets,
and the explanation is the whole point:

| | First stage (best) | + Rerank | Effect |
|---|---|---|---|
| **MS MARCO** (in-domain) | 0.842 (hybrid) | **0.941** | reranking dominates |
| **FiQA** (out-of-domain, financial) | 0.448 (dense) | 0.375 | reranking **degrades** |

The cross-encoder was trained on MS MARCO web-search relevance. It transfers well to
in-domain queries and underperforms a strong general embedding (`text-embedding-3-small`)
on out-of-domain financial text. This is empirical evidence that **reranker domain match
matters** — a finding about transfer, not a tooling mistake.

It is not a candidate-pool artifact either: `scripts/diagnose_rerank.py` shows that even
when the cross-encoder reranks dense's *own* top-100 on FiQA, it still loses to dense
(0.42 vs 0.49). On MS MARCO, dense edging out hybrid is the known RRF dilution effect:
fusion raises Recall@100 (0.961 → 0.993) but an equal-weight blend with the ~20-point
weaker BM25 nudges the single relevant doc just below rank 1; the reranker then reorders
that richer pool back to the top.

---

## Architecture

```
                ┌─────────────────────────────┐
                │       NEXT.JS FRONTEND       │
                │  Search comparison (4-way)   │
                │  A/B experiment runner       │
                │  Analytics dashboard         │
                └───────────────┬──────────────┘
                                │ HTTP / JSON (server-side proxy; no client secrets)
                ┌───────────────▼──────────────┐
                │        FASTAPI BACKEND        │
                │  POST /v1/search   (4 variants)
                │  POST /v1/experiments  (A/B)  │
                │  GET  /v1/results | /metrics  │
                │  Redis cache · rate limit · structlog
                └───┬───────────────┬───────┬───┘
                    │               │       │
          ┌─────────▼────────┐  ┌───▼───┐  ┌▼─────────┐
          │     ParadeDB     │  │ Redis │  │  OpenAI  │
          │   (PostgreSQL)   │  │ cache │  │  API     │
          │  pg_search BM25  │  │ + job │  │ (embed   │
          │  pgvector (IVF)  │  │ state │  │  only)   │
          └──────────────────┘  └───────┘  └──────────┘

   Cross-encoder rerank: in-process (sentence-transformers, CPU)
   Async A/B eval: FastAPI BackgroundTasks, state in Redis
```

**One database, two retrieval modes.** A single ParadeDB image
(`paradedb/paradedb:0.15.26-pg17`) provides both `pg_search` (Tantivy-backed BM25 with
real IDF weighting) and `pgvector` (dense cosine search via an IVFFlat index). This
removes Elasticsearch from the stack — one fewer service, and a local environment that
boots in under a minute. Verified in the running container: `pg_search` and `vector`
extensions coexist, and the `documents` table carries a BM25 index and an IVFFlat index
simultaneously.

> The first implementation used stock Postgres FTS (`ts_rank`), which lacks IDF weighting
> and scored NDCG@10 ≈ 0.06 on FiQA — far below the published BM25 baseline. Switching the
> sparse stage to `pg_search` brought it to 0.239, matching the literature, while keeping
> the single-Postgres design.

---

## The retrieval pipeline

```
Query
  ├─► Sparse retrieval (pg_search BM25) ──┐
  │                                       ├─► RRF fusion ─► top-100 ─► cross-encoder rerank ─► final
  └─► Dense retrieval (pgvector + embed) ─┘
```

| Stage | What it does | Why |
|---|---|---|
| **Sparse (BM25)** | Lexical match, IDF-weighted | Exact terms, rare tokens, identifiers — but blind to synonyms |
| **Dense** | Embedding cosine similarity (`text-embedding-3-small`, 1536-d) | Captures meaning ("couch" ≈ "sofa") — but fuzzy on exact terms |
| **Hybrid (RRF)** | Reciprocal Rank Fusion of the two lists | Lexical and semantic retrieval fail in complementary ways; fusing recovers both |
| **Rerank** | Cross-encoder rescores the top ~100 | Reads query + doc *together* — far more accurate, too slow for the full corpus |

**Why two stages?** Cost asymmetry. The accurate model (a cross-encoder) is too slow to
run over millions of documents, so stage 1 cheaply narrows the corpus to ~100 candidates
and stage 2 spends the expensive compute only on the survivors.

**Why RRF, not score addition?** BM25 scores are unbounded; cosine similarities are 0–1.
Adding them is meaningless. RRF fuses on rank position — `score(d) = Σ 1/(k + rankᵢ(d))`,
`k=60` — so no normalization is needed.

**Why a cross-encoder for reranking?** Stage-1 dense retrieval uses a bi-encoder (query
and document embedded independently, so vectors precompute and search with an ANN index).
A cross-encoder feeds query + document through the model together, modelling their
interaction directly: more accurate, but impossible to precompute — which is exactly why
it is confined to the top-100.

---

## Evaluation methodology

- **Tool:** `pytrec_eval` — the standard `trec_eval` binding, not hand-rolled metrics.
- **Metrics:** NDCG@{10,100}, MRR, MAP, Recall@{10,100}, Precision@10. NDCG is the
  headline: it rewards relevant results *and* rewards them appearing near the top via a
  logarithmic positional discount.
- **Significance:** paired t-test plus a bootstrap confidence interval on per-query
  metrics, so a lift is only claimed when it clears the noise.
- **Ground truth:** real qrels shipped with BEIR (FiQA and sampled MS MARCO). No invented
  labels, no LLM-as-judge.
- **Datasets:** FiQA is the full-corpus, literature-comparable absolute baseline; sampled
  MS MARCO is used for variant-to-variant deltas and the in-domain reranker lift (see the
  caveat under [Key results](#key-results)).

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
| Observability | Prometheus `/metrics` (per-stage latency histograms) · `structlog` correlation IDs |
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

# Frontend (in another shell)
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

The frontend exposes three pages: a four-column **search comparison** with divergence
highlighting, an **A/B experiment runner** with significance markers, and an **analytics
dashboard**.

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
uv run pytest                      # unit tests run without the stack
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run pytest   # incl. integration/e2e (needs seeded DB)
```

Unit tests cover RRF math, the `pytrec_eval` wrappers, significance, the cache, and API
validation. Integration/e2e tests run the pipeline against the seeded corpus and assert
NDCG clears a threshold (guarding against silent ranking regressions). CI (GitHub Actions)
runs lint, type-check, and the unit suite on push.

---

## Deployment

The binding constraint: **`pg_search` is ParadeDB-specific and is not available on any
managed Postgres add-on** (RDS, Supabase, Neon, Railway/Fly Postgres). The database must
run as the `paradedb/paradedb` container, so deploy targets that run arbitrary images
(Fly.io, Railway, a VPS) are required.

A practical setup: API + ParadeDB + Redis as containers on Fly.io or Railway (with a
persistent volume for the embeddings and indexes), and the frontend on Vercel pointing at
the API via `BACKEND_URL`. Secrets live in the platform secret store. Cost on the public
path is bounded by the Redis cache, `slowapi` rate limits, and pre-warmed demo queries.

---

## Design decisions and scope

**Deliberately not built**, and why:

| Not built | Rationale |
|---|---|
| Ad allocation / auction | Needs bid + pCTR + auction data. The reranker scores *relevance*; a real ad system multiplies by bid × pCTR under a latency budget. |
| Personalization | Honest evaluation needs real interaction data and online experiments; synthetic users can't be measured. |
| Custom-trained models | A pretrained cross-encoder gives most of the signal; training a two-tower model is a separate project. |
| Elasticsearch | ParadeDB covers both retrieval modes in one Postgres. ES is the horizontal-scale path beyond a single node. |
| Celery / task queue | A background task suffices for batch eval at this scale; a broker belongs when eval runs are concurrent and long. |
| Grafana stack | `/metrics` is Prometheus-ready; percentiles are surfaced directly rather than standing up the full stack. |

**Scope:** demo-scale corpora (≤57K docs), pretrained models only, no authentication
(public read-only demo, no PII). The architecture is built the way production systems are;
the scale is honestly stated as demo-scale.

---

## License

MIT. Built by Aditya Pimpalkar.
