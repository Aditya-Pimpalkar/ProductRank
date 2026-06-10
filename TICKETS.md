# ProductRank — Feature Ticket List

**Companion to:** PRD.md, ARCHITECTURE.md
**Convention:** `[PR-##]` ticket id · **Est** = focused hours · **DoD** = definition of done

Tickets are grouped by the three PRD milestones. Each milestone is independently shippable. Milestone 1 alone is portfolio-worthy.

Dependencies are noted as `(depends: PR-##)`. Do tickets roughly in id order within a milestone.

---

## Milestone 1 — Functional Core (resume-ready at the end)

Goal: four variants + eval engine producing the first **real** NDCG numbers over FiQA.

### PR-01 — Repo & local stack scaffold
- Init repo, `pyproject.toml` (uv or poetry), pre-commit (ruff/black), `.env.example`, `.gitignore`.
- `docker-compose.yml` with Postgres (pgvector image) + Redis.
- **Est:** 2h · **DoD:** `docker compose up` brings Postgres+Redis healthy in < 1 min; `.env` git-ignored.

### PR-02 — Database schema & migrations
- `products`/`documents` table (id, text, metadata), `tsvector` column + GIN index, `vector(1536)` column, `qrels` table, `queries` table.
- Migration tooling (Alembic).
- **Est:** 2h · (depends: PR-01) · **DoD:** schema applies cleanly from zero; pgvector + FTS columns present.

### PR-03 — FiQA ingestion + seed script
- `data/ingest/fiqa.py`: load BEIR FiQA corpus, queries, qrels into Postgres.
- `seed.py`: one command, idempotent (skip existing).
- **Est:** 3h · (depends: PR-02) · **DoD:** `python seed.py` yields a fully populated corpus + queries + qrels; re-run is a no-op.

### PR-04 — Batch embedding pipeline
- Embed documents via OpenAI `text-embedding-3-small` in rate-limited batches; resumable.
- Build `IVFFlat` cosine index after load.
- **Est:** 3h · (depends: PR-03) · **DoD:** all docs embedded; IVFFlat index built; interrupting and resuming does not re-embed.

### PR-05 — Sparse retrieval (tsvector)
- `retrieval/sparse.py`: top-k via `ts_rank_cd` over the GIN index.
- **Est:** 2h · (depends: PR-03) · **DoD:** returns ranked ids + scores for a query; unit test on a fixture.

### PR-06 — Dense retrieval (pgvector)
- `retrieval/dense.py`: embed query (cache-aware stub ok for now) → cosine top-k via pgvector.
- **Est:** 2h · (depends: PR-04) · **DoD:** returns ranked ids + scores; unit test on a fixture.

### PR-07 — RRF fusion
- `retrieval/fusion.py`: reciprocal rank fusion of two ranked lists, tunable `k`.
- **Est:** 1h · (depends: PR-05, PR-06) · **DoD:** unit test with hand-computed expected fused order passes.

### PR-08 — Cross-encoder rerank
- `retrieval/rerank.py`: `ms-marco-MiniLM-L-6-v2` rescoring of top-N (default 100) candidates.
- **Est:** 2h · (depends: PR-07) · **DoD:** reordered top-N returned with rerank scores; runs locally in reasonable time.

### PR-09 — Retrieval orchestration service
- `services/`: compose the four variants behind one interface; record per-stage latency + candidate counts.
- **Est:** 2h · (depends: PR-08) · **DoD:** one call returns ranked results + stage timings for any variant.

### PR-10 — Evaluation engine
- `evaluation/metrics.py`: pytrec_eval wrapper computing NDCG@{10,100}, MRR, MAP, Recall@{10,100}, P@10 from a run + qrels.
- Eval script runs all four variants over a query set.
- **Est:** 3h · (depends: PR-09) · **DoD:** produces a real metrics table for all four variants over FiQA. **First real numbers — record them.**

### PR-11 — BM25 baseline sanity check
- Compare sparse-variant NDCG@10 against published BEIR FiQA baseline range; note the gap (Postgres FTS ≈ BM25, not exact).
- **Est:** 1h · (depends: PR-10) · **DoD:** documented comparison; expected divergence explained in notes.

### PR-12 — Minimal README + results table
- Architecture diagram, quick start, the measured results table.
- **Est:** 2h · (depends: PR-10) · **DoD:** a stranger can clone, seed, and reproduce numbers from the README.

> **Milestone 1 exit:** real NDCG numbers exist. Write the resume bullet now, from these numbers, naming FiQA. Project is resume-ready.

---

## Milestone 2 — Comparison UI + Performance

Goal: make ranking differences *visible*; add caching and latency tracking.

### PR-13 — Redis caching layer
- `cache.py`: query-embedding cache (long TTL), result-set cache (short TTL), rerank cache.
- **Est:** 2h · (depends: PR-09) · **DoD:** repeat query served from cache; cache hit/miss observable; TTLs configurable.

### PR-14 — Per-stage latency tracking + /metrics
- Timers for retrieval/fusion/rerank/end-to-end; p50/p95/p99; Prometheus `/metrics` endpoint.
- **Est:** 2h · (depends: PR-09) · **DoD:** `/metrics` scrapeable; percentiles present per stage.

### PR-15 — structlog + correlation ids
- JSON structured logging; correlation id per request.
- **Est:** 1h · (depends: PR-09) · **DoD:** each request emits correlated structured logs.

### PR-16 — FastAPI hardening
- Pydantic models, `slowapi` rate limits, CORS to frontend origin, `top_k`/query-length caps, `/health`.
- **Est:** 2h · (depends: PR-09) · **DoD:** invalid input rejected with correct status; rate limit enforced; CORS scoped.

### PR-17 — `POST /v1/search` endpoint
- Wire orchestration + cache behind the versioned search route; OpenAPI auto-docs.
- **Est:** 2h · (depends: PR-13, PR-16) · **DoD:** returns ranked results + per-stage latency + candidate counts; Swagger renders.

### PR-18 — Next.js scaffold + API client
- App Router, Tailwind, shadcn/ui; typed API client; server-side calls only (no client secrets).
- **Est:** 2h · (depends: PR-17) · **DoD:** frontend boots, calls `/v1/search`, renders a raw result.

### PR-19 — Search Comparison page (four columns)
- Four variant columns, score badges, latency tags, product/document cards.
- **Est:** 4h · (depends: PR-18) · **DoD:** one query renders all four ranked lists side by side.

### PR-20 — Divergence highlighting
- Detect and visually link documents whose rank differs sharply across variants ("#3 vs #47").
- **Est:** 3h · (depends: PR-19) · **DoD:** divergent docs are unmistakably highlighted; the demo's core story is visible.

### PR-21 — Example-query buttons + warm path
- Pre-filled example queries (incl. one where dense loses to BM25); pre-compute/cache their results.
- **Est:** 2h · (depends: PR-20, PR-13) · **DoD:** tapping an example renders a comparison in < 1s.

> **Milestone 2 exit:** the LinkedIn comparison screenshot exists; warm queries are sub-second.

---

## Milestone 3 — Experiments, Async, Deploy, Story

Goal: A/B experimentation with significance, background jobs, live deploy, and the narrative artifacts.

### PR-22 — Significance testing
- `evaluation/significance.py`: paired t-test (and/or bootstrap CI) on per-query metrics between two variants.
- **Est:** 2h · (depends: PR-10) · **DoD:** returns p-value/CI; unit-tested on synthetic data.

### PR-23 — Background eval jobs
- `POST /v1/experiments` kicks off an async run (BackgroundTasks/worker); state + results in Redis; `GET /v1/experiments/{id}` polls.
- **Est:** 3h · (depends: PR-10, PR-13) · **DoD:** A/B over N queries runs without blocking the API; status + final metrics retrievable.

### PR-24 — Experiment Runner page
- Variant A/B + size selector; progress indicator; side-by-side metrics table with significance markers.
- **Est:** 4h · (depends: PR-23, PR-18) · **DoD:** user runs an A/B comparison end to end in the UI; significance shown.

### PR-25 — Analytics / Dashboard page
- All-time metrics, NDCG-by-variant bar chart, latency histograms, top query categories.
- **Est:** 3h · (depends: PR-14, PR-24) · **DoD:** dashboard renders the at-a-glance "ranking systems" visual.

### PR-26 — Tests + CI
- Unit (RRF, metrics, retrieval), integration (tiny fixture corpus), E2E (assert NDCG > threshold); GitHub Actions lint+type+test.
- **Est:** 4h · (depends: most of M1–M2) · **DoD:** CI green on push; E2E guards ranking quality.

### PR-27 — Live deployment
- Managed Postgres+pgvector, managed Redis, API container (Railway/Fly), frontend (Vercel); secrets in platform store; small rerank model; pre-computed demo-query rerank.
- **Est:** 4h · (depends: PR-21, PR-24) · **DoD:** public URL; example query returns in < 1s; OpenAI spend bounded by cache + rate limit.

### PR-28 — ARCHITECTURE.md finalize
- Lock decisions, alternatives, tradeoffs, "deliberately not built," Java-coverage note, security posture. Internalize for interviews.
- **Est:** 3h · (depends: deploy) · **DoD:** every component has a defensible written rationale; non-goals are explicit.

### PR-29 — Medium draft + LinkedIn post
- Teaching-narrative article (the candidate-gen→rerank story with real numbers); LinkedIn post leading with the hook, comparison screenshot, links in first comment.
- **Est:** 3h · (depends: PR-25, PR-27) · **DoD:** post drafted with *measured* numbers and dataset named; screenshot attached; demo + repo + article links ready.

### PR-30 — Demo prep (interview)
- 3–4 teaching queries (incl. a divergence case and one where the system is suboptimal + the fix); rehearse the architecture walkthrough.
- **Est:** 2h · (depends: PR-27, PR-28) · **DoD:** can drive a live 5–10 min demo narrating *why*, and whiteboard the pipeline cold.

> **Milestone 3 exit:** live demo, A/B with significance, full docs, LinkedIn + Medium shipped, interview-ready.

---

## Effort Summary

| Milestone | Tickets | Approx hours |
|---|---|---|
| M1 — Functional core | PR-01 … PR-12 | ~25h |
| M2 — Comparison UI + perf | PR-13 … PR-21 | ~20h |
| M3 — Experiments + deploy + story | PR-22 … PR-30 | ~28h |
| **Total** | **30** | **~73h** |

Claude Code accelerates the boilerplate (scaffold, schema, routers, Pydantic models, shadcn components, Recharts, test scaffolds, RRF/metric wrappers). It does **not** substitute for understanding the retrieval design choices, evaluation rigor, or latency debugging — read every generated file before moving on, because an interviewer will probe the exact line you didn't understand.
