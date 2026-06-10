# ProductRank

**Evaluation-first multi-stage retrieval & ranking platform** over BEIR/FiQA.

Four retrieval variants — **BM25 (sparse)**, **dense vector**, **hybrid (RRF)**, and
**hybrid + cross-encoder rerank** — measured against ground-truth relevance labels with
IR-standard metrics (NDCG, MRR, MAP, Recall, Precision).

> Portfolio / learning project. "Production" here means *production-grade architecture*,
> not production traffic. See [PRD.md](PRD.md), [ARCHITECTURE.md](ARCHITECTURE.md),
> [TICKETS.md](TICKETS.md).

## Stack

- **Backend:** FastAPI · SQLAlchemy · ParadeDB (Postgres + `pg_search` BM25 + `pgvector`) · Redis
- **Retrieval:** OpenAI `text-embedding-3-small` (dense) · ParadeDB `pg_search` real BM25 (sparse) ·
  RRF fusion · `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank
- **Eval:** `pytrec_eval` (trec_eval binding) · paired t-test + bootstrap CI for significance
- **Observability:** Prometheus `/metrics` (per-stage latency histograms) · `structlog` correlation ids
- **Frontend:** Next.js (App Router) + Tailwind + Recharts — search comparison, A/B runner, analytics

## Quick start

```bash
# 1. Bring up Postgres (pgvector) + Redis
cp .env.example .env          # then add your OPENAI_API_KEY
docker compose up -d

# 2. Python env (uv)
uv sync --extra dev

# 3. Migrate + seed a dataset (downloads + loads corpus/queries/qrels)
uv run alembic upgrade head
uv run python seed.py                 # FiQA  (financial QA, 57K docs, split=test)
# — or —
uv run python seed_msmarco.py         # MS MARCO sample (51K passages, split=dev) — wipes first

# 4. Embed the corpus + build the dense index (needs OPENAI_API_KEY)
uv run python -m productrank.cli embed

# 5. Evaluate all four variants → real NDCG numbers
#    (cross-encoder runs on CPU; MPS/Metal is force-disabled — see note below)
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run python -m productrank.cli eval
```

Run the API and frontend:

```bash
# API (CPU rerank, offline model load)
HF_HUB_OFFLINE=1 RERANK_DEVICE=cpu uv run uvicorn productrank.main:app --port 8000

# Frontend (needs Node ≥18 — Next.js 14)
cd frontend && npm install && BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

## Results — measured, not invented

Two datasets, two outcomes — and the *difference between them* is the project's most
defensible insight: **a reranker only helps when it's in-domain.** All numbers below are
real (`pytrec_eval`), with paired t-test + bootstrap-CI significance on per-query NDCG@10.

### MS MARCO (`dev`) — the textbook candidate-generation → rerank lift

1,000 dev queries, 51,070-passage sampled corpus (every judged passage + 50K distractors),
top-100 retrieval. The cross-encoder is **in-domain** here (it was trained on MS MARCO):

| Metric | BM25 | Dense | Hybrid (RRF) | **Hybrid + Rerank** |
|---|---|---|---|---|
| **NDCG@10** | 0.7460 | 0.8999 | 0.8420 | **0.9413** |
| NDCG@100 | 0.7671 | 0.9037 | 0.8595 | **0.9446** |
| MRR | 0.7275 | 0.8891 | 0.8245 | **0.9310** |
| MAP | 0.7185 | 0.8842 | 0.8191 | **0.9287** |
| Recall@10 | 0.8357 | 0.9462 | 0.9218 | **0.9790** |

**The reranker tops every metric, and every lift is significant:**

| Comparison | ΔNDCG@10 | p-value | Significant? |
|---|---|---|---|
| BM25 → Hybrid+Rerank | **+0.195** | 1e-62 | ✅ |
| Hybrid → Hybrid+Rerank | **+0.099** | 9e-30 | ✅ |
| Dense → Hybrid+Rerank | **+0.041** | 3e-9 | ✅ (beats the strong dense baseline) |

### FiQA (`test`) — where the reranker *doesn't* help, and why

648 test queries, 57,638 documents. **BM25 = 0.2391 matches the published BEIR FiQA baseline
(~0.236)** via ParadeDB `pg_search` real BM25 — see [docs/BASELINE.md](docs/BASELINE.md).

| Metric | BM25 | Dense | Hybrid (RRF) | Hybrid + Rerank |
|---|---|---|---|---|
| **NDCG@10** | 0.2391 | **0.4483** | 0.3677 | 0.3754 |
| NDCG@100 | 0.2938 | **0.5184** | 0.4499 | 0.4510 |
| MRR | 0.3083 | **0.5295** | 0.4516 | 0.4547 |
| MAP | 0.1908 | **0.3892** | 0.3137 | 0.3153 |
| Recall@100 | 0.5100 | **0.7797** | 0.7425 | 0.7425 |

Here **dense wins** and reranking adds no significant lift (Hybrid → Rerank Δ=+0.008, p=0.42).
Confirmed not a bug: even reranking dense's *own* top-100, the cross-encoder scores 0.42 < dense's
0.49 (`scripts/diagnose_rerank.py`), and a second reranker (BGE) loses too.

### The insight (the interview moment)

`ms-marco-MiniLM-L-6-v2` is trained on MS MARCO web-search QA. On MS MARCO it's **in-domain** and
reranking dominates. On FiQA (financial QA) it's **out-of-domain**, and a strong general embedding
(`text-embedding-3-small`) beats it — equal-weight RRF then only *dilutes* that strong dense signal.
Same pipeline, opposite conclusion, **explained and significance-tested rather than asserted**.
The fixes for the FiQA case: weighted RRF, rerank the dense pool directly, or a domain-tuned reranker.

> ⚠️ **Apple Silicon note:** sentence-transformers auto-selects the MPS (Metal) backend, which
> **deadlocks** for this cross-encoder (the process hangs in a `metal gpu stream` wait). The rerank
> path forces CPU (`RERANK_DEVICE=cpu`); set `HF_HUB_OFFLINE=1` once the model is cached so batch
> eval doesn't block on an HF Hub network check. Warm rerank is then ~1.1s/query.

> ⚠️ **Apple Silicon note:** sentence-transformers auto-selects the MPS (Metal) backend, which
> **deadlocks** for this cross-encoder (the process hangs in a `metal gpu stream` wait). The rerank
> path forces CPU (`RERANK_DEVICE=cpu`); set `HF_HUB_OFFLINE=1` once the model is cached so batch
> eval doesn't block on an HF Hub network check. Warm rerank is then ~1.1s/query.
