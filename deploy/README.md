# Deployment

Both datasets live, switchable in the UI (Path 2): **two databases inside one ParadeDB
instance** — `productrank_msmarco` and `productrank_fiqa` share one machine and one volume.
The tested retrieval code is unchanged; the API binds to the right database per request
based on a validated `dataset` parameter.

## Topology

| Service | Host | Notes |
|---|---|---|
| ParadeDB | Fly.io app `productrank-db` (`deploy/fly.paradedb.toml`) | One machine, one ~3 GB volume, both dataset databases. Internal-only (`productrank-db.internal:5432`). |
| API | Fly.io app `productrank-api` (`deploy/fly.api.toml` + `deploy/Dockerfile.api`) | FastAPI + cross-encoder baked in. 2 GB RAM, kept warm (`min_machines_running=1`). |
| Redis | Upstash (serverless) | Cache + A/B job state. Fail-soft. |
| Frontend | Vercel | Next.js; set env `BACKEND_URL=https://productrank-api.fly.dev`. |

## Secrets (never committed)

```bash
# API app
fly secrets set -a productrank-api \
  OPENAI_API_KEY=sk-... \
  POSTGRES_USER=productrank \
  POSTGRES_PASSWORD=<same as DB> \
  POSTGRES_HOST=productrank-db.internal \
  REDIS_URL=rediss://<upstash-url> \
  FRONTEND_ORIGIN=https://<your-vercel-app>.vercel.app

# DB app
fly secrets set -a productrank-db POSTGRES_PASSWORD=<same value>
```

`DB_NAME_MSMARCO` / `DB_NAME_FIQA` and `RERANK_DEVICE=cpu` / `HF_HUB_OFFLINE=1` are set in
`fly.api.toml` (non-secret). On Vercel, set only `BACKEND_URL`.

## What runs when

- **Build (image):** dependencies, app code, `results/*.json`, and the cross-encoder model
  (downloaded at build, then `HF_HUB_OFFLINE=1` at runtime).
- **Every deploy (release command, `deploy/migrate.sh`):** create the two databases if
  missing, then `alembic upgrade head` on each. Idempotent. **No seed/embed here** —
  seeding wipes.
- **One-time, manual:** seed + embed + index each database, then warm the demo caches.

## One-time host sequence (fresh host → ready)

```bash
# 0. Provision the DB volume + secrets, then deploy both apps.
fly volumes create pgdata --size 3 --region iad -a productrank-db
fly deploy -c deploy/fly.paradedb.toml          # ParadeDB up with persistent volume
fly deploy -c deploy/fly.api.toml               # API image; release cmd creates dbs + migrates

# 1. Seed + embed + index MS MARCO (primary), inside the API machine:
fly ssh console -a productrank-api
  uv run python seed_msmarco.py                          # download + load (wipes msmarco db only)
  uv run python -m productrank.cli embed --dataset msmarco   # embed ~51K + build IVFFlat

# 2. Seed + embed + index FiQA (secondary), same machine:
  uv run python seed.py                                  # FiQA → fiqa db
  uv run python -m productrank.cli embed --dataset fiqa  # embed ~57K (+$0.10–0.20) + IVFFlat

# 3. (optional) refresh the recorded dashboard numbers on-host:
  uv run python -m productrank.cli eval --dataset msmarco
  uv run python -m productrank.cli eval --dataset fiqa

# 4. Warm the demo example queries (instant, zero-spend public path):
  API_URL=http://127.0.0.1:8000 uv run python deploy/warm.py
  exit
```

Migrations re-run automatically on every `fly deploy` via the release command; the volume
persists, so seed/embed are never repeated. To rebuild a corpus, re-run the relevant
seed + embed steps (seeding wipes only that dataset's database).

## Cost & abuse control

- Embedding the public path is cache-first (30-day query-embedding TTL) and the demo
  queries are pre-warmed, so the common path costs nothing.
- `slowapi`: `/v1/search` 30/min/IP, `/v1/experiments` 5/min/IP.
- Experiments are capped at `query_set_size ≤ 100` and a global concurrency of 2
  simultaneous A/B jobs. Worst case under sustained abuse is bounded to a few dollars/day;
  CPU (rerank), not OpenAI spend, is the binding limit.
- One-time FiQA embedding is ~$0.10–0.20; MS MARCO ~$0.05–0.10.

## Faster path: restore precomputed embeddings (avoid re-embedding)

On a rate-limited OpenAI tier, embedding ~108K docs on-host can take hours. Since the
embeddings are deterministic for a fixed model, prefer reusing precomputed vectors:

- **Fly volume snapshot (recommended).** Once both DBs are embedded, the ParadeDB volume
  holds everything. Snapshot it (`fly volumes snapshots create <vol> -a productrank-db`)
  and restore future environments from that snapshot — a fresh DB comes up already
  populated in seconds, no re-embedding.
- **Dump / restore.** If you have the corpora embedded elsewhere (e.g. a local ParadeDB),
  `pg_dump --data-only` each database and load it on the DB machine. Do the load *on the
  Fly side* (over the private network), not from a laptop over `fly proxy` — the wireguard
  tunnel stalls under a sustained bulk COPY. For a big load, drop the `documents_bm25`
  index first and recreate it after (bulk build), then rebuild the IVFFlat index.

The one-time seed+embed sequence above is the from-scratch path; the snapshot is how you
redeploy quickly afterward.
