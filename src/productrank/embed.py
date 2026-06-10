"""Batch corpus embedding + dense (IVFFlat) index build (PR-04).

Resumability is the headline property: the pipeline only selects documents WHERE
embedding IS NULL, so interrupting it and re-running never re-embeds — a failure
mid-ingest (or hitting an API limit) does not force a restart, and re-runs are cheap.

Batching keeps requests under the embedding endpoint's token ceiling and amortizes
HTTP overhead. The IVFFlat index is built *after* vectors are loaded, because IVFFlat
learns its cluster centroids from existing data — building it on an empty table would
produce a useless index.
"""

from __future__ import annotations

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from productrank.config import settings
from productrank.db import SessionLocal, engine
from productrank.models import Document
from productrank.retrieval.embeddings import embed_texts

# Max characters of (title + text) sent per document. text-embedding-3-small accepts
# ~8191 tokens; ~8000 chars is a safe, simple cap that avoids a tokenizer dependency.
MAX_CHARS = 8000
# Documents per embedding request. Small enough to stay well under token limits,
# large enough to amortize round-trips.
EMBED_BATCH = 128


def _doc_input(title: str, body: str) -> str:
    combined = (title + "\n\n" + body).strip() if title else body.strip()
    # The embeddings API rejects empty strings. A handful of FiQA docs have no text;
    # embed a placeholder so every document gets a (junk) vector and the resumable
    # pipeline can complete — such docs simply never rank well, which is correct.
    return combined[:MAX_CHARS] if combined else "(empty document)"


def remaining(session: Session) -> int:
    return (
        session.scalar(
            select(func.count()).select_from(Document).where(Document.embedding.is_(None))
        )
        or 0
    )


def embed_corpus() -> int:
    """Embed every document missing an embedding. Returns count embedded this run."""
    embedded = 0
    with SessionLocal() as session:
        todo = remaining(session)
        if todo == 0:
            print("All documents already embedded.")
            return 0
        print(f"Embedding {todo} documents in batches of {EMBED_BATCH} …")

        while True:
            batch = session.execute(
                select(Document.id, Document.title, Document.text)
                .where(Document.embedding.is_(None))
                .order_by(Document.id)
                .limit(EMBED_BATCH)
            ).all()
            if not batch:
                break

            inputs = [_doc_input(t or "", b or "") for (_id, t, b) in batch]
            vectors = embed_texts(inputs)

            session.execute(
                update(Document),
                [{"id": row[0], "embedding": vec} for row, vec in zip(batch, vectors, strict=True)],
            )
            session.commit()
            embedded += len(batch)
            print(f"  embedded {embedded}/{todo}", end="\r", flush=True)

    print(f"\nDone. Embedded {embedded} documents this run.")
    return embedded


def build_ivfflat_index() -> None:
    """(Re)build the IVFFlat cosine index on the embedding column.

    Tuning: `lists` partitions the vectors into clusters; a common heuristic is
    ~sqrt(rows). `probes` (set per-query at search time) trades recall for latency.
    Both are config-driven (ARCHITECTURE §2.5).
    """
    lists = settings.ivfflat_lists
    with engine.begin() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM documents WHERE embedding IS NOT NULL")
        ).scalar_one()
        if n == 0:
            raise RuntimeError("No embeddings present — run embed_corpus() first.")
        print(f"Building IVFFlat index on {n} vectors (lists={lists}) …")
        # IVFFlat builds its cluster centroids in memory; the default 64MB is just shy
        # of what 57K×1536-dim vectors need. Raise it for this session only.
        conn.execute(text("SET maintenance_work_mem = '256MB'"))
        conn.execute(text("DROP INDEX IF EXISTS ix_documents_embedding_ivfflat"))
        conn.execute(
            text(
                "CREATE INDEX ix_documents_embedding_ivfflat ON documents "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            )
        )
        conn.execute(text("ANALYZE documents"))
    print("IVFFlat index built.")
