"""Dense (semantic) retrieval over pgvector.

The query is embedded with the same model used for the corpus (bi-encoder: query and
documents embedded independently), then nearest neighbours are found by cosine distance
using the IVFFlat index. `embedding <=> :vec` is pgvector's cosine-distance operator;
similarity is reported as `1 - distance` so higher = better, consistent with the other
stages.

`ivfflat.probes` is set per-session before the query: more probes → higher recall, more
latency (ARCHITECTURE §2.5). It's set with SET LOCAL inside the same transaction so it
doesn't leak to other sessions in the pool.
"""

from __future__ import annotations

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from productrank.config import settings
from productrank.retrieval.embeddings import embed_query
from productrank.retrieval.types import Hit, RankedList

_SQL = text(
    """
    SELECT id,
           1 - (embedding <=> :vec) AS score
    FROM documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> :vec
    LIMIT :k
    """
).bindparams(bindparam("vec"), bindparam("k"))


def search_dense(
    session: Session,
    query: str,
    top_k: int = 100,
    query_vector: list[float] | None = None,
) -> RankedList:
    """Dense top-k. Pass query_vector to reuse a cached embedding (PR-13)."""
    if not query.strip():
        return []
    vec = query_vector if query_vector is not None else embed_query(query)

    # pgvector expects the vector as a string literal like '[0.1,0.2,...]'.
    vec_literal = "[" + ",".join(repr(float(x)) for x in vec) + "]"

    # SET doesn't take bind params; set_config(..., is_local=true) is the parameterized
    # equivalent and scopes the setting to the current transaction only.
    session.execute(
        text("SELECT set_config('ivfflat.probes', :p, true)"),
        {"p": str(settings.ivfflat_probes)},
    )
    rows = session.execute(_SQL, {"vec": vec_literal, "k": top_k}).all()
    return [Hit(doc_id=r[0], score=float(r[1])) for r in rows]
