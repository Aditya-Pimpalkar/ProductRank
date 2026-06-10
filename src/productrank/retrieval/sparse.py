"""Sparse (lexical) retrieval via ParadeDB pg_search BM25.

This is the BM25 baseline in the variant set, backed by pg_search's Tantivy BM25 index
(migration 0002). Unlike stock Postgres FTS (`ts_rank`), pg_search applies real IDF
weighting, so a rare decisive term (a ticker, a model number) outweighs a common one —
which is exactly why it lands in range of the published BEIR FiQA BM25 baseline (NFR-3),
where the ts_rank approximation did not.

Query construction: `paradedb.match` tokenizes the query and matches with OR semantics
(BM25-scored). We search both `title` and `text` via `paradedb.boolean(should => …)`,
so a document matching in either field is retrieved and BM25-ranked. `paradedb.score(id)`
returns the BM25 score for ordering. (FiQA titles are mostly empty, so in practice the
text field carries the signal; including title is correct and free.)
"""

from __future__ import annotations

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from productrank.retrieval.types import Hit, RankedList

_SQL = text(
    """
    SELECT id, paradedb.score(id) AS score
    FROM documents
    WHERE id @@@ paradedb.boolean(
        should => ARRAY[
            paradedb.match('title', :q),
            paradedb.match('text', :q)
        ]
    )
    ORDER BY score DESC
    LIMIT :k
    """
).bindparams(bindparam("q"), bindparam("k"))


def search_sparse(session: Session, query: str, top_k: int = 100) -> RankedList:
    if not query.strip():
        return []
    rows = session.execute(_SQL, {"q": query, "k": top_k}).all()
    return [Hit(doc_id=r[0], score=float(r[1])) for r in rows]
