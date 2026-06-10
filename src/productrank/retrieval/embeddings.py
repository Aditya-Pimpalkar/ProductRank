"""OpenAI embedding client.

Thin wrapper around the OpenAI embeddings endpoint with batching and retry/backoff.
Two entry points:
  - embed_texts(list)  : batch path used by the corpus embedding pipeline (PR-04).
  - embed_query(str)   : single-query path used by dense retrieval (PR-06); a Redis
                         cache is layered on top of this in PR-13 (embeddings are
                         deterministic for a fixed model → long TTL).

Retry exists because the embedding endpoint is the one external dependency on the
request path; transient 429/5xx should back off, not fail the request.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from productrank.config import settings


@lru_cache
def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env before embedding "
            "(this is the only real secret the project needs)."
        )
    return OpenAI(api_key=settings.openai_api_key)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Order of returned vectors matches input order."""
    if not texts:
        return []
    resp = _client().embeddings.create(model=settings.embedding_model, input=texts)
    # The API guarantees response order matches input order; sort by index defensively.
    items = sorted(resp.data, key=lambda d: d.index)
    return [item.embedding for item in items]


def embed_query(text: str) -> list[float]:
    """Embed a single query string, using the Redis embedding cache when available.

    Embeddings are deterministic for a fixed model, so a repeated query is served from
    cache (long TTL) instead of paying another API call + ~200ms. Cache access is
    fail-soft (see cache.py)."""
    from productrank import cache

    cached = cache.get_query_embedding(text)
    if cached is not None:
        return cached
    vector = embed_texts([text])[0]
    cache.set_query_embedding(text, vector)
    return vector
