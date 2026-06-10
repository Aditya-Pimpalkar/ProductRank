"""Cross-encoder reranking (stage 2).

The cross-encoder reads query and document *together* (unlike the bi-encoder used for
dense retrieval, which embeds them independently), so it models their interaction
directly and is much more accurate — but quadratically expensive and impossible to
precompute. That cost is exactly why it runs only over the top-N RRF candidates, never
the full corpus (ARCHITECTURE §2.4).

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — small, fast, strong on passage ranking,
runs in-process on CPU within a reasonable latency budget for a demo.
"""

from __future__ import annotations

from functools import lru_cache

from productrank.config import settings
from productrank.retrieval.types import Hit, RankedList

# Truncate document text fed to the cross-encoder. MiniLM has a 512-token limit;
# ~2000 chars keeps us under it without a tokenizer dependency on the hot path.
MAX_DOC_CHARS = 2000


@lru_cache
def _model():
    # Imported lazily so importing the retrieval package doesn't pull in torch.
    import os

    import torch
    from sentence_transformers import CrossEncoder

    # Force CPU. On Apple Silicon, sentence-transformers auto-selects the MPS (Metal GPU)
    # backend, and MPS inference for this cross-encoder deadlocks here — the process hangs
    # in an uninterruptible "metal gpu stream" wait with ~0 progress (caught via `sample`).
    # The model is tiny (MiniLM-L6), so CPU is both reliable and fast enough for the demo
    # latency budget. Override with RERANK_DEVICE if a real CUDA GPU is available.
    device = os.getenv("RERANK_DEVICE", "cpu")
    torch.set_num_threads(int(os.getenv("RERANK_THREADS", "4")))
    return CrossEncoder(settings.rerank_model, device=device)


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int | None = None,
) -> RankedList:
    """Rescore (doc_id, text) candidates with the cross-encoder; return new order.

    `candidates` are the RRF survivors. The returned list is sorted by cross-encoder
    score descending, truncated to top_k (defaults to the full candidate set).
    """
    if not candidates:
        return []

    pairs = [(query, text[:MAX_DOC_CHARS]) for _doc_id, text in candidates]
    scores = _model().predict(pairs, convert_to_numpy=True, show_progress_bar=False)

    scored = [
        Hit(doc_id=doc_id, score=float(score))
        for (doc_id, _text), score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda h: (-h.score, h.doc_id))
    return scored[:top_k] if top_k is not None else scored
