"""Pre-warm the demo example queries so the public example-button path is instant and
costs zero OpenAI spend.

For each dataset and each example query, this hits the running API for all four variants
at the frontend's default top_k. That populates the Redis result-set cache and the
query-embedding cache with exactly the keys real example-button traffic will hit. Run it
once after the API is up and the corpora are embedded:

    API_URL=http://127.0.0.1:8000 uv run python deploy/warm.py

Keep EXAMPLE_QUERIES in sync with frontend/lib/api.ts.
"""

from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
VARIANTS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
TOP_K = 10

EXAMPLE_QUERIES: dict[str, list[str]] = {
    "msmarco": [
        "where did olives originate from",
        "what is the definition of pessimistic",
        "how long does it take corn to cook on the grill",
        "what causes spots on tree leaves",
        "where does microtubule formation occur",
        "what is the elevation of white pass in washington",
    ],
    "fiqa": [
        "What is the difference between a Roth and a traditional IRA?",
        "How does a 401k rollover work?",
        "Are stock dividends taxed as income?",
        "What is dollar cost averaging?",
        "Should I pay off debt or invest?",
        "How is a company's enterprise value calculated?",
    ],
}


def main() -> None:
    with httpx.Client(base_url=API_URL, timeout=120) as client:
        for dataset, queries in EXAMPLE_QUERIES.items():
            for q in queries:
                for variant in VARIANTS:
                    r = client.post(
                        "/v1/search",
                        json={"query": q, "variant": variant, "dataset": dataset, "top_k": TOP_K},
                    )
                    tag = "ok" if r.status_code == 200 else f"ERR {r.status_code}"
                    print(f"  [{dataset}/{variant}] {q[:48]:<48} {tag}")
    print("✓ demo queries warmed (result + embedding caches populated)")


if __name__ == "__main__":
    main()
