"""GET /v1/results — the latest saved evaluation results (for the Analytics page).

Reads results/eval_{split}.json produced by `productrank eval`. Returning the measured
table over an API keeps the frontend a thin renderer with no duplicated numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from productrank.config import DATASET_SPLIT
from productrank.schemas import Dataset

router = APIRouter(prefix="/v1", tags=["results"])

RESULTS_DIR = Path("results")


@router.get("/results")
def get_results(dataset: Dataset = Dataset.MSMARCO) -> dict:
    # dataset → split (msmarco→dev, fiqa→test). The dashboard toggles by dataset; this is
    # the single mapping point, so search and the dashboard always reflect the same corpus.
    split = DATASET_SPLIT[dataset.value]
    path = RESULTS_DIR / f"eval_{split}.json"
    if not path.exists():
        raise HTTPException(404, f"no results for dataset={dataset.value}; run `productrank eval`")
    data = json.loads(path.read_text())
    # Trim per-query payload — the dashboard only needs the aggregate table + timings.
    return {
        "split": data.get("split"),
        "num_queries": data.get("num_queries"),
        "top_k": data.get("top_k"),
        "aggregate": data.get("aggregate", {}),
        "wall_seconds": data.get("wall_seconds", {}),
    }
