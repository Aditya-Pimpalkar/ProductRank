"""A/B experiment endpoints (PR-23 / FR-5, FR-8).

POST kicks off a background job and returns its id immediately; GET polls status +
results. The request thread is never blocked on the eval (ARCHITECTURE §7).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from productrank import cache
from productrank.ratelimit import limiter
from productrank.schemas import ExperimentRequest, ExperimentResponse
from productrank.services.experiments import create_job, get_job, run_experiment

router = APIRouter(prefix="/v1", tags=["experiments"])


@router.post("/experiments", response_model=ExperimentResponse, status_code=202)
@limiter.limit("10/minute")
def start_experiment(
    request: Request,  # required by slowapi
    body: ExperimentRequest,
    background: BackgroundTasks,
) -> ExperimentResponse:
    if cache.get_client() is None:
        raise HTTPException(503, "Redis unavailable — experiment jobs require Redis for state.")
    job_id = create_job(
        body.variant_a.value, body.variant_b.value, body.query_set_size, body.split
    )
    background.add_task(run_experiment, job_id)
    state = get_job(job_id)
    return ExperimentResponse(**state)


@router.get("/experiments/{job_id}", response_model=ExperimentResponse)
def get_experiment(job_id: str) -> ExperimentResponse:
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, f"experiment {job_id} not found (or expired)")
    return ExperimentResponse(**state)
