"""FastAPI application factory (PR-16/17, FR-1/9).

Wires: CORS scoped to the known frontend origin, per-IP rate limiting (slowapi),
a correlation-id + structured-logging middleware, the versioned routers, and the
Prometheus /metrics endpoint. Secrets never reach the client — all model calls are
server-side (ARCHITECTURE §9.1).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from productrank.config import settings
from productrank.observability.logging import configure_logging, get_logger, new_correlation_id
from productrank.ratelimit import limiter
from productrank.routers import experiments, health, products, results, search

configure_logging(settings.log_level)
log = get_logger("app")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ProductRank API",
        version="0.1.0",
        description="Evaluation-first multi-stage retrieval and ranking over BEIR/FiQA.",
    )

    # Rate limiting (per-IP).
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # CORS scoped to the known frontend origin — never wildcarded in deploy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlation_id_mw(request: Request, call_next):
        cid = request.headers.get("x-correlation-id") or new_correlation_id()
        response = await call_next(request)
        response.headers["x-correlation-id"] = cid
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(products.router)
    app.include_router(experiments.router)
    app.include_router(results.router)
    return app


app = create_app()
