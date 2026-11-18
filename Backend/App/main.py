
"""FastAPI application entry point."""

from __future__ import annotations

import logging
import asyncio
import os
from contextlib import suppress
from contextlib import asynccontextmanager
from importlib import import_module
from importlib.util import find_spec
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.App.config import APP_NAME, APP_VERSION, CORS_ORIGINS, DEBUG
from Backend.App.utils.constants import API_DEFAULT_HOST, API_DEFAULT_PORT
from Backend.App.workers.news_outcome_worker import NewsOutcomeWorker
from Backend.routes.auth import router as auth_router
from Backend.routes.health import router as health_router
from Backend.routes.market import router as market_router
from Backend.routes.portfolio import router as portfolio_router
from Backend.routes.news import router as news_router
from Backend.routes.news_performance import router as news_performance_router
from Backend.routes.prediction import router as prediction_router
from Backend.routes.stocks import router as stocks_router

logger = logging.getLogger(__name__)


def _news_outcome_worker_config() -> tuple[bool, float]:
    enabled = os.getenv("NEWS_OUTCOME_WORKER_ENABLED", "true").strip().casefold() in {
        "1", "true", "yes", "on",
    }
    try:
        interval = float(os.getenv("NEWS_OUTCOME_WORKER_INTERVAL_SECONDS", "3600"))
    except ValueError as exc:
        raise RuntimeError("NEWS_OUTCOME_WORKER_INTERVAL_SECONDS must be numeric.") from exc
    if interval < 1:
        raise RuntimeError("NEWS_OUTCOME_WORKER_INTERVAL_SECONDS must be at least 1 second.")
    return enabled, interval


async def _run_news_outcome_worker(interval_seconds: float) -> None:
    worker = NewsOutcomeWorker()
    while True:
        await asyncio.to_thread(worker.run_once)
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Run the configurable news-outcome worker and application lifecycle hooks."""
    logger.info("%s API starting", APP_NAME)
    enabled, interval = _news_outcome_worker_config()
    worker_task = None
    if enabled:
        logger.info("Starting news outcome worker; interval=%s seconds", interval)
        worker_task = asyncio.create_task(
            _run_news_outcome_worker(interval),
            name="news-outcome-worker",
        )
    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        logger.info("%s API shutting down", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description="Backend for AI-powered Indian stock recommendations",
    version=APP_VERSION,
    debug=DEBUG,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "API availability and health checks"},
        {"name": "Authentication", "description": "User registration and JWT login"},
        {"name": "Predictions", "description": "Model-generated stock recommendations"},
        {"name": "Sentiment", "description": "Financial news sentiment analysis"},
        {"name": "Stocks", "description": "Supported Indian stock symbols"},
        {"name": "Market", "description": "Market overview information"},
        {"name": "Portfolio", "description": "Portfolio holding operations"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["Health"])
app.include_router(auth_router, tags=["Authentication"])
app.include_router(prediction_router, tags=["Predictions"])
app.include_router(stocks_router, tags=["Stocks"])
app.include_router(market_router, tags=["Market"])
app.include_router(portfolio_router, tags=["Portfolio"])
app.include_router(news_router, tags=["News"])
app.include_router(news_performance_router)

_sentiment_module = "Backend.routes.sentiment"
if find_spec(_sentiment_module) is None:
    logger.warning(
        "Sentiment router is unavailable: module %s was not found.",
        _sentiment_module,
    )
else:
    sentiment_router = getattr(import_module(_sentiment_module), "router")
    app.include_router(sentiment_router, tags=["Sentiment"])


@app.get("/")
def home() -> dict[str, str]:
    """Return basic API identity and status information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "success",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Backend.App.main:app",
        host=API_DEFAULT_HOST,
        port=API_DEFAULT_PORT,
        reload=DEBUG,
    )
