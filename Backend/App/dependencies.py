"""Reusable FastAPI dependencies for shared machine-learning services."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends

from .services.model_service import (
    StockSymbolNotFoundError,
    XGBoostModelService,
    get_xgboost_model_service,
)
from Src.Sentiment.news_sentiment import (
    DEFAULT_MODEL_NAME,
    FinancialNewsSentimentAnalyzer,
    NewsSentimentResult,
)


@lru_cache(maxsize=1)
def get_finbert_sentiment_service() -> FinancialNewsSentimentAnalyzer:
    """Provide one shared, lazily loaded FinBERT analyzer for the process."""
    return FinancialNewsSentimentAnalyzer(
        model_name=os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME),
        device=os.getenv("FINBERT_DEVICE") or None,
    )


def get_prediction_service(
    service: XGBoostModelService = Depends(get_xgboost_model_service),
) -> XGBoostModelService:
    """FastAPI dependency for routes that need stock predictions."""
    return service


def get_sentiment_service(
    service: FinancialNewsSentimentAnalyzer = Depends(get_finbert_sentiment_service),
) -> FinancialNewsSentimentAnalyzer:
    """FastAPI dependency for routes that need FinBERT sentiment analysis."""
    return service


__all__ = [
    "NewsSentimentResult",
    "StockSymbolNotFoundError",
    "XGBoostModelService",
    "get_finbert_sentiment_service",
    "get_prediction_service",
    "get_sentiment_service",
    "get_xgboost_model_service",
]
