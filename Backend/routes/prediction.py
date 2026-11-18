"""Stock prediction API route."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path as FastApiPath, Query, status

from Backend.App.config import PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from Backend.App.schemas.prediction import PredictionResponse
from Backend.App.dependencies import (
    StockSymbolNotFoundError,
    XGBoostModelService,
    get_prediction_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def _get_model_b_prediction_service() -> XGBoostModelService:
    """Load the separately saved experimental Model B without changing Model A."""
    return XGBoostModelService(
        model_path=PROJECT_ROOT / "models" / "xgboost_model_b.pkl",
        features_path=PROCESSED_DATA_DIR / "stock_features.csv",
        metadata_path=REPORTS_DIR / "model_b_metadata.json",
    )

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


@router.get(
    "/predict/{symbol}",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict a stock's direction",
    description=(
        "Return a BUY, HOLD, or SELL recommendation for a listed Indian stock "
        "symbol, such as RELIANCE, TCS, INFY, SBIN, or HDFCBANK."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Stock symbol is not in the dataset."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid stock symbol format."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Prediction model or required data is unavailable."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Prediction could not be generated."
        },
    },
)
def get_stock_prediction(
    symbol: str = FastApiPath(
        ...,
        min_length=1,
        max_length=20,
        description="Indian stock symbol, for example RELIANCE or TCS",
    ),
    prediction_service: XGBoostModelService = Depends(get_prediction_service),
    model: Literal["A", "B"] = Query(
        default="A", description="Frozen model to use (A or B)"
    ),
) -> PredictionResponse:
    """Return a model-generated recommendation for the requested stock."""
    normalized_symbol = symbol.strip().upper()
    if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized_symbol):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid stock symbol format.",
        )

    try:
        selected_service = (
            prediction_service if model == "A" else _get_model_b_prediction_service()
        )
        result = selected_service.predict(normalized_symbol)
    except StockSymbolNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction data is available for symbol '{normalized_symbol}'.",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Prediction model or data is unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model or required stock data is unavailable.",
        ) from exc
    except (RuntimeError, ValueError) as exc:
        logger.exception("Prediction service failed for symbol %s", normalized_symbol)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stock prediction could not be generated.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction service error for %s", normalized_symbol)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stock prediction could not be generated.",
        ) from exc

    return PredictionResponse(
        symbol=result.symbol,
        prediction=result.prediction,
        confidence=result.confidence,
        probabilities=result.probabilities,
        prediction_timestamp=datetime.now(timezone.utc),
        prediction_date=result.prediction_date,
    )
