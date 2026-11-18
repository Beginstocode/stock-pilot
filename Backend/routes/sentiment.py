"""Stock-wise financial news sentiment API endpoint."""

from __future__ import annotations

import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from Backend.App.schemas.sentiment import StockSentimentResponse
from Backend.App.services.sentiment_service import (
    FinancialNewsSentimentService,
    NoNewsAvailableError,
    SentimentDataUnavailableError,
    UnsupportedSentimentSymbolError,
    get_sentiment_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


@router.get(
    "/{symbol}",
    response_model=StockSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get stock news sentiment",
    description=(
        "Return the latest available aggregated FinBERT sentiment for a supported "
        "Indian stock symbol, including probabilities, sentiment score, and the "
        "timestamp of contributing news."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The symbol is unsupported or has no available news."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The stock symbol has an invalid format."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Processed sentiment data is unavailable or invalid."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Sentiment could not be retrieved."
        },
    },
)
def get_stock_sentiment(
    symbol: Annotated[
        str,
        Path(
            min_length=1,
            max_length=20,
            description="Supported Indian stock symbol, for example TCS or RELIANCE",
        ),
    ],
    sentiment_service: Annotated[
        FinancialNewsSentimentService, Depends(get_sentiment_service)
    ],
) -> StockSentimentResponse:
    """Return real processed FinBERT sentiment without filling missing values."""
    normalized_symbol = symbol.strip().upper()
    if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized_symbol):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid stock symbol format.",
        )

    try:
        result = sentiment_service.get_symbol_sentiment(normalized_symbol)
    except UnsupportedSentimentSymbolError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock symbol '{normalized_symbol}' is not supported.",
        ) from exc
    except NoNewsAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sentiment news is available for '{normalized_symbol}'.",
        ) from exc
    except SentimentDataUnavailableError as exc:
        logger.error(
            "Sentiment data is unavailable for %s: %s", normalized_symbol, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment data is currently unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected sentiment service error for %s", normalized_symbol
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stock sentiment could not be retrieved.",
        ) from exc

    return StockSentimentResponse(
        symbol=result.symbol,
        sentiment=result.sentiment,
        positive_probability=result.positive_probability,
        negative_probability=result.negative_probability,
        neutral_probability=result.neutral_probability,
        sentiment_score=result.sentiment_score,
        timestamp=result.timestamp,
        article_count=result.article_count,
    )


__all__ = ["router"]
