"""Market overview API endpoint."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from Backend.App.market_service import (
    MarketDataUnavailableError,
    MarketSnapshot,
    YahooFinanceMarketService,
    get_market_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class MarketIndexResponse(BaseModel):
    """Latest available information for a market index."""

    name: str = Field(..., description="Market index name")
    ticker: str = Field(..., description="Data provider's index ticker")
    value: float | None = Field(
        ..., description="Latest available index value, or null if unavailable"
    )
    change_percent: float | None = Field(
        ..., description="Change from the previous available close, or null if unavailable"
    )
    timestamp: datetime | None = Field(
        ..., description="Timestamp of the latest available value, or null if unavailable"
    )
    available: bool = Field(..., description="Whether an index value is available")


class MarketOverviewResponse(BaseModel):
    """Market overview and the time this response was assembled."""

    indices: list[MarketIndexResponse] = Field(
        ..., description="NIFTY 50, Bank NIFTY and India VIX"
    )
    retrieved_at: datetime = Field(
        ..., description="UTC timestamp when the market data was retrieved"
    )


def _to_response(snapshot: MarketSnapshot) -> MarketIndexResponse:
    return MarketIndexResponse(
        name=snapshot.name,
        ticker=snapshot.ticker,
        value=snapshot.value,
        change_percent=snapshot.change_percent,
        timestamp=snapshot.timestamp,
        available=snapshot.available,
    )


@router.get(
    "/market/overview",
    response_model=MarketOverviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Get market overview",
    description=(
        "Fetch the latest available NIFTY 50, Bank NIFTY and India VIX data "
        "from Yahoo Finance. Provider data may be delayed; unavailable values "
        "are returned as null."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "No usable market data is currently available."
        }
    },
)
def get_market_overview(
    market_service: YahooFinanceMarketService = Depends(get_market_service),
) -> MarketOverviewResponse:
    """Return latest market information without substituting fabricated values."""
    try:
        snapshots = market_service.get_overview()
    except MarketDataUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data is currently unavailable. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error retrieving the market overview")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The market data provider could not complete the request.",
        ) from exc

    return MarketOverviewResponse(
        indices=[_to_response(snapshot) for snapshot in snapshots],
        retrieved_at=datetime.now(timezone.utc),
    )
