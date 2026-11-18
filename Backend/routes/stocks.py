"""Stock catalog API endpoints."""

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from Backend.App.stock_catalog import STOCK_CATALOG
from Backend.App.schemas.stock import (
    HistoricalOHLCVRecord,
    HistoricalStockIndicatorRecord,
)
from Backend.App.services.stock_service import (
    StockDataNotFoundError,
    StockDataUnavailableError,
    StockService,
    UnsupportedStockSymbolError,
    get_stock_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class StockDetails(BaseModel):
    """Static identifying information for a supported stock."""

    symbol: str = Field(..., description="Supported Indian stock symbol")
    name: str = Field(..., description="Company name")


@router.get(
    "/stocks",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    summary="List supported stocks",
    description="Return the symbols currently supported by the recommendation system.",
)
def list_stocks() -> list[str]:
    """Return all supported stock symbols."""
    return list(STOCK_CATALOG)


@router.get(
    "/stocks/{symbol}",
    response_model=StockDetails,
    status_code=status.HTTP_200_OK,
    summary="Get stock details",
    description="Return static identifying information for a supported stock symbol.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The requested stock symbol is not supported."
        }
    },
)
def get_stock(
    symbol: Annotated[
        str,
        Path(
            min_length=1,
            max_length=20,
            description="Supported stock symbol, for example RELIANCE or TCS",
        ),
    ],
) -> StockDetails:
    """Return the catalog details for a supported symbol."""
    normalized_symbol = symbol.strip().upper()
    company_name = STOCK_CATALOG.get(normalized_symbol)
    if company_name is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock symbol '{normalized_symbol}' is not supported.",
        )

    return StockDetails(symbol=normalized_symbol, name=company_name)


@router.get(
    "/stocks/{symbol}/history",
    response_model=list[HistoricalOHLCVRecord],
    status_code=status.HTTP_200_OK,
    summary="Get historical stock prices",
    description=(
        "Return chronological historical OHLCV records from the processed stock "
        "data file. Optional start_date and end_date values are inclusive."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The symbol is unsupported or no matching history exists."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The symbol or date range is invalid."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Processed stock data is unavailable or invalid."
        },
    },
)
def get_stock_history(
    symbol: Annotated[
        str,
        Path(
            min_length=1,
            max_length=20,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9&.-]{0,19}$",
            description="Supported stock symbol, for example RELIANCE or TCS",
        ),
    ],
    stock_service: Annotated[StockService, Depends(get_stock_service)],
    start_date: Annotated[
        date | None,
        Query(description="Include records on or after this date (YYYY-MM-DD)"),
    ] = None,
    end_date: Annotated[
        date | None,
        Query(description="Include records on or before this date (YYYY-MM-DD)"),
    ] = None,
) -> list[HistoricalOHLCVRecord]:
    """Return historical OHLCV records from the existing processed CSV data."""
    normalized_symbol = symbol.strip().upper()
    try:
        points = stock_service.get_historical_data(
            normalized_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except UnsupportedStockSymbolError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock symbol '{normalized_symbol}' is not supported.",
        ) from exc
    except StockDataNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No historical stock data is available for '{normalized_symbol}' in the requested range.",
        ) from exc
    except StockDataUnavailableError as exc:
        logger.error("Historical stock data is unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical stock data is currently unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return [
        HistoricalOHLCVRecord(
            date=point.date,
            open=point.open,
            high=point.high,
            low=point.low,
            close=point.close,
            volume=point.volume,
        )
        for point in points
    ]


@router.get(
    "/stocks/{symbol}/indicators",
    response_model=list[HistoricalStockIndicatorRecord],
    status_code=status.HTTP_200_OK,
    summary="Get historical stock indicators",
    description=(
        "Return stored RSI, MACD, SMA, and EMA values from the processed feature "
        "data. Optional start_date and end_date values are inclusive."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The symbol is unsupported or no matching feature data exists."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The symbol or date range is invalid."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Processed stock feature data is unavailable or invalid."
        },
    },
)
def get_stock_indicators(
    symbol: Annotated[
        str,
        Path(
            min_length=1,
            max_length=20,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9&.-]{0,19}$",
            description="Supported stock symbol, for example RELIANCE or TCS",
        ),
    ],
    stock_service: Annotated[StockService, Depends(get_stock_service)],
    start_date: Annotated[
        date | None,
        Query(description="Include records on or after this date (YYYY-MM-DD)"),
    ] = None,
    end_date: Annotated[
        date | None,
        Query(description="Include records on or before this date (YYYY-MM-DD)"),
    ] = None,
) -> list[HistoricalStockIndicatorRecord]:
    """Return indicator values precomputed in the processed feature CSV."""
    normalized_symbol = symbol.strip().upper()
    try:
        points = stock_service.get_historical_data(
            normalized_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except UnsupportedStockSymbolError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock symbol '{normalized_symbol}' is not supported.",
        ) from exc
    except StockDataNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No historical indicator data is available for '{normalized_symbol}' in the requested range.",
        ) from exc
    except StockDataUnavailableError as exc:
        logger.error("Historical stock indicators are unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical stock indicators are currently unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return [
        HistoricalStockIndicatorRecord(
            date=point.date,
            close=point.close,
            rsi_14=point.rsi_14,
            macd=point.macd,
            macd_signal=point.macd_signal,
            macd_histogram=point.macd_histogram,
            sma_20=point.sma_20,
            sma_50=point.sma_50,
            sma_200=point.sma_200,
            ema_12=point.ema_12,
            ema_26=point.ema_26,
        )
        for point in points
    ]
