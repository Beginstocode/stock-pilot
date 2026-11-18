"""Pydantic v2 schemas for stock details and historical price records."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from Backend.App.stock_catalog import STOCK_CATALOG

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


class StockDataFields(BaseModel):
    """OHLCV prices and optional technical indicators for one observation."""

    model_config = ConfigDict(extra="forbid")

    open: float | None = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Opening price"
    )
    high: float | None = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Highest price"
    )
    low: float | None = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Lowest price"
    )
    close: float | None = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Closing price"
    )
    volume: float | None = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Trading volume"
    )
    rsi_14: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="14-period Relative Strength Index",
    )
    macd: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Moving Average Convergence Divergence value",
    )
    sma_20: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="20-period Simple Moving Average",
    )
    sma_50: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="50-period Simple Moving Average",
    )
    sma_200: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="200-period Simple Moving Average",
    )
    ema_12: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="12-period Exponential Moving Average",
    )
    ema_26: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="26-period Exponential Moving Average",
    )

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> StockDataFields:
        """Check available OHLC prices for an internally consistent range."""
        if self.high is not None:
            if self.low is not None and self.high < self.low:
                raise ValueError("High must be greater than or equal to Low.")
            if self.open is not None and self.high < self.open:
                raise ValueError("High must be greater than or equal to Open.")
            if self.close is not None and self.high < self.close:
                raise ValueError("High must be greater than or equal to Close.")
        if self.low is not None:
            if self.open is not None and self.low > self.open:
                raise ValueError("Low must be less than or equal to Open.")
            if self.close is not None and self.low > self.close:
                raise ValueError("Low must be less than or equal to Close.")
        return self


class StockDetailsResponse(StockDataFields):
    """Company details and its latest available processed market record."""

    symbol: str = Field(..., description="Supported Indian stock symbol")
    company_name: str = Field(..., min_length=1, description="Company's official name")
    date: datetime = Field(
        ..., description="Timestamp of the available processed stock observation"
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def validate_symbol(cls, value: object) -> object:
        """Normalize and validate a symbol from the supported stock catalog."""
        if not isinstance(value, str):
            raise ValueError("Stock symbol must be a string.")
        normalized = value.strip().upper()
        if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError("Stock symbol has an invalid format.")
        if normalized not in STOCK_CATALOG:
            raise ValueError(f"Stock symbol '{normalized}' is not supported.")
        return normalized

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, value: str) -> str:
        """Require the company name to match the supported symbol catalog."""
        name = value.strip()
        if not name:
            raise ValueError("Company name cannot be blank.")
        return name

    @model_validator(mode="after")
    def company_name_matches_symbol(self) -> StockDetailsResponse:
        """Prevent company names from contradicting the stock catalog."""
        if self.company_name != STOCK_CATALOG[self.symbol]:
            raise ValueError("Company name does not match the stock symbol.")
        return self


class HistoricalStockRecord(StockDataFields):
    """OHLCV data and available indicators for one historical date."""

    date: datetime = Field(..., description="Timestamp of the stock observation")


class HistoricalOHLCVRecord(BaseModel):
    """One historical date with its available OHLCV market data."""

    date: datetime = Field(..., description="Timestamp of the stock observation")
    open: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    high: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    low: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    close: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    volume: float | None = Field(..., ge=0.0, allow_inf_nan=False)


class HistoricalStockIndicatorRecord(BaseModel):
    """One processed price record with its stored technical indicators."""

    date: datetime = Field(..., description="Timestamp of the stock observation")
    close: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    rsi_14: float | None = Field(..., ge=0.0, le=100.0, allow_inf_nan=False)
    macd: float | None = Field(..., allow_inf_nan=False)
    macd_signal: float | None = Field(..., allow_inf_nan=False)
    macd_histogram: float | None = Field(..., allow_inf_nan=False)
    sma_20: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    sma_50: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    sma_200: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    ema_12: float | None = Field(..., ge=0.0, allow_inf_nan=False)
    ema_26: float | None = Field(..., ge=0.0, allow_inf_nan=False)


class HistoricalStockDataResponse(BaseModel):
    """Historical stock records for one supported company."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Supported Indian stock symbol")
    company_name: str = Field(..., min_length=1, description="Company's official name")
    records: list[HistoricalStockRecord] = Field(
        ..., description="Available historical price and indicator records"
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def validate_symbol(cls, value: object) -> object:
        """Normalize and validate a symbol from the supported stock catalog."""
        if not isinstance(value, str):
            raise ValueError("Stock symbol must be a string.")
        normalized = value.strip().upper()
        if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError("Stock symbol has an invalid format.")
        if normalized not in STOCK_CATALOG:
            raise ValueError(f"Stock symbol '{normalized}' is not supported.")
        return normalized

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, value: str) -> str:
        """Reject blank company names."""
        name = value.strip()
        if not name:
            raise ValueError("Company name cannot be blank.")
        return name

    @model_validator(mode="after")
    def validate_history(self) -> HistoricalStockDataResponse:
        """Ensure records are for this stock and listed chronologically."""
        if self.company_name != STOCK_CATALOG[self.symbol]:
            raise ValueError("Company name does not match the stock symbol.")
        dates = [record.date for record in self.records]
        if dates != sorted(dates):
            raise ValueError("Historical records must be ordered by date.")
        return self


__all__ = [
    "HistoricalOHLCVRecord",
    "HistoricalStockIndicatorRecord",
    "HistoricalStockDataResponse",
    "HistoricalStockRecord",
    "StockDataFields",
    "StockDetailsResponse",
]
