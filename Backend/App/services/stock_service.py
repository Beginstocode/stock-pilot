"""Read stock prices and technical indicators from processed CSV data."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from Backend.App.config import PROCESSED_DATA_DIR
from Backend.App.stock_catalog import STOCK_CATALOG

_DEFAULT_STOCK_DATA_PATH = PROCESSED_DATA_DIR / "stock_features.csv"
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")
_REQUIRED_COLUMNS = (
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI_14",
    "MACD",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
)


class UnsupportedStockSymbolError(ValueError):
    """Raised when a symbol is not in the supported stock catalog."""


class StockDataUnavailableError(RuntimeError):
    """Raised when the stock CSV is missing or cannot be used."""


class StockDataNotFoundError(LookupError):
    """Raised when the processed data has no rows for a supported stock."""


@dataclass(frozen=True)
class StockDataPoint:
    """One historical stock row with available prices and indicators."""

    date: datetime
    symbol: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    ema_12: float | None
    ema_26: float | None


class StockService:
    """Provide stock details and history from the project's processed data."""

    def __init__(self, data_path: Path = _DEFAULT_STOCK_DATA_PATH) -> None:
        self._data_path = data_path

    def list_supported_symbols(self) -> list[str]:
        """Return symbols supported by the application's stock catalog."""
        return list(STOCK_CATALOG)

    def get_stock_details(self, symbol: str) -> StockDataPoint:
        """Return the latest available processed row for a supported symbol."""
        history = self.get_historical_data(symbol, limit=1)
        return history[0]

    def get_historical_data(
        self,
        symbol: str,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        limit: int | None = None,
    ) -> list[StockDataPoint]:
        """Return chronological processed rows, optionally filtered by date and count."""
        normalized_symbol = self._normalize_symbol(symbol)
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero.")

        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        if start is not None and end is not None and start > end:
            raise ValueError("start_date must not be later than end_date.")

        frame = self._read_data()
        symbols = frame["Symbol"].astype("string").str.strip().str.upper()
        selected = frame.loc[symbols.eq(normalized_symbol)].copy()
        if selected.empty:
            raise StockDataNotFoundError(
                f"No processed stock data is available for {normalized_symbol}."
            )

        selected["Date"] = pd.to_datetime(
            selected["Date"], errors="coerce", utc=True
        )
        selected = selected.dropna(subset=["Date"])
        if start is not None:
            selected = selected.loc[selected["Date"] >= start]
        if end is not None:
            selected = selected.loc[selected["Date"] <= end]
        selected = selected.sort_values("Date", kind="mergesort")
        if limit is not None:
            selected = selected.tail(limit)
        if selected.empty:
            raise StockDataNotFoundError(
                f"No processed stock data is available for {normalized_symbol} "
                "in the requested date range."
            )

        return [
            self._to_data_point(row, normalized_symbol)
            for _, row in selected.iterrows()
        ]

    def _read_data(self) -> pd.DataFrame:
        """Read and validate the processed CSV without modifying it."""
        if not self._data_path.is_file():
            raise StockDataUnavailableError(
                f"Processed stock data was not found: {self._data_path}"
            )
        try:
            frame = pd.read_csv(self._data_path, low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise StockDataUnavailableError(
                f"Could not read processed stock data from {self._data_path}: {exc}"
            ) from exc

        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.columns.duplicated().any():
            raise StockDataUnavailableError(
                "Processed stock data contains duplicate column names."
            )
        missing_columns = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
        if missing_columns:
            raise StockDataUnavailableError(
                "Processed stock data is missing required columns: "
                f"{missing_columns}"
            )
        if frame.empty:
            raise StockDataNotFoundError("Processed stock data contains no rows.")
        return frame

    @staticmethod
    def _to_data_point(row: pd.Series, symbol: str) -> StockDataPoint:
        """Convert a CSV row while representing unavailable numeric values as None."""
        timestamp = pd.Timestamp(row["Date"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        def numeric(column: str) -> float | None:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        return StockDataPoint(
            date=timestamp.to_pydatetime(),
            symbol=symbol,
            open=numeric("Open"),
            high=numeric("High"),
            low=numeric("Low"),
            close=numeric("Close"),
            volume=numeric("Volume"),
            rsi_14=numeric("RSI_14"),
            macd=numeric("MACD"),
            macd_signal=numeric("MACD_Signal"),
            macd_histogram=numeric("MACD_Histogram"),
            sma_20=numeric("SMA_20"),
            sma_50=numeric("SMA_50"),
            sma_200=numeric("SMA_200"),
            ema_12=numeric("EMA_12"),
            ema_26=numeric("EMA_26"),
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise UnsupportedStockSymbolError("Stock symbol must be a string.")
        normalized_symbol = symbol.strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(normalized_symbol):
            raise UnsupportedStockSymbolError(
                "Stock symbol contains invalid characters or is too long."
            )
        if normalized_symbol not in STOCK_CATALOG:
            raise UnsupportedStockSymbolError(
                f"Stock symbol '{normalized_symbol}' is not supported."
            )
        return normalized_symbol

    @staticmethod
    def _parse_date(
        value: date | datetime | str | None, field_name: str
    ) -> pd.Timestamp | None:
        if value is None:
            return None
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            raise ValueError(f"{field_name} must be a valid date.")
        timestamp = pd.Timestamp(parsed)
        return timestamp.normalize()


@lru_cache(maxsize=1)
def get_stock_service() -> StockService:
    """Return a shared stock service configured for the processed data file."""
    return StockService()


__all__ = [
    "StockDataNotFoundError",
    "StockDataPoint",
    "StockDataUnavailableError",
    "StockService",
    "UnsupportedStockSymbolError",
    "get_stock_service",
]
