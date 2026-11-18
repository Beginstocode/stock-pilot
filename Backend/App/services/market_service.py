"""Historical market-index data access from project CSV datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal

import pandas as pd

from Backend.App.config import DATA_DIR

MarketName = Literal["NIFTY 50", "BANK NIFTY", "INDIA VIX"]

_MARKET_FILES: Final[dict[MarketName, Path]] = {
    "NIFTY 50": Path("raw") / "nifty50.csv",
    "BANK NIFTY": Path("raw") / "banknifty.csv",
    "INDIA VIX": Path("raw") / "india_vix.csv",
}
_MARKET_ALIASES: Final[dict[str, MarketName]] = {
    "NIFTY 50": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "BANK NIFTY": "BANK NIFTY",
    "BANKNIFTY": "BANK NIFTY",
    "NIFTY BANK": "BANK NIFTY",
    "INDIA VIX": "INDIA VIX",
    "INDIAVIX": "INDIA VIX",
    "VIX": "INDIA VIX",
}
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"Date", "Open", "High", "Low", "Close"}
)
_NUMERIC_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close")


class UnsupportedMarketError(ValueError):
    """Raised when a market name is not supported."""


class MarketDataUnavailableError(RuntimeError):
    """Raised when a configured market dataset is missing or unusable."""


class MarketDataNotFoundError(LookupError):
    """Raised when a dataset has no data in the requested date range."""


@dataclass(frozen=True)
class MarketDataPoint:
    """One historical market observation; this is not a live quote."""

    market: MarketName
    timestamp: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    change: float | None
    change_percent: float | None
    is_live: Literal[False] = False


class MarketService:
    """Read historical NIFTY 50, Bank NIFTY and India VIX data from CSV files."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        market_files: dict[MarketName, Path] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._market_files = market_files

    def get_latest(self, market: str) -> MarketDataPoint:
        """Return the latest valid historical observation and its change."""
        normalized_market = self._normalize_market(market)
        history = self._load_history(normalized_market)
        history = history.dropna(subset=["Close"])
        if history.empty:
            raise MarketDataNotFoundError(
                f"No valid closing values are available for {normalized_market}."
            )

        latest = history.iloc[-1]
        previous_close = (
            float(history.iloc[-2]["Close"]) if len(history) > 1 else None
        )
        close = float(latest["Close"])
        change = close - previous_close if previous_close is not None else None
        change_percent = (
            change / previous_close * 100
            if change is not None and previous_close != 0
            else None
        )
        return self._to_data_point(
            latest,
            normalized_market,
            change=change,
            change_percent=change_percent,
        )

    def get_overview(self) -> list[MarketDataPoint]:
        """Return the latest available historical observation for each market."""
        return [self.get_latest(market) for market in _MARKET_FILES]

    def get_historical_data(
        self,
        market: str,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        limit: int | None = None,
    ) -> list[MarketDataPoint]:
        """Return chronological historical observations, optionally filtered."""
        normalized_market = self._normalize_market(market)
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero.")

        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        if start is not None and end is not None and start > end:
            raise ValueError("start_date must not be later than end_date.")

        history = self._load_history(normalized_market)
        if start is not None:
            history = history.loc[history["Date"] >= start]
        if end is not None:
            history = history.loc[history["Date"] <= end]
        if limit is not None:
            history = history.tail(limit)
        if history.empty:
            raise MarketDataNotFoundError(
                f"No historical data is available for {normalized_market} "
                "in the requested date range."
            )

        points: list[MarketDataPoint] = []
        previous_close: float | None = None
        for _, row in history.iterrows():
            close = self._finite_float(row["Close"])
            change = (
                close - previous_close
                if close is not None and previous_close is not None
                else None
            )
            change_percent = (
                change / previous_close * 100
                if change is not None and previous_close != 0
                else None
            )
            points.append(
                self._to_data_point(
                    row,
                    normalized_market,
                    change=change,
                    change_percent=change_percent,
                )
            )
            if close is not None:
                previous_close = close
        return points

    def _load_history(self, market: MarketName) -> pd.DataFrame:
        """Read, normalize and validate a market dataset without modifying it."""
        path = self._resolve_path(market)
        if not path.is_file():
            raise MarketDataUnavailableError(
                f"Historical dataset for {market} was not found: {path}"
            )
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise MarketDataUnavailableError(
                f"Could not read historical dataset for {market} from {path}: {exc}"
            ) from exc

        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.columns.duplicated().any():
            raise MarketDataUnavailableError(
                f"Historical dataset for {market} contains duplicate column names."
            )
        missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
        if missing_columns:
            raise MarketDataUnavailableError(
                f"Historical dataset for {market} is missing required columns: "
                f"{missing_columns}"
            )
        if frame.empty:
            raise MarketDataNotFoundError(
                f"Historical dataset for {market} contains no rows."
            )

        frame = frame.loc[:, ["Date", *_NUMERIC_COLUMNS]].copy()
        date_text = frame["Date"].astype("string").str.strip()
        iso_dates = date_text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", na=False)
        parsed_dates = pd.to_datetime(
            date_text.where(iso_dates), errors="coerce", utc=True
        )
        other_dates = pd.to_datetime(
            date_text.where(~iso_dates),
            errors="coerce",
            dayfirst=True,
            format="mixed",
            utc=True,
        )
        frame["Date"] = parsed_dates.fillna(other_dates).dt.normalize()
        frame = frame.dropna(subset=["Date"])
        for column in _NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[:, list(_NUMERIC_COLUMNS)] = frame.loc[
            :, list(_NUMERIC_COLUMNS)
        ].replace([float("inf"), float("-inf")], float("nan"))

        if frame.empty:
            raise MarketDataNotFoundError(
                f"Historical dataset for {market} contains no valid dates."
            )
        frame = frame.drop_duplicates(
            subset=["Date", *_NUMERIC_COLUMNS], keep="first"
        )
        if frame["Date"].duplicated().any():
            raise MarketDataUnavailableError(
                f"Historical dataset for {market} contains conflicting rows "
                "for the same date."
            )
        return frame.sort_values("Date", kind="mergesort").reset_index(drop=True)

    def _resolve_path(self, market: MarketName) -> Path:
        if self._market_files is not None:
            configured_path = self._market_files.get(market)
            if configured_path is None:
                raise MarketDataUnavailableError(
                    f"No dataset path is configured for {market}."
                )
            if not configured_path.is_absolute():
                return (self._data_dir / configured_path).resolve()
            return configured_path
        return (self._data_dir / _MARKET_FILES[market]).resolve()

    @staticmethod
    def _normalize_market(market: str) -> MarketName:
        if not isinstance(market, str):
            raise UnsupportedMarketError("Market name must be a string.")
        normalized = " ".join(market.strip().upper().split())
        try:
            return _MARKET_ALIASES[normalized]
        except KeyError as exc:
            supported = ", ".join(_MARKET_FILES)
            raise UnsupportedMarketError(
                f"Unsupported market '{market}'. Supported markets: {supported}."
            ) from exc

    @staticmethod
    def _parse_date(
        value: date | datetime | str | None, field_name: str
    ) -> pd.Timestamp | None:
        if value is None:
            return None
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            raise ValueError(f"{field_name} must be a valid date.")
        return pd.Timestamp(parsed).normalize()

    @staticmethod
    def _finite_float(value: object) -> float | None:
        if pd.isna(value):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    @classmethod
    def _to_data_point(
        cls,
        row: pd.Series,
        market: MarketName,
        change: float | None,
        change_percent: float | None,
    ) -> MarketDataPoint:
        timestamp = pd.Timestamp(row["Date"]).to_pydatetime()
        return MarketDataPoint(
            market=market,
            timestamp=timestamp,
            open=cls._finite_float(row["Open"]),
            high=cls._finite_float(row["High"]),
            low=cls._finite_float(row["Low"]),
            close=cls._finite_float(row["Close"]),
            change=change,
            change_percent=change_percent,
        )


__all__ = [
    "MarketDataNotFoundError",
    "MarketDataPoint",
    "MarketDataUnavailableError",
    "MarketName",
    "MarketService",
    "UnsupportedMarketError",
]
