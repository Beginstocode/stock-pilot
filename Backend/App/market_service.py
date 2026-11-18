"""Market data access through Yahoo Finance."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Final

import yfinance as yf

logger = logging.getLogger(__name__)

MARKET_INSTRUMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("NIFTY 50", "^NSEI"),
    ("BANK NIFTY", "^NSEBANK"),
    ("INDIA VIX", "^INDIAVIX"),
)


@dataclass(frozen=True)
class MarketSnapshot:
    """Latest available values for a market index."""

    name: str
    ticker: str
    value: float | None
    change_percent: float | None
    timestamp: datetime | None

    @property
    def available(self) -> bool:
        """Whether a current index value was returned by the data provider."""
        return self.value is not None


class MarketDataUnavailableError(RuntimeError):
    """Raised when the provider has no usable data for any configured index."""


class YahooFinanceMarketService:
    """Fetch market index history from Yahoo Finance without caching values."""

    def get_overview(self) -> list[MarketSnapshot]:
        """Fetch the latest available daily quote for each configured index."""
        snapshots = [
            self._fetch_snapshot(name, ticker)
            for name, ticker in MARKET_INSTRUMENTS
        ]
        if not any(snapshot.available for snapshot in snapshots):
            raise MarketDataUnavailableError(
                "The market data provider returned no usable index data."
            )
        return snapshots

    @staticmethod
    def _fetch_snapshot(name: str, ticker: str) -> MarketSnapshot:
        """Fetch one index and leave unavailable fields empty rather than fabricating data."""
        try:
            history = yf.Ticker(ticker).history(
                period="5d",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            if history.empty or "Close" not in history.columns:
                logger.warning("No market history returned for %s (%s)", name, ticker)
                return MarketSnapshot(name, ticker, None, None, None)

            closes = history["Close"].dropna()
            if closes.empty:
                logger.warning("No closing values returned for %s (%s)", name, ticker)
                return MarketSnapshot(name, ticker, None, None, None)

            value = float(closes.iloc[-1])
            if not math.isfinite(value):
                logger.warning("Non-finite market value returned for %s (%s)", name, ticker)
                return MarketSnapshot(name, ticker, None, None, None)

            raw_timestamp = closes.index[-1]
            timestamp = (
                raw_timestamp.to_pydatetime()
                if hasattr(raw_timestamp, "to_pydatetime")
                else raw_timestamp
            )
            if not isinstance(timestamp, datetime):
                logger.warning("Invalid market timestamp returned for %s (%s)", name, ticker)
                return MarketSnapshot(name, ticker, value, None, None)

            change_percent: float | None = None
            if len(closes) >= 2:
                previous_value = float(closes.iloc[-2])
                if math.isfinite(previous_value) and previous_value != 0:
                    change_percent = (value - previous_value) / previous_value * 100

            return MarketSnapshot(
                name=name,
                ticker=ticker,
                value=value,
                change_percent=change_percent,
                timestamp=timestamp,
            )
        except Exception:
            logger.exception("Could not fetch market data for %s (%s)", name, ticker)
            return MarketSnapshot(name, ticker, None, None, None)


@lru_cache(maxsize=1)
def get_market_service() -> YahooFinanceMarketService:
    """Return a reusable service instance; market data itself is not cached."""
    return YahooFinanceMarketService()
