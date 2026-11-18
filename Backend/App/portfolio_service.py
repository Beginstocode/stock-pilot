"""Portfolio business logic and latest available stock-price lookup."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Protocol
from uuid import UUID, uuid4

from Backend.App.stock_catalog import STOCK_CATALOG
from Backend.App.services.stock_service import get_stock_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StockPrice:
    """Latest available provider price and its timestamp."""

    value: float
    timestamp: datetime


@dataclass(frozen=True)
class PortfolioHolding:
    """A portfolio lot, independent of price-provider availability."""

    holding_id: UUID
    symbol: str
    quantity: int
    purchase_price: float


class StockPriceProvider(Protocol):
    """Interface for retrieving real stock prices."""

    def get_latest_price(self, symbol: str) -> StockPrice | None:
        """Return the latest available price, or None when no quote is available."""


class UnsupportedStockSymbolError(ValueError):
    """Raised when a symbol is not in the supported stock catalog."""


class PortfolioHoldingNotFoundError(LookupError):
    """Raised when a requested holding ID is not in the portfolio."""


class ProcessedStockPriceProvider:
    """Read the latest available close from the same processed data as Stock Analysis."""

    def get_latest_price(self, symbol: str) -> StockPrice | None:
        """Return the latest processed close; never substitute a fabricated price."""
        try:
            history = get_stock_service().get_historical_data(symbol)
            if not history:
                return None
            for latest in reversed(history):
                value = latest.close
                timestamp = latest.date
                if value is None or not isinstance(timestamp, datetime):
                    continue
                value = float(value)
                if math.isfinite(value) and value > 0:
                    return StockPrice(value=value, timestamp=timestamp)
            return None
        except Exception:
            logger.exception("Could not retrieve latest price for %s", symbol)
            return None


class PortfolioService:
    """Manage process-local portfolio holdings and enrich them with market prices.

    Holdings are intentionally stored in memory until a database repository is
    introduced; they do not survive application restarts or worker changes.
    """

    def __init__(self, price_provider: StockPriceProvider | None = None) -> None:
        self._price_provider = price_provider or YahooFinanceStockPriceProvider()
        self._holdings: dict[UUID, PortfolioHolding] = {}
        self._lock = threading.RLock()

    def add_holding(
        self, symbol: str, quantity: int, purchase_price: float
    ) -> PortfolioHolding:
        """Add a holding after validating the symbol and values."""
        normalized_symbol = self._normalize_symbol(symbol)
        self._validate_holding_values(quantity, purchase_price)
        holding = PortfolioHolding(
            holding_id=uuid4(),
            symbol=normalized_symbol,
            quantity=quantity,
            purchase_price=purchase_price,
        )
        with self._lock:
            self._holdings[holding.holding_id] = holding
        return holding

    def list_holdings(self) -> list[PortfolioHolding]:
        """Return a stable snapshot of current holdings."""
        with self._lock:
            return list(self._holdings.values())

    def update_holding(
        self, holding_id: UUID, quantity: int, purchase_price: float
    ) -> PortfolioHolding:
        """Replace quantity and purchase price for an existing holding."""
        self._validate_holding_values(quantity, purchase_price)
        with self._lock:
            current = self._holdings.get(holding_id)
            if current is None:
                raise PortfolioHoldingNotFoundError(str(holding_id))
            updated = PortfolioHolding(
                holding_id=current.holding_id,
                symbol=current.symbol,
                quantity=quantity,
                purchase_price=purchase_price,
            )
            self._holdings[holding_id] = updated
            return updated

    def delete_holding(self, holding_id: UUID) -> None:
        """Delete an existing holding."""
        with self._lock:
            if holding_id not in self._holdings:
                raise PortfolioHoldingNotFoundError(str(holding_id))
            del self._holdings[holding_id]

    def get_latest_price(self, symbol: str) -> StockPrice | None:
        """Return the provider's latest price for a symbol."""
        return self._price_provider.get_latest_price(symbol)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol not in STOCK_CATALOG:
            raise UnsupportedStockSymbolError(normalized_symbol)
        return normalized_symbol

    @staticmethod
    def _validate_holding_values(quantity: int, purchase_price: float) -> None:
        if quantity <= 0:
            raise ValueError("Quantity must be greater than zero.")
        if not math.isfinite(purchase_price) or purchase_price <= 0:
            raise ValueError("Purchase price must be a finite value greater than zero.")


@lru_cache(maxsize=1)
def get_portfolio_service() -> PortfolioService:
    """Return the process-wide portfolio service instance."""
    return PortfolioService(price_provider=ProcessedStockPriceProvider())
