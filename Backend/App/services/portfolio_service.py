"""Portfolio management and valuation using the stock data service."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Protocol
from uuid import UUID, uuid4

from Backend.App.services.stock_service import (
    StockDataNotFoundError,
    StockDataUnavailableError,
    StockService,
    UnsupportedStockSymbolError,
)
from Backend.App.stock_catalog import STOCK_CATALOG

_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


class PortfolioHoldingNotFoundError(LookupError):
    """Raised when a requested holding does not exist."""


@dataclass(frozen=True)
class PortfolioHolding:
    """A stock holding recorded in the portfolio."""

    holding_id: UUID
    symbol: str
    quantity: int
    purchase_price: float

    @property
    def total_investment(self) -> float:
        """Return the quantity multiplied by purchase price."""
        return self.quantity * self.purchase_price


@dataclass(frozen=True)
class HoldingValuation:
    """A holding with an optional price from processed market data."""

    holding: PortfolioHolding
    current_price: float | None
    current_price_timestamp: datetime | None
    current_value: float | None
    profit_loss: float | None


@dataclass(frozen=True)
class PortfolioSummary:
    """Portfolio holdings and aggregate investment valuation."""

    holdings: tuple[HoldingValuation, ...]
    total_investment: float
    current_value: float | None
    profit_loss: float | None


class PortfolioRepository(Protocol):
    """Persistence contract for portfolio holdings.

    A database-backed repository can implement this interface without changing
    portfolio business logic.
    """

    def add(self, holding: PortfolioHolding) -> PortfolioHolding: ...

    def get(self, holding_id: UUID) -> PortfolioHolding | None: ...

    def list_all(self) -> list[PortfolioHolding]: ...

    def update(self, holding: PortfolioHolding) -> PortfolioHolding: ...

    def delete(self, holding_id: UUID) -> bool: ...


class InMemoryPortfolioRepository:
    """Thread-safe, process-local holding storage for pre-database use."""

    def __init__(self) -> None:
        self._holdings: dict[UUID, PortfolioHolding] = {}
        self._lock = threading.RLock()

    def add(self, holding: PortfolioHolding) -> PortfolioHolding:
        with self._lock:
            self._holdings[holding.holding_id] = holding
        return holding

    def get(self, holding_id: UUID) -> PortfolioHolding | None:
        with self._lock:
            return self._holdings.get(holding_id)

    def list_all(self) -> list[PortfolioHolding]:
        with self._lock:
            return list(self._holdings.values())

    def update(self, holding: PortfolioHolding) -> PortfolioHolding:
        with self._lock:
            if holding.holding_id not in self._holdings:
                raise PortfolioHoldingNotFoundError(str(holding.holding_id))
            self._holdings[holding.holding_id] = holding
        return holding

    def delete(self, holding_id: UUID) -> bool:
        with self._lock:
            return self._holdings.pop(holding_id, None) is not None


class PortfolioService:
    """Portfolio operations and valuations based on actual stored stock data."""

    def __init__(
        self,
        repository: PortfolioRepository | None = None,
        stock_service: StockService | None = None,
    ) -> None:
        self._repository = repository or InMemoryPortfolioRepository()
        self._stock_service = stock_service or StockService()

    def add_holding(
        self, symbol: str, quantity: int, purchase_price: int | float
    ) -> HoldingValuation:
        """Add a holding and return its valuation when a close is available."""
        normalized_symbol = self._normalize_symbol(symbol)
        self._validate_values(quantity, purchase_price)
        holding = PortfolioHolding(
            holding_id=uuid4(),
            symbol=normalized_symbol,
            quantity=quantity,
            purchase_price=float(purchase_price),
        )
        self._repository.add(holding)
        return self._value_holding(holding)

    def get_holding(self, holding_id: UUID) -> HoldingValuation:
        """Get one holding and its latest available processed close."""
        holding = self._repository.get(holding_id)
        if holding is None:
            raise PortfolioHoldingNotFoundError(str(holding_id))
        return self._value_holding(holding)

    def list_holdings(self) -> PortfolioSummary:
        """Return all holdings and aggregate their available valuations.

        Aggregate current value and profit/loss are None if any holding lacks a
        usable closing price; partial valuations are never presented as totals.
        """
        holdings = tuple(
            self._value_holding(holding)
            for holding in self._repository.list_all()
        )
        total_investment = math.fsum(
            item.holding.total_investment for item in holdings
        )
        if all(item.current_value is not None for item in holdings):
            current_value: float | None = math.fsum(
                item.current_value
                for item in holdings
                if item.current_value is not None
            )
            profit_loss: float | None = current_value - total_investment
        else:
            current_value = None
            profit_loss = None

        return PortfolioSummary(
            holdings=holdings,
            total_investment=total_investment,
            current_value=current_value,
            profit_loss=profit_loss,
        )

    def update_holding(
        self,
        holding_id: UUID,
        quantity: int,
        purchase_price: int | float,
    ) -> HoldingValuation:
        """Update quantity and purchase price for an existing holding."""
        self._validate_values(quantity, purchase_price)
        existing = self._repository.get(holding_id)
        if existing is None:
            raise PortfolioHoldingNotFoundError(str(holding_id))
        updated = PortfolioHolding(
            holding_id=existing.holding_id,
            symbol=existing.symbol,
            quantity=quantity,
            purchase_price=float(purchase_price),
        )
        self._repository.update(updated)
        return self._value_holding(updated)

    def delete_holding(self, holding_id: UUID) -> None:
        """Delete a holding or raise if it does not exist."""
        if not self._repository.delete(holding_id):
            raise PortfolioHoldingNotFoundError(str(holding_id))

    def _value_holding(self, holding: PortfolioHolding) -> HoldingValuation:
        """Value a holding from the latest processed close, if present."""
        try:
            stock_data = self._stock_service.get_stock_details(holding.symbol)
        except (StockDataNotFoundError, StockDataUnavailableError):
            stock_data = None

        current_price = (
            stock_data.close
            if stock_data is not None
            and stock_data.close is not None
            and math.isfinite(stock_data.close)
            and stock_data.close > 0
            else None
        )
        current_value = (
            current_price * holding.quantity if current_price is not None else None
        )
        return HoldingValuation(
            holding=holding,
            current_price=current_price,
            current_price_timestamp=(
                stock_data.date if current_price is not None and stock_data else None
            ),
            current_value=current_value,
            profit_loss=(
                current_value - holding.total_investment
                if current_value is not None
                else None
            ),
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
    def _validate_values(quantity: int, purchase_price: int | float) -> None:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        if (
            isinstance(purchase_price, bool)
            or not isinstance(purchase_price, (int, float))
            or not math.isfinite(float(purchase_price))
            or purchase_price <= 0
        ):
            raise ValueError("Purchase price must be a finite number greater than zero.")


@lru_cache(maxsize=1)
def get_portfolio_service() -> PortfolioService:
    """Return one process-local portfolio service instance."""
    return PortfolioService()


__all__ = [
    "HoldingValuation",
    "InMemoryPortfolioRepository",
    "PortfolioHolding",
    "PortfolioHoldingNotFoundError",
    "PortfolioRepository",
    "PortfolioService",
    "PortfolioSummary",
    "get_portfolio_service",
]
