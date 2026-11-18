"""Pydantic v2 schemas for portfolio holding requests and responses."""

from __future__ import annotations

import math
import re
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from Backend.App.stock_catalog import STOCK_CATALOG

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


class HoldingCreateRequest(BaseModel):
    """Request body for adding a stock holding."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Supported Indian stock symbol, such as RELIANCE or TCS",
        examples=["RELIANCE"],
    )
    quantity: int = Field(..., gt=0, description="Number of shares held")
    purchase_price: float = Field(
        ...,
        gt=0.0,
        allow_inf_nan=False,
        description="Purchase price per share in INR",
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

    @field_validator("quantity", mode="before")
    @classmethod
    def validate_quantity_type(cls, value: object) -> object:
        """Require whole-number quantities and reject booleans."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Quantity must be a positive integer.")
        return value


class HoldingUpdateRequest(BaseModel):
    """Request body for updating quantity and purchase price."""

    model_config = ConfigDict(extra="forbid")

    quantity: int = Field(..., gt=0, description="Updated number of shares held")
    purchase_price: float = Field(
        ...,
        gt=0.0,
        allow_inf_nan=False,
        description="Updated purchase price per share in INR",
    )

    @field_validator("quantity", mode="before")
    @classmethod
    def validate_quantity_type(cls, value: object) -> object:
        """Require whole-number quantities and reject booleans."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Quantity must be a positive integer.")
        return value


class HoldingDeleteResponse(BaseModel):
    """Confirmation details for a deleted holding."""

    model_config = ConfigDict(extra="forbid")

    holding_id: UUID = Field(..., description="ID of the deleted portfolio holding")
    deleted: bool = Field(..., description="Whether the holding was deleted")


class HoldingResponse(BaseModel):
    """A portfolio holding and its available price-based valuation."""

    model_config = ConfigDict(extra="forbid")

    holding_id: UUID = Field(..., description="Unique portfolio holding identifier")
    symbol: str = Field(..., description="Supported Indian stock symbol")
    quantity: int = Field(..., gt=0, description="Number of shares held")
    purchase_price: float = Field(
        ..., gt=0.0, allow_inf_nan=False, description="Purchase price per share in INR"
    )
    total_investment: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Quantity multiplied by purchase price"
    )
    current_price: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="Available market price per share in INR; null if unavailable",
    )
    current_value: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Current holding value; null if current price is unavailable",
    )
    profit_loss: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Current value minus investment; null if price is unavailable",
    )
    profit_loss_percentage: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Profit/loss as a percentage of investment; null if price is unavailable"
        ),
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

    @model_validator(mode="after")
    def validate_valuation(self) -> HoldingResponse:
        """Ensure unavailable prices do not have fabricated valuation fields."""
        investment = self.quantity * self.purchase_price
        if not math.isclose(
            self.total_investment, investment, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("total_investment must equal quantity times purchase_price.")

        valuation_fields = (
            self.current_price,
            self.current_value,
            self.profit_loss,
            self.profit_loss_percentage,
        )
        if self.current_price is None:
            if any(value is not None for value in valuation_fields[1:]):
                raise ValueError(
                    "Current value and profit/loss must be null when current price "
                    "is unavailable."
                )
            return self

        if any(value is None for value in valuation_fields[1:]):
            raise ValueError(
                "Current value and profit/loss are required when current price is available."
            )
        assert self.current_value is not None
        assert self.profit_loss is not None
        assert self.profit_loss_percentage is not None
        expected_value = self.quantity * self.current_price
        expected_profit_loss = expected_value - investment
        expected_percentage = expected_profit_loss / investment * 100
        if not math.isclose(
            self.current_value, expected_value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("current_value must equal quantity times current_price.")
        if not math.isclose(
            self.profit_loss, expected_profit_loss, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("profit_loss must equal current_value minus investment.")
        if not math.isclose(
            self.profit_loss_percentage,
            expected_percentage,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "profit_loss_percentage must match profit/loss over total investment."
            )
        return self


class PortfolioSummaryResponse(BaseModel):
    """Portfolio holdings and aggregate investment valuation."""

    model_config = ConfigDict(extra="forbid")

    holdings: list[HoldingResponse] = Field(
        ..., description="Portfolio holdings and their available valuations"
    )
    total_investment: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Total portfolio investment"
    )
    current_value: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Aggregate current value; null if any holding lacks a price",
    )
    profit_loss: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Aggregate profit/loss; null if any holding lacks a price",
    )
    profit_loss_percentage: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Aggregate profit/loss percentage; null if value is unavailable",
    )

    @model_validator(mode="after")
    def validate_summary(self) -> PortfolioSummaryResponse:
        """Check total investment and unavailable-price representation."""
        expected_investment = math.fsum(
            holding.total_investment for holding in self.holdings
        )
        if not math.isclose(
            self.total_investment,
            expected_investment,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("total_investment must equal the sum of holding investments.")

        has_missing_value = any(
            holding.current_value is None for holding in self.holdings
        )
        aggregate_fields = (
            self.current_value,
            self.profit_loss,
            self.profit_loss_percentage,
        )
        if has_missing_value:
            if any(value is not None for value in aggregate_fields):
                raise ValueError(
                    "Aggregate market value and profit/loss must be null when any "
                    "holding lacks current price data."
                )
            return self

        if any(value is None for value in aggregate_fields):
            raise ValueError(
                "Aggregate market value and profit/loss are required when all "
                "holding prices are available."
            )

        assert self.current_value is not None
        assert self.profit_loss is not None
        assert self.profit_loss_percentage is not None
        expected_value = math.fsum(
            holding.current_value
            for holding in self.holdings
            if holding.current_value is not None
        )
        expected_profit_loss = expected_value - self.total_investment
        expected_percentage = (
            expected_profit_loss / self.total_investment * 100
            if self.total_investment > 0
            else 0.0
        )
        if not math.isclose(
            self.current_value, expected_value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("current_value must equal the sum of holding values.")
        if not math.isclose(
            self.profit_loss, expected_profit_loss, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("profit_loss must equal current_value minus investment.")
        if not math.isclose(
            self.profit_loss_percentage,
            expected_percentage,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "profit_loss_percentage must match aggregate profit/loss over investment."
            )
        return self


__all__ = [
    "HoldingCreateRequest",
    "HoldingDeleteResponse",
    "HoldingResponse",
    "HoldingUpdateRequest",
    "PortfolioSummaryResponse",
]
