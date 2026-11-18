"""Pydantic v2 schemas for stock prediction requests and responses."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Recommendation = Literal["BUY", "HOLD", "SELL"]
_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


class PredictionRequest(BaseModel):
    """Request a recommendation for an Indian stock symbol."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock symbol, such as RELIANCE, TCS, INFY, SBIN, or HDFCBANK",
        examples=["RELIANCE"],
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        """Normalize symbols and reject malformed values."""
        if not isinstance(value, str):
            raise ValueError("Stock symbol must be a string.")
        normalized = value.strip().upper()
        if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError("Stock symbol has an invalid format.")
        return normalized


class PredictionProbabilities(BaseModel):
    """Model probabilities for each supported recommendation class."""

    model_config = ConfigDict(extra="forbid")

    BUY: float = Field(..., ge=0.0, le=1.0, description="Probability of BUY")
    HOLD: float = Field(..., ge=0.0, le=1.0, description="Probability of HOLD")
    SELL: float = Field(..., ge=0.0, le=1.0, description="Probability of SELL")

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> PredictionProbabilities:
        """Reject probability sets that do not form a distribution."""
        total = self.BUY + self.HOLD + self.SELL
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("BUY, HOLD, and SELL probabilities must sum to 1.")
        return self


class PredictionResponse(BaseModel):
    """Prediction output returned to API clients."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Normalized Indian stock symbol")
    prediction: Recommendation = Field(
        ..., description="Model-generated BUY, HOLD, or SELL recommendation"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability assigned to the predicted recommendation",
    )
    probabilities: PredictionProbabilities = Field(
        ..., description="Prediction probability for BUY, HOLD, and SELL"
    )
    prediction_timestamp: datetime = Field(
        ..., description="Timestamp when the prediction was generated"
    )
    prediction_date: date | None = Field(
        default=None,
        description="Optional date of the source market observation",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        """Normalize symbols and reject malformed values."""
        if not isinstance(value, str):
            raise ValueError("Stock symbol must be a string.")
        normalized = value.strip().upper()
        if not _STOCK_SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError("Stock symbol has an invalid format.")
        return normalized

    @model_validator(mode="after")
    def confidence_matches_prediction_probability(self) -> PredictionResponse:
        """Ensure confidence reflects the probability of the predicted class."""
        probability = getattr(self.probabilities, self.prediction)
        if not math.isclose(
            self.confidence, probability, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                "Confidence must match the probability of the predicted class."
            )
        return self


__all__ = [
    "PredictionProbabilities",
    "PredictionRequest",
    "PredictionResponse",
    "Recommendation",
]
