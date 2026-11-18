"""Pydantic v2 response schemas for stock news sentiment."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from Backend.App.stock_catalog import STOCK_CATALOG

SentimentClassification = Literal["Positive", "Negative", "Neutral"]

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")
_PROBABILITY_FIELDS = (
    "positive_probability",
    "negative_probability",
    "neutral_probability",
)


class SentimentFields(BaseModel):
    """Optional sentiment result fields shared by article and aggregate responses."""

    model_config = ConfigDict(extra="forbid")

    sentiment: SentimentClassification | None = Field(
        default=None,
        description="FinBERT classification; null when sentiment data is unavailable",
    )
    positive_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="FinBERT positive probability, or null when unavailable",
    )
    negative_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="FinBERT negative probability, or null when unavailable",
    )
    neutral_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="FinBERT neutral probability, or null when unavailable",
    )
    sentiment_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Positive probability minus negative probability; null when "
            "sentiment data is unavailable"
        ),
    )

    @model_validator(mode="after")
    def validate_sentiment_data(self) -> SentimentFields:
        """Require sentiment results to be complete and internally consistent."""
        values = [
            self.sentiment,
            self.positive_probability,
            self.negative_probability,
            self.neutral_probability,
            self.sentiment_score,
        ]
        present_count = sum(value is not None for value in values)
        if present_count == 0:
            return self
        if present_count != len(values):
            raise ValueError(
                "Sentiment classification, probabilities, and score must either "
                "all be provided or all be null."
            )

        assert self.positive_probability is not None
        assert self.negative_probability is not None
        assert self.neutral_probability is not None
        assert self.sentiment_score is not None

        probability_total = (
            self.positive_probability
            + self.negative_probability
            + self.neutral_probability
        )
        if not math.isclose(
            probability_total, 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError("Sentiment probabilities must sum to 1.")

        expected_score = self.positive_probability - self.negative_probability
        if not math.isclose(
            self.sentiment_score, expected_score, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                "Sentiment score must equal positive probability minus negative probability."
            )
        return self


class NewsArticleSentimentResponse(SentimentFields):
    """Sentiment result and source metadata for one financial news article."""

    article_id: str | None = Field(
        default=None, description="Optional source article identifier"
    )
    symbol: str = Field(..., description="Supported Indian stock symbol")
    headline: str = Field(..., min_length=1, description="Financial news headline")
    publication_timestamp: datetime = Field(
        ..., description="Article publication timestamp, including timezone"
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

    @field_validator("publication_timestamp")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        """Require an explicit timezone for publication timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Publication timestamp must include a timezone.")
        return value

    @field_validator("headline")
    @classmethod
    def validate_headline(cls, value: str) -> str:
        """Reject whitespace-only headlines."""
        headline = value.strip()
        if not headline:
            raise ValueError("News headline cannot be blank.")
        return headline


class NewsArticlesSentimentResponse(BaseModel):
    """Multiple article sentiment results for one stock symbol."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Supported Indian stock symbol")
    articles: list[NewsArticleSentimentResponse] = Field(
        ..., description="Available article sentiment records"
    )
    article_count: int = Field(
        ..., ge=0, description="Number of article records in this response"
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
    def validate_article_records(self) -> NewsArticlesSentimentResponse:
        """Keep the count and per-article symbol consistent."""
        if self.article_count != len(self.articles):
            raise ValueError("article_count must match the number of articles.")
        if any(article.symbol != self.symbol for article in self.articles):
            raise ValueError("Every article symbol must match the response symbol.")
        return self


class StockSentimentResponse(SentimentFields):
    """Aggregated latest sentiment for a stock, nullable when unavailable."""

    symbol: str = Field(..., description="Supported Indian stock symbol")
    timestamp: datetime | None = Field(
        default=None,
        description="Latest publication timestamp included in the aggregate",
    )
    article_count: int = Field(
        ..., ge=0, description="Number of articles contributing to the aggregate"
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

    @field_validator("timestamp")
    @classmethod
    def require_timezone_aware_timestamp(
        cls, value: datetime | None
    ) -> datetime | None:
        """Require a timezone whenever an aggregate timestamp is present."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Timestamp must include a timezone.")
        return value

    @model_validator(mode="after")
    def validate_aggregate_availability(self) -> StockSentimentResponse:
        """Represent missing aggregates with null fields, not Neutral sentiment."""
        has_sentiment = self.sentiment is not None
        if self.article_count == 0:
            if has_sentiment or self.timestamp is not None:
                raise ValueError(
                    "An aggregate without articles must have null sentiment data "
                    "and timestamp."
                )
        elif not has_sentiment or self.timestamp is None:
            raise ValueError(
                "An aggregate with articles must include sentiment data and a timestamp."
            )
        return self


__all__ = [
    "NewsArticleSentimentResponse",
    "NewsArticlesSentimentResponse",
    "SentimentClassification",
    "SentimentFields",
    "StockSentimentResponse",
]
