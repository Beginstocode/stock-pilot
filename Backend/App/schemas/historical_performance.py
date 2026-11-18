"""Validated response models for the persistent historical performance dashboard."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DashboardHorizon = Literal["1d", "3d", "5d"]
DashboardSentiment = Literal["Positive", "Negative", "Neutral"]
DashboardDirection = Literal["Positive", "Negative", "Neutral"]


class DashboardHorizonMetrics(BaseModel):
    horizon: DashboardHorizon
    eligible_events: int
    average_stock_return_pct: float | None
    average_nifty_return_pct: float | None
    average_excess_return_pct: float | None
    positive_outcomes: int
    negative_outcomes: int
    neutral_outcomes: int
    positive_outcome_pct: float | None
    negative_outcome_pct: float | None


class SentimentReturnAlignment(BaseModel):
    """Descriptive polarity agreement, excluding neutral and unavailable cases."""

    eligible_events: int = Field(description="All completed horizon outcomes, including neutral and unavailable sentiment.")
    sample_size: int = Field(description="Directional alignment denominator; excludes neutral sentiment, neutral excess returns, and missing sentiment.")
    aligned_events: int
    alignment_pct: float | None = Field(description="Aligned directional labels divided by sample_size; descriptive association, not forecast accuracy.")
    neutral_sentiment_events: int = Field(description="Completed outcomes with Neutral FinBERT sentiment; excluded from alignment sample.")
    neutral_excess_return_events: int = Field(description="Completed outcomes with excess return in the inclusive -0.05% to +0.05% neutral band; excluded from alignment sample.")
    unavailable_sentiment_events: int = Field(description="Completed outcomes without a recognized FinBERT label; excluded from alignment sample.")


class ImpactReturnGroup(BaseModel):
    """Observed returns grouped by descriptive business-impact category."""

    impact_direction: DashboardDirection
    sample_size: int = Field(description="Number of valid horizon outcomes in this impact-direction group.")
    average_stock_return_pct: float | None = Field(description="Arithmetic mean stock return for the group; null when sample_size is zero.")
    average_excess_return_pct: float | None = Field(description="Arithmetic mean stock-minus-NIFTY return for the group; null when sample_size is zero.")


class MagnitudeCorrelation(BaseModel):
    sample_size: int
    pearson_correlation: float | None


class DashboardHorizonEvaluation(BaseModel):
    horizon: DashboardHorizon
    sentiment_return_alignment: SentimentReturnAlignment = Field(description="Descriptive sentiment polarity alignment; not price prediction accuracy.")
    business_impact_returns: list[ImpactReturnGroup] = Field(description="Observed return summaries grouped by business impact, not BUY/SELL predictions.")
    impact_strength_vs_magnitude: MagnitudeCorrelation


class DashboardStockPerformance(BaseModel):
    symbol: str
    event_count: int
    eligible_events: int
    average_excess_return_pct: float | None
    sample_supported: bool = Field(description="At least five completed selected-horizon stock return observations.")


class DashboardEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int
    article_url: str
    headline: str
    symbol: str
    published_at: datetime
    finbert_sentiment: DashboardSentiment | None = Field(description="Article polarity label; not a stock-price prediction.")
    finbert_confidence: float | None
    ai_impact_direction: DashboardDirection = Field(description="Descriptive business-impact category; not a BUY/SELL signal.")
    ai_impact_strength: Literal["Low", "Medium", "High"]
    stock_returns_pct: dict[DashboardHorizon, float | None]
    nifty_returns_pct: dict[DashboardHorizon, float | None]
    excess_returns_pct: dict[DashboardHorizon, float | None]
    eligible: bool
    actual_excess_direction: DashboardDirection | None = Field(description="Observed stock-minus-NIFTY return direction using a +/-0.05% neutral band.")
    sentiment_return_alignment: Literal[
        "Aligned",
        "Not aligned",
        "Neutral sentiment",
        "Neutral excess return",
        "Sentiment unavailable",
        "Outcome pending",
    ] = Field(description="Descriptive comparison only. Neutral sentiment, neutral excess return, missing sentiment, and pending outcomes are explicit non-alignment statuses.")


class HistoricalPerformanceDashboardResponse(BaseModel):
    data_as_of: date
    filter_timezone: str
    selected_horizon: DashboardHorizon
    total_events: int
    eligible_events: int
    completed_outcomes: dict[DashboardHorizon, int]
    pending_outcomes: int
    performance_by_horizon: list[DashboardHorizonMetrics]
    evaluation_by_horizon: list[DashboardHorizonEvaluation]
    stock_comparison: list[DashboardStockPerformance]
    minimum_stock_sample_size: int
    events: list[DashboardEvent]
    page: int
    page_size: int
    total_pages: int
    methodology: list[str]


__all__ = [
    "DashboardEvent",
    "DashboardHorizonEvaluation",
    "DashboardHorizonMetrics",
    "DashboardStockPerformance",
    "HistoricalPerformanceDashboardResponse",
]
