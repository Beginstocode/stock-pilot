"""Structured responses for historical news event-study analytics."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from Backend.App.schemas.news_impact import ImpactDirection, ImpactStrength

Horizon = Literal["1d", "3d", "5d"]
NewsSentiment = Literal["Positive", "Negative", "Neutral"]


class HorizonOutcome(BaseModel):
    observation_date: date | None
    stock_return_pct: float | None
    nifty_return_pct: float | None
    excess_return_pct: float | None


class HistoricalNewsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    article_url: HttpUrl
    headline: str
    published_at: datetime
    symbol: str
    finbert_sentiment: NewsSentiment | None
    finbert_score: float | None
    ai_impact_direction: ImpactDirection | None = None
    ai_impact_strength: ImpactStrength | None = None
    anchor_date: date | None
    outcomes: dict[Horizon, HorizonOutcome]
    eligible: bool
    exclusion_reason: str | None = None


class HorizonMetrics(BaseModel):
    horizon: Horizon
    event_observations: int
    eligible_articles: int
    average_stock_return_pct: float | None
    average_nifty_return_pct: float | None
    average_excess_return_pct: float | None
    positive_outcomes: int
    negative_outcomes: int
    neutral_outcomes: int
    finbert_alignment_sample_size: int = Field(description="Directional sentiment/excess-return pairs; neutral and unavailable observations are excluded.")
    finbert_alignment_pct: float | None = Field(description="Descriptive polarity alignment, not stock-price prediction accuracy.")
    neutral_sentiment_observations: int
    neutral_excess_return_observations: int
    unavailable_sentiment_observations: int


class SentimentPerformance(BaseModel):
    sentiment: NewsSentiment
    event_observations_1d: int
    event_observations_3d: int
    event_observations_5d: int
    average_excess_return_1d_pct: float | None
    average_excess_return_3d_pct: float | None
    average_excess_return_5d_pct: float | None


class StockPerformance(BaseModel):
    symbol: str
    event_observations_1d: int
    event_observations_3d: int
    event_observations_5d: int
    average_excess_return_1d_pct: float | None
    average_excess_return_3d_pct: float | None
    average_excess_return_5d_pct: float | None


class BusinessImpactReturnGroup(BaseModel):
    impact_direction: ImpactDirection
    event_observations: int
    average_stock_return_1d_pct: float | None
    average_excess_return_1d_pct: float | None


class HistoricalImpactAvailability(BaseModel):
    available: bool = False
    event_observations: int = 0
    returns_by_impact_direction: list[BusinessImpactReturnGroup] = Field(
        default_factory=list,
        description="1D stock and excess returns grouped by business-impact direction; direction is not a price prediction.",
    )
    strength_magnitude_correlation: float | None = None
    reason: str


class HistoricalNewsPerformanceResponse(BaseModel):
    data_as_of: date
    stock_price_coverage_start: date
    stock_price_coverage_end: date
    nifty_coverage_start: date
    nifty_coverage_end: date
    available_symbols: list[str]
    raw_news_records: int
    finbert_records: int
    finbert_available: bool
    finbert_unavailable_reason: str | None = None
    total_events_analyzed: int
    excluded_events: int
    returns_by_horizon: list[HorizonMetrics]
    sentiment_performance: list[SentimentPerformance]
    ai_impact_performance: HistoricalImpactAvailability
    stock_comparison: list[StockPerformance]
    events: list[HistoricalNewsEvent] = Field(max_length=2000)
    disclaimer: str = (
        "Historical associations describe observed returns after publication; they do not establish that news caused price moves. "
        "Sentiment-return alignment is a descriptive association, not proven forecasting accuracy. Business impact categories are not BUY/SELL signals."
    )


__all__ = [
    "HistoricalImpactAvailability",
    "HistoricalNewsEvent",
    "HistoricalNewsPerformanceResponse",
    "HorizonMetrics",
    "HorizonOutcome",
    "SentimentPerformance",
    "StockPerformance",
]
