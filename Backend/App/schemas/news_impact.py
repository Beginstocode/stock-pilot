"""Validated request and response models for article impact analysis."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
)

ImpactDirection = Literal["Positive", "Negative", "Neutral"]
ImpactStrength = Literal["Low", "Medium", "High"]
ImpactTimeHorizon = Literal["Short-term", "Long-term", "Both"]
ContentSource = Literal["full_article", "rss_summary", "headline_only"]
ContentQuality = Literal["high", "medium", "low"]
Recommendation = Literal["BUY", "HOLD", "SELL"]
RiskText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class NewsImpactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_url: HttpUrl = Field(..., description="URL of an article in the Market News feed")


class NewsImpactAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1, max_length=500)
    affected_symbols: list[str] = Field(..., max_length=10)
    impact_direction: ImpactDirection
    impact_strength: ImpactStrength
    time_horizon: ImpactTimeHorizon
    explanation: str = Field(..., min_length=1, max_length=1500)
    risks: list[RiskText] = Field(..., min_length=1, max_length=8)

    @field_validator("affected_symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, value: object) -> object:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("Affected symbols must be a list of ticker strings.")
        return [item.strip().upper() for item in value]

    @field_validator("summary", "explanation", mode="before")
    @classmethod
    def trim_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RecommendationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    recommendation: Recommendation | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction_date: str | None = None


class NewsImpactResponse(NewsImpactAnalysis):
    content_basis: ContentSource
    content_quality: ContentQuality
    recommendations: list[RecommendationContext]
    disclaimer: str = (
        "Impact direction describes potential business impact, not a forecast of "
        "share-price movement. Existing model recommendations are shown as context "
        "and are not changed by news sentiment."
    )


__all__ = [
    "ImpactDirection",
    "ImpactStrength",
    "ImpactTimeHorizon",
    "ContentSource",
    "ContentQuality",
    "NewsImpactAnalysis",
    "NewsImpactRequest",
    "NewsImpactResponse",
    "RecommendationContext",
]
