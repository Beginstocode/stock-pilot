"""Historical event-study endpoint for Market News outcomes."""

from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.schemas.historical_news import HistoricalNewsPerformanceResponse
from Backend.App.schemas.historical_news import BusinessImpactReturnGroup, HistoricalImpactAvailability
from Backend.App.schemas.historical_performance import HistoricalPerformanceDashboardResponse
from Backend.App.services.historical_performance_service import HistoricalPerformanceService
from Backend.App.services.historical_news_service import (
    HistoricalNewsDataError,
    HistoricalNewsPerformanceService,
)
from Backend.App.services.news_intelligence_service import (
    expected_outcome_observations,
    load_nifty_session_dates,
)
from Backend.App.stock_catalog import STOCK_CATALOG

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/news", tags=["News Performance"])


@lru_cache(maxsize=1)
def get_historical_news_service() -> HistoricalNewsPerformanceService:
    """Create the reusable price/news analysis service."""
    return HistoricalNewsPerformanceService()


@lru_cache(maxsize=1)
def get_historical_performance_service() -> HistoricalPerformanceService:
    """Return the reusable persistent-event analytics service."""
    return HistoricalPerformanceService()


@router.get(
    "/historical-performance",
    response_model=HistoricalPerformanceDashboardResponse,
    summary="Analyze persisted news predictions against subsequent stock outcomes",
)
def get_historical_performance_dashboard(
    db: Session = Depends(get_db),
    stock_symbol: Annotated[str | None, Query(max_length=20)] = None,
    date_from: date | None = None,
    date_to: date | None = None,
    timezone_name: Annotated[str, Query(alias="timezone", min_length=1, max_length=100)] = "UTC",
    sentiment: Literal["Positive", "Negative", "Neutral"] | None = None,
    impact_direction: Literal["Positive", "Negative", "Neutral"] | None = None,
    horizon: Literal["1d", "3d", "5d"] = "1d",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
) -> HistoricalPerformanceDashboardResponse:
    """Return filtered aggregates and a page of stored article-symbol events."""
    symbol = stock_symbol.strip().upper() if stock_symbol else None
    if symbol and symbol not in STOCK_CATALOG:
        raise HTTPException(status_code=422, detail="Unsupported stock symbol.")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be on or before date_to.")
    try:
        return get_historical_performance_service().analyze(
            db,
            stock_symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            timezone_name=timezone_name,
            sentiment=sentiment,
            impact_direction=impact_direction,
            horizon=horizon,
            page=page,
            page_size=page_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Persistent historical news performance is unavailable.")
        raise HTTPException(status_code=503, detail="Historical performance data is unavailable.") from exc


def _stored_impact_metrics(db: Session) -> HistoricalImpactAvailability:
    rows = db.execute(
        select(
            NewsIntelligenceEvent.analysis_timestamp,
            NewsIntelligenceEvent.ai_impact_direction,
            NewsIntelligenceEvent.ai_impact_strength,
            NewsIntelligenceOutcome.stock_return_1d_pct,
            NewsIntelligenceOutcome.observed_at_1d,
            NewsIntelligenceOutcome.excess_return_1d_pct,
            NewsIntelligenceOutcome.calculated_at,
        )
        .join(
            NewsIntelligenceOutcome,
            NewsIntelligenceOutcome.event_id == NewsIntelligenceEvent.event_id,
        )
    ).all()
    scored: list[tuple[str, str, float, float]] = []
    session_dates = load_nifty_session_dates()
    observation_cutoff = datetime.now(timezone.utc)
    for analysis_at, direction, strength, stock_return, observed_at, excess_return, calculated_at in rows:
        if observed_at is None or stock_return is None or excess_return is None:
            continue
        if analysis_at.tzinfo is None:
            analysis_at = analysis_at.replace(tzinfo=timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if calculated_at.tzinfo is None:
            calculated_at = calculated_at.replace(tzinfo=timezone.utc)
        expected_observed_at = expected_outcome_observations(analysis_at, session_dates).get("1d")
        stock_value = float(stock_return)
        excess_value = float(excess_return)
        if (
            observed_at <= analysis_at
            or expected_observed_at is None
            or observed_at != expected_observed_at
            or observed_at > observation_cutoff
            or calculated_at < observed_at
            or not math.isfinite(stock_value)
            or not math.isfinite(excess_value)
        ):
            continue
        scored.append((direction, strength, stock_value, excess_value))

    if not scored:
        return HistoricalImpactAvailability(
            available=False,
            reason="No persisted business-impact event yet has a complete post-analysis 1-day stock and NIFTY outcome.",
        )
    strength_scale = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    strength_pairs = [
        (strength_scale[strength], abs(excess))
        for _, strength, _, excess in scored
        if strength in strength_scale
    ]
    magnitude_correlation = None
    if len(strength_pairs) >= 2:
        strengths, magnitudes = zip(*strength_pairs, strict=True)
        try:
            magnitude_correlation = round(statistics.correlation(strengths, magnitudes), 4)
        except statistics.StatisticsError:
            pass
    grouped = []
    for direction in ("Positive", "Negative", "Neutral"):
        selected = [(stock, excess) for impact, _, stock, excess in scored if impact == direction]
        grouped.append(BusinessImpactReturnGroup(
            impact_direction=direction,
            event_observations=len(selected),
            average_stock_return_1d_pct=round(sum(stock for stock, _ in selected) / len(selected), 4) if selected else None,
            average_excess_return_1d_pct=round(sum(excess for _, excess in selected) / len(selected), 4) if selected else None,
        ))
    return HistoricalImpactAvailability(
        available=True,
        event_observations=len(scored),
        returns_by_impact_direction=grouped,
        strength_magnitude_correlation=magnitude_correlation,
        reason="Returns are summarized by the recorded business-impact category; the categories are not stock-price predictions.",
    )


@router.get(
    "/performance",
    response_model=HistoricalNewsPerformanceResponse,
    summary="Compare historical news sentiment with subsequent stock returns",
)
def get_news_performance(db: Session = Depends(get_db)) -> HistoricalNewsPerformanceResponse:
    """Return post-publication returns, benchmark comparisons, and data coverage."""
    try:
        result = get_historical_news_service().analyze()
        try:
            impact = _stored_impact_metrics(db)
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Stored AI impact outcomes are unavailable; returning price-only analysis.")
            impact = HistoricalImpactAvailability(
                available=False,
                reason="The persistent AI impact outcome store is unavailable.",
            )
        return result.model_copy(update={"ai_impact_performance": impact})
    except HistoricalNewsDataError as exc:
        logger.exception("Historical news performance data is unavailable")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Historical news performance calculation failed")
        raise HTTPException(
            status_code=503,
            detail="Historical news performance could not be calculated.",
        ) from exc


__all__ = ["get_historical_news_service", "router"]
