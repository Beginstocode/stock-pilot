"""Database-backed analytics for persistent news intelligence outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.orm import Session

from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.schemas.historical_performance import (
    DashboardEvent,
    DashboardHorizonEvaluation,
    DashboardHorizonMetrics,
    DashboardStockPerformance,
    HistoricalPerformanceDashboardResponse,
    ImpactReturnGroup,
    MagnitudeCorrelation,
    SentimentReturnAlignment,
)
from Backend.App.services.news_intelligence_service import (
    expected_outcome_observations,
    load_nifty_session_dates,
)

HORIZONS = ("1d", "3d", "5d")
NEUTRAL_EXCESS_THRESHOLD_PCT = 0.05
MIN_STOCK_SAMPLE_SIZE = 5
STRENGTH_SCORE = {"Low": 1.0, "Medium": 2.0, "High": 3.0}


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _direction(value: float) -> str:
    if value > NEUTRAL_EXCESS_THRESHOLD_PCT:
        return "Positive"
    if value < -NEUTRAL_EXCESS_THRESHOLD_PCT:
        return "Negative"
    return "Neutral"


def local_date_boundaries_to_utc(
    date_from: date | None,
    date_to: date | None,
    timezone_name: str,
) -> tuple[datetime | None, datetime | None, str]:
    """Convert inclusive local calendar dates into a UTC half-open interval."""
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name!r}.") from exc
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from must be on or before date_to.")

    def start_of_local_day(day: date) -> datetime:
        local_midnight = datetime.combine(day, time.min)
        candidates = []
        for fold in (0, 1):
            candidate = local_midnight.replace(tzinfo=zone, fold=fold)
            round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
            round_trip_local = round_trip.replace(tzinfo=None)
            if round_trip_local == local_midnight:
                candidates.append(candidate.astimezone(timezone.utc))
            elif round_trip_local > local_midnight:
                # Midnight falls in a forward clock gap; its round trip is the
                # first valid local instant after the gap (or the next date if
                # a jurisdiction skipped the whole local calendar date).
                candidates.append(round_trip.astimezone(timezone.utc))
        if candidates:
            return min(candidates)
        raise ValueError(f"Could not resolve local midnight for {day} in {timezone_name}.")

    start_utc = start_of_local_day(date_from) if date_from else None
    end_utc = start_of_local_day(date_to + timedelta(days=1)) if date_to else None
    return start_utc, end_utc, zone.key


@dataclass
class _ReturnAccumulator:
    count: int = 0
    stock_sum: float = 0.0
    nifty_sum: float = 0.0
    excess_sum: float = 0.0
    positive: int = 0
    negative: int = 0
    neutral: int = 0

    def add(self, stock: float, nifty: float, excess: float) -> None:
        self.count += 1
        self.stock_sum += stock
        self.nifty_sum += nifty
        self.excess_sum += excess
        self.positive += stock > 0
        self.negative += stock < 0
        self.neutral += stock == 0

    @staticmethod
    def _avg(total: float, count: int) -> float | None:
        return round(total / count, 4) if count else None

    def response(self, horizon: str) -> DashboardHorizonMetrics:
        return DashboardHorizonMetrics(
            horizon=horizon,
            eligible_events=self.count,
            average_stock_return_pct=self._avg(self.stock_sum, self.count),
            average_nifty_return_pct=self._avg(self.nifty_sum, self.count),
            average_excess_return_pct=self._avg(self.excess_sum, self.count),
            positive_outcomes=self.positive,
            negative_outcomes=self.negative,
            neutral_outcomes=self.neutral,
            positive_outcome_pct=round(self.positive / self.count * 100, 2) if self.count else None,
            negative_outcome_pct=round(self.negative / self.count * 100, 2) if self.count else None,
        )


@dataclass
class _SentimentAlignmentAccumulator:
    eligible_events: int = 0
    sample_size: int = 0
    aligned_events: int = 0
    neutral_sentiment_events: int = 0
    neutral_excess_return_events: int = 0
    unavailable_sentiment_events: int = 0

    def add(self, sentiment: str | None, actual: str) -> None:
        self.eligible_events += 1
        if sentiment not in {"Positive", "Negative", "Neutral"}:
            self.unavailable_sentiment_events += 1
        if sentiment == "Neutral":
            self.neutral_sentiment_events += 1
        if actual == "Neutral":
            self.neutral_excess_return_events += 1
        if sentiment in {"Positive", "Negative"} and actual in {"Positive", "Negative"}:
            self.sample_size += 1
            self.aligned_events += sentiment == actual

    def response(self) -> SentimentReturnAlignment:
        return SentimentReturnAlignment(
            eligible_events=self.eligible_events,
            sample_size=self.sample_size,
            aligned_events=self.aligned_events,
            alignment_pct=round(self.aligned_events / self.sample_size * 100, 2) if self.sample_size else None,
            neutral_sentiment_events=self.neutral_sentiment_events,
            neutral_excess_return_events=self.neutral_excess_return_events,
            unavailable_sentiment_events=self.unavailable_sentiment_events,
        )


@dataclass
class _ImpactReturnAccumulator:
    count: int = 0
    stock_sum: float = 0.0
    excess_sum: float = 0.0

    def add(self, stock_return: float, excess_return: float) -> None:
        self.count += 1
        self.stock_sum += stock_return
        self.excess_sum += excess_return

    def response(self, direction: str) -> ImpactReturnGroup:
        return ImpactReturnGroup(
            impact_direction=direction,
            sample_size=self.count,
            average_stock_return_pct=round(self.stock_sum / self.count, 4) if self.count else None,
            average_excess_return_pct=round(self.excess_sum / self.count, 4) if self.count else None,
        )


@dataclass
class _CorrelationAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def response(self) -> MagnitudeCorrelation:
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator_x = self.count * self.sum_x2 - self.sum_x * self.sum_x
        denominator_y = self.count * self.sum_y2 - self.sum_y * self.sum_y
        correlation = None
        if self.count >= 2 and denominator_x > 0 and denominator_y > 0:
            correlation = round(numerator / math.sqrt(denominator_x * denominator_y), 4)
        return MagnitudeCorrelation(sample_size=self.count, pearson_correlation=correlation)


class HistoricalPerformanceService:
    """Stream filtered event rows once, aggregate all horizons, and page table rows."""

    def analyze(
        self,
        db: Session,
        *,
        stock_symbol: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        sentiment: str | None = None,
        impact_direction: str | None = None,
        horizon: str = "1d",
        page: int = 1,
        page_size: int = 50,
        as_of: date | None = None,
        now: datetime | None = None,
        timezone_name: str = "UTC",
    ) -> HistoricalPerformanceDashboardResponse:
        calculation_cutoff = _utc(now or datetime.now(timezone.utc))
        market_sessions = load_nifty_session_dates()
        try:
            date_start_utc, date_end_utc, resolved_timezone = local_date_boundaries_to_utc(
                date_from, date_to, timezone_name
            )
        except ValueError:
            raise
        statement = select(
            NewsIntelligenceEvent.event_id.label("event_id"),
            NewsIntelligenceEvent.article_url.label("article_url"),
            NewsIntelligenceEvent.article_title.label("headline"),
            NewsIntelligenceEvent.stock_symbol.label("symbol"),
            NewsIntelligenceEvent.published_at.label("published_at"),
            NewsIntelligenceEvent.analysis_timestamp.label("analysis_timestamp"),
            NewsIntelligenceEvent.finbert_sentiment.label("sentiment"),
            NewsIntelligenceEvent.finbert_confidence.label("finbert_confidence"),
            NewsIntelligenceEvent.ai_impact_direction.label("impact_direction"),
            NewsIntelligenceEvent.ai_impact_strength.label("impact_strength"),
            NewsIntelligenceOutcome.calculated_at.label("calculated_at"),
        )
        for key in HORIZONS:
            statement = statement.add_columns(
                getattr(NewsIntelligenceOutcome, f"stock_return_{key}_pct").label(f"stock_{key}"),
                getattr(NewsIntelligenceOutcome, f"nifty_return_{key}_pct").label(f"nifty_{key}"),
                getattr(NewsIntelligenceOutcome, f"excess_return_{key}_pct").label(f"excess_{key}"),
                getattr(NewsIntelligenceOutcome, f"observed_at_{key}").label(f"observed_{key}"),
            )
        statement = statement.outerjoin(
            NewsIntelligenceOutcome,
            NewsIntelligenceOutcome.event_id == NewsIntelligenceEvent.event_id,
        )
        if stock_symbol:
            statement = statement.where(NewsIntelligenceEvent.stock_symbol == stock_symbol.upper())
        if sentiment:
            statement = statement.where(NewsIntelligenceEvent.finbert_sentiment == sentiment)
        if impact_direction:
            statement = statement.where(NewsIntelligenceEvent.ai_impact_direction == impact_direction)
        if date_start_utc:
            statement = statement.where(NewsIntelligenceEvent.published_at >= date_start_utc)
        if date_end_utc:
            statement = statement.where(NewsIntelligenceEvent.published_at < date_end_utc)
        statement = statement.order_by(
            NewsIntelligenceEvent.published_at.desc(),
            NewsIntelligenceEvent.event_id.desc(),
        ).execution_options(yield_per=500)

        return_stats = {key: _ReturnAccumulator() for key in HORIZONS}
        sentiment_alignment_stats = {key: _SentimentAlignmentAccumulator() for key in HORIZONS}
        impact_return_stats = {
            key: {direction: _ImpactReturnAccumulator() for direction in ("Positive", "Negative", "Neutral")}
            for key in HORIZONS
        }
        strength_stats = {key: _CorrelationAccumulator() for key in HORIZONS}
        stock_stats: dict[str, dict[str, Any]] = {}
        completed = {key: 0 for key in HORIZONS}
        events_page: list[DashboardEvent] = []
        total_events = 0
        page_start = (page - 1) * page_size
        page_end = page_start + page_size

        for row in db.execute(statement).mappings():
            total_events += 1
            published_at = _utc(row["published_at"])
            analysis_at = _utc(row["analysis_timestamp"])
            expected_observations = expected_outcome_observations(analysis_at, market_sessions)
            symbol = row["symbol"]
            stock_group = stock_stats.setdefault(
                symbol,
                {
                    "event_count": 0,
                    "eligible": 0,
                    "excess_sum": 0.0,
                    "excess_count": 0,
                },
            )
            stock_group["event_count"] += 1
            values_by_horizon: dict[str, tuple[float, float, float] | None] = {}
            stock_returns: dict[str, float | None] = {}
            nifty_returns: dict[str, float | None] = {}
            excess_returns: dict[str, float | None] = {}
            for key in HORIZONS:
                stock_value = _number(row[f"stock_{key}"])
                nifty_value = _number(row[f"nifty_{key}"])
                excess_value = _number(row[f"excess_{key}"])
                observed_at = row[f"observed_{key}"]
                valid = (
                    stock_value is not None
                    and nifty_value is not None
                    and excess_value is not None
                    and observed_at is not None
                    and expected_observations.get(key) == _utc(observed_at)
                    and _utc(observed_at) > analysis_at
                    and _utc(observed_at) <= calculation_cutoff
                    and row["calculated_at"] is not None
                    and _utc(row["calculated_at"]) >= _utc(observed_at)
                    and math.isclose(stock_value - nifty_value, excess_value, abs_tol=0.0001)
                )
                values = (stock_value, nifty_value, excess_value) if valid else None
                values_by_horizon[key] = values
                stock_returns[key] = values[0] if values else None
                nifty_returns[key] = values[1] if values else None
                excess_returns[key] = values[2] if values else None
                if values:
                    completed[key] += 1
                    return_stats[key].add(*values)
                    actual = _direction(values[2])
                    sentiment_alignment_stats[key].add(row["sentiment"], actual)
                    impact_return_stats[key][row["impact_direction"]].add(values[0], values[2])
                    strength = STRENGTH_SCORE.get(row["impact_strength"])
                    if strength is not None:
                        strength_stats[key].add(strength, abs(values[2]))
                if key == horizon and values:
                    stock_group["eligible"] += 1
                    stock_group["excess_sum"] += values[2]
                    stock_group["excess_count"] += 1

            if page_start <= total_events - 1 < page_end:
                selected_values = values_by_horizon[horizon]
                actual_direction = _direction(selected_values[2]) if selected_values else None
                sentiment_label = row["sentiment"] if row["sentiment"] in {"Positive", "Negative", "Neutral"} else None
                if selected_values is None:
                    sentiment_alignment = "Outcome pending"
                elif sentiment_label is None:
                    sentiment_alignment = "Sentiment unavailable"
                elif sentiment_label == "Neutral":
                    sentiment_alignment = "Neutral sentiment"
                elif actual_direction == "Neutral":
                    sentiment_alignment = "Neutral excess return"
                else:
                    sentiment_alignment = "Aligned" if sentiment_label == actual_direction else "Not aligned"
                events_page.append(
                    DashboardEvent(
                        event_id=row["event_id"],
                        article_url=row["article_url"],
                        headline=row["headline"],
                        symbol=symbol,
                        published_at=published_at,
                        finbert_sentiment=sentiment_label,
                        finbert_confidence=_number(row["finbert_confidence"]),
                        ai_impact_direction=row["impact_direction"],
                        ai_impact_strength=row["impact_strength"],
                        stock_returns_pct=stock_returns,
                        nifty_returns_pct=nifty_returns,
                        excess_returns_pct=excess_returns,
                        eligible=selected_values is not None,
                        actual_excess_direction=actual_direction,
                        sentiment_return_alignment=sentiment_alignment,
                    )
                )

        stock_comparison = []
        for symbol, group in stock_stats.items():
            stock_comparison.append(
                DashboardStockPerformance(
                    symbol=symbol,
                    event_count=group["event_count"],
                    eligible_events=group["eligible"],
                    average_excess_return_pct=round(group["excess_sum"] / group["excess_count"], 4) if group["excess_count"] else None,
                    sample_supported=group["eligible"] >= MIN_STOCK_SAMPLE_SIZE,
                )
            )
        stock_comparison.sort(key=lambda item: (item.average_excess_return_pct is None, -(item.average_excess_return_pct or 0), -item.eligible_events, item.symbol))

        eligible_events = completed[horizon]
        return HistoricalPerformanceDashboardResponse(
            data_as_of=as_of or datetime.now(timezone.utc).date(),
            filter_timezone=resolved_timezone,
            selected_horizon=horizon,
            total_events=total_events,
            eligible_events=eligible_events,
            completed_outcomes=completed,
            pending_outcomes=total_events - eligible_events,
            performance_by_horizon=[return_stats[key].response(key) for key in HORIZONS],
            evaluation_by_horizon=[
                DashboardHorizonEvaluation(
                    horizon=key,
                    sentiment_return_alignment=sentiment_alignment_stats[key].response(),
                    business_impact_returns=[
                        impact_return_stats[key][direction].response(direction)
                        for direction in ("Positive", "Negative", "Neutral")
                    ],
                    impact_strength_vs_magnitude=strength_stats[key].response(),
                )
                for key in HORIZONS
            ],
            stock_comparison=stock_comparison,
            minimum_stock_sample_size=MIN_STOCK_SAMPLE_SIZE,
            events=events_page,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total_events / page_size) if total_events else 0,
            methodology=[
                "One persisted row represents one article and stock symbol. An article affecting multiple stocks creates multiple correlated observations, so event counts are not unique-article counts or independent samples.",
                "An outcome is eligible only when stock, NIFTY, and excess returns are present, internally consistent, and observed after the original AI analysis timestamp.",
                "Each outcome observation must match the expected 15:30 IST close for its 1D, 3D, or 5D NIFTY trading-session horizon; records with other dates or times are excluded. The horizon calendar comes from the existing NIFTY price dataset.",
                "The NIFTY session calendar could not be read, so stored outcomes cannot be validated and are excluded from metrics." if not market_sessions else "NIFTY session calendar validation is active for the existing price dataset.",
                "Stock returns use Adj Close consistently when that column exists in the stock dataset; rows without a valid adjusted close are unavailable. Close is used only when the dataset has no Adj Close column. NIFTY uses Close. The source does not record per-row price-basis provenance, and NIFTY index closes do not include a dividend-adjusted total-return series.",
                "Existing persisted outcomes do not record their historical price basis; rows calculated before consistent basis selection cannot be reclassified from stored fields alone.",
                "Business impact direction describes potential business impact and is not a BUY/SELL or stock-price prediction; subsequent returns are summarized by Positive, Negative, and Neutral impact group for each horizon.",
                "Sentiment-Return Alignment is descriptive polarity agreement for directional FinBERT labels against directional excess returns; neutral sentiment, neutral excess returns, and unavailable sentiment are excluded from its denominator and separately counted.",
                "Excess-return direction uses a neutral band from -0.05% through +0.05%; alignment uses only Positive/Negative sentiment and Positive/Negative excess-return directions. Positive and negative outcome shares use the sign of the stock return.",
                "Impact strength is an ordinal Low/Medium/High score compared with the absolute excess return using Pearson correlation; this is descriptive association, not causation or a price forecast.",
                f"Publication date filters are inclusive local calendar dates in {resolved_timezone}; database boundaries are converted to UTC, and displayed timestamps use the viewer's local timezone.",
                "Stock excess-return rows are marked sample-supported at five or more completed selected-horizon observations; missing historical prices remain unavailable rather than imputed.",
                "The selected horizon changes the alignment card, business-impact chart, stock-wise summary, pending count, and event-row interpretation. Overview returns and the business-impact table continue to show all three horizons.",
            ],
        )


__all__ = ["HistoricalPerformanceService", "local_date_boundaries_to_utc"]
