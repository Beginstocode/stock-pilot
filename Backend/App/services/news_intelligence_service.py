"""Persistence for original news impact assessments and later outcomes."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left, bisect_right
from datetime import date, time
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.config import DATA_DIR
from Backend.App.services.news_impact_service import NewsImpactContext
from Backend.App.schemas.news_impact import NewsImpactResponse


class NewsIntelligencePersistenceError(RuntimeError):
    """Raised when persistence fails for a reason other than a duplicate event."""


class HistoricalOutcomeDataError(RuntimeError):
    """Raised when existing price datasets cannot support outcome calculation."""


MARKET_TZ = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE = time(15, 30)


def _utc_timestamp(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


@lru_cache(maxsize=4)
def _cached_nifty_sessions(path: str, modified_ns: int, file_size: int) -> tuple[date, ...]:
    del modified_ns, file_size
    frame = pd.read_csv(path, usecols=["Date", "Close"])
    prices: dict[date, float] = {}
    for row in frame.to_dict("records"):
        try:
            session = pd.Timestamp(row["Date"]).date()
            close = float(row["Close"])
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(close) or close <= 0:
            continue
        if session in prices and not math.isclose(prices[session], close, abs_tol=1e-8):
            return ()
        prices[session] = close
    return tuple(sorted(prices))


def load_nifty_session_dates(nifty_path: Path | None = None) -> tuple[date, ...]:
    """Return the dates represented by the configured NIFTY price file."""
    if nifty_path is not None:
        path = nifty_path
    else:
        path = DATA_DIR / "cleaned" / "cleaned_nifty50.csv"
        if not path.is_file():
            path = DATA_DIR / "raw" / "nifty50.csv"
    try:
        stat = path.stat()
        return _cached_nifty_sessions(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return ()


def expected_outcome_observations(
    analysis_timestamp: datetime,
    sessions: Sequence[date],
) -> dict[str, datetime]:
    """Compute each horizon's exact expected NIFTY-session close from analysis time."""
    prediction_time = _utc_timestamp(analysis_timestamp)
    if prediction_time is None or not sessions:
        return {}
    local_prediction_time = prediction_time.astimezone(MARKET_TZ)
    anchor_index = (
        bisect_right(sessions, local_prediction_time.date())
        if local_prediction_time.time() >= MARKET_CLOSE
        else bisect_left(sessions, local_prediction_time.date())
    )
    observations = {}
    for horizon, distance in (("1d", 1), ("3d", 3), ("5d", 5)):
        outcome_index = anchor_index + distance - 1
        if outcome_index < len(sessions):
            observations[horizon] = datetime.combine(
                sessions[outcome_index], MARKET_CLOSE, MARKET_TZ
            ).astimezone(timezone.utc)
    return observations


def _article_identity(url: str) -> tuple[str, str]:
    parts = urlsplit(url.strip())
    normalized = urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, ""))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest(), normalized


def _finbert_confidence(context: NewsImpactContext) -> float | None:
    probabilities = {
        "Positive": context.positive_probability,
        "Negative": context.negative_probability,
        "Neutral": context.neutral_probability,
    }
    value = probabilities.get(context.sentiment or "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def persist_impact_analysis(
    db: Session,
    *,
    context: NewsImpactContext,
    response: NewsImpactResponse,
    analysis_timestamp: datetime | None = None,
    model_version: str = "unknown",
) -> list[NewsIntelligenceEvent]:
    """Insert one immutable event per affected symbol; existing events are retained."""
    published_at = _utc_timestamp(context.published_at)
    analyzed_at = _utc_timestamp(analysis_timestamp or datetime.now(timezone.utc))
    if published_at is None or analyzed_at is None:
        return []
    if analyzed_at < published_at:
        return []
    try:
        article_id, article_url = _article_identity(context.article_url)
    except ValueError:
        return []
    if not article_url or len(article_url) > 2048:
        return []
    title = (context.article_title or context.headline).strip()[:500]
    if not title:
        return []

    saved: list[NewsIntelligenceEvent] = []
    try:
        for symbol in dict.fromkeys(response.affected_symbols):
            normalized_symbol = symbol.strip().upper()
            existing = db.scalar(
                select(NewsIntelligenceEvent).where(
                    NewsIntelligenceEvent.article_id == article_id,
                    NewsIntelligenceEvent.stock_symbol == normalized_symbol,
                )
            )
            if existing is not None:
                saved.append(existing)
                continue
            event = NewsIntelligenceEvent(
                article_id=article_id,
                article_url=article_url,
                article_title=title,
                published_at=published_at,
                analysis_timestamp=analyzed_at,
                stock_symbol=normalized_symbol,
                finbert_sentiment=context.sentiment,
                finbert_confidence=_finbert_confidence(context),
                ai_impact_direction=response.impact_direction,
                ai_impact_strength=response.impact_strength,
                ai_explanation=response.explanation,
                content_source=response.content_basis,
                content_quality=response.content_quality,
                analysis_model_version=(model_version or "unknown")[:120],
            )
            db.add(event)
            saved.append(event)
        db.commit()
        for event in saved:
            db.refresh(event)
        return saved
    except IntegrityError:
        db.rollback()
        # A concurrent request may have inserted the same unique article-symbol row.
        rows = list(
            db.scalars(
                select(NewsIntelligenceEvent).where(
                    NewsIntelligenceEvent.article_id == article_id,
                    NewsIntelligenceEvent.stock_symbol.in_(response.affected_symbols),
                )
            ).all()
        )
        if rows:
            return rows
        raise NewsIntelligencePersistenceError("Could not persist the news impact event.")
    except SQLAlchemyError as exc:
        db.rollback()
        raise NewsIntelligencePersistenceError("Could not persist the news impact event.") from exc


def update_news_outcomes(
    db: Session,
    *,
    event_id: int,
    horizons: dict[str, dict[str, Any]],
    calculated_at: datetime | None = None,
    expected_observations: dict[str, datetime] | None = None,
) -> NewsIntelligenceOutcome:
    """Insert or update observed outcomes, rejecting prices before prediction time."""
    calculated = _utc_timestamp(calculated_at or datetime.now(timezone.utc))
    if calculated is None:
        raise ValueError("Outcome calculation timestamp must include a timezone.")
    if not horizons:
        raise ValueError("At least one observed outcome horizon is required.")
    event = db.get(NewsIntelligenceEvent, event_id)
    if event is None:
        raise LookupError(f"News event {event_id} was not found.")
    prediction_time = event.analysis_timestamp
    # SQLite drops timezone metadata for timezone=True columns; persisted values
    # are written in UTC, so restore that known convention on read.
    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.replace(tzinfo=timezone.utc)
    else:
        prediction_time = prediction_time.astimezone(timezone.utc)
    validated: dict[str, dict[str, Any]] = {}
    for horizon, values in horizons.items():
        if horizon not in {"1d", "3d", "5d"}:
            raise ValueError(f"Unsupported outcome horizon: {horizon}.")
        observed_at = _utc_timestamp(values.get("observed_at"))
        if observed_at is None or observed_at <= prediction_time:
            raise ValueError(f"{horizon} outcome price must be observed after the original prediction.")
        if expected_observations is not None and expected_observations.get(horizon) != observed_at:
            raise ValueError(f"{horizon} outcome must use the expected NIFTY trading-session close.")
        if calculated < observed_at:
            raise ValueError(f"{horizon} outcome cannot be calculated before its observation.")
        stock = values.get("stock_return_pct")
        nifty = values.get("nifty_return_pct")
        excess = values.get("excess_return_pct")
        if any(value is None or not math.isfinite(float(value)) for value in (stock, nifty, excess)):
            raise ValueError(f"{horizon} outcome requires finite stock, NIFTY, and excess returns.")
        if not math.isclose(float(stock) - float(nifty), float(excess), abs_tol=0.00001):
            raise ValueError(f"{horizon} excess return must equal stock return minus NIFTY return.")
        validated[horizon] = {
            "observed_at": observed_at,
            "stock_return_pct": Decimal(str(stock)),
            "nifty_return_pct": Decimal(str(nifty)),
            "excess_return_pct": Decimal(str(excess)),
        }

    row = db.scalar(select(NewsIntelligenceOutcome).where(NewsIntelligenceOutcome.event_id == event_id))
    if row is None:
        row = NewsIntelligenceOutcome(event_id=event_id, calculated_at=calculated)
        db.add(row)
    for horizon, values in validated.items():
        suffix = horizon
        setattr(row, f"stock_return_{suffix}_pct", values["stock_return_pct"])
        setattr(row, f"nifty_return_{suffix}_pct", values["nifty_return_pct"])
        setattr(row, f"excess_return_{suffix}_pct", values["excess_return_pct"])
        setattr(row, f"observed_at_{suffix}", values["observed_at"])
    row.calculated_at = calculated
    try:
        db.commit()
        db.refresh(row)
        return row
    except SQLAlchemyError as exc:
        db.rollback()
        raise NewsIntelligencePersistenceError("Could not persist historical news outcomes.") from exc


def calculate_ready_news_outcomes(
    db: Session,
    *,
    stock_prices_path: Path | None = None,
    nifty_path: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Calculate ready event outcomes from existing daily prices, anchored at AI analysis time."""
    stock_file = stock_prices_path or DATA_DIR / "processed" / "stock_price_combined.csv"
    benchmark_file = nifty_path or DATA_DIR / "cleaned" / "cleaned_nifty50.csv"
    if not benchmark_file.is_file() and (DATA_DIR / "raw" / "nifty50.csv").is_file():
        benchmark_file = DATA_DIR / "raw" / "nifty50.csv"
    try:
        stocks = pd.read_csv(stock_file, low_memory=False)
        nifty = pd.read_csv(benchmark_file, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HistoricalOutcomeDataError(f"Could not read existing price datasets: {exc}") from exc
    stocks.columns = stocks.columns.astype(str).str.strip()
    nifty.columns = nifty.columns.astype(str).str.strip()
    if not {"Date", "Symbol", "Close"}.issubset(stocks.columns):
        raise HistoricalOutcomeDataError("Historical stock prices are missing Date, Symbol, or Close columns.")
    if not {"Date", "Close"}.issubset(nifty.columns):
        raise HistoricalOutcomeDataError("NIFTY 50 prices are missing Date or Close columns.")

    calculated_at = _utc_timestamp(now or datetime.now(timezone.utc))
    assert calculated_at is not None
    price_cutoff_date = calculated_at.astimezone(MARKET_TZ).date()
    nifty_prices: dict[date, float] = {}
    for row in nifty.to_dict("records"):
        try:
            session_date = pd.Timestamp(row["Date"]).date()
            close = float(row["Close"])
        except (TypeError, ValueError, KeyError):
            continue
        if session_date > price_cutoff_date or not math.isfinite(close) or close <= 0:
            continue
        if session_date in nifty_prices and not math.isclose(nifty_prices[session_date], close, abs_tol=1e-8):
            raise HistoricalOutcomeDataError(f"NIFTY 50 has conflicting duplicate prices for {session_date}.")
        nifty_prices[session_date] = close
    sessions = sorted(nifty_prices)
    if not sessions:
        return 0

    stock_prices: dict[str, dict[date, float]] = {}
    stock_price_column = "Adj Close" if "Adj Close" in stocks.columns else "Close"
    for row in stocks.to_dict("records"):
        symbol = str(row.get("Symbol") or "").strip().upper()
        try:
            session_date = pd.Timestamp(row["Date"]).date()
        except (TypeError, ValueError, KeyError):
            continue
        if not symbol or session_date > price_cutoff_date:
            continue
        value = row.get(stock_price_column)
        try:
            close = float(value)
        except (TypeError, ValueError):
            close = math.nan
        if not math.isfinite(close) or close <= 0:
            continue
        by_date = stock_prices.setdefault(symbol, {})
        if session_date in by_date:
            raise HistoricalOutcomeDataError(f"{symbol} has duplicate price rows for {session_date}.")
        by_date[session_date] = close

    events = list(db.scalars(select(NewsIntelligenceEvent)).all())
    existing_outcomes = {
        row.event_id: row
        for row in db.scalars(select(NewsIntelligenceOutcome)).all()
    }
    updated = 0
    for event in events:
        prediction_time = event.analysis_timestamp
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=timezone.utc)
        else:
            prediction_time = prediction_time.astimezone(timezone.utc)
        local_prediction_time = prediction_time.astimezone(MARKET_TZ)
        if local_prediction_time.time() >= MARKET_CLOSE:
            anchor_index = bisect_right(sessions, local_prediction_time.date())
        else:
            anchor_index = bisect_left(sessions, local_prediction_time.date())
        baseline_index = anchor_index - 1
        if anchor_index >= len(sessions) or baseline_index < 0:
            continue
        baseline_date = sessions[baseline_index]
        baseline_observed = datetime.combine(baseline_date, MARKET_CLOSE, MARKET_TZ).astimezone(timezone.utc)
        if baseline_observed > prediction_time:
            continue
        symbol_series = stock_prices.get(event.stock_symbol, {})
        expected_observations = expected_outcome_observations(prediction_time, sessions)
        existing = existing_outcomes.get(event.event_id)
        stock_base = symbol_series.get(baseline_date)
        nifty_base = nifty_prices.get(baseline_date)
        if not stock_base or not nifty_base:
            continue

        horizons: dict[str, dict[str, Any]] = {}
        for horizon, distance in (("1d", 1), ("3d", 3), ("5d", 5)):
            if existing is not None and all(
                getattr(existing, f"{field}_{horizon}{suffix}") is not None
                for field, suffix in (
                    ("stock_return", "_pct"),
                    ("nifty_return", "_pct"),
                    ("excess_return", "_pct"),
                    ("observed_at", ""),
                )
            ):
                continue
            outcome_index = anchor_index + distance - 1
            if outcome_index >= len(sessions):
                continue
            outcome_date = sessions[outcome_index]
            observed_at = datetime.combine(outcome_date, MARKET_CLOSE, MARKET_TZ).astimezone(timezone.utc)
            if expected_observations.get(horizon) != observed_at:
                continue
            if observed_at <= prediction_time or observed_at > calculated_at:
                continue
            stock_outcome = symbol_series.get(outcome_date)
            nifty_outcome = nifty_prices.get(outcome_date)
            if not stock_outcome or not nifty_outcome:
                continue
            stock_return = (stock_outcome / stock_base - 1.0) * 100.0
            nifty_return = (nifty_outcome / nifty_base - 1.0) * 100.0
            horizons[horizon] = {
                "observed_at": observed_at,
                "stock_return_pct": stock_return,
                "nifty_return_pct": nifty_return,
                "excess_return_pct": stock_return - nifty_return,
            }
        if horizons:
            update_news_outcomes(
                db,
                event_id=event.event_id,
                horizons=horizons,
                calculated_at=calculated_at,
                expected_observations=expected_observations,
            )
            updated += 1
    return updated


__all__ = [
    "NewsIntelligencePersistenceError",
    "HistoricalOutcomeDataError",
    "calculate_ready_news_outcomes",
    "expected_outcome_observations",
    "load_nifty_session_dates",
    "persist_impact_analysis",
    "update_news_outcomes",
]
