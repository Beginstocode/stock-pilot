"""Persistence tests for immutable news predictions and future outcomes."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.schemas.news_impact import NewsImpactResponse
from Backend.App.services.news_impact_service import NewsImpactContext
from Backend.App.services.news_intelligence_service import (
    NewsIntelligencePersistenceError,
    calculate_ready_news_outcomes,
    persist_impact_analysis,
    update_news_outcomes,
)


def _context(**changes) -> NewsImpactContext:
    values = {
        "article_url": "https://news.example.test/story/one",
        "headline": "Company announces quarterly results",
        "source": "Example News",
        "published_at": "2026-11-10T10:00:00Z",
        "associated_symbols": ("TCS", "INFY"),
        "sentiment": "Positive",
        "sentiment_score": 0.6,
        "positive_probability": 0.7,
        "negative_probability": 0.1,
        "neutral_probability": 0.2,
        "article_text": "Revenue increased during the quarter.",
        "content_source": "rss_summary",
        "content_quality": "medium",
        "article_title": "Quarterly result update",
    }
    values.update(changes)
    return NewsImpactContext(**values)


def _response(*, explanation="The results may support growth.") -> NewsImpactResponse:
    return NewsImpactResponse(
        summary="Quarterly results were positive.",
        affected_symbols=["TCS", "INFY"],
        impact_direction="Positive",
        impact_strength="Medium",
        time_horizon="Short-term",
        explanation=explanation,
        risks=["The full report was not reviewed."],
        content_basis="rss_summary",
        content_quality="medium",
        recommendations=[],
    )


def test_persists_multiple_stock_events_and_prevents_duplicate_overwrite(db_session):
    timestamp = datetime(2026, 11, 10, 11, tzinfo=timezone.utc)
    first = persist_impact_analysis(
        db_session, context=_context(), response=_response(), analysis_timestamp=timestamp,
        model_version="impact-model-v1",
    )
    repeated = persist_impact_analysis(
        db_session, context=_context(), response=_response(explanation="Changed later."),
        analysis_timestamp=datetime(2026, 11, 11, 11, tzinfo=timezone.utc),
        model_version="impact-model-v2",
    )

    rows = list(db_session.scalars(select(NewsIntelligenceEvent).order_by(NewsIntelligenceEvent.stock_symbol)))
    assert len(first) == len(repeated) == len(rows) == 2
    assert {row.stock_symbol for row in rows} == {"INFY", "TCS"}
    assert {row.ai_explanation for row in rows} == {"The results may support growth."}
    assert {row.analysis_model_version for row in rows} == {"impact-model-v1"}
    assert all(
        (row.analysis_timestamp.replace(tzinfo=timezone.utc)
         if row.analysis_timestamp.tzinfo is None else row.analysis_timestamp)
        == timestamp
        for row in rows
    )
    assert all(float(row.finbert_confidence) == pytest.approx(0.7) for row in rows)


def test_missing_or_timezone_naive_publication_timestamp_is_not_persisted(db_session):
    missing = persist_impact_analysis(
        db_session, context=_context(published_at=""), response=_response()
    )
    naive = persist_impact_analysis(
        db_session, context=_context(published_at="2026-11-10T10:00:00"), response=_response()
    )
    assert missing == naive == []
    assert db_session.scalar(select(NewsIntelligenceEvent.event_id)) is None


def test_outcomes_are_separate_and_can_be_updated_only_after_prediction(db_session):
    event = persist_impact_analysis(
        db_session,
        context=_context(associated_symbols=("TCS",)),
        response=_response(),
        analysis_timestamp=datetime(2026, 11, 10, 11, tzinfo=timezone.utc),
    )[0]
    observed = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    first = update_news_outcomes(
        db_session,
        event_id=event.event_id,
        horizons={"1d": {
            "observed_at": observed,
            "stock_return_pct": 2.0,
            "nifty_return_pct": 0.5,
            "excess_return_pct": 1.5,
        }},
        calculated_at=datetime(2026, 11, 11, 11, tzinfo=timezone.utc),
    )
    updated = update_news_outcomes(
        db_session,
        event_id=event.event_id,
        horizons={"1d": {
            "observed_at": observed,
            "stock_return_pct": 3.0,
            "nifty_return_pct": 0.5,
            "excess_return_pct": 2.5,
        }},
        calculated_at=datetime(2026, 11, 12, 11, tzinfo=timezone.utc),
    )

    assert first.outcome_id == updated.outcome_id
    assert float(updated.stock_return_1d_pct) == pytest.approx(3.0)
    assert float(updated.excess_return_1d_pct) == pytest.approx(2.5)
    assert db_session.scalar(select(NewsIntelligenceOutcome.outcome_id)) == updated.outcome_id
    with pytest.raises(ValueError, match="expected NIFTY trading-session close"):
        update_news_outcomes(
            db_session,
            event_id=event.event_id,
            horizons={"1d": {
                "observed_at": observed,
                "stock_return_pct": 2.0,
                "nifty_return_pct": 0.5,
                "excess_return_pct": 1.5,
            }},
            calculated_at=datetime(2026, 11, 11, 11, tzinfo=timezone.utc),
            expected_observations={"1d": datetime(2026, 11, 12, 10, tzinfo=timezone.utc)},
        )
    with pytest.raises(ValueError, match="after the original prediction"):
        update_news_outcomes(
            db_session,
            event_id=event.event_id,
            horizons={"1d": {
                "observed_at": datetime(2026, 11, 10, 10, tzinfo=timezone.utc),
                "stock_return_pct": 2.0,
                "nifty_return_pct": 0.5,
                "excess_return_pct": 1.5,
            }},
        )


def test_database_failure_rolls_back_and_is_reported(db_session, monkeypatch):
    def fail_commit():
        raise SQLAlchemyError("simulated database outage")

    monkeypatch.setattr(db_session, "commit", fail_commit)
    with pytest.raises(NewsIntelligencePersistenceError, match="Could not persist"):
        persist_impact_analysis(db_session, context=_context(), response=_response())


def test_outcome_calculator_uses_prediction_time_and_only_observed_market_closes(db_session, tmp_path):
    event = persist_impact_analysis(
        db_session,
        context=_context(published_at="2024-01-01T12:00:00Z", associated_symbols=("TCS",)),
        response=_response(),
        analysis_timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )[0]
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
    stock_file = tmp_path / "stocks.csv"
    nifty_file = tmp_path / "nifty.csv"
    pd.DataFrame([
        {"Date": session, "Symbol": "TCS", "Adj Close": price, "Close": price * 2}
        for session, price in zip(sessions, [100, 110, 112, 115, 117, 120, 121], strict=True)
    ]).to_csv(stock_file, index=False)
    pd.DataFrame({"Date": sessions, "Close": [1000, 1005, 1010, 1020, 1030, 1040, 1045]}).to_csv(nifty_file, index=False)

    # At Jan 4 after close only the 1-session outcome has matured; 3d and 5d are future.
    changed = calculate_ready_news_outcomes(
        db_session,
        stock_prices_path=stock_file,
        nifty_path=nifty_file,
        now=datetime(2024, 1, 4, 12, tzinfo=timezone.utc),
    )
    partial = db_session.scalar(select(NewsIntelligenceOutcome).where(NewsIntelligenceOutcome.event_id == event.event_id))
    assert changed == 1
    assert float(partial.stock_return_1d_pct) == pytest.approx(10.0)
    assert partial.stock_return_3d_pct is None
    assert partial.stock_return_5d_pct is None

    calculate_ready_news_outcomes(
        db_session,
        stock_prices_path=stock_file,
        nifty_path=nifty_file,
        now=datetime(2024, 1, 9, 12, tzinfo=timezone.utc),
    )
    complete = db_session.scalar(select(NewsIntelligenceOutcome).where(NewsIntelligenceOutcome.event_id == event.event_id))
    assert float(complete.stock_return_3d_pct) == pytest.approx(15.0)
    assert float(complete.stock_return_5d_pct) == pytest.approx(20.0)


def test_outcome_calculator_does_not_fallback_from_missing_adjusted_close(db_session, tmp_path):
    event = persist_impact_analysis(
        db_session,
        context=_context(published_at="2024-01-01T12:00:00Z", associated_symbols=("TCS",)),
        response=_response(),
        analysis_timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )[0]
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    stocks = pd.DataFrame([
        {"Date": day, "Symbol": "TCS", "Adj Close": adjusted, "Close": close}
        for day, adjusted, close in zip(
            sessions, [100.0, None, 110.0, 120.0], [200.0, 300.0, 400.0, 500.0], strict=True
        )
    ])
    stocks_path = tmp_path / "mixed_basis_stocks.csv"
    nifty_path = tmp_path / "mixed_basis_nifty.csv"
    stocks.to_csv(stocks_path, index=False)
    pd.DataFrame({"Date": sessions, "Close": [1000.0, 1010.0, 1020.0, 1030.0]}).to_csv(nifty_path, index=False)

    calculate_ready_news_outcomes(
        db_session,
        stock_prices_path=stocks_path,
        nifty_path=nifty_path,
        now=datetime(2024, 1, 3, 12, tzinfo=timezone.utc),
    )
    outcome = db_session.scalar(select(NewsIntelligenceOutcome).where(NewsIntelligenceOutcome.event_id == event.event_id))
    assert outcome is None
