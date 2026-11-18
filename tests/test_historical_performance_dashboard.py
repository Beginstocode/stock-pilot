"""Aggregation and filtering tests for the persistent performance dashboard API."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.main import app
from Backend.App.services.historical_performance_service import (
    HistoricalPerformanceService,
    local_date_boundaries_to_utc,
)
from Backend.App.services.news_intelligence_service import (
    expected_outcome_observations,
    load_nifty_session_dates,
)


def _event(
    db,
    *,
    identity: str,
    symbol: str,
    sentiment: str,
    impact: str,
    published_at: datetime,
    outcome_1d: tuple[float, float, float] | None,
    outcome_3d: tuple[float, float, float] | None = None,
):
    analysis_at = published_at + timedelta(minutes=1)
    expected_observations = expected_outcome_observations(analysis_at, load_nifty_session_dates())
    event = NewsIntelligenceEvent(
        article_id=identity * 64,
        article_url=f"https://news.example.test/{identity}",
        article_title=f"Historical result {identity}",
        published_at=published_at,
        analysis_timestamp=analysis_at,
        stock_symbol=symbol,
        finbert_sentiment=sentiment,
        finbert_confidence=0.8,
        ai_impact_direction=impact,
        ai_impact_strength="High" if impact == "Positive" else "Low",
        ai_explanation="A persisted explanation.",
        content_source="rss_summary",
        content_quality="medium",
        analysis_model_version="test-model",
    )
    db.add(event)
    db.flush()
    outcome = NewsIntelligenceOutcome(
        event_id=event.event_id,
        calculated_at=max(expected_observations.values()),
    )
    for horizon, values in (("1d", outcome_1d), ("3d", outcome_3d)):
        if values is None:
            continue
        stock, nifty, excess = values
        setattr(outcome, f"stock_return_{horizon}_pct", stock)
        setattr(outcome, f"nifty_return_{horizon}_pct", nifty)
        setattr(outcome, f"excess_return_{horizon}_pct", excess)
        setattr(outcome, f"observed_at_{horizon}", expected_observations[horizon])
    db.add(outcome)
    return event


def _seed(db):
    _event(
        db,
        identity="a",
        symbol="TCS",
        sentiment="Positive",
        impact="Positive",
        published_at=datetime(2024, 1, 3, 11, tzinfo=timezone.utc),
        outcome_1d=(2.0, 1.0, 1.0),
        outcome_3d=(-1.0, 0.0, -1.0),
    )
    _event(
        db,
        identity="b",
        symbol="TCS",
        sentiment="Negative",
        impact="Negative",
        published_at=datetime(2024, 1, 4, 11, tzinfo=timezone.utc),
        outcome_1d=None,
        outcome_3d=(-2.0, 0.0, -2.0),
    )
    _event(
        db,
        identity="c",
        symbol="INFY",
        sentiment="Neutral",
        impact="Positive",
        published_at=datetime(2024, 1, 5, 11, tzinfo=timezone.utc),
        outcome_1d=(-0.1, -0.13, 0.03),
        outcome_3d=None,
    )
    db.commit()


def test_dashboard_aggregates_completed_outcomes_and_excludes_pending_from_alignment(db_session):
    _seed(db_session)
    response = HistoricalPerformanceService().analyze(db_session, horizon="1d")
    one_day = response.performance_by_horizon[0]
    evaluation = response.evaluation_by_horizon[0]

    assert response.total_events == 3
    assert response.eligible_events == 2
    assert response.pending_outcomes == 1
    assert response.completed_outcomes == {"1d": 2, "3d": 2, "5d": 0}
    assert one_day.average_stock_return_pct == pytest.approx(0.95)
    assert one_day.average_nifty_return_pct == pytest.approx(0.435)
    assert one_day.average_excess_return_pct == pytest.approx(0.515)
    assert one_day.positive_outcomes == 1
    assert one_day.negative_outcomes == 1
    assert evaluation.sentiment_return_alignment.eligible_events == 2
    assert evaluation.sentiment_return_alignment.sample_size == 1
    assert evaluation.sentiment_return_alignment.alignment_pct == 100.0
    positive_impact = next(group for group in evaluation.business_impact_returns if group.impact_direction == "Positive")
    assert positive_impact.sample_size == 2
    assert positive_impact.average_stock_return_pct == pytest.approx(0.95)
    assert positive_impact.average_excess_return_pct == pytest.approx(0.515)
    assert sum(group.sample_size for group in evaluation.business_impact_returns) == response.completed_outcomes["1d"]
    assert response.stock_comparison[0].sample_supported is False
    assert all(event.sentiment_return_alignment == "Outcome pending" for event in response.events if not event.eligible)
    assert not hasattr(evaluation, "ai_impact")


def test_dashboard_neutral_band_zero_returns_and_horizons_are_independent(db_session):
    for identity, symbol, sentiment, impact, published, one_day, three_day in (
        (
            "h", "TCS", "Neutral", "Neutral", datetime(2024, 1, 3, 11, tzinfo=timezone.utc),
            (0.0, 0.0, 0.0), (1.0, 0.8, 0.2),
        ),
        (
            "i", "INFY", "Neutral", "Neutral", datetime(2024, 1, 4, 11, tzinfo=timezone.utc),
            (0.05, 0.0, 0.05), (-1.0, -0.95, -0.05),
        ),
    ):
        event = _event(
            db_session,
            identity=identity,
            symbol=symbol,
            sentiment=sentiment,
            impact=impact,
            published_at=published,
            outcome_1d=one_day,
            outcome_3d=three_day,
        )
        outcome = db_session.query(NewsIntelligenceOutcome).filter_by(event_id=event.event_id).one()
    db_session.commit()

    response = HistoricalPerformanceService().analyze(db_session, horizon="1d")
    one_day = response.performance_by_horizon[0]
    one_day_eval = response.evaluation_by_horizon[0]
    three_day_eval = response.evaluation_by_horizon[1]
    five_day_eval = response.evaluation_by_horizon[2]

    # Zero stock return and +/-0.05 excess returns remain explicit neutral cases.
    assert one_day.neutral_outcomes == 1  # Only the exactly-zero stock return is neutral in the distribution.
    assert one_day.positive_outcomes == 1
    assert one_day.positive_outcome_pct == 50.0
    assert one_day.negative_outcome_pct == 0.0
    assert one_day_eval.sentiment_return_alignment.sample_size == 0
    assert one_day_eval.sentiment_return_alignment.alignment_pct is None
    assert one_day_eval.sentiment_return_alignment.neutral_sentiment_events == 2
    assert one_day_eval.sentiment_return_alignment.neutral_excess_return_events == 2
    neutral_impact = next(group for group in one_day_eval.business_impact_returns if group.impact_direction == "Neutral")
    assert neutral_impact.sample_size == 2
    assert neutral_impact.average_stock_return_pct == pytest.approx(0.025)
    assert neutral_impact.average_excess_return_pct == pytest.approx(0.025)
    assert three_day_eval.business_impact_returns[2].sample_size == 2
    assert five_day_eval.sentiment_return_alignment.eligible_events == 0
    assert five_day_eval.sentiment_return_alignment.alignment_pct is None
    assert all(group.sample_size == 0 and group.average_stock_return_pct is None for group in five_day_eval.business_impact_returns)


def test_dashboard_filters_stock_sentiment_direction_and_dates(db_session):
    _seed(db_session)
    response = HistoricalPerformanceService().analyze(
        db_session,
        stock_symbol="tcs",
        sentiment="Negative",
        impact_direction="Negative",
        date_from=datetime(2024, 1, 4).date(),
        date_to=datetime(2024, 1, 4).date(),
        horizon="3d",
    )

    assert response.total_events == 1
    assert response.eligible_events == 1
    assert response.events[0].symbol == "TCS"
    assert response.events[0].finbert_sentiment == "Negative"
    assert response.events[0].ai_impact_direction == "Negative"
    assert response.events[0].sentiment_return_alignment == "Aligned"


def test_local_calendar_day_filters_include_boundaries_across_utc_dates(db_session):
    # 2024-01-02 in Kolkata is [2024-01-01 18:30Z, 2024-01-02 18:30Z).
    for identity, published in (
        ("d", datetime(2024, 1, 1, 18, 29, 59, tzinfo=timezone.utc)),
        ("e", datetime(2024, 1, 1, 18, 30, tzinfo=timezone.utc)),
        ("f", datetime(2024, 1, 2, 18, 29, 59, tzinfo=timezone.utc)),
        ("g", datetime(2024, 1, 2, 18, 30, tzinfo=timezone.utc)),
    ):
        _event(
            db_session, identity=identity, symbol="TCS", sentiment="Neutral", impact="Neutral",
            published_at=published, outcome_1d=None,
        )
    db_session.commit()
    response = HistoricalPerformanceService().analyze(
        db_session, date_from=date(2024, 1, 2), date_to=date(2024, 1, 2), timezone_name="Asia/Kolkata"
    )
    assert response.total_events == 2
    assert {event.event_id for event in response.events} == {
        db_session.query(NewsIntelligenceEvent).filter_by(article_id="e" * 64).one().event_id,
        db_session.query(NewsIntelligenceEvent).filter_by(article_id="f" * 64).one().event_id,
    }
    assert response.filter_timezone == "Asia/Kolkata"


@pytest.mark.parametrize(
    ("day", "expected_start", "expected_end", "hours"),
    [
        (date(2024, 3, 10), datetime(2024, 3, 10, 5, tzinfo=timezone.utc), datetime(2024, 3, 11, 4, tzinfo=timezone.utc), 23),
        (date(2024, 11, 3), datetime(2024, 11, 3, 4, tzinfo=timezone.utc), datetime(2024, 11, 4, 5, tzinfo=timezone.utc), 25),
    ],
)
def test_local_day_boundaries_follow_dst(day, expected_start, expected_end, hours):
    start, end, _ = local_date_boundaries_to_utc(day, day, "America/New_York")
    assert start == expected_start
    assert end == expected_end
    assert (end - start).total_seconds() == hours * 3600


def test_empty_local_date_range_and_invalid_timezone(db_session):
    _seed(db_session)
    empty = HistoricalPerformanceService().analyze(
        db_session, date_from=date(2024, 2, 1), date_to=date(2024, 2, 2), timezone_name="Asia/Kolkata"
    )
    assert empty.total_events == 0
    with pytest.raises(ValueError):
        HistoricalPerformanceService().analyze(db_session, timezone_name="Not/A_Zone")


def test_dashboard_excludes_observation_not_on_expected_nifty_session_close(db_session):
    event = _event(
        db_session,
        identity="j",
        symbol="TCS",
        sentiment="Positive",
        impact="Positive",
        published_at=datetime(2024, 1, 3, 11, tzinfo=timezone.utc),
        outcome_1d=(2.0, 1.0, 1.0),
    )
    outcome = db_session.query(NewsIntelligenceOutcome).filter_by(event_id=event.event_id).one()
    outcome.observed_at_1d = datetime(2024, 2, 2, 12, tzinfo=timezone.utc)
    outcome.calculated_at = datetime(2024, 2, 3, 12, tzinfo=timezone.utc)
    db_session.commit()

    response = HistoricalPerformanceService().analyze(db_session)
    assert response.total_events == 1
    assert response.completed_outcomes["1d"] == 0
    assert response.eligible_events == 0


def test_dashboard_paginates_events_without_truncating_aggregates(db_session):
    _seed(db_session)
    response = HistoricalPerformanceService().analyze(db_session, page=2, page_size=2)
    assert response.total_events == 3
    assert response.total_pages == 2
    assert len(response.events) == 1
    assert response.eligible_events == 2


def test_dashboard_api_returns_validated_filtered_response(test_engine):
    with Session(test_engine) as db:
        _seed(db)

    def override_get_db():
        with Session(test_engine) as db:
            yield db

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.get("/news/historical-performance?stock_symbol=TCS&horizon=3d&page_size=1")
            invalid_symbol = client.get("/news/historical-performance?stock_symbol=NOPE")
            invalid_range = client.get("/news/historical-performance?date_from=2024-02-01&date_to=2024-01-01")
            invalid_date = client.get("/news/historical-performance?date_from=not-a-date")
            invalid_timezone = client.get("/news/historical-performance?timezone=Not/A_Zone")
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_events"] == 2
    assert payload["eligible_events"] == 2
    assert len(payload["events"]) == 1
    evaluation = payload["evaluation_by_horizon"][0]
    assert "sentiment_return_alignment" in evaluation
    assert "business_impact_returns" in evaluation
    assert "ai_impact" not in evaluation
    assert "impact_prediction_correct" not in payload["events"][0]
    assert "finbert_prediction_correct" not in payload["events"][0]
    assert "price_direction_prediction_accuracy" not in payload
    assert invalid_symbol.status_code == 422
    assert invalid_range.status_code == 422
    assert invalid_date.status_code == 422
    assert invalid_timezone.status_code == 422


def test_dashboard_api_is_registered_and_returns_empty_database_state(client):
    openapi = client.get("/openapi.json").json()
    assert "get" in openapi["paths"]["/news/historical-performance"]

    response = client.get(
        "/news/historical-performance?timezone=Asia%2FCalcutta&horizon=1d&page=1&page_size=50"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_events"] == 0
    assert payload["eligible_events"] == 0
    assert payload["events"] == []
