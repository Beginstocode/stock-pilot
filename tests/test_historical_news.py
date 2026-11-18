"""Historical news event-study behavior, data validation, and API tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.main import app
from Backend.App.schemas.historical_news import HistoricalNewsPerformanceResponse
from Backend.App.services.historical_news_service import (
    HistoricalNewsDataError,
    HistoricalNewsPerformanceService,
)
from Backend.App.services.news_intelligence_service import (
    expected_outcome_observations,
    load_nifty_session_dates,
)
from Backend.routes import news_performance


SESSIONS = ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]


def _make_service(tmp_path, *, as_of=date(2024, 1, 11), include_scores=True):
    news_path = tmp_path / "news.csv"
    sentiment_path = tmp_path / "sentiment.csv"
    stocks_path = tmp_path / "stocks.csv"
    nifty_path = tmp_path / "nifty.csv"

    news = pd.DataFrame(
        [
            {"Date": "2024-01-03T11:00:00Z", "Symbol": "TCS", "Headline": "Quarterly revenue rises", "URL": "https://news.example/story-a"},
            # Same article for a second symbol and a syndicated duplicate URL for TCS.
            {"Date": "2024-01-03T11:00:00Z", "Symbol": "INFY", "Headline": "Quarterly revenue rises", "URL": "https://news.example/story-a"},
            {"Date": "2024-01-03T11:30:00Z", "Symbol": "TCS", "Headline": "Quarterly revenue rises", "URL": "https://syndication.example/story-a"},
            # Weekend news and the Jan 4 market holiday align to the next NIFTY session.
            {"Date": "2024-01-06T07:00:00Z", "Symbol": "TCS", "Headline": "Weekend policy announcement", "URL": "https://news.example/story-b"},
        ]
    )
    news.to_csv(news_path, index=False)

    scores = pd.DataFrame(
        [
            {"article_id": "https://news.example/story-a", "symbol": "TCS", "sentiment": "Positive", "sentiment_score": 0.8},
            {"article_id": "https://news.example/story-a", "symbol": "INFY", "sentiment": "Positive", "sentiment_score": 0.7},
            {"article_id": "https://syndication.example/story-a", "symbol": "TCS", "sentiment": "Positive", "sentiment_score": 0.8},
            {"article_id": "https://news.example/story-b", "symbol": "TCS", "sentiment": "Negative", "sentiment_score": -0.5},
        ]
    )
    if include_scores:
        scores.to_csv(sentiment_path, index=False)

    stock_rows = []
    values = {
        "TCS": [100, 105, 110, 108, 115, 117, 120],
        "INFY": [200, 204, 210, 208, 215, 218, 222],
    }
    for symbol, adjusted in values.items():
        for session, adj_close in zip(SESSIONS, adjusted, strict=True):
            stock_rows.append({"Date": session, "Symbol": symbol, "Adj Close": adj_close, "Close": adj_close * 2})
    pd.DataFrame(stock_rows).to_csv(stocks_path, index=False)

    nifty_values = [1000, 1010, 1000, 1020, 1030, 1050, 1060]
    pd.DataFrame({"Date": SESSIONS, "Close": nifty_values}).to_csv(nifty_path, index=False)
    return HistoricalNewsPerformanceService(
        news_path=news_path,
        sentiment_path=sentiment_path,
        stock_prices_path=stocks_path,
        nifty_path=nifty_path,
        as_of=as_of,
    )


def test_after_market_weekend_holiday_and_adjusted_price_returns_are_aligned(tmp_path):
    result = _make_service(tmp_path).analyze()
    event = next(row for row in result.events if row.headline == "Quarterly revenue rises" and row.symbol == "TCS")

    assert event.anchor_date == date(2024, 1, 5)  # 16:30 IST: next session after close.
    assert event.outcomes["1d"].observation_date == date(2024, 1, 5)
    assert event.outcomes["1d"].stock_return_pct == pytest.approx((110 / 105 - 1) * 100)
    assert event.outcomes["1d"].nifty_return_pct == pytest.approx((1000 / 1010 - 1) * 100)
    assert event.outcomes["1d"].excess_return_pct == pytest.approx(
        ((110 / 105 - 1) - (1000 / 1010 - 1)) * 100
    )
    assert event.outcomes["3d"].observation_date == date(2024, 1, 9)
    assert event.finbert_sentiment == "Positive"

    weekend = next(row for row in result.events if row.headline == "Weekend policy announcement")
    assert weekend.anchor_date == date(2024, 1, 8)  # Weekend plus Jan 4 holiday.
    assert result.total_events_analyzed == 2
    assert result.returns_by_horizon[0].event_observations == 3
    assert "Adj Close" in result.disclaimer
    assert "NIFTY uses Close" in result.disclaimer


def test_missing_adjusted_close_is_not_filled_with_unadjusted_close(tmp_path):
    service = _make_service(tmp_path)
    stocks_path = tmp_path / "stocks.csv"
    stocks = pd.read_csv(stocks_path)
    stocks.loc[(stocks["Symbol"] == "TCS") & (stocks["Date"] == "2024-01-05"), "Adj Close"] = float("nan")
    stocks.to_csv(stocks_path, index=False)

    result = service.analyze()
    event = next(row for row in result.events if row.headline == "Quarterly revenue rises" and row.symbol == "TCS")
    assert event.outcomes["1d"].stock_return_pct is None


def test_duplicate_articles_are_one_event_and_multiple_symbols_remain_separate_rows(tmp_path):
    result = _make_service(tmp_path).analyze()
    articles = [row for row in result.events if row.headline == "Quarterly revenue rises"]
    assert len(articles) == 2
    assert {row.symbol for row in articles} == {"TCS", "INFY"}
    assert len({row.event_id for row in articles}) == 1
    assert result.total_events_analyzed == 2


def test_finbert_alignment_and_ai_impact_unavailability_are_explicit(tmp_path):
    result = _make_service(tmp_path).analyze()
    positive = next(row for row in result.returns_by_horizon if row.horizon == "1d")
    assert positive.finbert_alignment_sample_size == 3
    assert positive.finbert_alignment_pct is not None
    assert positive.neutral_sentiment_observations == 0
    assert result.ai_impact_performance.available is False
    assert result.ai_impact_performance.returns_by_impact_direction == []
    assert "not stored" in result.ai_impact_performance.reason


def test_missing_sentiment_dataset_does_not_invent_finbert_results(tmp_path):
    result = _make_service(tmp_path, include_scores=False).analyze()
    assert result.finbert_records == 0
    assert result.finbert_available is False
    assert result.finbert_unavailable_reason
    assert result.finbert_available is False
    assert result.finbert_unavailable_reason
    assert all(row.finbert_sentiment is None for row in result.events)
    assert all(metric.finbert_alignment_pct is None for metric in result.returns_by_horizon)


def test_future_price_rows_and_events_after_as_of_are_excluded(tmp_path):
    result = _make_service(tmp_path, as_of=date(2024, 1, 9)).analyze()
    assert result.stock_price_coverage_end == date(2024, 1, 9)
    assert all(
        outcome.observation_date is None or outcome.observation_date <= date(2024, 1, 9)
        for event in result.events
        for outcome in event.outcomes.values()
    )
    assert result.events[0].outcomes["5d"].excess_return_pct is None


def test_missing_market_prices_are_disclosed_without_fabricated_returns(tmp_path):
    service = _make_service(tmp_path)
    stocks = pd.read_csv(service.stock_prices_path)
    stocks = stocks[~((stocks.Symbol == "TCS") & (stocks.Date == "2024-01-05"))]
    stocks.to_csv(service.stock_prices_path, index=False)

    result = service.analyze()
    event = next(row for row in result.events if row.headline == "Quarterly revenue rises" and row.symbol == "TCS")
    assert event.outcomes["1d"].stock_return_pct is None
    assert event.eligible is False
    assert event.exclusion_reason


def test_api_returns_validated_historical_performance(monkeypatch, tmp_path):
    service = _make_service(tmp_path)
    monkeypatch.setattr(news_performance, "get_historical_news_service", lambda: service)
    with TestClient(app) as client:
        response = client.get("/news/performance")
    assert response.status_code == 200
    validated = HistoricalNewsPerformanceResponse.model_validate(response.json())
    assert validated.total_events_analyzed == 2
    assert validated.ai_impact_performance.available is False


def test_api_reports_missing_required_price_data(monkeypatch):
    class MissingData:
        def analyze(self):
            raise HistoricalNewsDataError("Required historical prices are missing.")

    monkeypatch.setattr(news_performance, "get_historical_news_service", lambda: MissingData())
    with TestClient(app) as client:
        response = client.get("/news/performance")
    assert response.status_code == 503
    assert response.json()["detail"] == "Required historical prices are missing."


def test_performance_api_includes_newly_persisted_ai_outcomes(monkeypatch, tmp_path, test_engine):
    service = _make_service(tmp_path)
    monkeypatch.setattr(news_performance, "get_historical_news_service", lambda: service)
    analysis_at = datetime(2024, 1, 3, 10, tzinfo=timezone.utc)
    observed_at = expected_outcome_observations(analysis_at, load_nifty_session_dates())["1d"]
    with Session(test_engine) as session:
        event = NewsIntelligenceEvent(
            article_id="a" * 64,
            article_url="https://news.example.test/persisted",
            article_title="Persisted news",
            published_at=datetime(2024, 1, 3, 9, tzinfo=timezone.utc),
            analysis_timestamp=analysis_at,
            stock_symbol="TCS",
            finbert_sentiment="Positive",
            finbert_confidence=0.8,
            ai_impact_direction="Positive",
            ai_impact_strength="High",
            ai_explanation="Persistent prediction for API evaluation.",
            content_source="rss_summary",
            content_quality="medium",
            analysis_model_version="test-model",
        )
        session.add(event)
        session.flush()
        session.add(NewsIntelligenceOutcome(
            event_id=event.event_id,
            stock_return_1d_pct=2.0,
            nifty_return_1d_pct=0.5,
            excess_return_1d_pct=1.5,
            observed_at_1d=observed_at,
            calculated_at=observed_at + timedelta(minutes=1),
        ))
        invalid_event = NewsIntelligenceEvent(
            article_id="b" * 64,
            article_url="https://news.example.test/invalid-observation",
            article_title="Invalid observation date",
            published_at=datetime(2024, 1, 3, 9, tzinfo=timezone.utc),
            analysis_timestamp=analysis_at,
            stock_symbol="INFY",
            finbert_sentiment="Negative",
            finbert_confidence=0.8,
            ai_impact_direction="Negative",
            ai_impact_strength="Low",
            ai_explanation="Persisted impact assessment.",
            content_source="rss_summary",
            content_quality="medium",
            analysis_model_version="test-model",
        )
        session.add(invalid_event)
        session.flush()
        session.add(NewsIntelligenceOutcome(
            event_id=invalid_event.event_id,
            stock_return_1d_pct=-2.0,
            nifty_return_1d_pct=-0.5,
            excess_return_1d_pct=-1.5,
            observed_at_1d=datetime(2024, 1, 3, 15, 30, tzinfo=timezone.utc),
            calculated_at=datetime(2024, 1, 3, 16, tzinfo=timezone.utc),
        ))
        session.commit()

    def override_get_db():
        with Session(test_engine) as session:
            yield session

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.get("/news/performance")
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    impact = response.json()["ai_impact_performance"]
    assert impact["available"] is True
    assert impact["event_observations"] == 1
    assert "directional_accuracy_1d_pct" not in impact
    positive_group = next(group for group in impact["returns_by_impact_direction"] if group["impact_direction"] == "Positive")
    assert positive_group["event_observations"] == 1
    assert positive_group["average_stock_return_1d_pct"] == 2.0
    assert positive_group["average_excess_return_1d_pct"] == 1.5
