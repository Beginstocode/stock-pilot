"""News impact analysis service and API tests without external LLM calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from Backend.App.main import app
from Backend.App.database.connection import get_db
from Backend.App.database.models import NewsIntelligenceEvent
from Backend.App.schemas.news_impact import NewsImpactResponse
from Backend.App.services.news_impact_service import (
    NewsImpactAnalysisService,
    NewsImpactContext,
    NewsImpactProviderError,
)
from Backend.App.services.news_intelligence_service import NewsIntelligencePersistenceError
from Backend.routes import news as news_route


def _context(*, article_text: str | None = "Company reported higher quarterly revenue."):
    return NewsImpactContext(
        article_url="https://news.example.com/story/1",
        headline="TCS reports quarterly results",
        source="Example Finance",
        published_at="2026-11-10T10:00:00Z",
        associated_symbols=("TCS",),
        sentiment="Positive",
        sentiment_score=0.6,
        positive_probability=0.7,
        negative_probability=0.1,
        neutral_probability=0.2,
        article_text=article_text,
        content_source="rss_summary" if article_text else "headline_only",
        content_quality="medium" if article_text else "low",
    )


def _analysis(
    direction="Positive",
    symbols=None,
):
    return {
        "summary": "The company reported higher quarterly revenue.",
        "affected_symbols": ["TCS"] if symbols is None else symbols,
        "impact_direction": direction,
        "impact_strength": "Medium",
        "time_horizon": "Short-term",
        "explanation": "The reported growth may support business momentum, subject to confirmation.",
        "risks": ["The headline does not provide full financial details."],
    }


class FakePredictionService:
    def __init__(self):
        self.calls: list[str] = []

    def predict(self, symbol: str):
        self.calls.append(symbol)
        return SimpleNamespace(
            prediction="HOLD",
            confidence=0.72,
            prediction_date="2026-11-09",
        )


@pytest.mark.parametrize("direction", ["Positive", "Negative", "Neutral"])
def test_analysis_supports_all_impact_directions(direction):
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(direction=direction),
        prediction_service_provider=FakePredictionService,
    )

    result = service.analyze(_context())

    assert result.impact_direction == direction
    assert result.affected_symbols == ["TCS"]
    assert "not a forecast" in result.disclaimer


def test_multiple_affected_stocks_include_existing_recommendations():
    predictions = FakePredictionService()
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(symbols=["TCS", "INFY"]),
        prediction_service_provider=lambda: predictions,
    )

    result = service.analyze(_context())

    assert result.affected_symbols == ["TCS", "INFY"]
    assert [item.symbol for item in result.recommendations] == ["TCS", "INFY"]
    assert [item.recommendation for item in result.recommendations] == ["HOLD", "HOLD"]
    assert predictions.calls == ["TCS", "INFY"]


def test_irrelevant_news_can_return_no_affected_stocks():
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(direction="Neutral", symbols=[]),
        prediction_service_provider=FakePredictionService,
    )

    result = service.analyze(_context())

    assert result.affected_symbols == []
    assert result.recommendations == []


def test_missing_article_content_is_explicitly_headline_only():
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(),
        prediction_service_provider=FakePredictionService,
    )

    result = service.analyze(_context(article_text=None))

    assert result.content_basis == "headline_only"


def test_unknown_llm_symbol_is_rejected():
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(symbols=["NOTASTOCK"]),
        prediction_service_provider=FakePredictionService,
    )

    with pytest.raises(NewsImpactProviderError, match="unsupported stock symbols"):
        service.analyze(_context())


def test_llm_failure_is_not_cached_and_surfaces_provider_error():
    calls = 0

    def failed_llm(_):
        nonlocal calls
        calls += 1
        raise NewsImpactProviderError("provider unavailable")

    service = NewsImpactAnalysisService(
        llm_call=failed_llm,
        prediction_service_provider=FakePredictionService,
    )
    for _ in range(2):
        with pytest.raises(NewsImpactProviderError, match="provider unavailable"):
            service.analyze(_context())
    assert calls == 2


def test_llm_adapter_rejects_remote_http_before_sending_api_key(monkeypatch):
    monkeypatch.setenv("NEWS_IMPACT_LLM_URL", "http://llm.example.com/v1/chat/completions")
    monkeypatch.setenv("NEWS_IMPACT_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("NEWS_IMPACT_LLM_MODEL", "test-model")

    with pytest.raises(NewsImpactProviderError, match="must use HTTPS"):
        NewsImpactAnalysisService._call_configured_llm({})


def test_repeated_article_analysis_uses_cache():
    calls = 0

    def successful_llm(_):
        nonlocal calls
        calls += 1
        return _analysis()

    service = NewsImpactAnalysisService(
        llm_call=successful_llm,
        prediction_service_provider=FakePredictionService,
    )
    first = service.analyze(_context())
    second = service.analyze(_context())

    assert calls == 1
    assert first.summary == second.summary


def test_impact_api_validates_request_and_response(monkeypatch, test_engine):
    service = NewsImpactAnalysisService(
        llm_call=lambda _: _analysis(symbols=["TCS", "INFY"]),
        prediction_service_provider=FakePredictionService,
    )
    monkeypatch.setattr(news_route, "_load_impact_context", lambda _url: _context())
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[news_route.get_news_impact_service] = lambda: service

    def override_get_db():
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            invalid = client.post("/news/impact", json={"article_url": "not a URL"})
            valid = client.post(
                "/news/impact",
                json={"article_url": "https://news.example.com/story/1"},
            )
            monkeypatch.setattr(
                news_route,
                "persist_impact_analysis",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    NewsIntelligencePersistenceError("simulated storage outage")
                ),
            )
            storage_failure = client.post(
                "/news/impact",
                json={"article_url": "https://news.example.com/story/1"},
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert invalid.status_code == 422
    assert valid.status_code == 200
    parsed = NewsImpactResponse.model_validate(valid.json())
    assert parsed.affected_symbols == ["TCS", "INFY"]
    assert parsed.recommendations[0].recommendation == "HOLD"
    assert parsed.content_basis == "rss_summary"
    assert parsed.content_quality == "medium"
    assert storage_failure.status_code == 200
    with Session(test_engine) as session:
        assert len(session.scalars(select(NewsIntelligenceEvent)).all()) == 2


def test_impact_api_handles_missing_article_and_llm_failure(monkeypatch):
    service = NewsImpactAnalysisService(
        llm_call=lambda _: (_ for _ in ()).throw(
            NewsImpactProviderError("provider unavailable")
        ),
        prediction_service_provider=FakePredictionService,
    )
    original_loader = news_route._load_impact_context
    monkeypatch.setattr(
        news_route,
        "_load_impact_context",
        lambda _url: (_ for _ in ()).throw(LookupError("article not found")),
    )
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[news_route.get_news_impact_service] = lambda: service
    try:
        with TestClient(app) as client:
            missing = client.post(
                "/news/impact", json={"article_url": "https://news.example.com/nope"}
            )
            monkeypatch.setattr(news_route, "_load_impact_context", lambda _url: _context())
            failed = client.post(
                "/news/impact", json={"article_url": "https://news.example.com/story/1"}
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
        monkeypatch.setattr(news_route, "_load_impact_context", original_loader)

    assert missing.status_code == 404
    assert failed.status_code == 503
