"""Shared isolated fixtures for backend API and database tests."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

# Set this before importing application modules so test collection never configures
# the PostgreSQL URL from the developer's ignored .env file.
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["NEWS_OUTCOME_WORKER_ENABLED"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TEST_RUN = PROJECT_ROOT / "tests" / ".pytest-temp" / f"run-{uuid4().hex}"
PROJECT_TEST_TEMP = PROJECT_TEST_RUN / "stdlib"


def pytest_configure(config: pytest.Config) -> None:
    """Use fresh project-local temp paths because this workspace denies child deletion."""
    pytest_basetemp = PROJECT_TEST_RUN / "basetemp"
    pytest_basetemp.mkdir(parents=True, exist_ok=False)
    PROJECT_TEST_TEMP.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(pytest_basetemp)
    tempfile.tempdir = str(PROJECT_TEST_TEMP)


for source_directory in (
    PROJECT_ROOT / "Src" / "data processing",
    PROJECT_ROOT / "Src" / "modeling",
):
    sys.path.insert(0, str(source_directory))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.models import Base
from Backend.App.dependencies import get_prediction_service
from Backend.App.main import app
from Backend.App.market_service import (
    MarketDataUnavailableError,
    MarketSnapshot,
    get_market_service,
)
from Backend.App.portfolio_service import get_portfolio_service
from Backend.App.services.model_service import StockSymbolNotFoundError
from Backend.App.services.sentiment_service import (
    NoNewsAvailableError,
    SentimentDataUnavailableError,
    StockSentiment,
    get_sentiment_service,
)

TEST_JWT_SECRET = "test-only-secret-that-is-never-used-outside-tests"


class FakePredictionService:
    """Controlled model-service substitute for API tests."""

    error: Exception | None = None

    def predict(self, symbol: str) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            symbol=symbol,
            prediction="BUY",
            confidence=0.8,
            probabilities={"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1},
            prediction_date="2026-11-12",
        )


class FakeSentimentService:
    """Controlled sentiment-service substitute for API tests."""

    error: Exception | None = None

    def get_symbol_sentiment(self, symbol: str) -> StockSentiment:
        if self.error is not None:
            raise self.error
        return StockSentiment(
            symbol=symbol,
            sentiment="Positive",
            positive_probability=0.7,
            negative_probability=0.1,
            neutral_probability=0.2,
            sentiment_score=0.6,
            timestamp=datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc),
            article_count=2,
        )


class FakeMarketService:
    """Controlled market-service substitute for API tests."""

    error: Exception | None = None

    def get_overview(self) -> list[MarketSnapshot]:
        if self.error is not None:
            raise self.error
        return [
            MarketSnapshot(
                name=name,
                ticker=ticker,
                value=value,
                change_percent=change,
                timestamp=datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc),
            )
            for name, ticker, value, change in (
                ("NIFTY 50", "^NSEI", 25000.0, 0.5),
                ("BANK NIFTY", "^NSEBANK", 54000.0, -0.2),
                ("INDIA VIX", "^INDIAVIX", 14.0, 1.1),
            )
        ]


class FakePortfolioService:
    """Price provider returning no quote, avoiding external market access."""

    def get_latest_price(self, symbol: str) -> None:
        return None


@pytest.fixture
def test_engine(tmp_path):
    """Create a fresh SQLite database file for each test; never use production DB."""
    database_path = tmp_path / "backend_test.sqlite3"
    test_database_url = URL.create(
        "sqlite+pysqlite",
        database=str(database_path),
    )
    engine = create_engine(test_database_url)
    if not engine.url.drivername.startswith("sqlite") or engine.url.database != str(
        database_path
    ):
        raise AssertionError("Tests must use the isolated temporary SQLite database.")
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Yield a session bound exclusively to the test database."""
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def fake_prediction_service():
    return FakePredictionService()


@pytest.fixture
def fake_sentiment_service():
    return FakeSentimentService()


@pytest.fixture
def fake_market_service():
    return FakeMarketService()


@pytest.fixture
def client(
    test_engine,
    monkeypatch: pytest.MonkeyPatch,
    fake_prediction_service: FakePredictionService,
    fake_sentiment_service: FakeSentimentService,
    fake_market_service: FakeMarketService,
):
    """Run API requests with database and external-service dependencies isolated."""
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)

    def override_get_db():
        with Session(test_engine) as session:
            yield session

    original_overrides: dict[Any, Any] = app.dependency_overrides.copy()
    app.dependency_overrides.update(
        {
            get_db: override_get_db,
            get_prediction_service: lambda: fake_prediction_service,
            get_sentiment_service: lambda: fake_sentiment_service,
            get_market_service: lambda: fake_market_service,
            get_portfolio_service: FakePortfolioService,
        }
    )
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
