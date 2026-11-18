"""Worker scheduling, retry, idempotency, and overlap behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Event, Thread

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from Backend.App.schemas.news_impact import NewsImpactResponse
from Backend.App.services.news_impact_service import NewsImpactContext
from Backend.App.services.news_intelligence_service import (
    HistoricalOutcomeDataError,
    calculate_ready_news_outcomes,
    persist_impact_analysis,
)
from Backend.App.workers.news_outcome_worker import NewsOutcomeWorker


SESSIONS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]


def _seed_event(db_session):
    context = NewsImpactContext(
        article_url="https://news.example.test/worker-story",
        headline="TCS quarterly results",
        source="Example",
        published_at="2024-01-01T10:00:00Z",
        associated_symbols=("TCS",),
        sentiment="Positive",
        sentiment_score=0.6,
        positive_probability=0.7,
        negative_probability=0.1,
        neutral_probability=0.2,
        article_text="Results summary",
        content_source="rss_summary",
        content_quality="medium",
    )
    response = NewsImpactResponse(
        summary="Results improved.",
        affected_symbols=["TCS"],
        impact_direction="Positive",
        impact_strength="Medium",
        time_horizon="Short-term",
        explanation="The results indicate stronger revenue.",
        risks=["The full report is unavailable."],
        content_basis="rss_summary",
        content_quality="medium",
        recommendations=[],
    )
    return persist_impact_analysis(
        db_session,
        context=context,
        response=response,
        analysis_timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )[0]


def _price_files(directory, *, short=False):
    sessions = SESSIONS[:2] if short else SESSIONS
    stock_file = directory / "stocks.csv"
    nifty_file = directory / "nifty.csv"
    prices = [100, 110] if short else [100, 110, 112, 115, 117, 120, 121]
    benchmark = [1000, 1005] if short else [1000, 1005, 1010, 1020, 1030, 1040, 1045]
    pd.DataFrame([
        {"Date": day, "Symbol": "TCS", "Adj Close": price, "Close": price * 2}
        for day, price in zip(sessions, prices, strict=True)
    ]).to_csv(stock_file, index=False)
    pd.DataFrame({"Date": sessions, "Close": benchmark}).to_csv(nifty_file, index=False)
    return stock_file, nifty_file


def test_worker_calculates_ready_outcomes_and_is_idempotent_across_restart(
    db_session, test_engine, tmp_path
):
    event = _seed_event(db_session)
    stocks, nifty = _price_files(tmp_path)
    now = datetime(2024, 1, 9, 12, tzinfo=timezone.utc)
    session_factory = lambda: Session(test_engine)
    lock_path = tmp_path / "news-worker.lock"
    worker = NewsOutcomeWorker(
        session_factory=session_factory,
        calculator=lambda session: calculate_ready_news_outcomes(
            session, stock_prices_path=stocks, nifty_path=nifty, now=now
        ),
        lock_path=lock_path,
    )

    first = worker.run_once()
    duplicate = worker.run_once()
    restarted = NewsOutcomeWorker(
        session_factory=session_factory,
        calculator=lambda session: calculate_ready_news_outcomes(
            session, stock_prices_path=stocks, nifty_path=nifty, now=now
        ),
        lock_path=lock_path,
    ).run_once()

    assert first.events_checked == 1
    assert first.outcomes_calculated == 3
    assert first.events_pending == 0
    assert first.events_skipped == 0
    assert duplicate.outcomes_calculated == restarted.outcomes_calculated == 0
    assert duplicate.events_skipped == restarted.events_skipped == 1


def test_worker_keeps_unavailable_horizons_pending_with_partial_price_history(
    db_session, test_engine, tmp_path
):
    _seed_event(db_session)
    stocks, nifty = _price_files(tmp_path, short=True)
    worker = NewsOutcomeWorker(
        session_factory=lambda: Session(test_engine),
        calculator=lambda session: calculate_ready_news_outcomes(
            session,
            stock_prices_path=stocks,
            nifty_path=nifty,
            now=datetime(2024, 1, 4, 12, tzinfo=timezone.utc),
        ),
        lock_path=tmp_path / "partial-worker.lock",
    )
    stats = worker.run_once()

    assert stats.outcomes_calculated == 1
    assert stats.events_pending == 1
    assert stats.failures == 0


def test_worker_retries_after_price_provider_failure(db_session, test_engine, tmp_path):
    _seed_event(db_session)
    calls = 0

    def failing_then_successful(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HistoricalOutcomeDataError("price file temporarily unavailable")
        return 0

    worker = NewsOutcomeWorker(
        session_factory=lambda: Session(test_engine),
        calculator=failing_then_successful,
        lock_path=tmp_path / "retry-worker.lock",
    )
    failed = worker.run_once()
    retried = worker.run_once()

    assert failed.failures == 1
    assert failed.events_pending == 1
    assert retried.failures == 0
    assert retried.events_skipped == 1


def test_worker_reports_database_failures(db_session, tmp_path):
    def unavailable_session():
        raise SQLAlchemyError("database offline")

    worker = NewsOutcomeWorker(
        session_factory=unavailable_session,
        lock_path=tmp_path / "database-failure.lock",
    )
    stats = worker.run_once()

    assert stats.failures == 1
    assert stats.status == "failed"
    assert "database offline" in stats.errors[0]


def test_worker_prevents_overlapping_runs(db_session, test_engine, tmp_path):
    started = Event()
    release = Event()

    def slow_calculator(_session):
        started.set()
        assert release.wait(timeout=5)
        return 0

    worker = NewsOutcomeWorker(
        session_factory=lambda: Session(test_engine),
        calculator=slow_calculator,
        lock_path=tmp_path / "overlap.lock",
    )
    results = []
    thread = Thread(target=lambda: results.append(worker.run_once()))
    thread.start()
    assert started.wait(timeout=5)
    concurrent = worker.run_once()
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert concurrent.status == "overlap_skipped"
    assert results[0].failures == 0
