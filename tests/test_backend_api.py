"""FastAPI route, validation, authentication, and ownership tests."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from Backend.App.database.models import Holding, Portfolio, User
from Backend.App.dependencies import StockSymbolNotFoundError
from Backend.App.main import app
from Backend.App.market_service import MarketDataUnavailableError
from Backend.App.services.stock_service import StockService, get_stock_service
from Backend.App.services.sentiment_service import (
    NoNewsAvailableError,
    SentimentDataUnavailableError,
)


def register_and_login(client, email: str) -> tuple[int, dict[str, str]]:
    password = "test-password-long-enough"
    registration = client.post(
        "/auth/register",
        json={"name": "Test User", "email": email, "password": password},
    )
    assert registration.status_code == 201
    user_id = registration.json()["user_id"]
    login = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200
    return user_id, {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_root_and_health_routes(client):
    root = client.get("/")
    health = client.get("/health")
    assert root.status_code == 200
    assert set(root.json()) == {"name", "version", "status"}
    assert health.status_code == 200
    assert health.json()["api_status"] == "running"
    assert health.json()["health_status"] == "healthy"


def test_stock_list_details_and_invalid_symbol(client):
    response = client.get("/stocks")
    assert response.status_code == 200
    assert set(response.json()) == {
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "ITC",
        "LT",
        "BHARTIARTL",
        "TATAMOTORS",
    }

    details = client.get("/stocks/tcs")
    assert details.status_code == 200
    assert details.json() == {
        "symbol": "TCS",
        "name": "Tata Consultancy Services Limited",
    }
    assert client.get("/stocks/NOTASTOCK").status_code == 404


def test_stock_history_returns_real_ohlcv_and_filters_dates(
    client,
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / "stock_features.csv"
    csv_path.write_text(
        "Date,Symbol,Open,High,Low,Close,Volume,RSI_14,MACD,SMA_20,SMA_50,SMA_200,EMA_12,EMA_26\n"
        "2024-01-01,TCS,100,110,90,105,1000,,,,,,,\n"
        "2024-01-02,TCS,105,115,100,112,1200,,,,,,,\n"
        "2024-01-03,TCS,112,120,108,118,1500,,,,,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_stock_service,
        lambda: StockService(data_path=csv_path),
    )

    response = client.get(
        "/stocks/tcs/history",
        params={"start_date": "2024-01-02", "end_date": "2024-01-03"},
    )

    assert response.status_code == 200
    records = response.json()
    assert len(records) == 2
    assert records[0] == {
        "date": "2024-01-02T00:00:00Z",
        "open": 105.0,
        "high": 115.0,
        "low": 100.0,
        "close": 112.0,
        "volume": 1200.0,
    }
    assert records[1]["date"] == "2024-01-03T00:00:00Z"
    assert set(records[0]) == {"date", "open", "high", "low", "close", "volume"}


def test_stock_history_rejects_invalid_symbols_and_date_ranges(
    client,
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / "stock_features.csv"
    csv_path.write_text(
        "Date,Symbol,Open,High,Low,Close,Volume,RSI_14,MACD,SMA_20,SMA_50,SMA_200,EMA_12,EMA_26\n"
        "2024-01-01,TCS,100,110,90,105,1000,,,,,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_stock_service,
        lambda: StockService(data_path=csv_path),
    )

    assert client.get("/stocks/NOTASTOCK/history").status_code == 404
    assert client.get("/stocks/bad%20symbol/history").status_code == 422
    assert client.get(
        "/stocks/TCS/history",
        params={"start_date": "2024-02-01", "end_date": "2024-01-01"},
    ).status_code == 422


def test_stock_indicators_returns_processed_values_and_filters_dates(
    client,
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / "stock_features.csv"
    csv_path.write_text(
        "Date,Symbol,Open,High,Low,Close,Volume,RSI_14,MACD,SMA_20,SMA_50,SMA_200,EMA_12,EMA_26,MACD_Signal,MACD_Histogram\n"
        "2024-01-01,TCS,100,110,90,105,1000,48.5,1.25,103,101,95,104,102,0.75,0.5\n"
        "2024-01-02,TCS,105,115,100,112,1200,53.25,2.5,106,103,96,108,105,1.5,1.0\n"
        "2024-01-03,TCS,112,120,108,118,1500,,,,,,,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_stock_service,
        lambda: StockService(data_path=csv_path),
    )

    response = client.get(
        "/stocks/tcs/indicators",
        params={"start_date": "2024-01-02", "end_date": "2024-01-03"},
    )

    assert response.status_code == 200
    records = response.json()
    assert len(records) == 2
    assert records[0] == {
        "date": "2024-01-02T00:00:00Z",
        "close": 112.0,
        "rsi_14": 53.25,
        "macd": 2.5,
        "macd_signal": 1.5,
        "macd_histogram": 1.0,
        "sma_20": 106.0,
        "sma_50": 103.0,
        "sma_200": 96.0,
        "ema_12": 108.0,
        "ema_26": 105.0,
    }
    assert records[1]["rsi_14"] is None
    assert records[1]["macd_signal"] is None


def test_stock_indicators_rejects_invalid_symbol_and_date_range(
    client,
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / "stock_features.csv"
    csv_path.write_text(
        "Date,Symbol,Open,High,Low,Close,Volume,RSI_14,MACD,SMA_20,SMA_50,SMA_200,EMA_12,EMA_26\n"
        "2024-01-01,TCS,100,110,90,105,1000,48.5,1.25,103,101,95,104,102\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_stock_service,
        lambda: StockService(data_path=csv_path),
    )

    assert client.get("/stocks/NOTASTOCK/indicators").status_code == 404
    assert client.get("/stocks/bad%20symbol/indicators").status_code == 422
    assert client.get(
        "/stocks/TCS/indicators",
        params={"start_date": "2024-02-01", "end_date": "2024-01-01"},
    ).status_code == 422


def test_prediction_success_and_invalid_symbol(
    client,
    fake_prediction_service,
):
    response = client.get("/predict/tcs")
    assert response.status_code == 200
    assert response.json()["symbol"] == "TCS"
    assert response.json()["prediction"] == "BUY"
    assert response.json()["confidence"] == 0.8
    assert response.json()["probabilities"] == {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}
    assert sum(response.json()["probabilities"].values()) == pytest.approx(1.0)
    assert set(response.json()) == {
        "symbol",
        "prediction",
        "confidence",
        "probabilities",
        "prediction_timestamp",
        "prediction_date",
    }
    assert client.get("/predict/bad%20symbol").status_code == 422


def test_prediction_can_select_model_b(client, fake_prediction_service, monkeypatch):
    from Backend.routes import prediction as prediction_routes

    monkeypatch.setattr(
        prediction_routes,
        "_get_model_b_prediction_service",
        lambda: fake_prediction_service,
    )
    response = client.get("/predict/TCS", params={"model": "B"})
    assert response.status_code == 200
    assert response.json()["symbol"] == "TCS"
    assert response.json()["probabilities"] == {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}
    assert client.get("/predict/TCS", params={"model": "C"}).status_code == 422


def test_prediction_service_errors(client, fake_prediction_service):
    fake_prediction_service.error = StockSymbolNotFoundError("missing")
    assert client.get("/predict/RELIANCE").status_code == 404
    fake_prediction_service.error = FileNotFoundError("model missing")
    assert client.get("/predict/RELIANCE").status_code == 503
    fake_prediction_service.error = RuntimeError("model failed")
    assert client.get("/predict/RELIANCE").status_code == 500


def test_sentiment_available_missing_and_invalid(client, fake_sentiment_service):
    response = client.get("/sentiment/tcs")
    assert response.status_code == 200
    assert response.json()["symbol"] == "TCS"
    assert response.json()["sentiment"] == "Positive"
    assert response.json()["positive_probability"] == 0.7
    assert response.json()["sentiment_score"] == 0.6

    fake_sentiment_service.error = NoNewsAvailableError("no news")
    assert client.get("/sentiment/TCS").status_code == 404
    fake_sentiment_service.error = SentimentDataUnavailableError("missing data")
    assert client.get("/sentiment/TCS").status_code == 503
    assert client.get("/sentiment/bad%20symbol").status_code == 422


def test_market_overview_success_and_provider_errors(client, fake_market_service):
    response = client.get("/market/overview")
    assert response.status_code == 200
    assert len(response.json()["indices"]) == 3
    assert response.json()["indices"][0]["name"] == "NIFTY 50"

    fake_market_service.error = MarketDataUnavailableError("unavailable")
    assert client.get("/market/overview").status_code == 503
    fake_market_service.error = RuntimeError("provider error")
    assert client.get("/market/overview").status_code == 502


def test_auth_registration_duplicate_login_and_invalid_password(client, db_session):
    email = "auth-flow@example.test"
    password = "test-password-long-enough"
    registration = client.post(
        "/auth/register",
        json={"name": "Auth User", "email": email, "password": password},
    )
    assert registration.status_code == 201
    assert "hashed_password" not in registration.json()
    user = db_session.get(User, registration.json()["user_id"])
    assert user is not None
    assert user.hashed_password != password
    assert user.hashed_password.startswith("$2")

    duplicate = client.post(
        "/auth/register",
        json={"name": "Another", "email": email, "password": password},
    )
    assert duplicate.status_code == 409
    assert client.post(
        "/auth/register",
        json={"name": "Short", "email": "short@example.test", "password": "short"},
    ).status_code == 422

    assert client.post(
        "/auth/login",
        json={"email": email, "password": "incorrect-password"},
    ).status_code == 401
    login = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["access_token"]

    current_user = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert current_user.status_code == 200
    assert current_user.json()["email"] == email
    assert client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    ).status_code == 401


def test_portfolio_and_holding_routes_require_auth_and_reject_owner_ids(client):
    assert client.get("/portfolio").status_code == 401
    assert client.get("/portfolio/holdings").status_code == 401
    assert client.post(
        "/portfolio/holdings",
        json={
            "symbol": "TCS",
            "quantity": 1,
            "purchase_price": 10,
            "portfolio_id": 1,
        },
    ).status_code == 401

    _, headers = register_and_login(client, "portfolio-owner@example.test")
    rejected_owner_field = client.post(
        "/portfolio/holdings",
        headers=headers,
        json={
            "symbol": "TCS",
            "quantity": 1,
            "purchase_price": 10,
            "portfolio_id": 1,
            "user_id": 999,
        },
    )
    assert rejected_owner_field.status_code == 422

    rejected_bad_holding = client.post(
        "/portfolio/holdings",
        headers=headers,
        json={
            "symbol": "TCS",
            "quantity": 0,
            "purchase_price": 10,
            "portfolio_id": 1,
        },
    )
    assert rejected_bad_holding.status_code == 422


def test_portfolio_holding_crud_and_user_isolation(client, test_engine):
    user_a_id, headers_a = register_and_login(client, "owner-a@example.test")
    user_b_id, headers_b = register_and_login(client, "owner-b@example.test")

    created_portfolio = client.post(
        "/portfolio",
        headers=headers_a,
        json={"portfolio_name": "Main"},
    )
    assert created_portfolio.status_code == 201
    portfolio_id = created_portfolio.json()["portfolio_id"]
    assert created_portfolio.json()["portfolio_name"] == "Main"

    own_portfolios = client.get("/portfolio", headers=headers_a)
    other_portfolios = client.get("/portfolio", headers=headers_b)
    assert own_portfolios.status_code == 200
    assert [item["portfolio_id"] for item in own_portfolios.json()] == [portfolio_id]
    assert other_portfolios.json() == []

    created_holding = client.post(
        "/portfolio/holdings",
        headers=headers_a,
        json={
            "symbol": "TCS",
            "quantity": 2,
            "purchase_price": 100.5,
            "portfolio_id": portfolio_id,
        },
    )
    assert created_holding.status_code == 201
    holding_data = created_holding.json()
    assert holding_data["symbol"] == "TCS"
    assert holding_data["quantity"] == 2
    assert holding_data["current_price"] is None
    holding_id = holding_data["holding_id"]
    database_holding_id = UUID(holding_id).int

    listed = client.get(
        "/portfolio/holdings",
        params={"portfolio_id": portfolio_id},
        headers=headers_a,
    )
    assert listed.status_code == 200
    assert len(listed.json()["holdings"]) == 1

    with Session(test_engine) as verification_session:
        persisted = verification_session.get(Holding, database_holding_id)
        assert persisted is not None
        assert persisted.portfolio_id == portfolio_id
        assert persisted.stock_symbol == "TCS"

    assert client.get(
        "/portfolio/holdings",
        params={"portfolio_id": portfolio_id},
        headers=headers_b,
    ).status_code == 404
    assert client.post(
        "/portfolio/holdings",
        headers=headers_b,
        json={
            "symbol": "INFY",
            "quantity": 1,
            "purchase_price": 20,
            "portfolio_id": portfolio_id,
        },
    ).status_code == 404
    assert client.put(
        f"/portfolio/holdings/{holding_id}",
        headers=headers_b,
        json={"quantity": 10, "purchase_price": 1},
    ).status_code == 404
    assert client.delete(
        f"/portfolio/holdings/{holding_id}",
        headers=headers_b,
    ).status_code == 404

    updated = client.put(
        f"/portfolio/holdings/{holding_id}",
        headers=headers_a,
        json={"quantity": 3, "purchase_price": 110},
    )
    assert updated.status_code == 200
    assert updated.json()["quantity"] == 3
    with Session(test_engine) as verification_session:
        persisted = verification_session.get(Holding, database_holding_id)
        assert persisted is not None
        assert persisted.quantity == 3
        assert float(persisted.purchase_price) == 110

    assert client.delete(
        f"/portfolio/holdings/{holding_id}",
        headers=headers_a,
    ).status_code == 204
    with Session(test_engine) as verification_session:
        assert verification_session.get(Holding, database_holding_id) is None

    assert client.get(
        "/portfolio/holdings",
        params={"portfolio_id": -1},
        headers=headers_a,
    ).status_code == 422
    assert user_a_id != user_b_id
