"""SQLAlchemy CRUD tests against an isolated temporary database."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from Backend.App.database.crud import (
    DuplicateRecordError,
    RecordNotFoundError,
    add_holding,
    add_watchlist_stock,
    create_portfolio,
    create_user,
    delete_holding,
    get_holding,
    get_prediction_history,
    get_user_by_email,
    get_user_by_id,
    get_user_portfolios,
    list_watchlist_stocks,
    remove_watchlist_stock,
    save_prediction,
    update_holding,
)


def test_user_portfolio_holding_and_missing_record_operations(db_session):
    user = create_user(
        db_session,
        name="Database Test",
        email="db-test@example.test",
        hashed_password="bcrypt-hash-for-test",
    )
    assert get_user_by_id(db_session, user.user_id) is not None
    assert get_user_by_email(db_session, "DB-TEST@example.test") == user
    with pytest.raises(DuplicateRecordError):
        create_user(
            db_session,
            name="Duplicate",
            email="db-test@example.test",
            hashed_password="bcrypt-hash",
        )

    portfolio = create_portfolio(
        db_session,
        user_id=user.user_id,
        portfolio_name="Primary",
    )
    assert get_user_portfolios(db_session, user.user_id) == [portfolio]

    holding = add_holding(
        db_session,
        portfolio_id=portfolio.portfolio_id,
        stock_symbol=" tcs ",
        quantity=2,
        purchase_price=Decimal("100.25"),
        purchase_date=date(2026, 11, 1),
    )
    assert holding.stock_symbol == "TCS"
    assert get_holding(db_session, holding.holding_id) == holding

    updated = update_holding(
        db_session,
        holding.holding_id,
        quantity=4,
        purchase_price=Decimal("120.50"),
    )
    assert updated.quantity == 4
    assert updated.purchase_price == Decimal("120.50")

    delete_holding(db_session, holding.holding_id)
    assert get_holding(db_session, holding.holding_id) is None
    with pytest.raises(RecordNotFoundError):
        delete_holding(db_session, holding.holding_id)
    with pytest.raises(RecordNotFoundError):
        create_portfolio(
            db_session,
            user_id=999999,
            portfolio_name="Missing user",
        )


def test_watchlist_and_prediction_history_operations(db_session):
    user = create_user(
        db_session,
        name="History Test",
        email="history-test@example.test",
        hashed_password="bcrypt-hash-for-test",
    )
    first = add_watchlist_stock(
        db_session,
        user_id=user.user_id,
        stock_symbol="INFY",
    )
    assert list_watchlist_stocks(db_session, user.user_id) == [first]
    with pytest.raises(DuplicateRecordError):
        add_watchlist_stock(
            db_session,
            user_id=user.user_id,
            stock_symbol="infy",
        )
    remove_watchlist_stock(
        db_session,
        user_id=user.user_id,
        stock_symbol="INFY",
    )
    assert list_watchlist_stocks(db_session, user.user_id) == []
    with pytest.raises(RecordNotFoundError):
        remove_watchlist_stock(
            db_session,
            user_id=user.user_id,
            stock_symbol="INFY",
        )

    timestamp = datetime(2026, 11, 12, tzinfo=timezone.utc)
    saved = save_prediction(
        db_session,
        stock_symbol="TCS",
        recommendation="buy",
        confidence_score=Decimal("0.875"),
        prediction_timestamp=timestamp,
    )
    assert saved.recommendation == "BUY"
    assert get_prediction_history(db_session, "TCS", limit=1) == [saved]
    with pytest.raises(ValueError):
        save_prediction(
            db_session,
            stock_symbol="TCS",
            recommendation="WAIT",
            confidence_score=0.5,
        )
    with pytest.raises(ValueError):
        add_holding(
            db_session,
            portfolio_id=1,
            stock_symbol="TCS",
            quantity=0,
            purchase_price=10,
        )
