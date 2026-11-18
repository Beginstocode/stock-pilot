"""Reusable SQLAlchemy CRUD operations for application database models."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from Backend.App.database.models import Holding, Portfolio, Prediction, User, Watchlist
from Backend.App.utils.constants import RECOMMENDATION_LABELS, SUPPORTED_STOCK_SYMBOLS


class DuplicateRecordError(ValueError):
    """Raised when a requested record conflicts with an existing unique record."""


class RecordNotFoundError(LookupError):
    """Raised when an update or delete targets a record that does not exist."""


class DatabaseOperationError(RuntimeError):
    """Raised when a database write fails for a reason other than a duplicate."""


def _commit(db: Session, *, duplicate_message: str | None = None) -> None:
    """Commit a write, roll back failures, and expose stable CRUD exceptions."""
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if duplicate_message is not None:
            raise DuplicateRecordError(duplicate_message) from exc
        raise DatabaseOperationError("Database constraints rejected the write.") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise DatabaseOperationError("The database write failed.") from exc


def _normalize_symbol(symbol: str) -> str:
    """Normalize and validate a symbol supported by the project."""
    if not isinstance(symbol, str):
        raise ValueError("Stock symbol must be a string.")
    normalized = symbol.strip().upper()
    if normalized not in SUPPORTED_STOCK_SYMBOLS:
        raise ValueError(f"Unsupported stock symbol: {normalized!r}.")
    return normalized


def create_user(
    db: Session,
    *,
    name: str,
    email: str,
    hashed_password: str,
) -> User:
    """Create a user using a precomputed password hash, never a plain password."""
    normalized_email = email.strip().casefold()
    if not name.strip() or not normalized_email or not hashed_password.strip():
        raise ValueError("Name, email, and hashed password are required.")
    if db.scalar(select(User.user_id).where(User.email == normalized_email)) is not None:
        raise DuplicateRecordError("A user with this email already exists.")

    user = User(
        name=name.strip(),
        email=normalized_email,
        hashed_password=hashed_password,
    )
    db.add(user)
    _commit(db, duplicate_message="A user with this email already exists.")
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Return a user by ID, or None when no matching user exists."""
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> User | None:
    """Return a user by case-insensitive email lookup, or None if absent."""
    normalized_email = email.strip().casefold()
    if not normalized_email:
        raise ValueError("Email must not be empty.")
    return db.scalar(select(User).where(User.email == normalized_email))


def create_portfolio(
    db: Session,
    *,
    user_id: int,
    portfolio_name: str,
) -> Portfolio:
    """Create a named portfolio for an existing user."""
    normalized_name = portfolio_name.strip()
    if not normalized_name:
        raise ValueError("Portfolio name must not be empty.")
    if get_user_by_id(db, user_id) is None:
        raise RecordNotFoundError(f"User {user_id} was not found.")

    portfolio = Portfolio(user_id=user_id, portfolio_name=normalized_name)
    db.add(portfolio)
    _commit(db)
    db.refresh(portfolio)
    return portfolio


def get_user_portfolios(db: Session, user_id: int) -> list[Portfolio]:
    """Return portfolios belonging to a user, ordered by portfolio ID."""
    statement = (
        select(Portfolio)
        .where(Portfolio.user_id == user_id)
        .order_by(Portfolio.portfolio_id)
    )
    return list(db.scalars(statement).all())


def add_holding(
    db: Session,
    *,
    portfolio_id: int,
    stock_symbol: str,
    quantity: int,
    purchase_price: Decimal | float | str,
    purchase_date: date | None = None,
) -> Holding:
    """Add a validated stock holding to an existing portfolio."""
    symbol = _normalize_symbol(stock_symbol)
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("Quantity must be a positive integer.")
    try:
        price = Decimal(str(purchase_price))
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError("Purchase price must be a valid positive number.") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError("Purchase price must be a valid positive number.")
    if db.get(Portfolio, portfolio_id) is None:
        raise RecordNotFoundError(f"Portfolio {portfolio_id} was not found.")

    values: dict[str, object] = {
        "portfolio_id": portfolio_id,
        "stock_symbol": symbol,
        "quantity": quantity,
        "purchase_price": price,
    }
    if purchase_date is not None:
        values["purchase_date"] = purchase_date
    holding = Holding(**values)
    db.add(holding)
    _commit(db)
    db.refresh(holding)
    return holding


def get_holding(db: Session, holding_id: int) -> Holding | None:
    """Return a holding by ID, or None when no matching holding exists."""
    return db.get(Holding, holding_id)


def update_holding(
    db: Session,
    holding_id: int,
    *,
    quantity: int | None = None,
    purchase_price: Decimal | float | str | None = None,
    purchase_date: date | None = None,
) -> Holding:
    """Update provided holding fields and return the persisted holding."""
    holding = db.get(Holding, holding_id)
    if holding is None:
        raise RecordNotFoundError(f"Holding {holding_id} was not found.")
    if quantity is None and purchase_price is None and purchase_date is None:
        raise ValueError("At least one holding field must be provided for update.")

    if quantity is not None:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        holding.quantity = quantity
    if purchase_price is not None:
        try:
            price = Decimal(str(purchase_price))
        except (ValueError, TypeError, ArithmeticError) as exc:
            raise ValueError("Purchase price must be a valid positive number.") from exc
        if not price.is_finite() or price <= 0:
            raise ValueError("Purchase price must be a valid positive number.")
        holding.purchase_price = price
    if purchase_date is not None:
        holding.purchase_date = purchase_date

    _commit(db)
    db.refresh(holding)
    return holding


def delete_holding(db: Session, holding_id: int) -> None:
    """Delete a holding, raising RecordNotFoundError if it does not exist."""
    holding = db.get(Holding, holding_id)
    if holding is None:
        raise RecordNotFoundError(f"Holding {holding_id} was not found.")
    db.delete(holding)
    _commit(db)


def add_watchlist_stock(
    db: Session,
    *,
    user_id: int,
    stock_symbol: str,
) -> Watchlist:
    """Add a supported symbol to a user's watchlist."""
    symbol = _normalize_symbol(stock_symbol)
    if get_user_by_id(db, user_id) is None:
        raise RecordNotFoundError(f"User {user_id} was not found.")
    existing = db.scalar(
        select(Watchlist).where(
            Watchlist.user_id == user_id,
            Watchlist.stock_symbol == symbol,
        )
    )
    if existing is not None:
        raise DuplicateRecordError(
            f"{symbol} is already in user {user_id}'s watchlist."
        )

    watchlist_entry = Watchlist(user_id=user_id, stock_symbol=symbol)
    db.add(watchlist_entry)
    _commit(
        db,
        duplicate_message=f"{symbol} is already in user {user_id}'s watchlist.",
    )
    db.refresh(watchlist_entry)
    return watchlist_entry


def list_watchlist_stocks(db: Session, user_id: int) -> list[Watchlist]:
    """Return a user's watchlist entries ordered by symbol."""
    statement = (
        select(Watchlist)
        .where(Watchlist.user_id == user_id)
        .order_by(Watchlist.stock_symbol)
    )
    return list(db.scalars(statement).all())


def remove_watchlist_stock(
    db: Session,
    *,
    user_id: int,
    stock_symbol: str,
) -> None:
    """Remove a symbol from a user's watchlist."""
    symbol = _normalize_symbol(stock_symbol)
    entry = db.scalar(
        select(Watchlist).where(
            Watchlist.user_id == user_id,
            Watchlist.stock_symbol == symbol,
        )
    )
    if entry is None:
        raise RecordNotFoundError(
            f"{symbol} was not found in user {user_id}'s watchlist."
        )
    db.delete(entry)
    _commit(db)


def save_prediction(
    db: Session,
    *,
    stock_symbol: str,
    recommendation: str,
    confidence_score: Decimal | float | str,
    prediction_timestamp: datetime | None = None,
) -> Prediction:
    """Persist an existing model prediction without generating one."""
    symbol = _normalize_symbol(stock_symbol)
    label = recommendation.strip().upper()
    if label not in RECOMMENDATION_LABELS:
        raise ValueError(f"Unsupported recommendation label: {label!r}.")
    try:
        confidence = Decimal(str(confidence_score))
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError("Confidence score must be between 0 and 1.") from exc
    if not confidence.is_finite() or not Decimal("0") <= confidence <= Decimal("1"):
        raise ValueError("Confidence score must be between 0 and 1.")

    prediction_fields: dict[str, object] = {
        "stock_symbol": symbol,
        "recommendation": label,
        "confidence_score": confidence,
    }
    if prediction_timestamp is not None:
        prediction_fields["prediction_timestamp"] = prediction_timestamp
    prediction = Prediction(**prediction_fields)
    db.add(prediction)
    _commit(db)
    db.refresh(prediction)
    return prediction


def get_prediction_history(
    db: Session,
    stock_symbol: str,
    *,
    limit: int | None = None,
) -> list[Prediction]:
    """Return a stock's prediction history, newest first."""
    symbol = _normalize_symbol(stock_symbol)
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero.")
    statement = (
        select(Prediction)
        .where(Prediction.stock_symbol == symbol)
        .order_by(Prediction.prediction_timestamp.desc(), Prediction.prediction_id.desc())
    )
    if limit is not None:
        statement = statement.limit(limit)
    return list(db.scalars(statement).all())


__all__ = [
    "DatabaseOperationError",
    "DuplicateRecordError",
    "RecordNotFoundError",
    "add_holding",
    "add_watchlist_stock",
    "create_portfolio",
    "create_user",
    "delete_holding",
    "get_holding",
    "get_prediction_history",
    "get_user_by_email",
    "get_user_by_id",
    "get_user_portfolios",
    "list_watchlist_stocks",
    "remove_watchlist_stock",
    "save_prediction",
    "update_holding",
]
