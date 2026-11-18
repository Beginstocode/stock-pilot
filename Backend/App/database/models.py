"""SQLAlchemy ORM models for users, portfolios, and stock recommendations."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from Backend.App.database.connection import Base
from Backend.App.utils.constants import RECOMMENDATION_LABELS, SUPPORTED_STOCK_SYMBOLS

_SUPPORTED_SYMBOL_SQL = ", ".join(
    f"'{symbol}'" for symbol in SUPPORTED_STOCK_SYMBOLS
)
_RECOMMENDATION_SQL = ", ".join(
    f"'{label}'" for label in RECOMMENDATION_LABELS
)


class User(Base):
    """Application user with a securely hashed password."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("length(name) > 0", name="ck_users_name_nonempty"),
        CheckConstraint("length(email) > 0", name="ck_users_email_nonempty"),
        CheckConstraint(
            "length(hashed_password) > 0",
            name="ck_users_hashed_password_nonempty",
        ),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    portfolios: Mapped[list[Portfolio]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    watchlists: Mapped[list[Watchlist]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Portfolio(Base):
    """Named portfolio owned by a user."""

    __tablename__ = "portfolios"
    __table_args__ = (
        CheckConstraint(
            "length(portfolio_name) > 0",
            name="ck_portfolios_name_nonempty",
        ),
        Index("ix_portfolios_user_id", "user_id"),
    )

    portfolio_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    portfolio_name: Mapped[str] = mapped_column(String(150), nullable=False)

    user: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list[Holding]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Holding(Base):
    """A stock holding and its purchase information in a portfolio."""

    __tablename__ = "holdings"
    __table_args__ = (
        CheckConstraint(
            f"stock_symbol IN ({_SUPPORTED_SYMBOL_SQL})",
            name="ck_holdings_supported_stock_symbol",
        ),
        CheckConstraint("quantity > 0", name="ck_holdings_quantity_positive"),
        CheckConstraint(
            "purchase_price > 0",
            name="ck_holdings_purchase_price_positive",
        ),
        Index("ix_holdings_portfolio_id", "portfolio_id"),
        Index("ix_holdings_stock_symbol", "stock_symbol"),
    )

    holding_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id", ondelete="CASCADE"),
        nullable=False,
    )
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    purchase_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    purchase_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        server_default=func.current_date(),
    )

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


class Watchlist(Base):
    """A stock symbol saved to a user's watchlist."""

    __tablename__ = "watchlists"
    __table_args__ = (
        CheckConstraint(
            f"stock_symbol IN ({_SUPPORTED_SYMBOL_SQL})",
            name="ck_watchlists_supported_stock_symbol",
        ),
        UniqueConstraint(
            "user_id",
            "stock_symbol",
            name="uq_watchlists_user_stock_symbol",
        ),
        Index("ix_watchlists_user_id", "user_id"),
        Index("ix_watchlists_stock_symbol", "stock_symbol"),
    )

    watchlist_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False)

    user: Mapped[User] = relationship(back_populates="watchlists")


class Prediction(Base):
    """Persisted stock recommendation output and confidence."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            f"stock_symbol IN ({_SUPPORTED_SYMBOL_SQL})",
            name="ck_predictions_supported_stock_symbol",
        ),
        CheckConstraint(
            f"recommendation IN ({_RECOMMENDATION_SQL})",
            name="ck_predictions_recommendation",
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="ck_predictions_confidence_range",
        ),
        Index("ix_predictions_stock_symbol", "stock_symbol"),
        Index("ix_predictions_prediction_timestamp", "prediction_timestamp"),
    )

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    recommendation: Mapped[str] = mapped_column(String(4), nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
    )
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class NewsIntelligenceEvent(Base):
    """Immutable first recorded AI assessment for one article and stock."""

    __tablename__ = "news_intelligence_events"
    __table_args__ = (
        UniqueConstraint("article_id", "stock_symbol", name="uq_news_event_article_stock"),
        CheckConstraint(
            f"stock_symbol IN ({_SUPPORTED_SYMBOL_SQL})",
            name="ck_news_event_supported_stock_symbol",
        ),
        CheckConstraint(
            "finbert_confidence IS NULL OR (finbert_confidence >= 0 AND finbert_confidence <= 1)",
            name="ck_news_event_finbert_confidence_range",
        ),
        Index("ix_news_events_article_id", "article_id"),
        Index("ix_news_events_published_at", "published_at"),
        Index("ix_news_events_analysis_timestamp", "analysis_timestamp"),
        Index("ix_news_events_stock_symbol", "stock_symbol"),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False)
    article_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    article_title: Mapped[str] = mapped_column(String(500), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    finbert_sentiment: Mapped[str | None] = mapped_column(String(20))
    finbert_confidence: Mapped[Decimal | None] = mapped_column(Numeric(7, 6))
    ai_impact_direction: Mapped[str] = mapped_column(String(20), nullable=False)
    ai_impact_strength: Mapped[str] = mapped_column(String(20), nullable=False)
    ai_explanation: Mapped[str] = mapped_column(String(1500), nullable=False)
    content_source: Mapped[str] = mapped_column(String(30), nullable=False)
    content_quality: Mapped[str] = mapped_column(String(20), nullable=False)
    analysis_model_version: Mapped[str] = mapped_column(String(120), nullable=False)

    outcome: Mapped[NewsIntelligenceOutcome | None] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        uselist=False,
    )


class NewsIntelligenceOutcome(Base):
    """Subsequent observed stock and benchmark returns for a stored event."""

    __tablename__ = "news_intelligence_outcomes"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_news_outcome_event"),
        Index("ix_news_outcomes_calculated_at", "calculated_at"),
    )

    outcome_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("news_intelligence_events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    stock_return_1d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    stock_return_3d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    stock_return_5d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    nifty_return_1d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    nifty_return_3d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    nifty_return_5d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    excess_return_1d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    excess_return_3d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    excess_return_5d_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    observed_at_1d: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at_3d: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at_5d: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    event: Mapped[NewsIntelligenceEvent] = relationship(back_populates="outcome")


__all__ = [
    "Base", "Holding", "NewsIntelligenceEvent", "NewsIntelligenceOutcome",
    "Portfolio", "Prediction", "User", "Watchlist",
]
