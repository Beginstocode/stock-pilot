"""Initialize missing database tables from the registered ORM models."""

from __future__ import annotations

import logging
import sys

from sqlalchemy.exc import SQLAlchemyError

from .connection import Base, engine
from .models import (
    Holding,
    NewsIntelligenceEvent,
    NewsIntelligenceOutcome,
    Portfolio,
    Prediction,
    User,
    Watchlist,
)

logger = logging.getLogger(__name__)

_REGISTERED_MODELS = (
    Holding, NewsIntelligenceEvent, NewsIntelligenceOutcome, Portfolio,
    Prediction, User, Watchlist,
)


def initialize_database() -> None:
    """Create tables that are not already present in the configured database."""
    if not _REGISTERED_MODELS:
        raise RuntimeError("No database models are registered.")
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        logger.error(
            "Database table initialization failed (%s).",
            type(exc).__name__,
        )
        raise RuntimeError(
            "Database table initialization failed. Check database availability "
            "and configuration."
        ) from exc


if __name__ == "__main__":
    try:
        initialize_database()
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)
    print("Database tables initialized successfully.")
