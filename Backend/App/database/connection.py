"""SQLAlchemy engine, session factory, and FastAPI database dependency."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from dotenv import load_dotenv
from fastapi import HTTPException, status
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import ArgumentError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from Backend.App.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

_database_url = os.getenv("DATABASE_URL")
if not _database_url:
    raise RuntimeError(
        f"DATABASE_URL is not configured. Set it in {PROJECT_ROOT / '.env'} "
        "or in the process environment."
    )

try:
    engine: Engine = create_engine(_database_url, pool_pre_ping=True)
except (ArgumentError, ModuleNotFoundError) as exc:
    logger.error("Could not configure the database engine: %s", exc)
    raise RuntimeError(
        "Could not configure the database engine. Check DATABASE_URL "
        "and ensure its database driver is installed."
    ) from exc

if engine.url.host is not None and "@" in engine.url.host:
    raise RuntimeError(
        "DATABASE_URL is malformed. Percent-encode reserved characters in "
        "credentials, such as '@' as '%40'."
    )

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and reliably roll back or close it."""
    db = SessionLocal()
    try:
        yield db
    except OperationalError as exc:
        logger.exception("Database connection or operation failed.")
        try:
            db.rollback()
        except SQLAlchemyError:
            logger.exception("Could not roll back the failed database session.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is currently unavailable.",
        ) from exc
    except SQLAlchemyError:
        logger.exception("Database operation failed.")
        db.rollback()
        raise
    finally:
        db.close()


__all__ = ["Base", "SessionLocal", "engine", "get_db"]
