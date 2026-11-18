"""Check PostgreSQL connectivity using the application's configured engine."""

from __future__ import annotations

import sys

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .connection import engine


def test_connection() -> bool:
    """Run ``SELECT 1`` and print useful connection error details."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar_one()
        if result != 1:
            print(
                "PostgreSQL connection failed: SELECT 1 returned "
                f"unexpected result {result!r}."
            )
            return False
    except SQLAlchemyError as exc:
        print(f"PostgreSQL connection failed ({type(exc).__name__}): {exc}")
        return False

    print("PostgreSQL connection successful (SELECT 1).")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_connection() else 1)
