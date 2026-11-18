"""Reusable formatting, conversion, and data validation helpers."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Real
from collections.abc import Iterable
from typing import TypeVar

import pandas as pd

from Backend.App.stock_catalog import STOCK_CATALOG

_STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")
T = TypeVar("T")


def normalize_stock_symbol(value: object) -> str | None:
    """Return a normalized supported stock symbol, or None for invalid input."""
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper()
    if not _STOCK_SYMBOL_PATTERN.fullmatch(symbol):
        return None
    return symbol if symbol in STOCK_CATALOG else None


def format_stock_symbol(value: object) -> str | None:
    """Format a valid supported stock symbol; return None if it is unsupported."""
    return normalize_stock_symbol(value)


def is_supported_stock_symbol(value: object) -> bool:
    """Return whether a value is a well-formed symbol in the supported catalog."""
    return normalize_stock_symbol(value) is not None


def to_datetime(
    value: object,
    *,
    default_timezone: timezone | None = timezone.utc,
) -> datetime | None:
    """Convert a date/time-like value to a datetime, or None if it is invalid.

    Naive parsed values are assigned ``default_timezone`` when provided;
    timezone-aware values retain their original timezone.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        try:
            timestamp = pd.to_datetime(value, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            return None
        if pd.isna(timestamp):
            return None
        if isinstance(timestamp, pd.Timestamp):
            parsed = timestamp.to_pydatetime()
        elif isinstance(timestamp, datetime):
            parsed = timestamp
        else:
            return None

    if parsed.tzinfo is None and default_timezone is not None:
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed


def format_date(
    value: object,
    *,
    date_format: str = "%Y-%m-%d",
) -> str | None:
    """Format a date-like value with ``date_format``; return None if invalid."""
    parsed = to_datetime(value)
    if parsed is None:
        return None
    try:
        return parsed.strftime(date_format)
    except (ValueError, TypeError):
        return None


def format_timestamp(
    value: object,
    *,
    timespec: str = "seconds",
) -> str | None:
    """Format a date-time as an ISO 8601 timestamp; return None if invalid."""
    parsed = to_datetime(value)
    if parsed is None:
        return None
    try:
        return parsed.isoformat(timespec=timespec)
    except (ValueError, TypeError):
        return None


def to_float(value: object, *, default: float | None = None) -> float | None:
    """Convert a finite real or numeric string to float, safely handling nulls."""
    if value is None or isinstance(value, bool):
        return default
    try:
        converted = (
            float(value)
            if isinstance(value, (Real, Decimal, str))
            else math.nan
        )
    except (TypeError, ValueError, OverflowError):
        return default
    return converted if math.isfinite(converted) else default


def to_int(value: object, *, default: int | None = None) -> int | None:
    """Convert a value to an integer only when it represents a whole number."""
    if value is None or isinstance(value, bool):
        return default
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            converted_float = float(text)
        elif isinstance(value, Real):
            converted_float = float(value)
        else:
            return default
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(converted_float) or not converted_float.is_integer():
        return default
    return int(converted_float)


def require_finite_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Convert and validate a finite number, raising ValueError when invalid."""
    converted = to_float(value)
    if converted is None:
        raise ValueError(f"{field_name} must be a finite number.")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{field_name} must be no greater than {maximum}.")
    return converted


def validate_required_columns(
    columns: Iterable[object] | None,
    required: set[str] | frozenset[str] | tuple[str, ...] | list[str],
) -> list[str]:
    """Return missing required column names in sorted order."""
    if columns is None:
        return sorted(set(required))
    try:
        available = {str(column).strip() for column in columns}
    except TypeError:
        return sorted(set(required))
    return sorted(set(required).difference(available))


def validate_non_empty_text(value: object, *, field_name: str) -> str:
    """Return stripped non-empty text or raise ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text.")
    return value.strip()


def first_non_none(value: T | None, fallback: T) -> T:
    """Return ``value`` when present, otherwise the supplied fallback."""
    return value if value is not None else fallback


__all__ = [
    "first_non_none",
    "format_date",
    "format_stock_symbol",
    "format_timestamp",
    "is_supported_stock_symbol",
    "normalize_stock_symbol",
    "require_finite_number",
    "to_datetime",
    "to_float",
    "to_int",
    "validate_non_empty_text",
    "validate_required_columns",
]
