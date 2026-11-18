"""Shared constants for the stock recommendation backend."""

from Backend.App.stock_catalog import STOCK_CATALOG

SUPPORTED_STOCK_SYMBOLS: tuple[str, ...] = tuple(STOCK_CATALOG)

RECOMMENDATION_LABELS: tuple[str, ...] = ("BUY", "HOLD", "SELL")

SENTIMENT_LABELS: tuple[str, ...] = ("Positive", "Negative", "Neutral")

TECHNICAL_INDICATOR_COLUMNS: tuple[str, ...] = (
    "RSI_14",
    "MACD",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
)

API_DEFAULT_HOST: str = "127.0.0.1"
API_DEFAULT_PORT: int = 8000

__all__ = [
    "API_DEFAULT_HOST",
    "API_DEFAULT_PORT",
    "RECOMMENDATION_LABELS",
    "SENTIMENT_LABELS",
    "SUPPORTED_STOCK_SYMBOLS",
    "TECHNICAL_INDICATOR_COLUMNS",
]
