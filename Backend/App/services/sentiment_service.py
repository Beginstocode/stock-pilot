"""Stock-wise FinBERT sentiment access and analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd

from Backend.App.config import PROCESSED_DATA_DIR
from Backend.App.dependencies import get_finbert_sentiment_service
from Backend.App.stock_catalog import STOCK_CATALOG
from Src.Sentiment.news_sentiment import (
    FinancialNewsSentimentAnalyzer,
    NewsSentimentResult,
)

SentimentLabel = Literal["Positive", "Negative", "Neutral"]

_REQUIRED_COLUMNS = {
    "published_at",
    "symbol",
    "positive_probability",
    "negative_probability",
    "neutral_probability",
}
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")
_PROBABILITY_COLUMNS = (
    "positive_probability",
    "negative_probability",
    "neutral_probability",
)


class UnsupportedSentimentSymbolError(ValueError):
    """Raised when a requested symbol is not in the supported stock catalog."""


class SentimentDataUnavailableError(RuntimeError):
    """Raised when processed sentiment data is missing or unusable."""


class NoNewsAvailableError(LookupError):
    """Raised when no scored news is available for a supported stock."""


@dataclass(frozen=True)
class StockSentiment:
    """Aggregate FinBERT sentiment for the latest news date of a stock."""

    symbol: str
    sentiment: SentimentLabel
    positive_probability: float
    negative_probability: float
    neutral_probability: float
    sentiment_score: float
    timestamp: datetime
    article_count: int


class FinancialNewsSentimentService:
    """Read processed FinBERT scores and reuse the shared FinBERT analyzer.

    Existing per-article scores are read from the processed CSV. Raw article
    analysis delegates to the application's cached analyzer, which lazily loads
    ProsusAI/finbert once and reuses it.
    """

    def __init__(
        self,
        sentiment_path: Path = PROCESSED_DATA_DIR / "news_sentiment.csv",
        analyzer_provider: Callable[[], FinancialNewsSentimentAnalyzer] = (
            get_finbert_sentiment_service
        ),
    ) -> None:
        self._sentiment_path = sentiment_path
        self._analyzer_provider = analyzer_provider

    def get_symbol_sentiment(self, symbol: str) -> StockSentiment:
        """Return aggregate probabilities from the latest available news date."""
        normalized_symbol = self._normalize_symbol(symbol)
        frame = self._read_processed_sentiment()

        symbols = frame["symbol"].astype("string").str.strip().str.upper()
        selected = frame.loc[symbols.eq(normalized_symbol)].copy()
        if selected.empty:
            raise NoNewsAvailableError(
                f"No processed financial news sentiment is available for "
                f"{normalized_symbol}."
            )

        selected["published_at"] = pd.to_datetime(
            selected["published_at"], errors="coerce", utc=True
        )
        selected = selected.dropna(subset=["published_at"])
        if selected.empty:
            raise SentimentDataUnavailableError(
                f"Processed sentiment data for {normalized_symbol} has no valid "
                "publication timestamps."
            )

        for column in _PROBABILITY_COLUMNS:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")

        probabilities = selected.loc[:, list(_PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        )
        valid_rows = (
            pd.notna(probabilities).all(axis=1)
            & (probabilities >= 0.0).all(axis=1)
            & (probabilities <= 1.0).all(axis=1)
            & (abs(probabilities.sum(axis=1) - 1.0) <= 1e-5)
        )
        selected = selected.loc[valid_rows].copy()
        if selected.empty:
            raise SentimentDataUnavailableError(
                f"Processed sentiment data for {normalized_symbol} contains no "
                "valid probability records."
            )

        if "article_id" in selected.columns:
            selected = selected.drop_duplicates(
                subset=["article_id", "published_at"], keep="last"
            )

        latest_date = selected["published_at"].dt.date.max()
        latest = selected.loc[selected["published_at"].dt.date.eq(latest_date)]
        if latest.empty:
            raise NoNewsAvailableError(
                f"No processed financial news sentiment is available for "
                f"{normalized_symbol}."
            )

        probability_means = latest.loc[:, list(_PROBABILITY_COLUMNS)].mean()
        positive = float(probability_means["positive_probability"])
        negative = float(probability_means["negative_probability"])
        neutral = float(probability_means["neutral_probability"])
        probability_total = positive + negative + neutral
        if not math.isfinite(probability_total) or probability_total <= 0.0:
            raise SentimentDataUnavailableError(
                f"Processed sentiment probabilities for {normalized_symbol} are invalid."
            )

        positive, negative, neutral = (
            positive / probability_total,
            negative / probability_total,
            neutral / probability_total,
        )
        label = max(
            (
                ("Positive", positive),
                ("Negative", negative),
                ("Neutral", neutral),
            ),
            key=lambda item: item[1],
        )[0]
        timestamp = latest["published_at"].max().to_pydatetime()

        return StockSentiment(
            symbol=normalized_symbol,
            sentiment=label,
            positive_probability=positive,
            negative_probability=negative,
            neutral_probability=neutral,
            sentiment_score=positive - negative,
            timestamp=timestamp,
            article_count=int(len(latest)),
        )

    def analyze_articles(
        self,
        symbol: str,
        articles: Sequence[Mapping[str, Any]],
        as_of: str | datetime | pd.Timestamp | None = None,
    ) -> StockSentiment:
        """Run supplied, explicitly associated articles through the shared FinBERT pipeline."""
        normalized_symbol = self._normalize_symbol(symbol)
        analyzer: FinancialNewsSentimentAnalyzer = self._analyzer_provider()
        try:
            results: list[NewsSentimentResult] = analyzer.analyze(
                articles, as_of=as_of
            )
        except (RuntimeError, ValueError) as exc:
            raise SentimentDataUnavailableError(
                f"FinBERT could not analyze the supplied news for {normalized_symbol}: {exc}"
            ) from exc

        stock_results = [
            result
            for result in results
            if result.symbol.strip().upper() == normalized_symbol
        ]
        if not stock_results:
            raise NoNewsAvailableError(
                f"No valid, explicitly associated news is available for "
                f"{normalized_symbol}."
            )

        latest_date = max(result.publication_date for result in stock_results)
        latest_results = [
            result
            for result in stock_results
            if result.publication_date == latest_date
        ]
        positive = math.fsum(
            result.positive_probability for result in latest_results
        ) / len(latest_results)
        negative = math.fsum(
            result.negative_probability for result in latest_results
        ) / len(latest_results)
        neutral = math.fsum(
            result.neutral_probability for result in latest_results
        ) / len(latest_results)
        label = max(
            (
                ("Positive", positive),
                ("Negative", negative),
                ("Neutral", neutral),
            ),
            key=lambda item: item[1],
        )[0]
        latest_timestamp = max(
            pd.Timestamp(result.published_at) for result in latest_results
        ).to_pydatetime()

        return StockSentiment(
            symbol=normalized_symbol,
            sentiment=label,
            positive_probability=positive,
            negative_probability=negative,
            neutral_probability=neutral,
            sentiment_score=positive - negative,
            timestamp=latest_timestamp,
            article_count=len(latest_results),
        )

    def _read_processed_sentiment(self) -> pd.DataFrame:
        """Read and validate the existing processed per-article sentiment file."""
        if not self._sentiment_path.is_file():
            raise SentimentDataUnavailableError(
                f"Processed sentiment data was not found: {self._sentiment_path}"
            )
        try:
            frame = pd.read_csv(self._sentiment_path, low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise SentimentDataUnavailableError(
                f"Could not read processed sentiment data from "
                f"{self._sentiment_path}: {exc}"
            ) from exc

        frame.columns = [str(column).strip().casefold() for column in frame.columns]
        if frame.columns.duplicated().any():
            raise SentimentDataUnavailableError(
                "Processed sentiment data contains duplicate column names."
            )
        missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
        if missing_columns:
            raise SentimentDataUnavailableError(
                "Processed sentiment data is missing required columns: "
                f"{missing_columns}"
            )
        if frame.empty:
            raise NoNewsAvailableError("Processed sentiment data contains no news.")
        return frame

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise ValueError("Stock symbol must be a string.")
        normalized_symbol = symbol.strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(normalized_symbol):
            raise ValueError("Stock symbol contains invalid characters or is too long.")
        if normalized_symbol not in STOCK_CATALOG:
            raise UnsupportedSentimentSymbolError(
                f"Stock symbol '{normalized_symbol}' is not supported."
            )
        return normalized_symbol


@lru_cache(maxsize=1)
def get_sentiment_service() -> FinancialNewsSentimentService:
    """Return one reusable stock sentiment service per application process."""
    return FinancialNewsSentimentService()


__all__ = [
    "FinancialNewsSentimentService",
    "NoNewsAvailableError",
    "SentimentDataUnavailableError",
    "SentimentLabel",
    "StockSentiment",
    "UnsupportedSentimentSymbolError",
    "get_sentiment_service",
]
