"""Historical news event study against adjusted stock prices and NIFTY 50."""

from __future__ import annotations

import hashlib
import math
import re
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd

from Backend.App.config import DATA_DIR
from Backend.App.schemas.historical_news import (
    HistoricalImpactAvailability,
    HistoricalNewsEvent,
    HistoricalNewsPerformanceResponse,
    HorizonMetrics,
    HorizonOutcome,
    SentimentPerformance,
    StockPerformance,
)
from Backend.App.stock_catalog import STOCK_CATALOG

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE = time(15, 30)
NEUTRAL_EXCESS_RETURN_THRESHOLD_PCT = 0.05
HORIZONS: tuple[tuple[str, int], ...] = (("1d", 1), ("3d", 3), ("5d", 5))


class HistoricalNewsDataError(RuntimeError):
    """Raised when required historical files are missing or malformed."""


def _parse_calendar_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).date()
    text = str(value).strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).date()
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(text, dayfirst=True, errors="raise").date()
        except (TypeError, ValueError):
            return None


def _normalize_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.casefold()
    if not parts.scheme or not parts.netloc:
        return text.casefold()
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, "")
    )


def _normalize_headline(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: Path, description: str, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise HistoricalNewsDataError(f"Required {description} dataset is missing: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HistoricalNewsDataError(f"Could not read {description} dataset: {exc}") from exc
    frame.columns = frame.columns.astype(str).str.strip()
    missing = required.difference(frame.columns)
    if missing:
        raise HistoricalNewsDataError(
            f"{description} dataset is missing required columns: {sorted(missing)}"
        )
    return frame


class HistoricalNewsPerformanceService:
    """Compute look-ahead-safe post-publication stock and benchmark returns."""

    def __init__(
        self,
        *,
        news_path: Path | None = None,
        sentiment_path: Path | None = None,
        stock_prices_path: Path | None = None,
        nifty_path: Path | None = None,
        as_of: date | None = None,
    ) -> None:
        self.news_path = news_path or DATA_DIR / "raw" / "financial_news.csv"
        self.sentiment_path = sentiment_path or DATA_DIR / "processed" / "news_sentiment.csv"
        self.stock_prices_path = stock_prices_path or DATA_DIR / "processed" / "stock_price_combined.csv"
        self.nifty_path = nifty_path or DATA_DIR / "cleaned" / "cleaned_nifty50.csv"
        self._as_of_override = as_of
        self.as_of = as_of or datetime.now(MARKET_TIMEZONE).date()
        self.publication_cutoff = datetime.now(MARKET_TIMEZONE)
        self._signature: tuple[int, ...] | None = None
        self._frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None] | None = None
        self._sentiment_unavailable_reason: str | None = None

    def _load_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        paths = (self.news_path, self.sentiment_path, self.stock_prices_path, self.nifty_path)
        signature = tuple(path.stat().st_mtime_ns if path.is_file() else -1 for path in paths)
        if self._frames is not None and signature == self._signature:
            return self._frames

        news = _read_csv(self.news_path, "raw financial news", {"Date", "Symbol", "Headline", "URL"})
        stocks = _read_csv(
            self.stock_prices_path,
            "historical stock prices",
            {"Date", "Symbol", "Close"},
        )
        stocks.attrs["stock_price_column"] = "Adj Close" if "Adj Close" in stocks else "Close"
        nifty_path = self.nifty_path
        if not nifty_path.is_file() and (DATA_DIR / "raw" / "nifty50.csv").is_file():
            nifty_path = DATA_DIR / "raw" / "nifty50.csv"
        nifty = _read_csv(nifty_path, "NIFTY 50 prices", {"Date", "Close"})
        sentiment: pd.DataFrame | None
        try:
            sentiment = _read_csv(
                self.sentiment_path,
                "FinBERT sentiment",
                {"article_id", "symbol", "sentiment"},
            )
        except HistoricalNewsDataError as exc:
            sentiment = None
            self._sentiment_unavailable_reason = str(exc)
        if sentiment is not None:
            self._sentiment_unavailable_reason = None

        self._frames = (news, stocks, nifty, sentiment)
        self._signature = signature
        return self._frames

    def analyze(self) -> HistoricalNewsPerformanceResponse:
        current_time = datetime.now(MARKET_TIMEZONE)
        self.publication_cutoff = current_time
        self.as_of = self._as_of_override or (
            current_time.date()
            if current_time.time() >= MARKET_CLOSE
            else current_time.date() - timedelta(days=1)
        )
        news, stocks, nifty, sentiment = self._load_frames()
        sessions, nifty_prices = self._prepare_nifty(nifty)
        stock_prices, stock_symbols, stock_dates = self._prepare_stocks(stocks)
        if not sessions or not stock_dates:
            raise HistoricalNewsDataError("Price datasets contain no valid trading sessions.")
        if len(nifty_prices) != len(sessions):
            raise HistoricalNewsDataError("NIFTY 50 data contains duplicate or invalid session rows.")

        sentiment_map = self._sentiment_map(sentiment)
        deduped_events = self._prepare_events(news)
        records: list[HistoricalNewsEvent] = []
        raw_observations: list[dict[str, Any]] = []
        for event in deduped_events:
            publication = event["published_at"]
            anchor_index, exclusion_reason = self._find_anchor(publication, sessions)
            anchor_date = sessions[anchor_index] if anchor_index is not None else None
            if anchor_date and anchor_date > self.as_of:
                anchor_index, anchor_date = None, None
                exclusion_reason = "The first eligible trading session is after the analysis date."
            for symbol in event["symbols"]:
                if symbol not in stock_symbols:
                    event_reason = f"No historical price series is available for {symbol}."
                    record = self._event_record(event, symbol, sentiment_map, None, {}, False, event_reason)
                    records.append(record)
                    continue
                outcomes: dict[str, HorizonOutcome] = {}
                base_date: date | None = None
                if anchor_index is not None and anchor_index > 0:
                    base_date = sessions[anchor_index - 1]
                event_eligible = False
                event_reason = exclusion_reason
                if base_date is None and event_reason is None:
                    event_reason = "No benchmark session exists before the eligible event session."
                for key, trading_days in HORIZONS:
                    outcome_date: date | None = None
                    stock_return = nifty_return = excess_return = None
                    if anchor_index is not None:
                        outcome_index = anchor_index + trading_days - 1
                        if outcome_index >= len(sessions):
                            event_reason = event_reason or "Insufficient future NIFTY sessions for this horizon."
                        else:
                            outcome_date = sessions[outcome_index]
                            if outcome_date > self.as_of:
                                event_reason = event_reason or "Future price data is not yet observable."
                                outcome_date = None
                            elif base_date is not None:
                                symbol_series = stock_prices[symbol]
                                stock_base = symbol_series.get(base_date)
                                stock_outcome = symbol_series.get(outcome_date) if outcome_date else None
                                nifty_base = nifty_prices.get(base_date)
                                nifty_outcome = nifty_prices.get(outcome_date) if outcome_date else None
                                if stock_base and stock_outcome and nifty_base and nifty_outcome:
                                    stock_return = (stock_outcome / stock_base - 1.0) * 100.0
                                    nifty_return = (nifty_outcome / nifty_base - 1.0) * 100.0
                                    excess_return = stock_return - nifty_return
                                    event_eligible = event_eligible or key == "1d"
                                else:
                                    event_reason = event_reason or "Stock or NIFTY close is missing for the baseline or horizon date."
                    outcomes[key] = HorizonOutcome(
                        observation_date=outcome_date,
                        stock_return_pct=stock_return,
                        nifty_return_pct=nifty_return,
                        excess_return_pct=excess_return,
                    )
                record = self._event_record(
                    event,
                    symbol,
                    sentiment_map,
                    anchor_date,
                    outcomes,
                    event_eligible,
                    None if event_eligible else event_reason or "No complete stock and NIFTY return observation.",
                )
                records.append(record)
                raw_observations.append(
                    {
                        "event_id": event["event_id"],
                        "symbol": symbol,
                        "sentiment": record.finbert_sentiment,
                        "score": record.finbert_score,
                        "outcomes": outcomes,
                    }
                )

        metrics = [self._horizon_metrics(key, raw_observations) for key, _ in HORIZONS]
        sentiment_performance = [
            self._sentiment_performance(label, raw_observations)
            for label in ("Positive", "Negative", "Neutral")
        ]
        stock_comparison = [
            self._stock_performance(symbol, raw_observations)
            for symbol in sorted(stock_symbols)
        ]
        analyzed_articles = {
            row["event_id"]
            for row in raw_observations
            if row["outcomes"]["1d"].excess_return_pct is not None
        }
        event_article_ids = {event["event_id"] for event in deduped_events}
        first_news_date = min(event["published_at"].date() for event in deduped_events) if deduped_events else self.as_of
        eligible_stock_dates = [d for d in stock_dates if d <= self.as_of]
        eligible_nifty_dates = [d for d in sessions if d <= self.as_of]
        if not eligible_stock_dates or not eligible_nifty_dates:
            raise HistoricalNewsDataError("No stock and benchmark prices are available as of the analysis date.")

        return HistoricalNewsPerformanceResponse(
            data_as_of=self.as_of,
            stock_price_coverage_start=min(eligible_stock_dates),
            stock_price_coverage_end=max(eligible_stock_dates),
            nifty_coverage_start=min(eligible_nifty_dates),
            nifty_coverage_end=max(eligible_nifty_dates),
            available_symbols=sorted(stock_symbols.intersection(STOCK_CATALOG)),
            raw_news_records=len(news),
            finbert_records=0 if sentiment is None else len(sentiment),
            finbert_available=sentiment is not None,
            finbert_unavailable_reason=self._sentiment_unavailable_reason,
            total_events_analyzed=len(analyzed_articles),
            excluded_events=len(event_article_ids - analyzed_articles),
            returns_by_horizon=metrics,
            sentiment_performance=sentiment_performance,
            ai_impact_performance=HistoricalImpactAvailability(
                available=False,
                reason=(
                    "Historical AI impact analyses are not stored in this project. The current impact endpoint analyzes articles on demand and keeps only an in-memory cache, so no time-aligned impact directions or strengths are available for a valid historical comparison."
                ),
            ),
            stock_comparison=stock_comparison,
            events=records,
            disclaimer=(
                "Observed post-publication returns are associations, not evidence that news caused price changes. "
                "Daily closes cannot measure intraday reactions. Events published after the 15:30 IST market close are assigned to the next NIFTY trading session. "
                f"Stock prices consistently use {stocks.attrs['stock_price_column']} for this dataset; rows missing that value are excluded, with no per-row fallback. "
                "NIFTY uses Close, not a dividend-adjusted total-return index, so benchmark returns and stock adjustment basis may differ."
            ),
        )

    def _prepare_nifty(self, frame: pd.DataFrame) -> tuple[list[date], dict[date, float]]:
        prices: dict[date, float] = {}
        for row in frame.to_dict("records"):
            session = _parse_calendar_date(row.get("Date"))
            close = _finite_number(row.get("Close"))
            if session is None or close is None or close <= 0 or session > self.as_of:
                continue
            if session in prices:
                if not math.isclose(prices[session], close, rel_tol=0.0, abs_tol=1e-8):
                    raise HistoricalNewsDataError(f"NIFTY 50 has conflicting duplicate rows for {session}.")
                continue
            prices[session] = close
        sessions = sorted(prices)
        return sessions, prices

    def _prepare_stocks(
        self, frame: pd.DataFrame
    ) -> tuple[dict[str, dict[date, float]], set[str], list[date]]:
        prices: dict[str, dict[date, float]] = {}
        all_dates: set[date] = set()
        price_column = frame.attrs.get("stock_price_column", "Adj Close" if "Adj Close" in frame else "Close")
        for row in frame.to_dict("records"):
            symbol = str(row.get("Symbol") or "").strip().upper()
            session = _parse_calendar_date(row.get("Date"))
            close = _finite_number(row.get(price_column))
            if not symbol or session is None or session > self.as_of or close is None or close <= 0:
                continue
            if session in prices.setdefault(symbol, {}):
                raise HistoricalNewsDataError(f"{symbol} has duplicate price rows for {session}.")
            prices[symbol][session] = close
            all_dates.add(session)
        return prices, set(prices), sorted(all_dates)

    def _sentiment_map(self, frame: pd.DataFrame | None) -> dict[tuple[str, str], tuple[str | None, float | None]]:
        if frame is None:
            return {}
        grouped: dict[tuple[str, str], list[tuple[str | None, float | None]]] = {}
        for row in frame.to_dict("records"):
            key = (_normalize_url(row.get("article_id")), str(row.get("symbol") or "").strip().upper())
            label = str(row.get("sentiment") or "").strip().title()
            if label not in {"Positive", "Negative", "Neutral"}:
                label = ""
            grouped.setdefault(key, []).append((label or None, _finite_number(row.get("sentiment_score"))))
        results = {}
        for key, values in grouped.items():
            labels = {item[0] for item in values if item[0]}
            scores = [item[1] for item in values if item[1] is not None]
            label = next(iter(labels)) if len(labels) == 1 else None
            score = sum(scores) / len(scores) if scores and len(labels) <= 1 else None
            results[key] = (label, score)
        return results

    def _prepare_events(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        required = {"Date", "Symbol", "Headline", "URL"}
        if not required.issubset(frame.columns):
            raise HistoricalNewsDataError(f"Raw news is missing columns: {sorted(required - set(frame.columns))}")
        events: dict[str, dict[str, Any]] = {}
        duplicate_index: dict[tuple[str, str], str] = {}
        for row in frame.to_dict("records"):
            try:
                published = pd.Timestamp(row.get("Date"))
            except (TypeError, ValueError):
                continue
            if pd.isna(published) or published.tzinfo is None:
                continue
            published = published.tz_convert("UTC")
            if published > pd.Timestamp(self.publication_cutoff.astimezone(ZoneInfo("UTC"))):
                continue
            local_date = published.tz_convert(MARKET_TIMEZONE).date()
            headline_value, symbol_value, url_value = row.get("Headline"), row.get("Symbol"), row.get("URL")
            headline = "" if pd.isna(headline_value) else str(headline_value).strip()
            symbol = "" if pd.isna(symbol_value) else str(symbol_value).strip().upper()
            url = "" if pd.isna(url_value) else str(url_value).strip()
            if not headline or symbol not in STOCK_CATALOG or not url.casefold().startswith(("http://", "https://")):
                continue
            normalized_url = _normalize_url(url)
            normalized_headline = _normalize_headline(headline)
            duplicate_key = (local_date.isoformat(), normalized_headline)
            event_id = hashlib.sha256(
                (normalized_url or "|".join(duplicate_key)).encode("utf-8")
            ).hexdigest()[:20]
            existing_id = event_id if event_id in events else duplicate_index.get(duplicate_key)
            if existing_id is None:
                events[event_id] = {
                    "event_id": event_id,
                    "article_url": url,
                    "normalized_urls": {normalized_url},
                    "headline": headline,
                    "normalized_headline": normalized_headline,
                    "published_at": published,
                    "symbols": {symbol},
                }
                duplicate_index[duplicate_key] = event_id
            else:
                event = events[existing_id]
                event["symbols"].add(symbol)
                if normalized_url:
                    event["normalized_urls"].add(normalized_url)
                if published < event["published_at"]:
                    event["published_at"] = published
                    event["article_url"] = url
        prepared = []
        for event in events.values():
            event["symbols"] = sorted(event["symbols"])
            prepared.append(event)
        return sorted(prepared, key=lambda item: (item["published_at"], item["event_id"]))

    def _find_anchor(self, published: pd.Timestamp, sessions: list[date]) -> tuple[int | None, str | None]:
        local = published.tz_convert(MARKET_TIMEZONE)
        if local.time() >= MARKET_CLOSE:
            index = bisect_right(sessions, local.date())
        else:
            index = bisect_left(sessions, local.date())
        if index >= len(sessions):
            return None, "No NIFTY trading session follows the article publication date."
        return index, None

    def _event_record(
        self,
        event: dict[str, Any],
        symbol: str,
        sentiment_map: dict[tuple[str, str], tuple[str | None, float | None]],
        anchor_date: date | None,
        outcomes: dict[str, HorizonOutcome],
        eligible: bool,
        exclusion_reason: str | None,
    ) -> HistoricalNewsEvent:
        scores = [sentiment_map.get((url, symbol)) for url in event["normalized_urls"]]
        scores = [item for item in scores if item is not None]
        labels = {item[0] for item in scores if item[0]}
        sentiment_label = next(iter(labels)) if len(labels) == 1 else None
        finite_scores = [item[1] for item in scores if item[1] is not None]
        sentiment_score = sum(finite_scores) / len(finite_scores) if finite_scores and len(labels) <= 1 else None
        return HistoricalNewsEvent(
            event_id=event["event_id"],
            article_url=event["article_url"],
            headline=event["headline"],
            published_at=event["published_at"].to_pydatetime(),
            symbol=symbol,
            finbert_sentiment=sentiment_label,
            finbert_score=sentiment_score,
            anchor_date=anchor_date,
            outcomes=outcomes,
            eligible=eligible,
            exclusion_reason=exclusion_reason,
        )

    def _horizon_metrics(self, horizon: str, rows: list[dict[str, Any]]) -> HorizonMetrics:
        eligible = [row for row in rows if row["outcomes"][horizon].excess_return_pct is not None]
        returns = [row["outcomes"][horizon].stock_return_pct for row in eligible]
        nifty = [row["outcomes"][horizon].nifty_return_pct for row in eligible]
        excess = [row["outcomes"][horizon].excess_return_pct for row in eligible]
        aligned = evaluated = neutral_sentiment = neutral_return = unavailable_sentiment = 0
        for row in eligible:
            predicted, actual = row["sentiment"], self._return_direction(row["outcomes"][horizon].excess_return_pct)
            if predicted is None:
                unavailable_sentiment += 1
            if predicted == "Neutral":
                neutral_sentiment += 1
            if actual == "Neutral":
                neutral_return += 1
            if predicted in {"Positive", "Negative"} and actual in {"Positive", "Negative"}:
                evaluated += 1
                aligned += predicted == actual
        return HorizonMetrics(
            horizon=horizon,
            event_observations=len(eligible),
            eligible_articles=len({row["event_id"] for row in eligible}),
            average_stock_return_pct=self._mean(returns),
            average_nifty_return_pct=self._mean(nifty),
            average_excess_return_pct=self._mean(excess),
            positive_outcomes=sum(value > 0 for value in returns),
            negative_outcomes=sum(value < 0 for value in returns),
            neutral_outcomes=sum(value == 0 for value in returns),
            finbert_alignment_sample_size=evaluated,
            finbert_alignment_pct=round(aligned / evaluated * 100.0, 2) if evaluated else None,
            neutral_sentiment_observations=neutral_sentiment,
            neutral_excess_return_observations=neutral_return,
            unavailable_sentiment_observations=unavailable_sentiment,
        )

    def _sentiment_performance(self, label: str, rows: list[dict[str, Any]]) -> SentimentPerformance:
        selected = [row for row in rows if row["sentiment"] == label]
        returns = {
            key: [row["outcomes"][key].excess_return_pct for row in selected if row["outcomes"][key].excess_return_pct is not None]
            for key, _ in HORIZONS
        }
        return SentimentPerformance(
            sentiment=label,
            event_observations_1d=len(returns["1d"]),
            event_observations_3d=len(returns["3d"]),
            event_observations_5d=len(returns["5d"]),
            average_excess_return_1d_pct=self._mean(returns["1d"]),
            average_excess_return_3d_pct=self._mean(returns["3d"]),
            average_excess_return_5d_pct=self._mean(returns["5d"]),
        )

    def _stock_performance(self, symbol: str, rows: list[dict[str, Any]]) -> StockPerformance:
        selected = [row for row in rows if row["symbol"] == symbol]
        means = {
            key: self._mean([
                row["outcomes"][key].excess_return_pct
                for row in selected if row["outcomes"][key].excess_return_pct is not None
            ])
            for key, _ in HORIZONS
        }
        return StockPerformance(
            symbol=symbol,
            event_observations_1d=sum(row["outcomes"]["1d"].excess_return_pct is not None for row in selected),
            event_observations_3d=sum(row["outcomes"]["3d"].excess_return_pct is not None for row in selected),
            event_observations_5d=sum(row["outcomes"]["5d"].excess_return_pct is not None for row in selected),
            average_excess_return_1d_pct=means["1d"],
            average_excess_return_3d_pct=means["3d"],
            average_excess_return_5d_pct=means["5d"],
        )

    @staticmethod
    def _return_direction(value: float) -> str:
        if value > NEUTRAL_EXCESS_RETURN_THRESHOLD_PCT:
            return "Positive"
        if value < -NEUTRAL_EXCESS_RETURN_THRESHOLD_PCT:
            return "Negative"
        return "Neutral"

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        finite = [value for value in values if value is not None and math.isfinite(value)]
        return round(sum(finite) / len(finite), 4) if finite else None


__all__ = ["HistoricalNewsDataError", "HistoricalNewsPerformanceService"]
