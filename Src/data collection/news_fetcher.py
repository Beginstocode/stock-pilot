"""Collect recent company-specific financial headlines from Google News RSS.

The script fetches current/recent RSS items only; Google News RSS is not a
seven-year historical news source. Results are attributed to the stock whose
company-specific RSS query returned the item. This search association is not a
claim that the article materially affects that stock.

Install the dependency:
    python -m pip install feedparser

Run from the project root:
    python news_fetcher.py
    python news_fetcher.py --limit-per-company 30 --delay 1.5

The CSV is written to data/raw/financial_news.csv. Publication timestamps are
preserved as UTC ISO-8601 values; entries without a valid publication date,
headline, or article URL are skipped rather than assigned invented values.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from article_extractor import ExtractedArticle, extract_article
except ModuleNotFoundError:  # Support loading this standalone script from test harnesses.
    import importlib.util
    import sys

    _extractor_path = Path(__file__).resolve().with_name("article_extractor.py")
    _extractor_spec = importlib.util.spec_from_file_location("stockpilot_article_extractor", _extractor_path)
    if _extractor_spec is None or _extractor_spec.loader is None:
        raise
    _extractor_module = importlib.util.module_from_spec(_extractor_spec)
    sys.modules[_extractor_spec.name] = _extractor_module
    _extractor_spec.loader.exec_module(_extractor_module)
    ExtractedArticle = _extractor_module.ExtractedArticle
    extract_article = _extractor_module.extract_article


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "financial_news.csv"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
RSS_LOCALE_PARAMETERS = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
COMPANIES: dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "ITC": "ITC Limited",
    "LT": "Larsen & Toubro",
    "BHARTIARTL": "Bharti Airtel",
    "TATAMOTORS": "Tata Motors",
}
CSV_COLUMNS = (
    "Date", "Symbol", "Headline", "Text", "Source", "URL", "RSSURL", "RSSSummary",
    "ArticleTitle", "Author", "ArticlePublishedAt", "ContentSource", "ContentQuality",
)
LOGGER = logging.getLogger(__name__)
@lru_cache(maxsize=512)
def _cached_article(url: str) -> ExtractedArticle | None:
    """Cache both successful and failed extraction attempts within this run."""
    try:
        return extract_article(url)
    except Exception as exc:
        LOGGER.info("Article extraction unavailable for %s: %s", url, exc)
        return None


class _TextExtractor(HTMLParser):
    """Collect readable text from the HTML fragments present in RSS summaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)


def _rss_summary_text(entry: Any) -> str:
    """Return plain-text RSS summary content without fetching the article page."""
    summary = str(getattr(entry, "summary", "") or "").strip()
    if not summary:
        content = getattr(entry, "content", None) or []
        if content:
            summary = str(getattr(content[0], "value", "") or "").strip()
    if not summary:
        return ""
    extractor = _TextExtractor()
    try:
        extractor.feed(summary)
        extractor.close()
    except (AssertionError, ValueError):
        LOGGER.warning("Could not parse HTML in an RSS summary; using plain text.")
        return " ".join(summary.split())
    return " ".join(extractor.parts)


@dataclass(frozen=True)
class CollectionSummary:
    """Outcome of one collection run."""

    collected_by_symbol: dict[str, int]
    failed_symbols: tuple[str, ...]
    output_path: Path

    @property
    def total_articles(self) -> int:
        return sum(self.collected_by_symbol.values())


def build_feed_url(company_name: str, limit: int) -> str:
    """Build a bounded Google News RSS search URL for one company."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query = f'"{company_name}" financial news'
    parameters = {**RSS_LOCALE_PARAMETERS, "q": query, "num": str(limit)}
    return f"{GOOGLE_NEWS_RSS_URL}?{urlencode(parameters)}"


def _publication_timestamp(entry: Any) -> str | None:
    """Return a real feed publication timestamp normalized to UTC ISO-8601."""
    published_struct = getattr(entry, "published_parsed", None)
    if published_struct:
        try:
            timestamp = datetime.fromtimestamp(
                calendar.timegm(published_struct), tz=timezone.utc
            )
        except (OverflowError, OSError, TypeError, ValueError):
            return None
    else:
        published_text = getattr(entry, "published", None)
        if not isinstance(published_text, str) or not published_text.strip():
            return None
        try:
            timestamp = datetime.fromisoformat(
                published_text.strip().replace("Z", "+00:00")
            )
        except ValueError:
            try:
                timestamp = parsedate_to_datetime(published_text.strip())
            except (TypeError, ValueError, OverflowError):
                return None
        if timestamp.tzinfo is None:
            return None
        timestamp = timestamp.astimezone(timezone.utc)

    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalize_url(url: str) -> str:
    """Normalize URL fragments and trailing slashes for duplicate checks."""
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, "")
    )


def _entry_to_record(
    entry: Any, symbol: str, extracted: ExtractedArticle | None = None,
    *, extraction_attempted: bool = False,
) -> dict[str, str] | None:
    """Validate an RSS entry and convert it to the required CSV shape."""
    headline = str(getattr(entry, "title", "") or "").strip()
    link = str(getattr(entry, "link", "") or "").strip()
    published_at = _publication_timestamp(entry)
    if not headline:
        LOGGER.warning("Skipping a %s feed item with no headline.", symbol)
        return None
    if not published_at:
        LOGGER.warning("Skipping %s headline with a missing or invalid publication date: %s", symbol, headline)
        return None
    if not link:
        LOGGER.warning("Skipping %s headline with no article URL: %s", symbol, headline)
        return None

    source_info = getattr(entry, "source", None)
    source = str(getattr(source_info, "title", "") or "").strip()
    summary = _rss_summary_text(entry)
    if not extraction_attempted:
        extracted = _cached_article(link)
    if extracted is not None:
        content, content_source, content_quality = extracted.text, "full_article", "high"
        article_url = extracted.url
        article_title, author, article_date = extracted.title or "", extracted.author or "", extracted.published_at or ""
    elif summary:
        content, content_source, content_quality = summary, "rss_summary", "medium"
        article_url = link
        article_title, author, article_date = "", "", ""
    else:
        content, content_source, content_quality = headline, "headline_only", "low"
        article_url = link
        article_title, author, article_date = "", "", ""
    return {
        "Date": published_at,
        "Symbol": symbol,
        "Headline": headline,
        "Text": content,
        "Source": source,
        "URL": article_url,
        "RSSURL": link,
        "RSSSummary": summary,
        "ArticleTitle": article_title,
        "Author": author,
        "ArticlePublishedAt": article_date,
        "ContentSource": content_source,
        "ContentQuality": content_quality,
    }


def _fetch_feed(url: str, timeout: float) -> Any:
    """Download and parse one RSS response using feedparser."""
    try:
        import feedparser
    except ImportError as exc:
        raise RuntimeError(
            "The news fetcher requires feedparser. Install it with: "
            "python -m pip install feedparser"
        ) from exc

    request = Request(
        url,
        headers={"User-Agent": "StockRecommendationNewsFetcher/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download Google News RSS feed: {exc}") from exc
    return feedparser.parse(payload)


def collect_news(
    *,
    limit_per_company: int = 20,
    delay_seconds: float = 1.0,
    timeout_seconds: float = 20.0,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    companies: dict[str, str] | None = None,
) -> CollectionSummary:
    """Fetch separate RSS feeds for each stock, deduplicate, and save results.

    A failed or unavailable company feed is logged and does not stop the other
    feeds. A delay is applied between requests. ``companies`` may be supplied
    by callers to collect a different ticker/company subset.
    """
    selected_companies = companies if companies is not None else COMPANIES
    if not selected_companies:
        raise ValueError("At least one company must be configured")
    if limit_per_company < 1:
        raise ValueError("limit_per_company must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    all_records: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    failed_symbols: list[str] = []
    seen: set[tuple[str, str]] = set()
    existing_articles: dict[str, ExtractedArticle] = {}
    if DEFAULT_OUTPUT_PATH.is_file():
        try:
            with DEFAULT_OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as existing_file:
                for cached_row in csv.DictReader(existing_file):
                    rss_url = (cached_row.get("RSSURL") or "").strip()
                    body = (cached_row.get("Text") or "").strip()
                    if rss_url and cached_row.get("ContentSource") == "full_article" and len(body) >= 200:
                        existing_articles[rss_url] = ExtractedArticle(
                            url=(cached_row.get("URL") or rss_url).strip(),
                            title=(cached_row.get("ArticleTitle") or "").strip() or None,
                            author=(cached_row.get("Author") or "").strip() or None,
                            published_at=(cached_row.get("ArticlePublishedAt") or "").strip() or None,
                            text=body,
                        )
        except (OSError, csv.Error, UnicodeError) as exc:
            LOGGER.warning("Could not reuse previously extracted article text: %s", exc)

    for index, (symbol, company_name) in enumerate(selected_companies.items()):
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol or not company_name.strip():
            raise ValueError("Company symbols and names must be non-empty strings")
        try:
            feed = _fetch_feed(
                build_feed_url(company_name, limit_per_company),
                timeout_seconds,
            )
        except RuntimeError as exc:
            LOGGER.error("RSS collection failed for %s (%s): %s", normalized_symbol, company_name, exc)
            failed_symbols.append(normalized_symbol)
            counts[normalized_symbol] = 0
        else:
            status = getattr(feed, "status", 200)
            if status and int(status) >= 400:
                LOGGER.error(
                    "Google News returned HTTP %s for %s (%s).",
                    status,
                    normalized_symbol,
                    company_name,
                )
                failed_symbols.append(normalized_symbol)
                counts[normalized_symbol] = 0
            else:
                if getattr(feed, "bozo", False):
                    LOGGER.warning(
                        "Google News RSS for %s was malformed: %s",
                        normalized_symbol,
                        getattr(feed, "bozo_exception", "unknown parse error"),
                    )
                entries = list(getattr(feed, "entries", []))
                candidate_links = {
                    str(getattr(entry, "link", "") or "").strip()
                    for entry in entries
                    if str(getattr(entry, "title", "") or "").strip()
                    and _publication_timestamp(entry)
                    and str(getattr(entry, "link", "") or "").strip()
                }
                with ThreadPoolExecutor(max_workers=12) as executor:
                    extraction_futures = {
                        link: executor.submit(_cached_article, link)
                        for link in candidate_links
                        if link not in existing_articles
                    }
                    extracted_by_url = {
                        link: future.result()
                        for link, future in extraction_futures.items()
                    }
                extracted_by_url.update(
                    {link: existing_articles[link] for link in candidate_links if link in existing_articles}
                )
                company_records: list[dict[str, str]] = []
                seen_for_company: set[str] = set()
                for entry in entries:
                    link = str(getattr(entry, "link", "") or "").strip()
                    record = _entry_to_record(
                        entry, normalized_symbol, extracted_by_url.get(link),
                        extraction_attempted=link in extracted_by_url,
                    )
                    if record is None:
                        continue
                    dedupe_key = (
                        record["Symbol"],
                        _normalize_url(record["URL"]),
                    )
                    if dedupe_key in seen or dedupe_key in seen_for_company:
                        continue
                    seen_for_company.add(dedupe_key)
                    company_records.append(record)
                    if len(company_records) >= limit_per_company:
                        break
                seen.update(seen_for_company)
                all_records.extend(company_records)
                counts[normalized_symbol] = len(company_records)

        print(f"{normalized_symbol} ({company_name}): {counts[normalized_symbol]} articles")
        if index < len(selected_companies) - 1 and delay_seconds:
            time.sleep(delay_seconds)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_records)
    except OSError as exc:
        raise RuntimeError(f"Could not save news CSV to {output_path}: {exc}") from exc

    return CollectionSummary(
        collected_by_symbol=counts,
        failed_symbols=tuple(failed_symbols),
        output_path=output_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-per-company",
        type=int,
        default=20,
        help="Maximum number of unique articles to save per company (default: 20)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between company feed requests (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-feed network timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        summary = collect_news(
            limit_per_company=args.limit_per_company,
            delay_seconds=args.delay,
            timeout_seconds=args.timeout,
            output_path=args.output,
        )
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(f"Total unique company/article records: {summary.total_articles}")
    print(f"Saved CSV: {summary.output_path}")
    if summary.failed_symbols:
        LOGGER.error("Feeds failed for: %s", ", ".join(summary.failed_symbols))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
