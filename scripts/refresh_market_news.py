"""Fetch Google News articles, score them with FinBERT, and publish both CSVs."""

from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_NEWS_PATH = RAW_DIR / "financial_news.csv"
SENTIMENT_PATH = PROCESSED_DIR / "news_sentiment.csv"
FETCH_SCRIPT = PROJECT_ROOT / "Src" / "data collection" / "news_fetcher.py"
SENTIMENT_SCRIPT = PROJECT_ROOT / "Src" / "Sentiment" / "news_sentiment.py"
LOG_PATH = PROJECT_ROOT / "reports" / "market_news_pipeline.log"

RAW_COLUMNS = {
    "Date", "Symbol", "Headline", "Text", "Source", "URL", "RSSURL", "RSSSummary",
    "ArticleTitle", "Author", "ArticlePublishedAt", "ContentSource", "ContentQuality",
}
SENTIMENT_COLUMNS = {
    "article_id",
    "published_at",
    "symbol",
    "headline",
    "sentiment",
    "positive_probability",
    "negative_probability",
    "neutral_probability",
    "sentiment_score",
}
PROBABILITY_COLUMNS = (
    "positive_probability",
    "negative_probability",
    "neutral_probability",
)
LOGGER = logging.getLogger("market_news_pipeline")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def _run_stage(command: list[str], stage_name: str) -> None:
    LOGGER.info("Starting %s: %s", stage_name, subprocess.list2cmdline(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout.strip():
        LOGGER.info("%s output:\n%s", stage_name, result.stdout.strip())
    if result.stderr.strip():
        log = LOGGER.error if result.returncode else LOGGER.warning
        log("%s stderr:\n%s", stage_name, result.stderr.strip())
    if result.returncode:
        raise RuntimeError(
            f"{stage_name} failed with exit code {result.returncode}. "
            f"stderr: {result.stderr.strip() or '(empty)'}. "
            f"stdout: {result.stdout.strip() or '(empty)'}"
        )
    LOGGER.info("%s completed successfully", stage_name)


def _read_csv(path: Path, required_columns: set[str], description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"{description} output was not created: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = [str(column).strip() for column in (reader.fieldnames or [])]
            if not columns:
                raise ValueError(f"{description} CSV has no header: {path}")
            if len(columns) != len(set(columns)):
                raise ValueError(f"{description} CSV has duplicate column names")
            missing = required_columns.difference(columns)
            if missing:
                raise ValueError(
                    f"{description} CSV is missing required columns: {sorted(missing)}"
                )
            if columns != reader.fieldnames:
                raise ValueError(
                    f"{description} CSV column names contain unexpected whitespace"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read {description} CSV {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{description} CSV contains no records")
    return rows


def _valid_timestamp(value: str, description: str) -> None:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} has an invalid timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{description} timestamp has no timezone: {value!r}")


def _validate_raw_news(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, str]]]:
    rows = _read_csv(path, RAW_COLUMNS, "Fetched news")
    associations: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        symbol = (row.get("Symbol") or "").strip().upper()
        headline = (row.get("Headline") or "").strip()
        url = (row.get("URL") or "").strip()
        if not symbol or not headline:
            raise ValueError(f"Fetched news row {row_number} has an empty symbol or headline")
        _valid_timestamp(row.get("Date") or "", f"Fetched news row {row_number}")
        if urlsplit(url).scheme.casefold() not in {"http", "https"} or not urlsplit(url).netloc:
            raise ValueError(f"Fetched news row {row_number} has an invalid URL: {url!r}")
        association = (symbol, url)
        if association in associations:
            raise ValueError(
                f"Fetched news contains a duplicate symbol/article URL at row {row_number}"
            )
        associations.add(association)
    LOGGER.info("Validated %d fetched article-symbol records", len(rows))
    return rows, associations


def _validate_sentiment(
    path: Path, raw_associations: set[tuple[str, str]]
) -> int:
    rows = _read_csv(path, SENTIMENT_COLUMNS, "FinBERT sentiment")
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        symbol = (row.get("symbol") or "").strip().upper()
        article_id = (row.get("article_id") or "").strip()
        headline = (row.get("headline") or "").strip()
        sentiment = (row.get("sentiment") or "").strip().title()
        if not symbol or not headline or sentiment not in {"Positive", "Negative", "Neutral"}:
            raise ValueError(f"FinBERT row {row_number} has missing or invalid article fields")
        _valid_timestamp(row.get("published_at") or "", f"FinBERT row {row_number}")
        if (symbol, article_id) not in raw_associations:
            raise ValueError(
                f"FinBERT row {row_number} does not match a fetched symbol/article URL"
            )
        association = (symbol, article_id)
        if association in seen:
            raise ValueError(f"FinBERT CSV contains a duplicate symbol/article at row {row_number}")
        seen.add(association)

        probabilities: dict[str, float] = {}
        for column in PROBABILITY_COLUMNS:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"FinBERT row {row_number} has an invalid {column}"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"FinBERT row {row_number} has an out-of-range {column}")
            probabilities[column] = value
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise ValueError(f"FinBERT row {row_number} probabilities do not sum to one")
        try:
            score = float(row.get("sentiment_score", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FinBERT row {row_number} has an invalid sentiment score") from exc
        expected_score = (
            probabilities["positive_probability"] - probabilities["negative_probability"]
        )
        if not math.isfinite(score) or abs(score - expected_score) > 1e-4:
            raise ValueError(f"FinBERT row {row_number} has an inconsistent sentiment score")

    LOGGER.info("Validated %d FinBERT article-symbol records", len(rows))
    return len(rows)


def _publish_pair(staged_raw: Path, staged_sentiment: Path) -> None:
    targets = (RAW_NEWS_PATH, SENTIMENT_PATH)
    staged = (staged_raw, staged_sentiment)
    backup_roots = (
        staged_raw.parent,
        staged_sentiment.parent,
    )
    backups: dict[Path, Path | None] = {}
    for target, backup_root in zip(targets, backup_roots, strict=True):
        if target.exists():
            backup = backup_root / f"{target.name}.previous"
            shutil.copy2(target, backup)
            backups[target] = backup
        else:
            backups[target] = None

    replaced: list[Path] = []
    try:
        for source, target in zip(staged, targets, strict=True):
            os.replace(source, target)
            replaced.append(target)
            LOGGER.info("Published %s", target)
    except OSError as publish_error:
        rollback_errors: list[str] = []
        for target in reversed(replaced):
            backup = backups[target]
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    restore_path = backup.with_name(f"{backup.name}.restore")
                    shutil.copy2(backup, restore_path)
                    os.replace(restore_path, target)
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        detail = f"Could not publish both CSV files: {publish_error}"
        if rollback_errors:
            detail += "; rollback errors: " + "; ".join(rollback_errors)
        raise RuntimeError(detail) from publish_error


def run_pipeline() -> tuple[int, int]:
    """Run both existing stages, validate their outputs, and publish as a pair."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-news-stage-", dir=RAW_DIR) as raw_temp:
        with tempfile.TemporaryDirectory(
            prefix=".market-news-stage-", dir=PROCESSED_DIR
        ) as processed_temp:
            staged_raw = Path(raw_temp) / RAW_NEWS_PATH.name
            staged_sentiment = Path(processed_temp) / SENTIMENT_PATH.name
            _run_stage(
                [sys.executable, str(FETCH_SCRIPT), "--output", str(staged_raw)],
                "Google News RSS fetch",
            )
            _, raw_associations = _validate_raw_news(staged_raw)
            _run_stage(
                [
                    sys.executable,
                    str(SENTIMENT_SCRIPT),
                    str(staged_raw),
                    "--output",
                    str(staged_sentiment),
                ],
                "FinBERT sentiment analysis",
            )
            scored_count = _validate_sentiment(staged_sentiment, raw_associations)
            _publish_pair(staged_raw, staged_sentiment)
            return len(raw_associations), scored_count


def main() -> int:
    _configure_logging()
    started_at = _utc_now()
    LOGGER.info("News pipeline started at %s", started_at)
    status = "failure"
    try:
        raw_count, scored_count = run_pipeline()
    except Exception:
        LOGGER.exception("News pipeline failed; published CSV preservation was attempted")
        finished_at = _utc_now()
        LOGGER.error("News pipeline finished at %s with status=%s", finished_at, status)
        return 1
    status = "success"
    LOGGER.info(
        "News pipeline succeeded: %d raw and %d scored article-symbol records",
        raw_count,
        scored_count,
    )
    LOGGER.info("News pipeline finished at %s with status=%s", _utc_now(), status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
