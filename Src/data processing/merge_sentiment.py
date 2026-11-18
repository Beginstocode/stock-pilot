"""Merge point-in-time financial-news sentiment into stock feature data.

Input sentiment timestamps must contain a timezone (the FinBERT output uses
UTC ISO-8601). For each stock trading date, news is assigned to the interval
after the previous trading date's cutoff and through the current date's cutoff.
The default cutoff is 09:15 Asia/Kolkata, before the regular market session;
configure it to match the actual time your model makes predictions.

Days without eligible articles retain NaN sentiment/probabilities and zero
article counts. They are not represented as neutral sentiment.

Run from the project root:
    python merge_sentiment.py
    python merge_sentiment.py --cutoff-time 15:30 --overwrite-output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "stock_features.csv"
DEFAULT_SENTIMENT_PATH = PROJECT_ROOT / "data" / "processed" / "news_sentiment.csv"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_stock_features.csv"
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
SENTIMENT_COLUMNS = (
    "Average_Sentiment_Score",
    "Average_Positive_Probability",
    "Average_Negative_Probability",
    "Average_Neutral_Probability",
    "Positive_News_Count",
    "Negative_News_Count",
    "Neutral_News_Count",
    "Total_News_Count",
    "Sentiment_Score_5D",
    "News_Publication_Timestamps_UTC",
)
PROBABILITY_COLUMNS = (
    "positive_probability",
    "negative_probability",
    "neutral_probability",
)
LOGGER = logging.getLogger(__name__)


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{description} CSV was not found: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read {description} CSV {path}: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(f"{description} CSV contains duplicate column names")
    if frame.empty:
        raise ValueError(f"{description} CSV contains no rows: {path}")
    return frame


def _standardize_stock_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"Date", "Symbol"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Stock features are missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["Symbol"] = result["Symbol"].astype("string").str.strip().str.upper()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    invalid = result["Date"].isna() | result["Symbol"].isna() | result["Symbol"].eq("")
    if invalid.any():
        raise ValueError(
            f"Stock features contain {int(invalid.sum())} invalid Date/Symbol key row(s)"
        )
    if result.duplicated(["Symbol", "Date"]).any():
        raise ValueError("Stock features contain duplicate Symbol-Date rows")
    collisions = sorted(set(SENTIMENT_COLUMNS).intersection(result.columns))
    if collisions:
        raise ValueError(
            "Stock features already contain generated sentiment columns; refusing "
            f"to replace them: {collisions}"
        )
    return result


def _merge_target_columns(
    features: pd.DataFrame,
    targets_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve existing labels and attach training labels when available."""
    existing_target_columns = [
        column for column in ("Future_Return_5D", "Target") if column in features.columns
    ]
    if targets_path is None:
        return features, {
            "target_columns_preserved": existing_target_columns,
            "target_rows_missing": None,
        }

    targets_frame = _read_csv(targets_path, "Training targets")
    missing_keys = {"Date", "Symbol"}.difference(targets_frame.columns)
    if missing_keys:
        raise ValueError(
            f"Training targets are missing required columns: {sorted(missing_keys)}"
        )
    target_columns = [
        column for column in ("Future_Return_5D", "Target")
        if column in targets_frame.columns and column not in features.columns
    ]
    if not target_columns:
        return features, {
            "target_columns_preserved": existing_target_columns,
            "target_rows_missing": None,
        }

    targets = targets_frame.loc[:, ["Date", "Symbol", *target_columns]].copy()
    targets["Symbol"] = targets["Symbol"].astype("string").str.strip().str.upper()
    targets["Date"] = pd.to_datetime(targets["Date"], errors="coerce").dt.normalize()
    invalid = targets["Date"].isna() | targets["Symbol"].isna() | targets["Symbol"].eq("")
    if invalid.any():
        raise ValueError(
            f"Training targets contain {int(invalid.sum())} invalid Date/Symbol key row(s)"
        )
    if targets.duplicated(["Symbol", "Date"]).any():
        raise ValueError("Training targets contain duplicate Symbol-Date rows")

    merged = features.merge(
        targets,
        on=["Symbol", "Date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    missing_target_rows = int(merged[target_columns].isna().all(axis=1).sum())
    return merged, {
        "target_columns_preserved": existing_target_columns + target_columns,
        "target_rows_missing": missing_target_rows,
    }


def _parse_publication_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse only explicitly timezone-aware publication timestamps."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        timestamp = pd.Timestamp(value.strip())
    except (TypeError, ValueError, OverflowError):
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC")


def _prepare_news(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    aliases = {
        column.casefold(): column
        for column in frame.columns
    }
    required = {
        "symbol",
        "published_at",
        "sentiment",
        "sentiment_score",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(aliases))
    if missing:
        raise ValueError(f"News sentiment CSV is missing required columns: {missing}")

    news = pd.DataFrame(
        {
            "Symbol": frame[aliases["symbol"]].astype("string").str.strip().str.upper(),
            "Published_At": frame[aliases["published_at"]].map(
                _parse_publication_timestamp
            ),
            "Sentiment": frame[aliases["sentiment"]]
            .astype("string")
            .str.strip()
            .str.casefold(),
            "Sentiment_Score": pd.to_numeric(
                frame[aliases["sentiment_score"]], errors="coerce"
            ),
            "Positive_Probability": pd.to_numeric(
                frame[aliases["positive_probability"]], errors="coerce"
            ),
            "Negative_Probability": pd.to_numeric(
                frame[aliases["negative_probability"]], errors="coerce"
            ),
            "Neutral_Probability": pd.to_numeric(
                frame[aliases["neutral_probability"]], errors="coerce"
            ),
        }
    )
    invalid_key = news["Symbol"].isna() | news["Symbol"].eq("")
    invalid_date = news["Published_At"].isna()
    valid_sentiments = {"positive", "negative", "neutral"}
    invalid_sentiment = ~news["Sentiment"].isin(valid_sentiments)
    invalid_numbers = (
        ~np.isfinite(news["Sentiment_Score"])
        | ~np.isfinite(news["Positive_Probability"])
        | ~np.isfinite(news["Negative_Probability"])
        | ~np.isfinite(news["Neutral_Probability"])
    )
    invalid_probability = (
        news[list(
            ("Positive_Probability", "Negative_Probability", "Neutral_Probability")
        )]
        .lt(0)
        .any(axis=1)
        | news[list(
            ("Positive_Probability", "Negative_Probability", "Neutral_Probability")
        )]
        .gt(1)
        .any(axis=1)
    )
    invalid_score_range = ~news["Sentiment_Score"].between(-1, 1)
    invalid = (
        invalid_key
        | invalid_date
        | invalid_sentiment
        | invalid_numbers
        | invalid_probability
        | invalid_score_range
    )
    stats = {
        "invalid_symbol_rows": int(invalid_key.sum()),
        "invalid_or_timezone_naive_timestamps": int(invalid_date.sum()),
        "invalid_sentiment_or_score_rows": int(
            (invalid_sentiment | invalid_numbers | invalid_probability | invalid_score_range).sum()
        ),
    }
    news = news.loc[~invalid].copy()

    # Prefer the source URL/ID when present. Otherwise use content/time to avoid
    # counting a repeated sentiment export as another article.
    id_column = aliases.get("article_id") or aliases.get("url")
    if id_column:
        article_id = frame.loc[news.index, id_column].astype("string").str.strip()
    else:
        article_id = pd.Series(pd.NA, index=news.index, dtype="string")
    news["Article_ID"] = article_id
    fallback_id = (
        news["Symbol"].astype(str)
        + "|"
        + news["Published_At"].astype(str)
        + "|"
        + news["Sentiment_Score"].astype(str)
        + "|"
        + news["Positive_Probability"].astype(str)
        + "|"
        + news["Negative_Probability"].astype(str)
        + "|"
        + news["Neutral_Probability"].astype(str)
    )
    news["Dedupe_ID"] = news["Article_ID"].where(
        news["Article_ID"].notna() & news["Article_ID"].ne(""),
        fallback_id,
    )
    before_deduplication = len(news)
    news = news.drop_duplicates(["Symbol", "Dedupe_ID"], keep="first").copy()
    stats["duplicate_rows_removed"] = before_deduplication - len(news)
    return news, stats


def _cutoff_utc(trading_date: pd.Timestamp, cutoff_time: time) -> pd.Timestamp:
    local_cutoff = datetime.combine(trading_date.date(), cutoff_time, tzinfo=MARKET_TIMEZONE)
    return pd.Timestamp(local_cutoff).tz_convert("UTC")


def _exclude_pre_history_news(
    news: pd.DataFrame,
    feature_dates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop articles before each symbol's first feature-history date."""
    starts = feature_dates.groupby("Symbol")["Date"].min()
    start_utc = {
        symbol: pd.Timestamp(
            datetime.combine(pd.Timestamp(first_date).date(), time.min, tzinfo=MARKET_TIMEZONE)
        ).tz_convert("UTC")
        for symbol, first_date in starts.items()
    }
    known_symbol = news["Symbol"].isin(start_utc)
    before_history = pd.Series(
        [
            symbol in start_utc and published_at < start_utc[symbol]
            for symbol, published_at in zip(
                news["Symbol"],
                news["Published_At"],
                strict=True,
            )
        ],
        index=news.index,
        dtype=bool,
    )
    excluded = {
        "pre_history_articles_excluded": int(before_history.sum()),
        "news_symbols_without_feature_history_excluded": int((~known_symbol).sum()),
    }
    eligible = news.loc[known_symbol & ~before_history].copy()
    return eligible, excluded


def _aggregate_for_stock_dates(
    feature_keys: pd.DataFrame,
    news: pd.DataFrame,
    cutoff_time: time,
) -> pd.DataFrame:
    """Assign each article once to the first prediction cutoff after publication."""
    output_rows: list[dict[str, Any]] = []
    news_by_symbol = {
        symbol: group.sort_values("Published_At")
        for symbol, group in news.groupby("Symbol", sort=False)
    }
    for symbol, keys in feature_keys.groupby("Symbol", sort=False):
        dates = sorted(keys["Date"].unique())
        stock_news = news_by_symbol.get(symbol)
        previous_cutoff: pd.Timestamp | None = None
        for raw_date in dates:
            trading_date = pd.Timestamp(raw_date)
            cutoff = _cutoff_utc(trading_date, cutoff_time)
            if stock_news is None:
                eligible = news.iloc[0:0]
            else:
                current_window = stock_news["Published_At"].le(cutoff)
                if previous_cutoff is not None:
                    current_window &= stock_news["Published_At"].gt(previous_cutoff)
                eligible = stock_news.loc[current_window]
            if eligible.empty:
                counts = {"positive": 0, "negative": 0, "neutral": 0}
                publication_timestamps = ""
                averages = {
                    "Average_Sentiment_Score": np.nan,
                    "Average_Positive_Probability": np.nan,
                    "Average_Negative_Probability": np.nan,
                    "Average_Neutral_Probability": np.nan,
                }
            else:
                counts = eligible["Sentiment"].value_counts().to_dict()
                publication_timestamps = json.dumps(
                    [
                        timestamp.isoformat().replace("+00:00", "Z")
                        for timestamp in eligible["Published_At"]
                    ],
                    separators=(",", ":"),
                )
                averages = {
                    "Average_Sentiment_Score": float(eligible["Sentiment_Score"].mean()),
                    "Average_Positive_Probability": float(
                        eligible["Positive_Probability"].mean()
                    ),
                    "Average_Negative_Probability": float(
                        eligible["Negative_Probability"].mean()
                    ),
                    "Average_Neutral_Probability": float(
                        eligible["Neutral_Probability"].mean()
                    ),
                }
            output_rows.append(
                {
                    "Symbol": symbol,
                    "Date": trading_date,
                    **averages,
                    "Positive_News_Count": int(counts.get("positive", 0)),
                    "Negative_News_Count": int(counts.get("negative", 0)),
                    "Neutral_News_Count": int(counts.get("neutral", 0)),
                    "Total_News_Count": int(len(eligible)),
                    "News_Publication_Timestamps_UTC": publication_timestamps,
                }
            )
            previous_cutoff = cutoff

    aggregate = pd.DataFrame(output_rows)
    aggregate.sort_values(["Symbol", "Date"], kind="mergesort", inplace=True)
    aggregate["Sentiment_Score_5D"] = aggregate.groupby(
        "Symbol", sort=False
    )["Average_Sentiment_Score"].transform(
        lambda values: values.rolling(window=5, min_periods=1).mean()
    )
    return aggregate


def merge_sentiment(
    features_path: Path = DEFAULT_FEATURES_PATH,
    sentiment_path: Path = DEFAULT_SENTIMENT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    targets_path: Path | None = DEFAULT_TARGETS_PATH,
    cutoff_time: time = time(9, 15),
    overwrite_output: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge point-in-time sentiment into features and persist the result.

    ``cutoff_time`` is interpreted in Asia/Kolkata. News between a prior
    trading date's cutoff and the current date's cutoff is assigned to the
    current date. This includes after-hours and non-trading-day publications
    at the next cutoff, but never includes news from after that cutoff.
    Existing inputs are never modified. An existing output is protected unless
    ``overwrite_output=True`` is explicitly set.
    """
    if not isinstance(cutoff_time, time):
        raise TypeError("cutoff_time must be a datetime.time value")
    if cutoff_time.tzinfo is not None:
        raise ValueError("cutoff_time must be a local Asia/Kolkata wall-clock time")
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_path}. "
            "Use --overwrite-output only if replacing this generated output is intended."
        )
    protected_paths = {features_path.resolve(), sentiment_path.resolve()}
    if targets_path is not None:
        protected_paths.add(targets_path.resolve())
    if output_path.resolve() in protected_paths:
        raise ValueError("Output path must not overwrite an input CSV")

    features = _standardize_stock_features(_read_csv(features_path, "Stock features"))
    features, target_stats = _merge_target_columns(features, targets_path)
    sentiment_frame = _read_csv(sentiment_path, "News sentiment")
    news, stats = _prepare_news(sentiment_frame)
    feature_keys = features[["Symbol", "Date"]].drop_duplicates()
    news, history_stats = _exclude_pre_history_news(news, feature_keys)

    aggregates = _aggregate_for_stock_dates(
        feature_keys,
        news,
        cutoff_time,
    )
    # Verify each stored source timestamp belongs to the stock-date session
    # and is no later than its configured prediction cutoff.
    feature_starts = feature_keys.groupby("Symbol")["Date"].min().to_dict()
    timestamp_violations = 0
    for row in aggregates.itertuples(index=False):
        timestamps = json.loads(row.News_Publication_Timestamps_UTC or "[]")
        session_cutoff = _cutoff_utc(pd.Timestamp(row.Date), cutoff_time)
        session_start = pd.Timestamp(
            datetime.combine(
                pd.Timestamp(feature_starts[row.Symbol]).date(),
                time.min,
                tzinfo=MARKET_TIMEZONE,
            )
        ).tz_convert("UTC")
        for timestamp_text in timestamps:
            published_at = pd.Timestamp(timestamp_text)
            if (
                published_at.tzinfo is None
                or published_at > session_cutoff
                or published_at < session_start
            ):
                timestamp_violations += 1
    if timestamp_violations:
        raise AssertionError(
            f"{timestamp_violations} news timestamp(s) violate pre-history or cutoff rules"
        )

    result = features.merge(
        aggregates,
        on=["Symbol", "Date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result["Total_News_Count"] = result["Total_News_Count"].fillna(0).astype("int64")
    for column in (
        "Positive_News_Count",
        "Negative_News_Count",
        "Neutral_News_Count",
    ):
        result[column] = result[column].fillna(0).astype("int64")

    matched_news_count = int(result["Total_News_Count"].sum())
    rows_with_news = int(result["Total_News_Count"].gt(0).sum())
    statistics: dict[str, Any] = {
        "stock_feature_rows": len(features),
        "output_rows": len(result),
        "news_input_rows": len(sentiment_frame),
        "valid_deduplicated_news_rows": len(news),
        "news_assigned_to_feature_dates": matched_news_count,
        "feature_rows_with_news": rows_with_news,
        "feature_rows_without_news": len(result) - rows_with_news,
        "cutoff_time_asia_kolkata": cutoff_time.isoformat(),
        "news_timestamps_after_cutoff": timestamp_violations,
        **target_stats,
        **history_stats,
        **stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".tmp",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            result.to_csv(temporary_file, index=False)
        if output_path.exists() and not overwrite_output:
            raise FileExistsError(
                f"Output appeared during processing and will not be overwritten: {output_path}"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Could not write merged features to {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not remove temporary output file %s", temporary_path)

    statistics["output_missing_values"] = {
        str(column): int(count)
        for column, count in result.isna().sum().items()
        if count
    }
    return result, statistics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--sentiment", type=Path, default=DEFAULT_SENTIMENT_PATH)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_PATH)
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="Do not attach target columns from training_dataset.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--cutoff-time",
        default="09:15",
        help="Daily prediction cutoff in Asia/Kolkata (HH:MM; default: 09:15)",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing merged output; inputs are never overwritten",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        parsed_cutoff = time.fromisoformat(args.cutoff_time)
        if parsed_cutoff.second or parsed_cutoff.microsecond or parsed_cutoff.tzinfo:
            raise ValueError("cutoff time must be HH:MM without seconds or timezone")
        result, statistics = merge_sentiment(
            features_path=args.features,
            sentiment_path=args.sentiment,
            output_path=args.output,
            targets_path=None if args.no_targets else args.targets,
            cutoff_time=parsed_cutoff,
            overwrite_output=args.overwrite_output,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(f"Output: {args.output}")
    print(f"Rows: {statistics['stock_feature_rows']:,} -> {len(result):,}")
    print(f"Total stock-date rows: {len(result):,}")
    print(f"Rows with sentiment: {statistics['feature_rows_with_news']:,}")
    print(f"Rows without sentiment: {statistics['feature_rows_without_news']:,}")
    print(
        "Sentiment coverage: "
        f"{statistics['feature_rows_with_news'] / max(1, len(result)) * 100:.2f}%"
    )
    print(
        "Articles excluded due to pre-history dates: "
        f"{statistics['pre_history_articles_excluded']:,}"
    )
    print(
        "Articles excluded due to invalid timestamps: "
        f"{statistics['invalid_or_timezone_naive_timestamps']:,}"
    )
    print(
        "Eligible news assigned: "
        f"{statistics['news_assigned_to_feature_dates']:,} across "
        f"{statistics['feature_rows_with_news']:,} feature rows"
    )
    print(f"Rows without eligible news: {statistics['feature_rows_without_news']:,}")
    print("Missing values by column:")
    missing = statistics["output_missing_values"]
    print(pd.Series(missing, dtype="int64").to_string() if missing else "None")
    print(
        "Input cleanup: "
        f"{statistics['invalid_symbol_rows']} invalid symbols, "
        f"{statistics['invalid_or_timezone_naive_timestamps']} invalid/naive timestamps, "
        f"{statistics['invalid_sentiment_or_score_rows']} invalid sentiment rows, "
        f"{statistics['duplicate_rows_removed']} duplicates removed."
    )
    print(
        "Point-in-time timestamp verification: "
        f"{statistics['news_timestamps_after_cutoff']} violations."
    )
    print(f"Target columns preserved: {statistics['target_columns_preserved']}")
    if statistics["target_rows_missing"] is not None:
        print(f"Feature rows without matching target values: {statistics['target_rows_missing']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
