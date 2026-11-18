"""Clean and audit stock, market, feature, and training CSV datasets.

Run from the project directory:
    python data_cleaning.py

Original inputs are never modified. Missing feature values (including rolling
warm-up values) are preserved; no forward or backward filling is performed.
Statistical outliers are reported but retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


ROOT_DIR: Final[Path] = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR: Final[Path] = ROOT_DIR / "data" / "cleaned"
TARGET_LABELS: Final[set[str]] = {"BUY", "HOLD", "SELL"}
CATEGORICAL_COLUMNS: Final[set[str]] = {"date", "symbol", "target", "index name"}
FUTURE_COLUMN_TOKENS: Final[tuple[str, ...]] = (
    "future",
    "forward",
    "lead",
    "next",
    "target",
    "label",
    "outcome",
    "ahead",
)
OHLC_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close")
REQUIRED_STOCK_KEYS: Final[tuple[str, ...]] = ("Date", "Symbol")


@dataclass
class DatasetAudit:
    """Diagnostics and result table for one input dataset."""

    name: str
    source: Path
    original_rows: int
    original_columns: list[str]
    original_dtypes: dict[str, str]
    original_missing: dict[str, int]
    cleaned: pd.DataFrame
    original_date_range: str | None = None
    duplicate_rows: int = 0
    invalid_key_rows: int = 0
    invalid_ohlc_rows: int = 0
    negative_volume_rows: int = 0
    invalid_numeric_cells: dict[str, int] = field(default_factory=dict)
    infinite_cells: dict[str, int] = field(default_factory=dict)
    outlier_cells: dict[str, int] = field(default_factory=dict)

    @property
    def final_rows(self) -> int:
        return len(self.cleaned)

    @property
    def invalid_records(self) -> int:
        return self.invalid_key_rows + self.invalid_ohlc_rows + self.negative_volume_rows


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO and common Indian market date strings without swapping ISO dates."""
    text = values.astype("string").str.strip()
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", na=False)
    iso_dates = pd.to_datetime(text.where(iso_mask), errors="coerce")
    other_dates = pd.to_datetime(
        text.where(~iso_mask), errors="coerce", dayfirst=True, format="mixed"
    )
    return iso_dates.fillna(other_dates).dt.normalize()


def resolve_input_paths(
    input_dir: Path = ROOT_DIR,
    stock_path: Path | None = None,
    nifty_path: Path | None = None,
    vix_path: Path | None = None,
    features_path: Path | None = None,
    training_path: Path | None = None,
) -> dict[str, Path | None]:
    """Resolve requested files and this project's known stock-file fallback."""
    if stock_path is None:
        preferred = input_dir / "stock_prices_combined.csv"
        fallback = input_dir / "data" / "market" / "all_stocks.csv"
        stock_path = preferred if preferred.exists() else fallback
    return {
        "stock_prices": stock_path,
        "nifty50": nifty_path or input_dir / "data" / "market" / "nifty50.csv",
        "india_vix": vix_path or input_dir / "data" / "market" / "india_vix.csv",
        "stock_features": features_path or input_dir / "stock_features.csv",
        "training_dataset": training_path or input_dir / "training_dataset.csv",
    }


def _read_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name} input file was not found: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read {dataset_name} from {path}: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError(f"{dataset_name} input is empty: {path}")
    if frame.columns.duplicated().any():
        raise ValueError(f"{dataset_name} has duplicate column names")
    return frame


def inspect_frame(frame: pd.DataFrame, name: str) -> str:
    """Return a compact pre-cleaning schema and coverage inspection."""
    lines = [
        f"Dataset: {name}",
        f"Dimensions: {len(frame)} rows x {len(frame.columns)} columns",
        f"Columns: {', '.join(frame.columns)}",
        "Data types: "
        + "; ".join(f"{column}={dtype}" for column, dtype in frame.dtypes.items()),
    ]
    if "Date" in frame.columns:
        dates = parse_dates(frame["Date"])
        valid = dates.dropna()
        date_range = (
            f"{valid.min().date()} to {valid.max().date()}" if not valid.empty else "no valid dates"
        )
        lines.append(f"Date range: {date_range}")
    return "\n".join(lines)


def _coerce_numeric_values(frame: pd.DataFrame, audit: DatasetAudit) -> None:
    """Coerce numerical fields; malformed and infinite cells become NaN."""
    for column in frame.columns:
        if column.casefold() in CATEGORICAL_COLUMNS or column == "Date":
            continue
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_mask = original.notna() & converted.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            audit.invalid_numeric_cells[column] = invalid_count
        if pd.api.types.is_numeric_dtype(original) or converted.notna().any() or invalid_count:
            infinite_mask = converted.isin([np.inf, -np.inf])
            infinite_count = int(infinite_mask.sum())
            if infinite_count:
                audit.infinite_cells[column] = infinite_count
                converted = converted.mask(infinite_mask)
            frame[column] = converted


def _deduplicate(
    frame: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, int]:
    """Drop repeated stock/date or market/date keys, preserving the first row."""
    if "Date" not in frame.columns:
        raise ValueError(f"{dataset_name} must contain a Date column")
    if dataset_name in {"stock_prices", "stock_features", "training_dataset"}:
        if "Symbol" not in frame.columns:
            raise ValueError(f"{dataset_name} must contain a Symbol column")
        key_columns = ["Symbol", "Date"]
    else:
        key_columns = ["Date"]
    duplicate_count = int(frame.duplicated(key_columns, keep="first").sum())
    return frame.drop_duplicates(key_columns, keep="first").copy(), duplicate_count


def _ohlc_invalid_mask(frame: pd.DataFrame) -> pd.Series:
    """Find missing, non-positive, or logically inconsistent OHLC records."""
    if not set(OHLC_COLUMNS).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    open_price, high, low, close = (frame[column] for column in OHLC_COLUMNS)
    missing_or_nonpositive = (
        open_price.isna()
        | high.isna()
        | low.isna()
        | close.isna()
        | open_price.le(0)
        | high.le(0)
        | low.le(0)
        | close.le(0)
    )
    inconsistent = (
        high.lt(open_price)
        | high.lt(low)
        | high.lt(close)
        | low.gt(open_price)
        | low.gt(high)
        | low.gt(close)
    )
    return missing_or_nonpositive | inconsistent


def _outlier_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count robust 3-IQR outliers in numeric columns; do not remove them."""
    counts: dict[str, int] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna()
        if len(values) < 8:
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        spread = third_quartile - first_quartile
        if not np.isfinite(spread) or spread == 0:
            continue
        lower_limit = first_quartile - 3 * spread
        upper_limit = third_quartile + 3 * spread
        count = int((frame[column].lt(lower_limit) | frame[column].gt(upper_limit)).sum())
        if count:
            counts[column] = count
    return counts


def clean_dataset(frame: pd.DataFrame, name: str, source: Path) -> DatasetAudit:
    """Normalize and validate one dataset while preserving usable missing values."""
    original_missing = frame.isna().sum()
    original_date_range: str | None = None
    if "Date" in frame.columns:
        original_dates = parse_dates(frame["Date"]).dropna()
        if not original_dates.empty:
            original_date_range = f"{original_dates.min().date()} to {original_dates.max().date()}"
    audit = DatasetAudit(
        name=name,
        source=source,
        original_rows=len(frame),
        original_columns=list(frame.columns),
        original_dtypes={column: str(dtype) for column, dtype in frame.dtypes.items()},
        original_missing={column: int(count) for column, count in original_missing.items() if count},
        cleaned=frame.copy(),
        original_date_range=original_date_range,
    )
    cleaned = frame.copy()
    if "Date" not in cleaned.columns:
        raise ValueError(f"{name} must contain a Date column")
    cleaned["Date"] = parse_dates(cleaned["Date"])

    key_columns = ["Date"]
    if name in {"stock_prices", "stock_features", "training_dataset"}:
        if "Symbol" not in cleaned.columns:
            raise ValueError(f"{name} must contain a Symbol column")
        cleaned["Symbol"] = cleaned["Symbol"].astype("string").str.strip()
        key_columns.append("Symbol")
    invalid_key_mask = cleaned[key_columns].isna().any(axis=1)
    if "Symbol" in key_columns:
        invalid_key_mask |= cleaned["Symbol"].eq("")
    audit.invalid_key_rows = int(invalid_key_mask.sum())
    cleaned = cleaned.loc[~invalid_key_mask].copy()

    if name == "training_dataset" and "Target" in cleaned.columns:
        cleaned["Target"] = cleaned["Target"].astype("string").str.strip().str.upper()

    _coerce_numeric_values(cleaned, audit)
    cleaned, audit.duplicate_rows = _deduplicate(cleaned, name)

    invalid_ohlc = _ohlc_invalid_mask(cleaned)
    audit.invalid_ohlc_rows = int(invalid_ohlc.sum())
    cleaned = cleaned.loc[~invalid_ohlc].copy()
    if "Volume" in cleaned.columns:
        negative_volume = cleaned["Volume"].lt(0).fillna(False)
        audit.negative_volume_rows = int(negative_volume.sum())
        cleaned = cleaned.loc[~negative_volume].copy()

    cleaned = cleaned.sort_values(
        [column for column in ("Date", "Symbol") if column in cleaned.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    audit.cleaned = cleaned
    audit.outlier_cells = _outlier_counts(cleaned)
    return audit


def is_future_derived_column(column: str) -> bool:
    """Conservatively identify target, label, and future-named columns."""
    normalized = column.strip().casefold()
    return any(token in normalized for token in FUTURE_COLUMN_TOKENS)


def _date_consistency_lines(audits: dict[str, DatasetAudit]) -> list[str]:
    """Report stock-wise gaps and exact-date alignment of market datasets."""
    lines: list[str] = []
    stock_audit = audits.get("stock_prices")
    if stock_audit is not None and not stock_audit.cleaned.empty:
        stocks = stock_audit.cleaned
        all_dates = set(stocks["Date"].dropna().unique())
        lines.append(f"Stock trading-session dates in combined universe: {len(all_dates)}")
        for symbol, group in stocks.groupby("Symbol", sort=True):
            dates = set(group["Date"].dropna().unique())
            lines.append(
                f"  {symbol}: rows={len(group)}, unique_dates={len(dates)}, "
                f"dates absent versus combined universe={len(all_dates - dates)}"
            )

        for market_name in ("nifty50", "india_vix"):
            market_audit = audits.get(market_name)
            if market_audit is None:
                lines.append(f"{market_name}: input unavailable; alignment not checked")
                continue
            market_dates = set(market_audit.cleaned["Date"].dropna().unique())
            overlap = all_dates & market_dates
            lines.append(
                f"{market_name}: dates={len(market_dates)}, overlap with stock dates={len(overlap)}, "
                f"stock dates without market observation={len(all_dates - market_dates)}, "
                f"market dates outside stock calendar={len(market_dates - all_dates)}"
            )
    else:
        lines.append("Stock date consistency could not be calculated; cleaned stock input is unavailable.")
    nifty_audit = audits.get("nifty50")
    vix_audit = audits.get("india_vix")
    if nifty_audit is not None and vix_audit is not None:
        nifty_dates = set(nifty_audit.cleaned["Date"].dropna().unique())
        vix_dates = set(vix_audit.cleaned["Date"].dropna().unique())
        lines.append(
            "NIFTY/VIX direct alignment: "
            f"common_dates={len(nifty_dates & vix_dates)}, "
            f"NIFTY_only_dates={len(nifty_dates - vix_dates)}, "
            f"VIX_only_dates={len(vix_dates - nifty_dates)}"
        )
    else:
        lines.append("NIFTY/VIX direct alignment not checked because one or both inputs are unavailable.")
    return lines


def _feature_and_target_lines(audits: dict[str, DatasetAudit]) -> list[str]:
    """Summarize rolling NaNs, target balance, and safe predictor candidates."""
    lines: list[str] = []
    features_audit = audits.get("stock_features")
    if features_audit is not None:
        features = features_audit.cleaned
        rolling_columns = [
            column
            for column in features.columns
            if any(token in column.casefold() for token in ("sma", "ema", "rsi", "macd", "roc", "volatility", "bb_", "atr", "return_", "relative_volume", "avg_volume"))
        ]
        lines.append("Feature missing counts (rolling warm-up and other missingness):")
        rolling_missing = features[rolling_columns].isna().sum()
        rolling_missing = rolling_missing.loc[rolling_missing.gt(0)]
        lines.extend(
            f"  {column}: {int(count)}" for column, count in rolling_missing.items()
        ) or lines.append("  None")

        candidate_features = [
            column
            for column in features.columns
            if column.casefold() not in {"date", "symbol"} and not is_future_derived_column(column)
        ]
        excluded = [column for column in features.columns if is_future_derived_column(column)]
        lines.append(f"Feature predictor candidates excluding Date/Symbol: {len(candidate_features)}")
        lines.append(f"Future/label-like columns excluded from model inputs: {excluded or 'None'}")

    training_audit = audits.get("training_dataset")
    if training_audit is not None:
        training = training_audit.cleaned
        if "Target" in training.columns:
            targets = training["Target"].astype("string").str.upper()
            counts = targets.value_counts(dropna=False)
            lines.append("Training target distribution:")
            lines.extend(f"  {label}: {int(counts.get(label, 0))}" for label in ("BUY", "HOLD", "SELL"))
            unexpected = sorted(set(targets.dropna().unique()) - TARGET_LABELS)
            lines.append(f"Unexpected target labels: {unexpected or 'None'}")
        if "Future_Return_5D" in training.columns:
            lines.append("Future_Return_5D is present as a label-support column, not a model predictor.")
        training_predictors = [
            column
            for column in training.columns
            if column.casefold() not in {"date", "symbol"} and not is_future_derived_column(column)
        ]
        training_excluded = [column for column in training.columns if is_future_derived_column(column)]
        lines.append(f"Training predictor candidates excluding Date/Symbol: {len(training_predictors)}")
        lines.append(f"Training future/label-like columns excluded from model inputs: {training_excluded or 'None'}")
    return lines


def build_report(audits: dict[str, DatasetAudit]) -> str:
    """Generate a human-readable report with per-dataset cleaning decisions."""
    sections = [
        "STOCK DATA CLEANING REPORT",
        "==========================",
        "Policy: originals are not modified; no forward/backward fill is used;",
        "rolling-window NaNs are preserved; statistical outliers are flagged, not removed.",
        "",
    ]
    for name, audit in audits.items():
        cleaned = audit.cleaned
        sections.extend(
            [
                f"[{name}]",
                f"Source: {audit.source}",
                f"Original row count: {audit.original_rows}",
                f"Final row count: {audit.final_rows}",
                f"Original dimensions: {audit.original_rows} x {len(audit.original_columns)}",
                f"Final dimensions: {cleaned.shape[0]} x {cleaned.shape[1]}",
                f"Original columns: {audit.original_columns}",
                f"Original data types: {audit.original_dtypes}",
                f"Original date range: {audit.original_date_range or 'not available'}",
                f"Duplicate records removed: {audit.duplicate_rows}",
                f"Invalid key/date records removed: {audit.invalid_key_rows}",
                f"OHLC-invalid records removed: {audit.invalid_ohlc_rows}",
                f"Negative-volume records removed: {audit.negative_volume_rows}",
                f"Total invalid records removed: {audit.invalid_records}",
                f"Original missing values by column: {audit.original_missing or 'None'}",
                f"Final missing values by column: {cleaned.isna().sum().loc[lambda values: values.gt(0)].astype(int).to_dict() or 'None'}",
                f"Malformed numeric cells converted to missing by column: {audit.invalid_numeric_cells or 'None'}",
                f"Infinite values converted to missing by column: {audit.infinite_cells or 'None'}",
                f"Suspicious outlier counts by column (retained): {audit.outlier_cells or 'None'}",
            ]
        )
        if "Date" in cleaned.columns and not cleaned.empty:
            sections.append(f"Final date range: {cleaned['Date'].min().date()} to {cleaned['Date'].max().date()}")
        sections.append("")

    sections.extend(["[Date and stock-wise consistency]", *_date_consistency_lines(audits), ""])
    sections.extend(["[Features, targets, and leakage checks]", *_feature_and_target_lines(audits), ""])
    sections.extend(
        [
            "Interpretation",
            "- Removed duplicate-key and structurally invalid price records are counted above.",
            "- Outlier counts are diagnostic only; those observations remain in cleaned outputs.",
            "- Missing feature values are intentionally retained for model-side handling.",
            "- Never pass Date, Symbol, Target, Future_Return_5D, or other future-derived columns as model inputs.",
        ]
    )
    return "\n".join(sections) + "\n"


def run_cleaning(
    paths: dict[str, Path | None],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, DatasetAudit]:
    """Load available inputs, clean each dataset, export separate files and report."""
    output_names = {
        "stock_prices": "cleaned_stock_prices.csv",
        "nifty50": "cleaned_nifty50.csv",
        "india_vix": "cleaned_india_vix.csv",
        "stock_features": "cleaned_stock_features.csv",
        "training_dataset": "cleaned_training_dataset.csv",
    }
    audits: dict[str, DatasetAudit] = {}
    missing_inputs: list[str] = []
    for name, output_name in output_names.items():
        path = paths.get(name)
        if path is None or not path.is_file():
            missing_inputs.append(f"{name}: {path if path is not None else 'not configured'}")
            continue
        original = _read_csv(path, name)
        print(inspect_frame(original, name))
        audits[name] = clean_dataset(original, name, path)

    if not audits:
        raise FileNotFoundError("None of the configured CSV inputs are available")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, audit in audits.items():
        output_path = output_dir / output_names[name]
        try:
            audit.cleaned.to_csv(output_path, index=False)
        except OSError as exc:
            raise RuntimeError(f"Could not save cleaned {name} CSV to {output_path}: {exc}") from exc
        print(f"Saved {output_path} ({len(audit.cleaned):,} rows)")

    report = build_report(audits)
    if missing_inputs:
        report += "\n[Unavailable inputs]\n" + "\n".join(f"- {entry}" for entry in missing_inputs) + "\n"
    report_path = output_dir / "cleaning_report.txt"
    try:
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not save cleaning report to {report_path}: {exc}") from exc
    print(f"Saved {report_path}")
    if missing_inputs:
        print("Unavailable inputs: " + "; ".join(missing_inputs))
    return audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR)
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--nifty", type=Path)
    parser.add_argument("--vix", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = resolve_input_paths(
        input_dir=args.input_dir,
        stock_path=args.stock,
        nifty_path=args.nifty,
        vix_path=args.vix,
        features_path=args.features,
        training_path=args.training,
    )
    try:
        run_cleaning(paths, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(status=1, message=f"Data cleaning failed: {exc}\n")


if __name__ == "__main__":
    main()