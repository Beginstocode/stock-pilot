"""Create leakage-aware stock features and five-session classification labels.

Run from the project root with:
    python feature_engineering.py

By default the script reads stock_prices_combined.csv. If it is not present,
it uses data/market/all_stocks.csv, which is the combined stock file in this
workspace. The output files are written to the project root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


ROOT_DIR: Final[Path] = Path(__file__).resolve().parent
DEFAULT_STOCK_INPUT: Final[Path] = ROOT_DIR / "stock_prices_combined.csv"
FALLBACK_STOCK_INPUT: Final[Path] = ROOT_DIR / "data" / "market" / "all_stocks.csv"
DEFAULT_NIFTY_INPUT: Final[Path] = ROOT_DIR / "data" / "market" / "nifty50.csv"
DEFAULT_VIX_INPUT: Final[Path] = ROOT_DIR / "data" / "market" / "india_vix.csv"
OUTPUT_DIR: Final[Path] = ROOT_DIR

TARGET_COLUMNS: Final[tuple[str, ...]] = ("Future_Return_5D", "Target")
PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)
REQUIRED_STOCK_COLUMNS: Final[set[str]] = {
    "Date",
    "Symbol",
    *PRICE_COLUMNS,
    "Adj Close",
}
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "Daily_Return",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "ROC_10",
    "Volatility_10D",
    "Volatility_20D",
    "Volatility_30D",
    "BB_Upper_20",
    "BB_Middle_20",
    "BB_Lower_20",
    "ATR_14",
    "Volume_Change",
    "Avg_Volume_20D",
    "Relative_Volume",
    "High_Low_Pct",
    "Open_Close_Pct",
    "Daily_Price_Range",
    "NIFTY_Daily_Return",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "India_VIX_Close",
    "India_VIX_Daily_Change",
)
MODEL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    *PRICE_COLUMNS,
    "Adj Close",
    *FEATURE_COLUMNS,
)


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO and common Indian-market date strings to date-only values."""
    text = values.astype("string").str.strip()
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", na=False)
    parsed = pd.to_datetime(text.where(iso_mask), errors="coerce")
    other_dates = pd.to_datetime(
        text.where(~iso_mask), errors="coerce", dayfirst=True, format="mixed"
    )
    parsed = parsed.fillna(other_dates)
    return parsed.dt.normalize()


def _clean_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)


def clean_stock_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, de-duplicate, and chronologically sort raw prices."""
    missing_columns = REQUIRED_STOCK_COLUMNS.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"Stock data is missing required columns: {sorted(missing_columns)}")

    cleaned = prices.copy()
    cleaned["Date"] = _parse_dates(cleaned["Date"])
    cleaned["Symbol"] = cleaned["Symbol"].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["Date", "Symbol"])
    cleaned = cleaned.loc[cleaned["Symbol"].ne("")].copy()
    _clean_numeric_columns(cleaned, list(PRICE_COLUMNS) + ["Adj Close"])
    cleaned = cleaned.drop_duplicates(subset=["Symbol", "Date"], keep="first")
    return cleaned.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.ne(0))
    return numerator.div(denominator)


def _stock_features(group: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators for one symbol, with no cross-symbol state."""
    group = group.sort_values("Date", kind="mergesort").copy()
    close = group["Close"]
    high = group["High"]
    low = group["Low"]
    open_price = group["Open"]
    volume = group["Volume"]
    daily_return = close.pct_change(fill_method=None)

    features = group.copy()
    features["Daily_Return"] = daily_return
    for period in (5, 10, 20):
        features[f"Return_{period}D"] = close.pct_change(periods=period, fill_method=None)

    for period in (5, 10, 20, 50, 100, 200):
        features[f"SMA_{period}"] = close.rolling(window=period, min_periods=period).mean()
    for period in (12, 26):
        features[f"EMA_{period}"] = close.ewm(
            span=period, adjust=False, min_periods=period
        ).mean()

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    relative_strength = _safe_divide(average_gain, average_loss)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50)
    features["RSI_14"] = rsi

    macd = features["EMA_12"] - features["EMA_26"]
    features["MACD"] = macd
    features["MACD_Signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["MACD_Histogram"] = features["MACD"] - features["MACD_Signal"]
    features["ROC_10"] = close.pct_change(periods=10, fill_method=None) * 100

    for period in (10, 20, 30):
        features[f"Volatility_{period}D"] = daily_return.rolling(
            window=period, min_periods=period
        ).std()
    bb_middle = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    features["BB_Upper_20"] = bb_middle + 2 * bb_std
    features["BB_Middle_20"] = bb_middle
    features["BB_Lower_20"] = bb_middle - 2 * bb_std

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    true_range = true_range.where(previous_close.notna(), high - low)
    features["ATR_14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    features["Volume_Change"] = volume.pct_change(fill_method=None)
    features["Avg_Volume_20D"] = volume.rolling(window=20, min_periods=20).mean()
    features["Relative_Volume"] = _safe_divide(volume, features["Avg_Volume_20D"])
    features["High_Low_Pct"] = _safe_divide(high - low, close) * 100
    features["Open_Close_Pct"] = _safe_divide(close - open_price, open_price) * 100
    features["Daily_Price_Range"] = high - low
    return features


def build_market_features(
    nifty: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build market features indexed by exact observation date, without look-ahead."""
    market_frames: list[pd.DataFrame] = []

    if nifty is not None:
        required = {"Date", "Close"}
        if not required.issubset(nifty.columns):
            raise ValueError(f"NIFTY data must contain columns: {sorted(required)}")
        index_data = nifty.loc[:, ["Date", "Close"]].copy()
        index_data["Date"] = _parse_dates(index_data["Date"])
        index_data["Close"] = pd.to_numeric(index_data["Close"], errors="coerce")
        index_data = index_data.dropna(subset=["Date"]).drop_duplicates("Date", keep="first")
        index_data = index_data.sort_values("Date", kind="mergesort")
        index_close = index_data["Close"]
        index_data["NIFTY_Daily_Return"] = index_close.pct_change(fill_method=None)
        index_data["NIFTY_Return_5D"] = index_close.pct_change(5, fill_method=None)
        index_data["NIFTY_Return_20D"] = index_close.pct_change(20, fill_method=None)
        market_frames.append(
            index_data[["Date", "NIFTY_Daily_Return", "NIFTY_Return_5D", "NIFTY_Return_20D"]]
        )

    if vix is not None:
        if not {"Date", "Close"}.issubset(vix.columns):
            raise ValueError("India VIX data must contain Date and Close columns")
        vix_data = vix.loc[:, ["Date", "Close"]].copy()
        vix_data["Date"] = _parse_dates(vix_data["Date"])
        vix_data["Close"] = pd.to_numeric(vix_data["Close"], errors="coerce")
        vix_data = vix_data.dropna(subset=["Date"]).drop_duplicates("Date", keep="first")
        vix_data = vix_data.sort_values("Date", kind="mergesort")
        vix_data.rename(columns={"Close": "India_VIX_Close"}, inplace=True)
        vix_data["India_VIX_Daily_Change"] = vix_data["India_VIX_Close"].diff()
        market_frames.append(vix_data)

    if not market_frames:
        return pd.DataFrame(columns=["Date"])

    combined_market = market_frames[0]
    for market_frame in market_frames[1:]:
        combined_market = combined_market.merge(
            market_frame, on="Date", how="outer", validate="one_to_one"
        )
    combined_market.replace([np.inf, -np.inf], np.nan, inplace=True)
    return combined_market.sort_values("Date").reset_index(drop=True)


def build_datasets(
    prices: pd.DataFrame,
    nifty: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    buy_threshold: float = 0.03,
    sell_threshold: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return features-only rows and a supervised dataset with separate labels."""
    if buy_threshold < 0 or sell_threshold < 0:
        raise ValueError("BUY and SELL thresholds must be non-negative")

    cleaned = clean_stock_data(prices)
    if cleaned.empty:
        raise ValueError("No valid stock rows remain after cleaning")

    engineered = pd.concat(
        [
            _stock_features(group)
            for _, group in cleaned.groupby("Symbol", sort=False)
        ],
        ignore_index=True,
    )

    market = build_market_features(nifty=nifty, vix=vix)
    if not market.empty:
        engineered = engineered.merge(
            market, on="Date", how="left", validate="many_to_one", sort=False
        )

    for column in FEATURE_COLUMNS:
        if column not in engineered.columns:
            engineered[column] = np.nan
    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    engineered = engineered.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)

    feature_columns = [
        column
        for column in engineered.columns
        if column not in TARGET_COLUMNS
    ]
    features = engineered[feature_columns].copy()

    training = engineered.copy()
    training["Future_Return_5D"] = training.groupby("Symbol", sort=False)["Close"].transform(
        lambda close: _safe_divide(close.shift(-5), close).sub(1)
    )
    training["Target"] = np.select(
        [
            training["Future_Return_5D"].gt(buy_threshold),
            training["Future_Return_5D"].lt(-sell_threshold),
        ],
        ["BUY", "SELL"],
        default="HOLD",
    )
    training = training.dropna(subset=["Future_Return_5D"]).reset_index(drop=True)
    return features, training


def validate_datasets(
    features: pd.DataFrame,
    training: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    """Raise an error if key data-integrity or leakage checks fail."""
    for name, dataset in (("features", features), ("training", training)):
        if dataset.duplicated(["Symbol", "Date"]).any():
            raise AssertionError(f"{name} contains duplicate Symbol-Date rows")
        numeric = dataset.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy()).any():
            raise AssertionError(f"{name} contains infinite numeric values")

    if any(column in features.columns for column in TARGET_COLUMNS):
        raise AssertionError("Future target columns leaked into the features-only dataset")
    if not set(MODEL_INPUT_COLUMNS).isdisjoint(TARGET_COLUMNS):
        raise AssertionError("A future target column is included in the model-input whitelist")
    if training["Target"].isna().any() or training["Future_Return_5D"].isna().any():
        raise AssertionError("Training data contains rows without a valid future target")

    expected_target = np.select(
        [
            training["Future_Return_5D"].gt(buy_threshold),
            training["Future_Return_5D"].lt(-sell_threshold),
        ],
        ["BUY", "SELL"],
        default="HOLD",
    )
    if not np.array_equal(training["Target"].to_numpy(), expected_target):
        raise AssertionError("Target classes do not match the configured thresholds")


def _read_optional_csv(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"Warning: {label} file not found at {path}; continuing without it.")
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read {label} file {path}: {exc}") from exc


def _resolve_stock_input(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_STOCK_INPUT.exists():
        return DEFAULT_STOCK_INPUT
    if FALLBACK_STOCK_INPUT.exists():
        return FALLBACK_STOCK_INPUT
    raise FileNotFoundError(
        f"No stock input found. Expected {DEFAULT_STOCK_INPUT} or {FALLBACK_STOCK_INPUT}."
    )


def print_summary(features: pd.DataFrame, training: pd.DataFrame, source: Path) -> None:
    """Print dataset dimensions, coverage, missingness, and label balance."""
    symbols = features["Symbol"].nunique()
    dates = pd.to_datetime(features["Date"], errors="coerce")
    print(f"Input: {source}")
    print(f"Features shape: {features.shape}")
    print(f"Training shape: {training.shape}")
    print(f"Stocks: {symbols}")
    if dates.notna().any():
        print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print("Missing values in features (largest first):")
    missing = features.isna().sum().sort_values(ascending=False)
    print(missing[missing.gt(0)].to_string() if missing.gt(0).any() else "None")
    print("Target distribution:")
    print(training["Target"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0).to_string())


def run_pipeline(
    input_path: Path | None = None,
    nifty_path: Path | None = DEFAULT_NIFTY_INPUT,
    vix_path: Path | None = DEFAULT_VIX_INPUT,
    output_dir: Path = OUTPUT_DIR,
    buy_threshold: float = 0.03,
    sell_threshold: float = 0.03,
) -> tuple[Path, Path]:
    """Read source files, generate datasets, validate, and export CSVs."""
    source = _resolve_stock_input(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Stock input file does not exist: {source}")
    try:
        prices = pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read stock input file {source}: {exc}") from exc

    nifty = _read_optional_csv(nifty_path, "NIFTY")
    vix = _read_optional_csv(vix_path, "India VIX")
    features, training = build_datasets(
        prices,
        nifty=nifty,
        vix=vix,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    validate_datasets(features, training, buy_threshold, sell_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "stock_features.csv"
    training_path = output_dir / "training_dataset.csv"
    features.to_csv(features_path, index=False)
    training.to_csv(training_path, index=False)
    print_summary(features, training, source)
    print(f"Saved: {features_path}")
    print(f"Saved: {training_path}")
    return features_path, training_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Combined stock CSV (default: auto-detected)")
    parser.add_argument("--nifty", type=Path, default=DEFAULT_NIFTY_INPUT)
    parser.add_argument("--vix", type=Path, default=DEFAULT_VIX_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--buy-threshold", type=float, default=0.03)
    parser.add_argument("--sell-threshold", type=float, default=0.03)
    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        nifty_path=args.nifty,
        vix_path=args.vix,
        output_dir=args.output_dir,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )


if __name__ == "__main__":
    main()