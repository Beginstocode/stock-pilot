"""Stage a validated daily market-data refresh without retraining models.

The command downloads the supported NSE stocks and the index series used by the
feature pipeline, merges them with existing history, rebuilds feature-only
rows with the training feature-engineering code, and predicts with frozen
Model A and Model B. Outputs are written to a timestamped run directory. Use
``--promote`` only after validation to atomically replace the processed price
and feature files; timestamped originals are backed up first. The training
dataset and model artifacts are never written.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
	PROJECT_ROOT,
	PROJECT_ROOT / "Src" / "data processing",
	PROJECT_ROOT / "Src" / "modeling",
	PROJECT_ROOT / "Src" / "prediction",
):
	if str(_path) not in sys.path:
		sys.path.insert(0, str(_path))

from Backend.App.stock_catalog import STOCK_CATALOG
import feature_engineering
from predict import predict_stock

LOGGER = logging.getLogger("stockpilot.market_refresh")

STOCK_YAHOO_TICKERS: dict[str, str] = {
	"RELIANCE": "RELIANCE.NS",
	"TCS": "TCS.NS",
	"INFY": "INFY.NS",
	"HDFCBANK": "HDFCBANK.NS",
	"ICICIBANK": "ICICIBANK.NS",
	"SBIN": "SBIN.NS",
	"ITC": "ITC.NS",
	"LT": "LT.NS",
	"BHARTIARTL": "BHARTIARTL.NS",
	# NSE changed the former TATAMOTORS symbol to TMPV effective 2025-10-24.
	"TATAMOTORS": "TMPV.NS",
}
MARKET_YAHOO_TICKERS: dict[str, str] = {
	"NIFTY50_Close": "^NSEI",
	"India_VIX_Close": "^INDIAVIX",
}
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
OHLC_COLUMNS = ("Open", "High", "Low", "Close")
MODEL_A_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
MODEL_A_METADATA = PROJECT_ROOT / "reports" / "class_label_mapping.json"
MODEL_B_PATH = PROJECT_ROOT / "models" / "xgboost_model_b.pkl"
MODEL_B_METADATA = PROJECT_ROOT / "reports" / "model_b_metadata.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MARKET_PATH = PROJECT_ROOT / "data" / "market" / "combined_market_data.csv"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "processed" / "refresh_runs"


def configure_yfinance_cache(cache_dir: Path | None = None) -> Path:
	"""Point yfinance's SQLite caches at a writable local temporary directory."""
	import yfinance as yf

	cache_dir = Path(cache_dir or (Path(tempfile.gettempdir()) / "stockpilot-yfinance-cache"))
	cache_dir.mkdir(parents=True, exist_ok=True)
	yf.set_tz_cache_location(str(cache_dir))
	# yfinance versions without a public cookie-cache location setter expose this
	# manager internally. Failure to configure it must not block price requests.
	try:
		from yfinance import cache

		manager = getattr(cache, "_CookieDBManager", None)
		if manager is not None and callable(getattr(manager, "set_location", None)):
			manager.set_location(str(cache_dir))
	except (ImportError, AttributeError, OSError):
		LOGGER.debug("Could not set the optional yfinance cookie-cache location")
	return cache_dir


def completed_session_end(now: datetime | None = None) -> date:
	"""Return an exclusive Yahoo end date that avoids today's partial candle."""
	local_now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
	if local_now.tzinfo is None:
		local_now = local_now.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
	local_now = local_now.astimezone(ZoneInfo("Asia/Kolkata"))
	# Yahoo's daily candle is only eligible after the NSE cash session is closed.
	local_time = local_now.timetz().replace(tzinfo=None)
	last_included_day = local_now.date() if local_time >= day_time(16, 0) else local_now.date() - timedelta(days=1)
	return last_included_day + timedelta(days=1)


def _flatten_download_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
	if not isinstance(frame.columns, pd.MultiIndex):
		return frame.copy()
	for level in range(frame.columns.nlevels):
		if ticker in frame.columns.get_level_values(level):
			return frame.xs(ticker, axis=1, level=level, drop_level=True).copy()
	if frame.columns.nlevels == 2 and frame.columns.get_level_values(1).nunique() == 1:
		return frame.copy().set_axis(frame.columns.get_level_values(0), axis=1)
	raise ValueError(f"Could not select {ticker} from Yahoo's multi-index response")


def normalize_yahoo_history(
	frame: pd.DataFrame,
	ticker: str,
	*,
	symbol: str | None = None,
	close_column: str | None = None,
) -> pd.DataFrame:
	"""Normalize a yfinance daily response without filling or inventing values."""
	if frame is None or frame.empty:
		raise ValueError(f"Yahoo Finance returned no daily rows for {ticker}")
	data = _flatten_download_columns(frame, ticker)
	if "Date" in data.columns:
		date_values = pd.to_datetime(data.pop("Date"), errors="coerce")
	else:
		date_values = pd.to_datetime(data.index, errors="coerce")
	date_values = pd.DatetimeIndex(date_values)
	if date_values.tz is not None:
		date_values = date_values.tz_convert("Asia/Kolkata").tz_localize(None)
	# Yahoo may name both the index and an explicit field "Date". Keep the
	# normalized date only in our output column, avoiding pandas label ambiguity.
	data = data.reset_index(drop=True)
	data["Date"] = date_values.normalize()
	data = data.dropna(subset=["Date"]).copy()
	if close_column is not None:
		if close_column not in data.columns:
			raise ValueError(f"Yahoo response for {ticker} is missing {close_column}")
		result = data[[close_column]].rename(columns={close_column: "Close"}).copy()
	else:
		missing = set(PRICE_COLUMNS).difference(data.columns)
		if missing:
			raise ValueError(f"Yahoo response for {ticker} lacks columns: {sorted(missing)}")
		result = data.loc[:, list(PRICE_COLUMNS)].copy()
	result.insert(0, "Date", data["Date"].to_numpy())
	for column in result.columns:
		if column != "Date":
			result[column] = pd.to_numeric(result[column], errors="coerce")
	result.replace([np.inf, -np.inf], np.nan, inplace=True)
	if symbol is not None:
		result.insert(1, "Symbol", symbol)
	result = result.sort_values("Date", kind="mergesort").reset_index(drop=True)
	if result.duplicated("Date").any():
		raise ValueError(f"Yahoo response contains duplicate dates for {ticker}")
	return result


def download_daily_history(
	ticker: str,
	start: date,
	end_exclusive: date,
	*,
	max_attempts: int = 3,
	sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
	"""Fetch real daily bars, retry transient failures, and fail on empty data."""
	import yfinance as yf

	last_error: Exception | None = None
	for attempt in range(1, max_attempts + 1):
		try:
			raw = yf.download(
				ticker,
				start=start.isoformat(),
				end=end_exclusive.isoformat(),
				interval="1d",
				auto_adjust=False,
				progress=False,
				threads=False,
				timeout=25,
			)
			if raw is None or raw.empty:
				raise ValueError(f"No daily rows returned for {ticker} from {start} to {end_exclusive}")
			return normalize_yahoo_history(raw, ticker)
		except Exception as exc:
			last_error = exc
			if attempt < max_attempts:
				wait_seconds = float(2 ** (attempt - 1))
				LOGGER.warning("Fetch failed for %s (attempt %d/%d); retrying in %.0fs", ticker, attempt, max_attempts, wait_seconds)
				sleep(wait_seconds)
	if last_error is not None:
		raise RuntimeError(f"Could not fetch daily history for {ticker}: {last_error}") from last_error
	raise RuntimeError(f"Could not fetch daily history for {ticker}")


def validate_stock_prices(frame: pd.DataFrame, *, label: str = "stock data") -> None:
	"""Reject invalid OHLCV, keys, or duplicate Symbol-Date records."""
	required = {"Date", "Symbol", *PRICE_COLUMNS}
	missing = required.difference(frame.columns)
	if missing:
		raise ValueError(f"{label} is missing columns: {sorted(missing)}")
	if frame.empty:
		raise ValueError(f"{label} contains no rows")
	if frame["Date"].isna().any() or frame["Symbol"].isna().any():
		raise ValueError(f"{label} contains missing Date or Symbol keys")
	if frame["Symbol"].astype("string").str.strip().eq("").any():
		raise ValueError(f"{label} contains an empty Symbol")
	if frame.duplicated(["Symbol", "Date"]).any():
		raise ValueError(f"{label} contains duplicate Symbol-Date rows")
	numeric = frame.loc[:, list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
	if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
		raise ValueError(f"{label} contains missing, malformed, or infinite OHLCV values")
	prices = numeric.loc[:, list(OHLC_COLUMNS)]
	if (prices <= 0).any().any():
		raise ValueError(f"{label} contains non-positive OHLC prices")
	if (
		prices["High"].lt(prices["Open"])
		| prices["High"].lt(prices["Low"])
		| prices["High"].lt(prices["Close"])
		| prices["Low"].gt(prices["Open"])
		| prices["Low"].gt(prices["Close"])
	).any():
		raise ValueError(f"{label} contains inconsistent OHLC values")
	if (numeric["Volume"] < 0).any():
		raise ValueError(f"{label} contains negative volume")
	if (numeric["Adj Close"] <= 0).any():
		raise ValueError(f"{label} contains a non-positive adjusted close")


def merge_preserving_history(
	existing: pd.DataFrame,
	updates: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
	"""Append new rows; for overlaps retain the original historical record."""
	if not existing.empty:
		old = existing.copy()
		old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.normalize()
		old["Symbol"] = old["Symbol"].astype("string").str.strip().str.upper()
	else:
		old = pd.DataFrame(columns=updates.columns)
	new = updates.copy()
	new["Date"] = pd.to_datetime(new["Date"], errors="coerce").dt.normalize()
	new["Symbol"] = new["Symbol"].astype("string").str.strip().str.upper()
	joined = pd.concat([old, new], ignore_index=True, sort=False)
	duplicate_count = int(joined.duplicated(["Symbol", "Date"], keep="first").sum())
	joined = joined.drop_duplicates(["Symbol", "Date"], keep="first")
	joined = joined.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)
	validate_stock_prices(joined, label="merged stock data")
	return joined, duplicate_count


def _latest_date(frame: pd.DataFrame, symbol: str | None = None) -> date | None:
	data = frame if symbol is None else frame.loc[frame["Symbol"].eq(symbol)]
	if data.empty:
		return None
	return pd.to_datetime(data["Date"]).max().date()


def merge_market_close(
	existing_combined: pd.DataFrame,
	updated_closes: dict[str, pd.DataFrame],
	*,
	source_vix: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], int]:
	"""Merge NIFTY/VIX closes from their real sources, preserving old values."""
	old = existing_combined.copy()
	old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.normalize()
	old = old.dropna(subset=["Date"])
	old = old.sort_values("Date", kind="mergesort").drop_duplicates("Date", keep="first")
	# The combined inference cache may predate observations already present in
	# the authoritative VIX source. Restore only absent values from that source.
	if source_vix is not None and "India_VIX_Close" in updated_closes:
		source = source_vix.loc[:, ["Date", "Close"]].copy()
		source["Date"] = feature_engineering._parse_dates(source["Date"])
		source["Close"] = pd.to_numeric(source["Close"], errors="coerce")
		source = source.dropna(subset=["Date", "Close"]).drop_duplicates("Date", keep="first")
		old = old.merge(source.rename(columns={"Close": "_source_vix"}), on="Date", how="outer", validate="one_to_one")
		old["India_VIX_Close"] = old["India_VIX_Close"].combine_first(old["_source_vix"])
		old = old.drop(columns="_source_vix")
	new = pd.DataFrame({"Date": pd.to_datetime([], errors="coerce")})
	for column, bars in updated_closes.items():
		series = bars.loc[:, ["Date", "Close"]].rename(columns={"Close": column})
		new = series if new.empty else new.merge(series, on="Date", how="outer", validate="one_to_one")
	# A field-by-field merge preserves existing index values while appending dates.
	for column in updated_closes:
		if column not in old.columns:
			old[column] = np.nan
		if column not in new.columns:
			new[column] = np.nan
	combined = old.loc[:, ["Date", *updated_closes]].merge(
		new.loc[:, ["Date", *updated_closes]],
		on="Date",
		how="outer",
		suffixes=("_old", "_new"),
		validate="one_to_one",
	)
	for field in updated_closes:
		combined[field] = combined[f"{field}_old"].combine_first(combined[f"{field}_new"])
	combined = combined.loc[:, ["Date", *updated_closes]].sort_values("Date", kind="mergesort")
	for column in updated_closes:
		combined[column] = pd.to_numeric(combined[column], errors="coerce")
	if combined.duplicated("Date").any():
		raise ValueError("Merged market history contains duplicate dates")
	if combined[list(updated_closes)].isna().all(axis=1).any():
		# Rows with neither market source are not useful for feature engineering.
		combined = combined.loc[~combined[list(updated_closes)].isna().all(axis=1)].copy()
	separate: dict[str, pd.DataFrame] = {}
	for column in updated_closes:
		valid = combined.loc[:, ["Date", column]].dropna(subset=[column]).copy()
		if (~np.isfinite(valid[column])).any() or (valid[column] <= 0).any():
			raise ValueError(f"Market series {column} contains invalid close values")
		separate[column] = valid.rename(columns={column: "Close"})
	old_keys = set(old["Date"])
	new_keys = set(new["Date"]) if not new.empty else set()
	duplicates = len(old_keys & new_keys)
	return combined.reset_index(drop=True), separate, duplicates


def get_stock_updates(
	existing: pd.DataFrame,
	*,
	start_date: date,
	end_exclusive: date,
	fetcher: Callable[[str, date, date], pd.DataFrame] = download_daily_history,
) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for symbol in STOCK_CATALOG:
		ticker = STOCK_YAHOO_TICKERS[symbol]
		latest = _latest_date(existing, symbol)
		start = max(start_date, latest) if latest is not None else start_date
		bars = fetcher(ticker, start, end_exclusive)
		bars.insert(1, "Symbol", symbol)
		validate_stock_prices(bars, label=f"fetched data for {symbol}")
		frames.append(bars)
	return pd.concat(frames, ignore_index=True)


def get_market_updates(
	existing_combined: pd.DataFrame,
	*,
	start_date: date,
	end_exclusive: date,
	fetcher: Callable[[str, date, date], pd.DataFrame] = download_daily_history,
) -> dict[str, pd.DataFrame]:
	frames: dict[str, pd.DataFrame] = {}
	last_date = pd.to_datetime(existing_combined["Date"], errors="coerce").max()
	start = max(start_date, last_date.date()) if pd.notna(last_date) else start_date
	for column, ticker in MARKET_YAHOO_TICKERS.items():
		bars = fetcher(ticker, start, end_exclusive)
		if "Close" not in bars.columns:
			raise ValueError(f"Fetched market data for {ticker} has no Close column")
		frames[column] = bars.loc[:, ["Date", "Close"]].copy()
	return frames


def validate_feature_output(
	features: pd.DataFrame,
	model_feature_columns: list[str],
	supported_symbols: tuple[str, ...] = tuple(STOCK_CATALOG),
) -> dict[str, Any]:
	"""Check feature schema, keys, latest row coverage, and missing values."""
	if not set(model_feature_columns).issubset(features.columns):
		missing = sorted(set(model_feature_columns).difference(features.columns))
		raise ValueError(f"Engineered data lacks model features: {missing}")
	if features.duplicated(["Symbol", "Date"]).any():
		raise ValueError("Engineered features contain duplicate Symbol-Date rows")
	if "Target" in features.columns or "Future_Return_5D" in features.columns:
		raise ValueError("Feature-only prediction data contains future-derived target columns")
	latest: dict[str, str] = {}
	missing_features: dict[str, list[str]] = {}
	for symbol in supported_symbols:
		rows = features.loc[features["Symbol"].astype("string").str.upper().eq(symbol)]
		if rows.empty:
			raise ValueError(f"No engineered rows for supported symbol {symbol}")
		last_date = pd.to_datetime(rows["Date"], errors="coerce").max()
		latest[symbol] = last_date.date().isoformat()
		last_row = rows.loc[pd.to_datetime(rows["Date"]).eq(last_date)].iloc[0]
		bad = [column for column in model_feature_columns if not np.isfinite(pd.to_numeric(pd.Series([last_row[column]]), errors="coerce").iloc[0])]
		missing_features[symbol] = bad
		if bad:
			raise ValueError(f"Latest feature row for {symbol} has unavailable values: {bad}")
	return {
		"feature_count": len(model_feature_columns),
		"latest_market_date_by_symbol": latest,
		"latest_feature_missing_count_by_symbol": {symbol: len(cols) for symbol, cols in missing_features.items()},
	}


def missing_session_counts(prices: pd.DataFrame, sessions: pd.Series) -> dict[str, int]:
	"""Count absent observed market sessions by symbol; never fill them."""
	expected = set(pd.to_datetime(sessions, errors="coerce").dropna().dt.normalize())
	counts: dict[str, int] = {}
	for symbol in STOCK_CATALOG:
		observed = set(pd.to_datetime(prices.loc[prices["Symbol"].eq(symbol), "Date"]).dt.normalize())
		first = min(observed) if observed else None
		counts[symbol] = len({day for day in expected if first is not None and day >= first} - observed)
	return counts


def recent_missing_sessions(
	prices: pd.DataFrame,
	sessions: pd.Series,
	*,
	window: int = 30,
) -> dict[str, list[str]]:
	"""List recent benchmark sessions absent from each symbol, without filling."""
	expected = pd.DatetimeIndex(pd.to_datetime(sessions, errors="coerce").dropna().dt.normalize().unique()).sort_values()
	recent = expected[-window:]
	result: dict[str, list[str]] = {}
	for symbol in STOCK_CATALOG:
		observed = set(pd.to_datetime(prices.loc[prices["Symbol"].eq(symbol), "Date"]).dt.normalize())
		result[symbol] = [session.date().isoformat() for session in recent if session not in observed]
	return result


def missing_session_dates(prices: pd.DataFrame, sessions: pd.Series) -> dict[str, list[str]]:
	"""Return every missing benchmark session after each symbol's first row."""
	expected = pd.DatetimeIndex(pd.to_datetime(sessions, errors="coerce").dropna().dt.normalize().unique()).sort_values()
	result: dict[str, list[str]] = {}
	for symbol in STOCK_CATALOG:
		observed = set(pd.to_datetime(prices.loc[prices["Symbol"].eq(symbol), "Date"]).dt.normalize())
		first = min(observed) if observed else None
		result[symbol] = [
			session.date().isoformat()
			for session in expected
			if first is not None and session >= first and session not in observed
		]
	return result


def run_predictions(
	features_path: Path,
	*,
	predictor: Callable[..., Any] = predict_stock,
) -> pd.DataFrame:
	"""Run both frozen model artifacts against feature-only data for all stocks."""
	rows: list[dict[str, Any]] = []
	for model_name, model_path, metadata_path in (
		("Model A", MODEL_A_PATH, MODEL_A_METADATA),
		("Model B", MODEL_B_PATH, MODEL_B_METADATA),
	):
		for symbol in STOCK_CATALOG:
			result = predictor(
				symbol,
				model_path=model_path,
				features_path=features_path,
				metadata_path=metadata_path,
			)
			rows.append({
				"model": model_name,
				"model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
				"symbol": result.symbol,
				"prediction_date": result.prediction_date,
				"prediction": result.prediction,
				"confidence": result.confidence,
				"probability_buy": result.probabilities["BUY"],
				"probability_hold": result.probabilities["HOLD"],
				"probability_sell": result.probabilities["SELL"],
			})
	return pd.DataFrame(rows)


def _read_existing_prices(path: Path) -> pd.DataFrame:
	frame = pd.read_csv(path, low_memory=False)
	frame.columns = [str(column).strip() for column in frame.columns]
	frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
	frame["Symbol"] = frame["Symbol"].astype("string").str.strip().str.upper()
	return frame


def _read_existing_market(path: Path) -> pd.DataFrame:
	frame = pd.read_csv(path, low_memory=False)
	frame.columns = [str(column).strip() for column in frame.columns]
	if not {"Date", *MARKET_YAHOO_TICKERS}.issubset(frame.columns):
		raise ValueError("Existing combined market data lacks Date, NIFTY50_Close, or India_VIX_Close")
	frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
	return frame


def _market_close_frame(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	nifty = combined.loc[:, ["Date", "NIFTY50_Close"]].rename(columns={"NIFTY50_Close": "Close"})
	vix = combined.loc[:, ["Date", "India_VIX_Close"]].rename(columns={"India_VIX_Close": "Close"})
	return nifty.dropna(subset=["Close"]), vix.dropna(subset=["Close"])


def _sha256(path: Path) -> str:
	import hashlib

	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
	frame.to_csv(path, index=False, date_format="%Y-%m-%d")
	if not path.is_file() or path.stat().st_size == 0:
		raise OSError(f"Could not write non-empty staged file {path}")


def promote_staged_data(
	run_dir: Path,
	data_dir: Path,
	market_path: Path = DEFAULT_MARKET_PATH,
) -> Path:
	"""Back up current live files, then replace prices, features, and market closes."""
	backup_dir = run_dir / "backup"
	backup_dir.mkdir(parents=True, exist_ok=False)
	targets = {
		"stock_price_combined.csv": data_dir / "stock_price_combined.csv",
		"stock_features.csv": data_dir / "stock_features.csv",
		"combined_market_data.csv": market_path,
	}
	for name, target in targets.items():
		if target.exists():
			shutil.copy2(target, backup_dir / name)
	temporary: dict[str, Path] = {}
	for name in targets:
		tmp = targets[name].with_name(f".{name}.{uuid.uuid4().hex}.tmp")
		shutil.copy2(run_dir / name, tmp)
		temporary[name] = tmp
	replaced: list[str] = []
	try:
		for name, target in targets.items():
			os.replace(temporary[name], target)
			replaced.append(name)
	except OSError:
		for name in reversed(replaced):
			backup = backup_dir / name
			if backup.exists():
				recovery = targets[name].with_name(f".{name}.{uuid.uuid4().hex}.restore")
				shutil.copy2(backup, recovery)
				os.replace(recovery, targets[name])
			else:
				targets[name].unlink(missing_ok=True)
		raise
	finally:
		for tmp in temporary.values():
			if tmp.exists():
				tmp.unlink()
	return backup_dir


def run_refresh(
	*,
	data_dir: Path = DEFAULT_DATA_DIR,
	market_path: Path = DEFAULT_MARKET_PATH,
	runs_dir: Path = DEFAULT_RUNS_DIR,
	start_date: date = date(2019, 1, 1),
	end_exclusive: date | None = None,
	promote: bool = False,
	fetcher: Callable[[str, date, date], pd.DataFrame] = download_daily_history,
) -> tuple[Path, dict[str, Any]]:
	"""Fetch, merge, validate, engineer, predict, and optionally promote data."""
	end_exclusive = end_exclusive or completed_session_end()
	if len(STOCK_YAHOO_TICKERS) != 10 or set(STOCK_YAHOO_TICKERS) != set(STOCK_CATALOG):
		raise ValueError("Yahoo source mapping must cover the ten supported catalog symbols exactly")
	for required in (data_dir / "stock_price_combined.csv", market_path, MODEL_A_PATH, MODEL_A_METADATA, MODEL_B_PATH, MODEL_B_METADATA):
		if not required.is_file():
			raise FileNotFoundError(f"Required refresh input is missing: {required}")
	configure_yfinance_cache()
	existing = _read_existing_prices(data_dir / "stock_price_combined.csv")
	validate_stock_prices(existing, label="existing stock price history")
	existing_market = _read_existing_market(market_path)
	stock_updates = get_stock_updates(existing, start_date=start_date, end_exclusive=end_exclusive, fetcher=fetcher)
	merged_prices, price_overlap_count = merge_preserving_history(existing, stock_updates)
	market_updates = get_market_updates(existing_market, start_date=start_date, end_exclusive=end_exclusive, fetcher=fetcher)
	vix_source_path = PROJECT_ROOT / "data" / "raw" / "india_vix.csv"
	vix_source = pd.read_csv(vix_source_path, usecols=["Date", "Close"]) if vix_source_path.is_file() else None
	market_merged, _, market_overlap_count = merge_market_close(
		existing_market, market_updates, source_vix=vix_source
	)
	market_nifty, market_vix = _market_close_frame(market_merged)
	features, _unused_training_labels = feature_engineering.build_datasets(
		merged_prices,
		nifty=market_nifty,
		vix=market_vix,
		buy_threshold=0.03,
		sell_threshold=0.03,
	)
	feature_engineering.validate_datasets(
		features,
		_unused_training_labels,
		buy_threshold=0.03,
		sell_threshold=0.03,
	)
	meta_a = json.loads(MODEL_A_METADATA.read_text(encoding="utf-8"))
	meta_b = json.loads(MODEL_B_METADATA.read_text(encoding="utf-8"))
	feature_columns = meta_a["feature_columns"]
	if feature_columns != meta_b.get("feature_columns"):
		raise ValueError("Frozen Model A and Model B feature lists do not match")
	if len(feature_columns) != 41:
		raise ValueError(f"Expected 41 frozen model inputs, found {len(feature_columns)}")
	feature_report = validate_feature_output(features, feature_columns)
	if set(feature_report["latest_market_date_by_symbol"].values()) != {max(feature_report["latest_market_date_by_symbol"].values())}:
		raise ValueError("The latest actual market-data date differs across the ten symbols")
	latest_market_date = max(feature_report["latest_market_date_by_symbol"].values())
	calendar_counts = missing_session_counts(merged_prices, market_merged["Date"])
	latest_market_date_obj = pd.Timestamp(latest_market_date)
	if latest_market_date_obj < pd.Timestamp(merged_prices["Date"].max()).normalize():
		raise ValueError("Market index data ends before the latest stock data")

	stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
	runs_dir.mkdir(parents=True, exist_ok=True)
	stage = runs_dir / f".staging-{stamp}"
	final_run = runs_dir / stamp
	stage.mkdir()
	try:
		_write_csv(merged_prices, stage / "stock_price_combined.csv")
		_write_csv(features, stage / "stock_features.csv")
		_write_csv(market_merged, stage / "combined_market_data.csv")
		predictions = run_predictions(stage / "stock_features.csv")
		if len(predictions) != 20 or predictions[["prediction", "confidence", "probability_buy", "probability_hold", "probability_sell"]].isna().any().any():
			raise ValueError("Model predictions are incomplete for the supported stock universe")
		_write_csv(predictions, stage / "predictions.csv")
		latest_by_symbol = {
			symbol: _latest_date(merged_prices, symbol).isoformat()
			for symbol in STOCK_CATALOG
		}
		report: dict[str, Any] = {
			"run_id": stamp,
			"created_at_utc": datetime.now(timezone.utc).isoformat(),
			"provider": "Yahoo Finance via yfinance; auto_adjust=False",
			"requested_end_exclusive": end_exclusive.isoformat(),
			"supported_symbols": list(STOCK_CATALOG),
			"source_tickers": STOCK_YAHOO_TICKERS,
			"latest_actual_market_date_by_symbol": latest_by_symbol,
			"latest_engineered_features": feature_report,
			"price_rows": len(merged_prices),
			"price_duplicate_rows_resolved_existing_priority": price_overlap_count,
		"duplicate_symbol_date_rows_after_merge": int(merged_prices.duplicated(["Symbol", "Date"]).sum()),
		"invalid_ohlcv_rows": 0,
			"market_rows": len(market_merged),
			"market_duplicate_dates_resolved_existing_priority": market_overlap_count,
			"missing_nifty_session_count_by_symbol": calendar_counts,
		"missing_nifty_session_dates_by_symbol": missing_session_dates(merged_prices, market_merged["Date"]),
		"missing_nifty_sessions_last_30_by_symbol": recent_missing_sessions(merged_prices, market_merged["Date"]),
			"prediction_rows": len(predictions),
			"prediction_counts_by_model": predictions.groupby("model").size().to_dict(),
			"model_a_sha256": _sha256(MODEL_A_PATH),
			"model_b_sha256": _sha256(MODEL_B_PATH),
			"feature_columns": feature_columns,
			"feature_only_output": True,
			"future_outcomes_used_for_prediction": False,
			"original_training_dataset_modified": False,
			"promoted": False,
		}
		(stage / "refresh_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
		# Renaming the complete staging directory publishes a self-contained run.
		os.replace(stage, final_run)
		if promote:
			backup = promote_staged_data(final_run, data_dir, market_path)
			report["promoted"] = True
			report["backup_directory"] = str(backup.relative_to(PROJECT_ROOT))
			(final_run / "refresh_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
		return final_run, report
	except Exception:
		if stage.exists():
			shutil.rmtree(stage)
		raise


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--market", type=Path, default=DEFAULT_MARKET_PATH)
	parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
	parser.add_argument("--start-date", type=date.fromisoformat, default=date(2019, 1, 1))
	parser.add_argument("--promote", action="store_true", help="Back up and replace the current processed price and feature CSVs after successful validation")
	args = parser.parse_args(argv)
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	try:
		run_dir, report = run_refresh(
			data_dir=args.data_dir,
			market_path=args.market,
			runs_dir=args.runs_dir,
			start_date=args.start_date,
			promote=args.promote,
		)
	except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
		print(f"Refresh failed; current datasets were not replaced: {exc}", file=sys.stderr)
		return 1
	print(f"Refresh saved: {run_dir}")
	print(f"Predictions: {run_dir / 'predictions.csv'}")
	for symbol, as_of in report["latest_actual_market_date_by_symbol"].items():
		print(f"{symbol}: {as_of}")
	if report["promoted"]:
		print(f"Previous price/features saved in: {report['backup_directory']}")
	else:
		print("Current processed datasets were not replaced. Pass --promote to promote these validated outputs.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
