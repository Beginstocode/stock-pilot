from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Src" / "data collection"))

from refresh_market_data import (
	MARKET_YAHOO_TICKERS,
	STOCK_YAHOO_TICKERS,
	completed_session_end,
	merge_market_close,
	merge_preserving_history,
	normalize_yahoo_history,
	run_predictions,
	validate_feature_output,
	validate_stock_prices,
)


def _price_row(symbol: str, day: str, close: float) -> dict[str, object]:
	return {
		"Date": pd.Timestamp(day),
		"Symbol": symbol,
		"Open": close - 1,
		"High": close + 1,
		"Low": close - 2,
		"Close": close,
		"Adj Close": close,
		"Volume": 100,
	}


def test_yahoo_multiindex_is_normalized_without_filling_values():
	columns = pd.MultiIndex.from_product(
		[["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["ABC.NS"]]
	)
	frame = pd.DataFrame([[10, 12, 9, 11, 10.5, 100]], columns=columns, index=pd.to_datetime(["2026-11-13"]))
	frame.index.name = "Date"
	frame["Date"] = frame.index
	actual = normalize_yahoo_history(frame, "ABC.NS", symbol="ABC")
	assert actual.loc[0, "Close"] == 11
	assert actual.loc[0, "Adj Close"] == 10.5
	assert actual.loc[0, "Symbol"] == "ABC"
	assert len(actual) == 1


def test_merge_keeps_existing_history_and_resolves_overlaps_without_duplicates():
	existing = pd.DataFrame([_price_row("RELIANCE", "2026-11-12", 100)])
	updates = pd.DataFrame([
		_price_row("RELIANCE", "2026-11-12", 101),
		_price_row("RELIANCE", "2026-11-13", 102),
	])
	merged, overlap_count = merge_preserving_history(existing, updates)
	assert overlap_count == 1
	assert len(merged) == 2
	assert not merged.duplicated(["Symbol", "Date"]).any()
	assert merged.loc[merged.Date.eq(pd.Timestamp("2026-11-12")), "Close"].item() == 100
	assert merged.loc[merged.Date.eq(pd.Timestamp("2026-11-13")), "Close"].item() == 102


@pytest.mark.parametrize(
	"change",
	[
		{"High": 8},
		{"Low": 12},
		{"Volume": -1},
		{"Close": np.nan},
		{"Adj Close": 0},
	],
)
def test_invalid_ohlcv_is_rejected(change):
	row = _price_row("TCS", "2026-11-13", 10)
	row.update(change)
	with pytest.raises(ValueError):
		validate_stock_prices(pd.DataFrame([row]))


def test_duplicate_symbol_date_is_rejected_at_source_validation():
	row = _price_row("TCS", "2026-11-13", 10)
	with pytest.raises(ValueError, match="duplicate"):
		validate_stock_prices(pd.DataFrame([row, row]))


def test_market_merge_preserves_old_closes_and_appends_new_sessions():
	old = pd.DataFrame({
		"Date": pd.to_datetime(["2026-11-12"]),
		"NIFTY50_Close": [25000.0],
		"India_VIX_Close": [15.0],
	})
	updates = {
		"NIFTY50_Close": pd.DataFrame({"Date": pd.to_datetime(["2026-11-12", "2026-11-13"]), "Close": [25001.0, 25100.0]}),
		"India_VIX_Close": pd.DataFrame({"Date": pd.to_datetime(["2026-11-12", "2026-11-13"]), "Close": [15.1, 16.0]}),
	}
	merged, separate, overlap_count = merge_market_close(old, updates)
	assert len(merged) == 2
	assert overlap_count == 1
	assert merged.loc[merged.Date.eq(pd.Timestamp("2026-11-12")), "NIFTY50_Close"].item() == 25000
	assert merged.loc[merged.Date.eq(pd.Timestamp("2026-11-13")), "India_VIX_Close"].item() == 16
	assert len(separate["NIFTY50_Close"]) == 2


def test_completed_session_cutoff_avoids_partial_daily_candle():
	before_cutoff = datetime(2026, 11, 13, 9, 59, tzinfo=timezone.utc)  # 15:29 IST
	after_cutoff = datetime(2026, 11, 13, 10, 30, tzinfo=timezone.utc)  # 16:00 IST
	# The function uses Asia/Kolkata session time; ensure exclusive end includes
	# a full current-session bar only after its close threshold.
	assert completed_session_end(before_cutoff) == date(2026, 11, 13)
	assert completed_session_end(after_cutoff) == date(2026, 11, 14)


def test_feature_schema_rejects_future_targets_and_requires_latest_values():
	features = pd.DataFrame([
		{"Date": pd.Timestamp("2026-11-13"), "Symbol": symbol, "x": 1.0}
		for symbol in STOCK_YAHOO_TICKERS
	])
	report = validate_feature_output(features, ["x"])
	assert report["feature_count"] == 1
	assert set(report["latest_market_date_by_symbol"]) == set(STOCK_YAHOO_TICKERS)
	with pytest.raises(ValueError, match="future-derived"):
		validate_feature_output(features.assign(Target="HOLD"), ["x"])


def test_prediction_runner_calls_both_frozen_models_for_all_supported_symbols():
	predictor = Mock(side_effect=lambda symbol, **kwargs: SimpleNamespace(
		symbol=symbol,
		prediction_date="2026-11-13",
		prediction="HOLD",
		confidence=0.5,
		probabilities={"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25},
	))
	features_path = Path("unused_features.csv")
	output = run_predictions(features_path, predictor=predictor)
	assert len(output) == 20
	assert output.groupby("model").size().to_dict() == {"Model A": 10, "Model B": 10}
	assert set(output.symbol) == set(STOCK_YAHOO_TICKERS)
	assert {call.kwargs["metadata_path"].name for call in predictor.call_args_list} == {
		"class_label_mapping.json",
		"model_b_metadata.json",
	}
	assert len(STOCK_YAHOO_TICKERS) == 10
	assert set(MARKET_YAHOO_TICKERS) == {"NIFTY50_Close", "India_VIX_Close"}
