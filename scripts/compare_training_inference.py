"""Compare saved training and inference feature values for RELIANCE and TCS."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURES = json.loads((ROOT / "reports" / "class_label_mapping.json").read_text(encoding="utf-8"))["feature_columns"]

training = pd.read_csv(PROCESSED / "training_dataset.csv", parse_dates=["Date"])
inference = pd.read_csv(PROCESSED / "stock_features.csv", parse_dates=["Date"])

records: list[dict[str, object]] = []
for symbol in ("RELIANCE", "TCS"):
    train_rows = training.loc[training["Symbol"].eq(symbol), ["Symbol", "Date", *FEATURES]]
    infer_rows = inference.loc[inference["Symbol"].eq(symbol), ["Symbol", "Date", *FEATURES]]
    joined = train_rows.merge(infer_rows, on=["Symbol", "Date"], how="inner", suffixes=("_training", "_inference"), validate="one_to_one")
    for feature in FEATURES:
        train_values = pd.to_numeric(joined[f"{feature}_training"], errors="coerce")
        infer_values = pd.to_numeric(joined[f"{feature}_inference"], errors="coerce")
        both_nan = train_values.isna() & infer_values.isna()
        equal = train_values.eq(infer_values) | both_nan
        abs_diff = (train_values - infer_values).abs()
        mismatch = ~equal
        for idx in joined.index[mismatch]:
            records.append({
                "symbol": symbol,
                "date": joined.at[idx, "Date"].date().isoformat(),
                "feature": feature,
                "training_value": train_values.at[idx],
                "inference_value": infer_values.at[idx],
                "absolute_difference": abs_diff.at[idx],
            })

report = pd.DataFrame(records, columns=["symbol", "date", "feature", "training_value", "inference_value", "absolute_difference"])
if not report.empty:
    report.sort_values(["symbol", "absolute_difference"], ascending=[True, False], na_position="first", inplace=True)
output = ROOT / "reports" / "feature_value_mismatch.csv"
report.to_csv(output, index=False)
print(f"Saved {len(report)} mismatches to {output}")
for symbol in ("RELIANCE", "TCS"):
    print(f"\n{symbol}: first 10 mismatches by absolute difference")
    subset = report.loc[report["symbol"].eq(symbol)].head(10)
    if subset.empty:
        print("No mismatches")
    else:
        print(subset[["feature", "date", "training_value", "inference_value", "absolute_difference"]].to_string(index=False))
