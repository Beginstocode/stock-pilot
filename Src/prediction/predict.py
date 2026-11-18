"""Generate a latest-row stock prediction from the trained XGBoost model.

Run from the project root:
    python "Src/prediction/predict.py" RELIANCE

If no symbol is supplied, the script prompts for one. The feature list and
class mapping are read from the metadata written during model training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
if str(_MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELING_DIR))
from feature_preprocessing import prepare_feature_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "stock_features.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "reports" / "class_label_mapping.json"
EXPECTED_LABELS = {"BUY", "HOLD", "SELL"}
FUTURE_NAME_PATTERN = re.compile(
    r"future|forward|lead|next|target|label|outcome|ahead|t\s*\+\s*\d",
    flags=re.IGNORECASE,
)


class StockSymbolNotFoundError(ValueError):
    """Raised when the requested symbol has no rows in the feature dataset."""


@dataclass(frozen=True)
class PredictionResult:
    """Serializable prediction details for CLI and API consumers."""

    symbol: str
    prediction: str
    probabilities: dict[str, float]
    confidence: float
    prediction_date: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly prediction data."""
        return {
            "symbol": self.symbol,
            "prediction": self.prediction,
            "probabilities": self.probabilities.copy(),
            "confidence": self.confidence,
            "prediction_date": self.prediction_date,
        }


def _is_excluded_feature(column: str) -> bool:
    """Identify metadata, labels, and future-derived columns."""
    normalized = column.strip().casefold()
    return (
        normalized in {"date", "symbol", "target", "future_return_5d"}
        or FUTURE_NAME_PATTERN.search(column) is not None
    )


def load_metadata(path: Path) -> tuple[list[str], dict[int, str]]:
    """Load and validate training-time feature order and label mapping."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training metadata was not found: {path}. "
            "Feature columns and class labels cannot be safely inferred."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read training metadata from {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Training metadata must be a JSON object: {path}")

    feature_columns = metadata.get("feature_columns")
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or any(not isinstance(column, str) or not column.strip() for column in feature_columns)
    ):
        raise ValueError(f"Training metadata has no valid feature_columns list: {path}")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Training metadata contains duplicate feature names")
    excluded = [column for column in feature_columns if _is_excluded_feature(column)]
    if excluded:
        raise ValueError(
            "Training metadata includes prohibited metadata or future-derived "
            f"features: {excluded}"
        )

    raw_mapping = metadata.get("id_to_label")
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Training metadata is missing the id_to_label mapping: {path}")
    if any(not isinstance(label, str) for label in raw_mapping.values()):
        raise ValueError("Training metadata has an invalid class label")
    try:
        id_to_label = {int(class_id): label for class_id, label in raw_mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Training metadata has an invalid class ID mapping") from exc
    if (
        len(id_to_label) != len(raw_mapping)
        or set(id_to_label.values()) != EXPECTED_LABELS
    ):
        raise ValueError(
            "Training metadata must map exactly three class IDs to BUY, HOLD, and SELL"
        )
    return feature_columns, id_to_label


def load_model(path: Path, feature_columns: list[str], id_to_label: dict[int, str]) -> Any:
    """Load the trained model and verify its available schema metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {path}")
    try:
        model = joblib.load(path)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
    ) as exc:
        raise RuntimeError(f"Could not load trained model from {path}: {exc}") from exc

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ValueError(f"Model at {path} must support predict and predict_proba")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != feature_columns:
        raise ValueError(
            "Model feature names/order do not match training metadata. "
            "Use the metadata created with this exact model."
        )
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(feature_columns):
        raise ValueError(
            f"Model expects {model_feature_count} features, but metadata lists "
            f"{len(feature_columns)}."
        )

    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError("Model does not expose classes_; probability order cannot be verified")
    raw_class_ids = list(model_classes)
    try:
        class_ids = [int(class_id) for class_id in raw_class_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("Model contains an invalid class ID") from exc
    if any(raw_id != class_id for raw_id, class_id in zip(raw_class_ids, class_ids)):
        raise ValueError("Model contains a non-integer class ID")
    if len(set(class_ids)) != len(class_ids) or set(class_ids) != set(id_to_label):
        raise ValueError(
            "Model class IDs do not match the BUY/HOLD/SELL IDs in training metadata."
        )
    return model


def load_latest_feature_row(
    path: Path, symbol: str, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Select and convert the latest feature row for a symbol."""
    if not path.is_file():
        raise FileNotFoundError(f"Stock features file was not found: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read stock features from {path}: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(f"Stock features file has duplicate column names: {path}")
    missing_columns = [
        column
        for column in ("Date", "Symbol", *feature_columns)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Stock features file is missing required training columns: "
            f"{missing_columns}"
        )

    requested_symbol = symbol.strip().casefold()
    symbols = frame["Symbol"].astype("string").str.strip()
    selected = frame.loc[symbols.str.casefold().eq(requested_symbol)].copy()
    if selected.empty:
        available = sorted(symbols.dropna().unique().tolist())
        raise StockSymbolNotFoundError(
            f"Stock symbol {symbol.strip().upper()!r} was not found in {path}. "
            f"Available symbols include: {', '.join(available[:10]) or 'none'}"
        )

    selected["Date"] = pd.to_datetime(selected["Date"], errors="coerce").dt.normalize()
    selected = selected.dropna(subset=["Date"])
    if selected.empty:
        raise ValueError(f"No valid prediction dates are available for {symbol.strip().upper()}")
    latest_date = selected["Date"].max()
    latest_rows = selected.loc[selected["Date"].eq(latest_date)]
    if len(latest_rows) != 1:
        raise ValueError(
            f"Multiple rows exist for {symbol.strip().upper()} on {latest_date.date()}; "
            "cannot choose an unambiguous latest observation."
        )

    matrix = prepare_feature_matrix(latest_rows, feature_columns)
    if matrix.isna().all(axis=1).iloc[0]:
        raise ValueError(
            f"All training features are missing for {symbol.strip().upper()} "
            f"on {latest_date.date()}."
        )
    return matrix, latest_date


def predict_stock(
    symbol: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> PredictionResult:
    """Predict BUY/HOLD/SELL for the latest available row of one stock."""
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("Stock symbol cannot be empty")

    feature_columns, id_to_label = load_metadata(metadata_path)
    model = load_model(model_path, feature_columns, id_to_label)
    feature_row, prediction_date = load_latest_feature_row(
        features_path, normalized_symbol, feature_columns
    )

    try:
        probabilities = np.asarray(model.predict_proba(feature_row), dtype=float)
        predictions = np.asarray(model.predict(feature_row)).reshape(-1)
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Model prediction failed for {normalized_symbol}: {exc}") from exc

    model_classes = [int(class_id) for class_id in model.classes_]
    if probabilities.shape != (1, len(model_classes)):
        raise ValueError(
            "Model returned probabilities with an unexpected shape: "
            f"{probabilities.shape}"
        )
    row_probabilities = probabilities[0]
    if (
        not np.isfinite(row_probabilities).all()
        or (row_probabilities < 0).any()
        or (row_probabilities > 1).any()
        or not np.isclose(row_probabilities.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("Model returned invalid class probabilities")
    if len(predictions) != 1:
        raise ValueError("Model returned an unexpected number of predictions")

    predicted_value = predictions[0]
    try:
        predicted_id = int(predicted_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Model returned an invalid predicted class: {predicted_value!r}") from exc
    if predicted_value != predicted_id or predicted_id not in id_to_label:
        raise ValueError(f"Model returned an unknown predicted class: {predicted_value!r}")

    probability_by_id = dict(zip(model_classes, row_probabilities, strict=True))
    probability_by_label = {
        label: float(probability_by_id[class_id])
        for class_id, label in id_to_label.items()
    }
    return PredictionResult(
        symbol=normalized_symbol,
        prediction=id_to_label[predicted_id],
        probabilities=probability_by_label,
        confidence=probability_by_label[id_to_label[predicted_id]],
        prediction_date=prediction_date.date().isoformat(),
    )


def format_prediction(result: PredictionResult) -> str:
    """Format the result for a terminal without coupling inference to display."""
    return "\n".join(
        [
            f"Stock: {result.symbol}",
            f"Prediction: {result.prediction}",
            *(f"{label} Probability: {result.probabilities[label]:.0%}" for label in ("BUY", "HOLD", "SELL")),
            f"Confidence: {result.confidence:.0%}",
            f"Prediction Date: {result.prediction_date}",
            "",
            "Research-based model prediction; not a guaranteed investment recommendation.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbol",
        nargs="?",
        help="Indian stock symbol, for example RELIANCE, TCS, INFY, or SBIN",
    )
    args = parser.parse_args(argv)
    try:
        symbol = args.symbol if args.symbol is not None else input("Enter stock symbol: ")
        result = predict_stock(symbol)
    except (EOFError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_prediction(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
