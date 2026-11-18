"""Cached XGBoost inference service for stock recommendations."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from Backend.App.config import MODEL_PATH, PROCESSED_DATA_DIR, REPORTS_DIR
from Src.prediction.predict import (
    PredictionResult,
    StockSymbolNotFoundError,
    load_latest_feature_row,
    load_metadata,
    load_model,
)

_EXPECTED_LABELS = {"BUY", "HOLD", "SELL"}
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,19}\Z")


class XGBoostModelService:
    """Load a trained model once per service instance and reuse it for inference."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        features_path: Path = PROCESSED_DATA_DIR / "stock_features.csv",
        metadata_path: Path = REPORTS_DIR / "class_label_mapping.json",
    ) -> None:
        self._feature_columns, self._id_to_label = load_metadata(metadata_path)
        if set(self._id_to_label.values()) != _EXPECTED_LABELS:
            raise ValueError(
                "Model metadata must map classes to BUY, HOLD, and SELL."
            )
        self._model = load_model(
            model_path, self._feature_columns, self._id_to_label
        )
        self._features_path = features_path

    def predict(self, symbol: str) -> PredictionResult:
        """Predict from the latest saved, training-format feature row for a symbol."""
        if not isinstance(symbol, str):
            raise ValueError("Stock symbol must be a string.")

        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("Stock symbol cannot be empty.")
        if not _SYMBOL_PATTERN.fullmatch(normalized_symbol):
            raise ValueError("Stock symbol contains invalid characters or is too long.")

        feature_row, prediction_date = load_latest_feature_row(
            self._features_path, normalized_symbol, self._feature_columns
        )

        try:
            probabilities = np.asarray(
                self._model.predict_proba(feature_row), dtype=float
            )
            predictions = np.asarray(self._model.predict(feature_row)).reshape(-1)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"Model prediction failed for {normalized_symbol}: {exc}"
            ) from exc

        model_classes = [int(class_id) for class_id in self._model.classes_]
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
            raise ValueError("Model returned invalid class probabilities.")
        if len(predictions) != 1:
            raise ValueError("Model returned an unexpected number of predictions.")

        predicted_value = predictions[0]
        try:
            predicted_id = int(predicted_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Model returned an invalid predicted class: {predicted_value!r}"
            ) from exc
        if predicted_value != predicted_id or predicted_id not in self._id_to_label:
            raise ValueError(
                f"Model returned an unknown predicted class: {predicted_value!r}"
            )

        probability_by_id = dict(zip(model_classes, row_probabilities, strict=True))
        probability_by_label = {
            label: float(probability_by_id[class_id])
            for class_id, label in self._id_to_label.items()
        }
        prediction = self._id_to_label[predicted_id]
        return PredictionResult(
            symbol=normalized_symbol,
            prediction=prediction,
            probabilities=probability_by_label,
            confidence=probability_by_label[prediction],
            prediction_date=prediction_date.date().isoformat(),
        )


@lru_cache(maxsize=1)
def get_xgboost_model_service() -> XGBoostModelService:
    """Provide one lazily initialized model service per application process."""
    return XGBoostModelService()


def get_model_service() -> XGBoostModelService:
    """FastAPI dependency provider for the cached XGBoost model service."""
    return get_xgboost_model_service()


__all__ = [
    "StockSymbolNotFoundError",
    "XGBoostModelService",
    "get_model_service",
    "get_xgboost_model_service",
]
