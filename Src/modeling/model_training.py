"""Train and evaluate a leakage-aware multiclass XGBoost stock classifier.

Run from the project directory:
	python model_training.py

XGBoost handles missing feature values natively. This script converts infinite
values to missing values and never fits preprocessing on validation or test data.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from feature_preprocessing import prepare_feature_matrix


ROOT_DIR: Final[Path] = Path(__file__).resolve().parent
DEFAULT_INPUT: Final[Path] = ROOT_DIR / "training_dataset.csv"
DEFAULT_MODEL_PATH: Final[Path] = ROOT_DIR / "models" / "xgboost_model.pkl"
DEFAULT_REPORT_DIR: Final[Path] = ROOT_DIR / "reports"
SEED: Final[int] = 42
PURGE_TRADING_DAYS: Final[int] = 5
LABEL_TO_ID: Final[dict[str, int]] = {"BUY": 0, "HOLD": 1, "SELL": 2}
ID_TO_LABEL: Final[dict[int, str]] = {value: key for key, value in LABEL_TO_ID.items()}
METADATA_COLUMNS: Final[set[str]] = {"date", "symbol"}
EXPLICITLY_EXCLUDED_COLUMNS: Final[set[str]] = {
	"target",
	"future_return_5d",
}
FUTURE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
	r"future|forward|lead|next|target|label|outcome|ahead|t\s*\+\s*\d",
	flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DataSplits:
	"""Date-disjoint model partitions and the omitted purge dates."""

	train: pd.DataFrame
	validation: pd.DataFrame
	test: pd.DataFrame
	purged_dates: tuple[pd.Timestamp, ...]


def load_dataset(path: Path) -> pd.DataFrame:
	"""Load the CSV, validate its required fields, and normalize dates/labels."""
	if not path.is_file():
		raise FileNotFoundError(f"Training dataset was not found: {path}")
	try:
		frame = pd.read_csv(path)
	except (OSError, pd.errors.ParserError, UnicodeError) as exc:
		raise RuntimeError(f"Could not read training dataset {path}: {exc}") from exc

	frame.columns = [str(column).strip() for column in frame.columns]
	required = {"Date", "Symbol", "Target"}
	missing = required.difference(frame.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
	if frame.empty:
		raise ValueError("Training dataset contains no rows")

	frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
	frame["Symbol"] = frame["Symbol"].astype("string").str.strip()
	frame["Target"] = frame["Target"].astype("string").str.strip().str.upper()
	frame = frame.dropna(subset=["Date", "Symbol", "Target"])
	frame = frame.loc[frame["Symbol"].ne("")].copy()
	unknown_labels = sorted(set(frame["Target"].unique()).difference(LABEL_TO_ID))
	if unknown_labels:
		raise ValueError(f"Unexpected Target labels: {unknown_labels}; expected BUY, HOLD, SELL")
	if frame.duplicated(["Symbol", "Date"]).any():
		raise ValueError("Dataset contains duplicate Symbol-Date rows")
	if set(frame["Target"].unique()) != set(LABEL_TO_ID):
		raise ValueError("Dataset must contain BUY, HOLD, and SELL labels")
	return frame.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)


def inspect_dataset(frame: pd.DataFrame) -> None:
	"""Print dataset dimensions, coverage, target balance, and missingness."""
	print(f"Rows: {len(frame):,} | Columns: {len(frame.columns)}")
	print(f"Stocks: {frame['Symbol'].nunique()} | Trading dates: {frame['Date'].nunique():,}")
	print(f"Date range: {frame['Date'].min().date()} to {frame['Date'].max().date()}")
	print("Target counts:")
	print(frame["Target"].value_counts().reindex(LABEL_TO_ID, fill_value=0).to_string())
	missing = frame.isna().sum().sort_values(ascending=False)
	missing = missing.loc[missing.gt(0)]
	print("Columns with missing values:")
	print(missing.to_string() if not missing.empty else "None")


def chronological_purged_split(
	frame: pd.DataFrame,
	purge_days: int = PURGE_TRADING_DAYS,
) -> DataSplits:
	"""Split by shared dates and omit purge_days before validation and test.

	The 70%/15%/15% boundaries are computed over unique trading dates. The
	five dates immediately before validation and test are omitted from all
	partitions, preventing five-session labels from crossing a boundary.
	"""
	if purge_days < 5:
		raise ValueError("purge_days must be at least five trading sessions")
	dates = pd.DatetimeIndex(frame["Date"].dropna().sort_values().unique())
	if len(dates) < (2 * purge_days + 10):
		raise ValueError("Not enough distinct dates for chronological splits and purge gaps")

	first_boundary = int(np.floor(len(dates) * 0.70))
	second_boundary = int(np.floor(len(dates) * 0.85))
	train_end = first_boundary - purge_days
	validation_start = first_boundary
	validation_end = second_boundary - purge_days
	test_start = second_boundary
	if train_end <= 0 or validation_end <= validation_start or test_start >= len(dates):
		raise ValueError("Dataset is too small to form non-empty purged partitions")

	train_dates = dates[:train_end]
	first_gap = dates[train_end:first_boundary]
	validation_dates = dates[validation_start:validation_end]
	second_gap = dates[validation_end:second_boundary]
	test_dates = dates[test_start:]
	splits = DataSplits(
		train=frame.loc[frame["Date"].isin(train_dates)].copy(),
		validation=frame.loc[frame["Date"].isin(validation_dates)].copy(),
		test=frame.loc[frame["Date"].isin(test_dates)].copy(),
		purged_dates=tuple(first_gap) + tuple(second_gap),
	)
	if any(part.empty for part in (splits.train, splits.validation, splits.test)):
		raise ValueError("At least one chronological partition has no rows")
	partitions = [set(part["Date"].unique()) for part in (splits.train, splits.validation, splits.test)]
	if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
		raise AssertionError("Date overlap detected across data partitions")
	if len(first_gap) < purge_days or len(second_gap) < purge_days:
		raise AssertionError("A purge gap is shorter than requested")
	return splits


def is_future_derived_column(column: str, target_column: str = "Target") -> bool:
	"""Return true for metadata, label, explicit future, or future-named fields."""
	normalized = column.strip().casefold()
	return (
		normalized in METADATA_COLUMNS
		or normalized in EXPLICITLY_EXCLUDED_COLUMNS
		or normalized == target_column.casefold()
		or FUTURE_NAME_PATTERN.search(column) is not None
	)


def make_model_matrices(
	splits: DataSplits,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
	"""Build numeric predictor matrices; infer usable columns from training only."""
	candidate_columns = [
		column for column in splits.train.columns if not is_future_derived_column(column)
	]
	if not candidate_columns:
		raise ValueError("No candidate feature columns remain after leakage exclusions")

	train_numeric = splits.train[candidate_columns].apply(pd.to_numeric, errors="coerce")
	# Feature availability is decided from training rows only, not validation/test.
	feature_columns = [column for column in candidate_columns if train_numeric[column].notna().any()]
	if not feature_columns:
		raise ValueError("No numeric feature columns have usable values in the training partition")

	return (
		prepare_feature_matrix(splits.train, feature_columns),
		prepare_feature_matrix(splits.validation, feature_columns),
		prepare_feature_matrix(splits.test, feature_columns),
		feature_columns,
	)


def encode_labels(labels: pd.Series) -> np.ndarray:
	"""Encode labels using the persistent BUY=0, HOLD=1, SELL=2 mapping."""
	encoded = labels.map(LABEL_TO_ID)
	if encoded.isna().any():
		raise ValueError("Encountered an unknown or missing class label")
	return encoded.to_numpy(dtype=np.int32)


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
	"""Build a 3x3 confusion matrix in the fixed class-id order."""
	matrix = np.zeros((len(LABEL_TO_ID), len(LABEL_TO_ID)), dtype=np.int64)
	np.add.at(matrix, (y_true, y_pred), 1)
	return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
	"""Calculate accuracy, class precision/recall/F1 and macro/weighted averages."""
	matrix = confusion_matrix_fixed(y_true, y_pred)
	support = matrix.sum(axis=1)
	predicted_count = matrix.sum(axis=0)
	true_positive = np.diag(matrix).astype(float)
	precision = np.divide(true_positive, predicted_count, out=np.zeros(3), where=predicted_count != 0)
	recall = np.divide(true_positive, support, out=np.zeros(3), where=support != 0)
	f1 = np.divide(
		2 * precision * recall,
		precision + recall,
		out=np.zeros(3),
		where=(precision + recall) != 0,
	)
	accuracy = float(true_positive.sum() / max(1, matrix.sum()))
	rows: list[dict[str, Any]] = []
	for label, class_id in LABEL_TO_ID.items():
		rows.append(
			{
				"class": label,
				"precision": precision[class_id],
				"recall": recall[class_id],
				"f1_score": f1[class_id],
				"support": int(support[class_id]),
				"accuracy": np.nan,
			}
		)
	rows.extend(
		[
			{
				"class": "macro avg",
				"precision": float(precision.mean()),
				"recall": float(recall.mean()),
				"f1_score": float(f1.mean()),
				"support": int(support.sum()),
				"accuracy": accuracy,
			},
			{
				"class": "weighted avg",
				"precision": float(np.average(precision, weights=support)) if support.sum() else 0.0,
				"recall": float(np.average(recall, weights=support)) if support.sum() else 0.0,
				"f1_score": float(np.average(f1, weights=support)) if support.sum() else 0.0,
				"support": int(support.sum()),
				"accuracy": accuracy,
			},
		]
	)
	return pd.DataFrame(rows), matrix


def parameter_candidates() -> list[dict[str, Any]]:
	"""Small reproducible search space; validation selects among these candidates."""
	return [
		{
			"n_estimators": estimators,
			"max_depth": depth,
			"learning_rate": 0.05,
			"class_weight": class_weight,
		}
		for estimators, depth in ((200, 3), (300, 3), (200, 5), (300, 5))
		for class_weight in ("none", "balanced")
	]


def make_sample_weights(labels: np.ndarray, strategy: str) -> np.ndarray | None:
	"""Calculate optional inverse-frequency weights using only the fit labels."""
	if strategy == "none":
		return None
	if strategy != "balanced":
		raise ValueError(f"Unsupported class weighting strategy: {strategy}")
	class_counts = np.bincount(labels, minlength=len(LABEL_TO_ID))
	present = class_counts > 0
	class_weights = np.zeros(len(LABEL_TO_ID), dtype=float)
	class_weights[present] = len(labels) / (int(present.sum()) * class_counts[present])
	return class_weights[labels]


def tune_on_validation(
	x_train: pd.DataFrame,
	y_train: np.ndarray,
	x_validation: pd.DataFrame,
	y_validation: np.ndarray,
	seed: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
	"""Train candidate models on train and select the best macro-F1 on validation."""
	try:
		from xgboost import XGBClassifier
	except ImportError as exc:
		raise RuntimeError(
			"XGBoost is required. Install it in the selected environment with: "
			"python -m pip install xgboost"
		) from exc

	trial_rows: list[dict[str, Any]] = []
	best_params: dict[str, Any] | None = None
	best_score = -1.0
	for candidate in parameter_candidates():
		model_params = {key: value for key, value in candidate.items() if key != "class_weight"}
		sample_weight = make_sample_weights(y_train, candidate["class_weight"])
		model = XGBClassifier(
			objective="multi:softprob",
			num_class=len(LABEL_TO_ID),
			eval_metric="mlogloss",
			tree_method="hist",
			random_state=seed,
			n_jobs=-1,
			subsample=0.9,
			colsample_bytree=0.9,
			reg_lambda=1.0,
			**model_params,
		)
		model.fit(
			x_train,
			y_train,
			sample_weight=sample_weight,
			eval_set=[(x_validation, y_validation)],
			verbose=False,
		)
		predictions = model.predict(x_validation).astype(np.int32)
		metrics, _ = classification_metrics(y_validation, predictions)
		macro = metrics.loc[metrics["class"].eq("macro avg")].iloc[0]
		score = float(macro["f1_score"])
		trial_rows.append(
			{
				**candidate,
				"validation_accuracy": float(macro["accuracy"]),
				"validation_macro_f1": score,
			}
		)
		if score > best_score:
			best_score = score
			best_params = candidate.copy()

	if best_params is None:
		raise RuntimeError("Hyperparameter search did not produce a candidate model")
	return best_params, pd.DataFrame(trial_rows).sort_values(
		["validation_macro_f1", "validation_accuracy"], ascending=False
	).reset_index(drop=True)


def train_final_model(
	x_train: pd.DataFrame,
	y_train: np.ndarray,
	x_validation: pd.DataFrame,
	y_validation: np.ndarray,
	params: dict[str, Any],
	seed: int = SEED,
) -> Any:
	"""Refit the chosen configuration on train+validation after selection."""
	from xgboost import XGBClassifier

	model_params = {key: value for key, value in params.items() if key != "class_weight"}
	model = XGBClassifier(
		objective="multi:softprob",
		num_class=len(LABEL_TO_ID),
		eval_metric="mlogloss",
		tree_method="hist",
		random_state=seed,
		n_jobs=-1,
		subsample=0.9,
		colsample_bytree=0.9,
		reg_lambda=1.0,
		**model_params,
	)
	x_fit = pd.concat([x_train, x_validation], axis=0, ignore_index=True)
	y_fit = np.concatenate([y_train, y_validation])
	model.fit(
		x_fit,
		y_fit,
		sample_weight=make_sample_weights(y_fit, params["class_weight"]),
		verbose=False,
	)
	return model


def save_reports(
	report_dir: Path,
	model_path: Path,
	model: Any,
	feature_columns: list[str],
	params: dict[str, Any],
	tuning_results: pd.DataFrame,
	metrics: pd.DataFrame,
	matrix: np.ndarray,
	y_test: np.ndarray,
	y_pred: np.ndarray,
	split_summary: dict[str, Any],
) -> None:
	"""Persist the model and API-friendly metadata/reports."""
	model_path.parent.mkdir(parents=True, exist_ok=True)
	report_dir.mkdir(parents=True, exist_ok=True)
	try:
		with model_path.open("wb") as model_file:
			pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
	except OSError as exc:
		raise RuntimeError(f"Could not save model to {model_path}: {exc}") from exc

	metrics.to_csv(report_dir / "evaluation_metrics.csv", index=False)
	tuning_results.to_csv(report_dir / "validation_tuning.csv", index=False)
	confusion = pd.DataFrame(
		matrix,
		index=[f"actual_{label}" for label in LABEL_TO_ID],
		columns=[f"predicted_{label}" for label in LABEL_TO_ID],
	)
	confusion.to_csv(report_dir / "confusion_matrix.csv", index=True, index_label="actual")
	importance = pd.DataFrame(
		{
			"feature": feature_columns,
			"importance_gain": model.feature_importances_,
		}
	).sort_values("importance_gain", ascending=False)
	importance.to_csv(report_dir / "feature_importance.csv", index=False)

	actual_counts = pd.Series(y_test).map(ID_TO_LABEL).value_counts().reindex(LABEL_TO_ID, fill_value=0)
	predicted_counts = pd.Series(y_pred).map(ID_TO_LABEL).value_counts().reindex(LABEL_TO_ID, fill_value=0)
	pd.DataFrame(
		{
			"class": list(LABEL_TO_ID),
			"actual_count": actual_counts.to_numpy(),
			"predicted_count": predicted_counts.to_numpy(),
		}
	).to_csv(report_dir / "class_distribution.csv", index=False)

	metadata = {
		"label_to_id": LABEL_TO_ID,
		"id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
		"feature_columns": feature_columns,
		"best_parameters": params,
		"random_seed": SEED,
		"purge_trading_days": PURGE_TRADING_DAYS,
		"split": split_summary,
	}
	(report_dir / "class_label_mapping.json").write_text(
		json.dumps(metadata, indent=2), encoding="utf-8"
	)


def train_and_evaluate(
	input_path: Path = DEFAULT_INPUT,
	model_path: Path = DEFAULT_MODEL_PATH,
	report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
	"""Execute load, split, tune, final fit, test evaluation, and artifact export."""
	frame = load_dataset(input_path)
	inspect_dataset(frame)
	splits = chronological_purged_split(frame)
	x_train, x_validation, x_test, feature_columns = make_model_matrices(splits)
	y_train = encode_labels(splits.train["Target"])
	y_validation = encode_labels(splits.validation["Target"])
	y_test = encode_labels(splits.test["Target"])

	print(
		"Split rows: "
		f"train={len(splits.train):,}, validation={len(splits.validation):,}, "
		f"test={len(splits.test):,}; purged trading dates={len(splits.purged_dates)}"
	)
	print(f"Predictor columns ({len(feature_columns)}): {feature_columns}")

	best_params, tuning_results = tune_on_validation(
		x_train, y_train, x_validation, y_validation
	)
	model = train_final_model(
		x_train, y_train, x_validation, y_validation, best_params
	)
	y_pred = model.predict(x_test).astype(np.int32)
	metrics, matrix = classification_metrics(y_test, y_pred)

	train_dates = sorted(splits.train["Date"].unique())
	validation_dates = sorted(splits.validation["Date"].unique())
	test_dates = sorted(splits.test["Date"].unique())
	split_summary = {
		"train_date_range": [str(train_dates[0].date()), str(train_dates[-1].date())],
		"validation_date_range": [str(validation_dates[0].date()), str(validation_dates[-1].date())],
		"test_date_range": [str(test_dates[0].date()), str(test_dates[-1].date())],
		"train_rows": len(splits.train),
		"validation_rows": len(splits.validation),
		"test_rows": len(splits.test),
		"purged_trading_dates": len(splits.purged_dates),
	}
	save_reports(
		report_dir,
		model_path,
		model,
		feature_columns,
		best_params,
		tuning_results,
		metrics,
		matrix,
		y_test,
		y_pred,
		split_summary,
	)

	print("Best validation parameters:")
	print(json.dumps(best_params, indent=2))
	print("Test metrics:")
	print(metrics.to_string(index=False))
	print("Test confusion matrix (rows=actual, columns=predicted; BUY, HOLD, SELL):")
	print(matrix)
	print(f"Saved model: {model_path}")
	print(f"Saved reports: {report_dir}")
	return {"model": model, "metrics": metrics, "confusion_matrix": matrix}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
	args = parser.parse_args()
	try:
		train_and_evaluate(args.input, args.model_path, args.report_dir)
	except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
		parser.exit(status=1, message=f"Training failed: {exc}\n")


if __name__ == "__main__":
	main()
