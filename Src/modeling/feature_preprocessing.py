"""Shared feature matrix preparation for training and model inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def prepare_feature_matrix(
	frame: pd.DataFrame,
	feature_columns: Sequence[str],
) -> pd.DataFrame:
	"""Select ordered model features and apply the training-time conversion rules.

	Values are coerced to numeric, infinities become missing values, and the
	result is returned as float32. XGBoost handles those missing values natively.
	"""
	missing_columns = [column for column in feature_columns if column not in frame.columns]
	if missing_columns:
		raise ValueError(f"Feature data is missing model columns: {missing_columns}")

	matrix = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
	matrix = matrix.replace([np.inf, -np.inf], np.nan)
	return matrix.astype("float32")
