"""Ensure prediction and training prepare identical feature matrices."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from Src.modeling.feature_preprocessing import prepare_feature_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PredictionPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processed_dir = PROJECT_ROOT / "data" / "processed"
        metadata = json.loads(
            (PROJECT_ROOT / "reports" / "class_label_mapping.json").read_text(
                encoding="utf-8"
            )
        )
        cls.feature_columns = metadata["feature_columns"]
        cls.training = pd.read_csv(
            cls.processed_dir / "training_dataset.csv", parse_dates=["Date"]
        )
        cls.inference = pd.read_csv(
            cls.processed_dir / "stock_features.csv", parse_dates=["Date"]
        )

    def test_shared_preprocessor_preserves_order_dtype_and_missing_value_rules(self):
        frame = pd.DataFrame(
            {
                "b": ["2.5", "not-a-number"],
                "a": [np.inf, None],
            }
        )
        result = prepare_feature_matrix(frame, ["a", "b"])

        self.assertEqual(list(result.columns), ["a", "b"])
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in result.dtypes))
        self.assertTrue(np.isnan(result.loc[0, "a"]))
        self.assertTrue(np.isnan(result.loc[1, "a"]))
        self.assertEqual(float(result.loc[0, "b"]), 2.5)
        self.assertTrue(np.isnan(result.loc[1, "b"]))

    def test_reliance_and_tcs_historical_rows_match_training_preparation(self):
        self.assertEqual(len(self.feature_columns), 41)

        for symbol in ("RELIANCE", "TCS"):
            with self.subTest(symbol=symbol):
                training_rows = (
                    self.training.loc[self.training["Symbol"].eq(symbol)]
                    .sort_values("Date")
                    .set_index("Date")
                )
                inference_rows = (
                    self.inference.loc[self.inference["Symbol"].eq(symbol)]
                    .sort_values("Date")
                    .set_index("Date")
                )
                common_dates = training_rows.index.intersection(inference_rows.index)
                self.assertGreater(len(common_dates), 0)

                expected = prepare_feature_matrix(
                    training_rows.loc[common_dates], self.feature_columns
                )
                actual = prepare_feature_matrix(
                    inference_rows.loc[common_dates], self.feature_columns
                )

                self.assertEqual(list(actual.columns), self.feature_columns)
                self.assertEqual(actual.shape[1], 41)
                self.assertTrue(
                    np.allclose(
                        expected.to_numpy(),
                        actual.to_numpy(),
                        rtol=1e-6,
                        atol=1e-6,
                        equal_nan=True,
                    ),
                    f"Training and inference features differ for {symbol}",
                )


if __name__ == "__main__":
    unittest.main()
