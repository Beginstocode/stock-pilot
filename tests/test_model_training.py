from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import model_training as pipeline


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        training_dataset = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "training_dataset.csv"
        )
        cls.frame = pipeline.load_dataset(training_dataset)
        cls.splits = pipeline.chronological_purged_split(cls.frame)

    def test_splits_are_chronological_and_have_five_date_gaps(self):
        train_dates = pd.Index(self.splits.train["Date"].unique())
        validation_dates = pd.Index(self.splits.validation["Date"].unique())
        test_dates = pd.Index(self.splits.test["Date"].unique())

        self.assertLess(train_dates.max(), validation_dates.min())
        self.assertLess(validation_dates.max(), test_dates.min())
        self.assertEqual(len(self.splits.purged_dates), 10)
        self.assertEqual(len(set(train_dates) & set(validation_dates)), 0)
        self.assertEqual(len(set(validation_dates) & set(test_dates)), 0)

    def test_model_inputs_exclude_metadata_and_future_columns(self):
        _, _, _, columns = pipeline.make_model_matrices(self.splits)

        self.assertTrue(
            {"Date", "Symbol", "Future_Return_5D", "Target"}.isdisjoint(columns)
        )
        self.assertTrue(all(not pipeline.is_future_derived_column(c) for c in columns))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(self.splits.train[c]) for c in columns))

    def test_label_encoding_and_balanced_weights_are_stable(self):
        encoded = pipeline.encode_labels(pd.Series(["BUY", "HOLD", "SELL"]))
        np.testing.assert_array_equal(encoded, np.array([0, 1, 2]))
        weights = pipeline.make_sample_weights(np.array([0, 0, 1, 1, 1, 2]), "balanced")
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[2]), 2 / 3)
        self.assertAlmostEqual(float(weights[5]), 2.0)

    def test_confusion_matrix_keeps_all_three_classes(self):
        matrix = pipeline.confusion_matrix_fixed(
            np.array([0, 1, 2]), np.array([0, 2, 1])
        )
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(int(matrix.sum()), 3)
        self.assertEqual(int(matrix[0, 0]), 1)


if __name__ == "__main__":
    unittest.main()