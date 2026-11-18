import unittest

import numpy as np
import pandas as pd

import feature_engineering as pipeline


def make_prices(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=len(closes), freq="B"),
            "Symbol": symbol,
            "Open": closes,
            "High": np.asarray(closes) + 1,
            "Low": np.asarray(closes) - 1,
            "Close": closes,
            "Adj Close": closes,
            "Volume": np.arange(100, 100 + len(closes)),
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_features_are_independent_by_symbol_and_target_rows_are_separate(self):
        prices = pd.concat(
            [make_prices("AAA", [100, 101, 102, 103, 104, 120, 121]),
             make_prices("BBB", [200, 201, 202, 203, 204, 205, 206])],
            ignore_index=True,
        )
        features, training = pipeline.build_datasets(prices)

        first_a = features.loc[features["Symbol"].eq("AAA")].iloc[0]
        first_b = features.loc[features["Symbol"].eq("BBB")].iloc[0]
        self.assertTrue(pd.isna(first_a["Daily_Return"]))
        self.assertTrue(pd.isna(first_b["Daily_Return"]))
        self.assertNotIn("Future_Return_5D", features.columns)
        self.assertNotIn("Target", features.columns)
        self.assertEqual(len(features), len(prices))
        self.assertEqual(len(training), len(prices) - 10)
        self.assertEqual(training.loc[training["Symbol"].eq("AAA"), "Target"].iloc[0], "BUY")

    def test_market_features_join_only_on_matching_dates(self):
        prices = make_prices("AAA", [100, 101, 102, 103, 104, 105, 106])
        nifty = pd.DataFrame(
            {"Date": ["2024-01-02", "2024-01-03"], "Close": [200, 220]}
        )
        vix = pd.DataFrame(
            {"Date": ["2024-01-02"], "Close": [15.0]}
        )
        features, _ = pipeline.build_datasets(prices, nifty=nifty, vix=vix)

        jan_2 = features.loc[features["Date"].eq(pd.Timestamp("2024-01-02"))].iloc[0]
        jan_3 = features.loc[features["Date"].eq(pd.Timestamp("2024-01-03"))].iloc[0]
        jan_4 = features.loc[features["Date"].eq(pd.Timestamp("2024-01-04"))].iloc[0]
        self.assertAlmostEqual(jan_3["NIFTY_Daily_Return"], 0.1)
        self.assertEqual(jan_2["India_VIX_Close"], 15.0)
        self.assertTrue(pd.isna(jan_4["NIFTY_Daily_Return"]))
        self.assertTrue(pd.isna(jan_3["India_VIX_Close"]))

    def test_duplicate_rows_and_infinite_values_are_cleaned(self):
        prices = make_prices("AAA", [100, 101, 102, 103, 104, 105, 106])
        prices["Open"] = prices["Open"].astype(float)
        prices.loc[1, "Open"] = np.inf
        duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
        features, training = pipeline.build_datasets(duplicated)

        self.assertFalse(features.duplicated(["Symbol", "Date"]).any())
        self.assertFalse(np.isinf(features.select_dtypes(include=[np.number])).any().any())
        self.assertTrue(pd.isna(features.loc[features["Date"].eq(pd.Timestamp("2024-01-02")), "Open"]).all())
        self.assertEqual(len(training), len(prices) - 5)

    def test_threshold_equality_is_hold(self):
        prices = make_prices("AAA", [100, 100, 100, 100, 100, 103, 100])
        exact_return = 103 / 100 - 1
        _, training = pipeline.build_datasets(
            prices, buy_threshold=exact_return, sell_threshold=exact_return
        )
        self.assertEqual(training.iloc[0]["Target"], "HOLD")

    def test_zero_close_does_not_create_infinite_target(self):
        prices = make_prices("AAA", [0, 100, 101, 102, 103, 104, 105])
        features, training = pipeline.build_datasets(prices)

        self.assertFalse(np.isinf(features.select_dtypes(include=[np.number])).any().any())
        self.assertFalse(np.isinf(training.select_dtypes(include=[np.number])).any().any())
        self.assertNotIn(pd.Timestamp("2024-01-01"), training["Date"].tolist())


if __name__ == "__main__":
    unittest.main()