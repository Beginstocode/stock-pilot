import unittest

import numpy as np
import pandas as pd

import data_cleaning as cleaner


class DataCleaningTests(unittest.TestCase):
    def test_date_parser_preserves_iso_and_parses_market_text_dates(self):
        parsed = cleaner.parse_dates(pd.Series(["2024-01-03", "25 Sep 2019"]))

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-01-03"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2019-09-25"))

    def test_stock_duplicates_invalid_ohlc_and_outliers_are_audited(self):
        stock = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"],
                "Symbol": ["AAA"] * 10,
                "Open": [9, 9, 10, 11, 9, 10, 11, 9, 10, 1000],
                "High": [10, 10, 9, 12, 10, 11, 12, 10, 11, 1001],
                "Low": [8, 8, 8, 10, 8, 9, 10, 8, 9, 999],
                "Close": [9, 9, 9, 11, 9, 10, 11, 9, 10, 1000],
                "Volume": [100] * 10,
                "SMA_200": [np.nan] * 10,
            }
        )

        audit = cleaner.clean_dataset(stock, "stock_prices", cleaner.Path("sample.csv"))

        self.assertEqual(audit.duplicate_rows, 1)
        self.assertEqual(audit.invalid_ohlc_rows, 1)
        self.assertEqual(len(audit.cleaned), 8)
        self.assertIn("Open", audit.outlier_cells)
        self.assertEqual(audit.cleaned["Open"].max(), 1000)
        self.assertTrue(audit.cleaned["SMA_200"].isna().all())

    def test_training_target_columns_are_future_derived_not_predictors(self):
        columns = ["Date", "Symbol", "Close", "Future_Return_5D", "Target", "Next_Week_Label"]
        excluded = [column for column in columns if cleaner.is_future_derived_column(column)]

        self.assertEqual(excluded, ["Future_Return_5D", "Target", "Next_Week_Label"])

    def test_infinite_and_invalid_numeric_values_become_missing(self):
        stock = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Symbol": ["AAA", "AAA"],
                "Open": [10, "bad"],
                "High": [11, 11],
                "Low": [9, 9],
                "Close": [10, 10],
                "Volume": [100, 100],
            }
        )
        stock["Volume"] = stock["Volume"].astype(float)
        stock.loc[0, "Volume"] = np.inf

        audit = cleaner.clean_dataset(stock, "stock_prices", cleaner.Path("sample.csv"))

        self.assertEqual(audit.infinite_cells.get("Volume"), 1)
        self.assertEqual(audit.invalid_numeric_cells.get("Open"), 1)
        self.assertEqual(len(audit.cleaned), 1)
        self.assertTrue(pd.isna(audit.cleaned.iloc[0]["Volume"]))


if __name__ == "__main__":
    unittest.main()