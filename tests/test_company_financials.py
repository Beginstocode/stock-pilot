from pathlib import Path
import importlib.util
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Src"
    / "data collection"
    / "company_financials.py"
)
SPEC = importlib.util.spec_from_file_location("company_financials", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CompanyFinancialsTests(unittest.TestCase):
    def test_stock_list_contains_all_working_stocks(self):
        self.assertEqual(
            module.STOCKS,
            [
                "RELIANCE.NS",
                "TCS.NS",
                "INFY.NS",
                "HDFCBANK.NS",
                "ICICIBANK.NS",
                "SBIN.NS",
                "ITC.NS",
                "LT.NS",
                "BHARTIARTL.NS",
                "TATAMOTORS.NS",
            ],
        )

    def test_quarterly_schema_is_stable(self):
        required_columns = {
            "Ticker",
            "StatementType",
            "Period",
            "FiscalYear",
            "FiscalQuarter",
            "SourceCurrency",
            "Metric",
            "Value",
            "RawData",
        }
        self.assertTrue(required_columns.issubset(module.QUARTERLY_COLUMNS))


if __name__ == "__main__":
    unittest.main()
