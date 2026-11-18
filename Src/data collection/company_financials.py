"""Collect quarterly financial statements for the configured working stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


STOCKS = [
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
]

QUARTERLY_COLUMNS = [
    "Ticker",
    "StatementType",
    "Period",
    "FiscalYear",
    "FiscalQuarter",
    "SourceCurrency",
    "Metric",
    "Value",
    "RawData",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "market" / "financials"
COMBINED_FILE = OUTPUT_DIR / "quarterly_results.csv"


def _normalise_statement(
    ticker: str,
    statement_type: str,
    statement: Any,
    source_currency: str,
) -> Any:
    """Convert a Yahoo Finance quarterly statement into the common row schema."""
    import pandas as pd

    if statement.empty:
        return pd.DataFrame(columns=QUARTERLY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for period in statement.columns:
        period_value = pd.Timestamp(period)
        fiscal_year = int(period_value.year)
        fiscal_quarter = int(period_value.quarter)
        period_text = period_value.strftime("%Y-%m-%d")

        for metric, value in statement[period].items():
            if pd.isna(value):
                continue
            rows.append(
                {
                    "Ticker": ticker,
                    "StatementType": statement_type,
                    "Period": period_text,
                    "FiscalYear": fiscal_year,
                    "FiscalQuarter": fiscal_quarter,
                    "SourceCurrency": source_currency,
                    "Metric": str(metric),
                    "Value": value,
                    "RawData": json.dumps(
                        {"metric": str(metric), "value": value},
                        default=str,
                    ),
                }
            )

    return pd.DataFrame(rows, columns=QUARTERLY_COLUMNS)


def collect_quarterly_results(
    stocks: list[str] = STOCKS,
    output_dir: Path | None = None,
) -> tuple[Any, list[str]]:
    """Download and normalize quarterly financial statements for configured stocks."""
    import pandas as pd
    import yfinance as yf

    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for ticker in stocks:
        try:
            company = yf.Ticker(ticker)
            source_currency = str(company.info.get("currency", "UNKNOWN"))
            statement_methods = {
                "IncomeStatement": company.quarterly_income_stmt,
                "BalanceSheet": company.quarterly_balance_sheet,
                "CashFlow": company.quarterly_cashflow,
            }

            stock_frames: list[pd.DataFrame] = []
            for statement_type, statement in statement_methods.items():
                if statement.empty:
                    continue
                stock_frames.append(
                    _normalise_statement(
                        ticker,
                        statement_type,
                        statement,
                        source_currency,
                    )
                )

            if not stock_frames:
                raise RuntimeError("All quarterly statement APIs returned empty data")

            stock_output = output_dir / f"{ticker.replace('.NS', '').replace('.BO', '')}_quarterly_results.csv"
            combined_stock = pd.concat(stock_frames, ignore_index=True)
            combined_stock.to_csv(stock_output, index=False)
            frames.append(combined_stock)
            print(f"Saved {stock_output} ({len(combined_stock)} rows)")
        except Exception as exc:
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            print(f"Failed {ticker}: {type(exc).__name__}: {exc}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUARTERLY_COLUMNS)
    combined = combined.sort_values(
        ["Ticker", "StatementType", "Period", "Metric"],
        kind="stable",
    ).reset_index(drop=True)
    combined.to_csv(output_dir / "quarterly_results.csv", index=False)
    return combined, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    results, failures = collect_quarterly_results(output_dir=args.output_dir)
    print(f"Collected {len(results)} quarterly rows for {len(STOCKS)} stocks")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
