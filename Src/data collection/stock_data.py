import yfinance as yf
import pandas as pd
import os

stocks = [
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

fallbacks = {
    "TATAMOTORS.NS": ["TATAMOTORS.NS", "TATAMOTORS.BO", "TTM.NS", "TATAMOTORS", "TATAMOTORS.NSE"],
}

os.makedirs("data/market", exist_ok=True)

for ticker in stocks:
    candidates = fallbacks.get(ticker, [ticker])
    downloaded = False

    for symbol in candidates:
        print(f"Downloading {symbol}...")

        try:
            df = yf.download(
                symbol,
                start="2019-01-01",
                end="2026-09-23",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(f"  {symbol} failed: {exc}")
            continue

        if df is None or df.empty:
            print(f"  {symbol} returned no data.")
            continue

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Symbol"] = symbol.replace(".NS", "").replace(".BO", "")

        output_path = os.path.join("data", "market", f"{symbol.replace('.NS', '').replace('.BO', '')}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
        downloaded = True
        break

    if not downloaded:
        stale_path = os.path.join("data", "market", f"{ticker.replace('.NS', '').replace('.BO', '')}.csv")
        if os.path.exists(stale_path):
            os.remove(stale_path)
        print(f"No usable data found for {ticker}. Skipping CSV creation.")

print("Done!")