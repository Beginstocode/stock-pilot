import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
src_dir = root / "data" / "market" / "stock data"
out_file = root / "data" / "market" / "all_stocks.csv"

csv_files = sorted(src_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No stock CSV files found in {src_dir}")

frames = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    frames.append(df)

if not frames:
    raise ValueError("No usable stock data found to combine.")

combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
combined.to_csv(out_file, index=False)

print(f"Merged {len(csv_files)} files into {out_file}")
print(f"Rows: {len(combined)} | Columns: {list(combined.columns)}")
