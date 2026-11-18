import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
src_dir = root / "data" / "market" / "india_vix data"
out_file = root / "data" / "market" / "india_vix.csv"

csv_files = sorted(src_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No India VIX CSV files found in {src_dir}")

frames = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
combined.to_csv(out_file, index=False)

print(f"Merged {len(csv_files)} India VIX files into {out_file} with {len(combined)} rows.")
