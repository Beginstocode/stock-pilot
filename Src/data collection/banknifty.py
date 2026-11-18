import csv
from datetime import datetime
from pathlib import Path


root = Path(__file__).resolve().parent
src_dir = root / "data" / "market" / "banknifty"
out_file = root / "data" / "market" / "banknifty.csv"

csv_files = sorted(src_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No Bank Nifty CSV files found in {src_dir}")

rows_by_date = {}
fieldnames = None
for csv_path in csv_files:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if fieldnames is None:
            fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items() if key}
            try:
                parsed_date = datetime.strptime(row["Date"], "%d %b %Y")
            except (KeyError, ValueError):
                continue
            row["Date"] = parsed_date.strftime("%Y-%m-%d")
            rows_by_date[parsed_date] = row

if not rows_by_date or not fieldnames:
    raise ValueError("No usable Bank Nifty data found to combine.")

with out_file.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_by_date[date] for date in sorted(rows_by_date))

print(f"Merged {len(csv_files)} Bank Nifty files into {out_file}")
print(f"Rows: {len(rows_by_date)} | Columns: {fieldnames}")
