from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent
infile = base_dir / "Calendari_laboral_de_Catalunya.csv"
outfile = base_dir / "Calendari_laboral_de_Catalunya_2026.csv"

df = pd.read_csv(infile)

df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
df = df[df["Data"].dt.year == 2026].copy()

df["date"] = df["Data"].dt.strftime("%Y-%m-%d")
df["is_holiday"] = 1

df[["date", "is_holiday"]].drop_duplicates().to_csv(outfile, index=False)
