#!/usr/bin/env python3
"""
table_stats.py
==============
Computes the summary statistics for the LaTeX table in the report.

Reads the latest features_YYYYMMDD_HHMMSS.csv from data/processed/,
mirrors exactly the direction filter used in the modelling notebooks
(df_model = df[df["direction"].notna()]), and prints each row value.

Usage (from PROJECT root or any subdirectory):
    python table_stats.py
    python table_stats.py --features path/to/features_file.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ── path resolution ────────────────────────────────────────────────────────────
def find_project_root(start: Path, max_levels: int = 5) -> Path:
    for p in [start, *start.parents[:max_levels]]:
        if (p / "data" / "processed").exists():
            return p
    return start


def latest_features(processed_dir: Path) -> Path:
    candidates = sorted(
        f for f in processed_dir.glob("features_*.csv")
        if re.fullmatch(r"features_\d{8}_\d{6}\.csv", f.name)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped features CSV found in {processed_dir}\n"
            "Expected pattern: features_YYYYMMDD_HHMMSS.csv"
        )
    return candidates[-1]


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compute features dataset summary stats")
    p.add_argument(
        "--features", type=Path, default=None,
        help="Explicit path to features CSV (default: latest in data/processed/)"
    )
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.features:
        features_path = args.features.resolve()
    else:
        root = find_project_root(Path(__file__).resolve().parent)
        features_path = latest_features(root / "data" / "processed")

    print(f"Reading: {features_path.name}  …", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(features_path, low_memory=False)

    for col in ["planned_arrival_dt", "actual_arrival_dt", "hour_trunc", "service_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    total_raw = len(df)

    # "Missing direction" — reported on the FULL dataset before filtering
    missing_dir_n   = df["direction"].isna().sum()
    missing_dir_pct = missing_dir_n / total_raw * 100

    # ── direction filter (mirrors the notebooks exactly) ──────────────────────
    df_model = df[df["direction"].notna()].copy()

    # ── "weather from raw observation" ────────────────────────────────────────
    # The features pipeline sets a boolean flag when weather came from an actual
    # API observation rather than being interpolated / forward-filled.
    # Column name is weather_from_obs, weather_observed, or similar.
    weather_flag_col = next(
        (c for c in df_model.columns
         if "weather" in c.lower() and ("obs" in c.lower() or "raw" in c.lower() or "flag" in c.lower())),
        None
    )
    if weather_flag_col:
        weather_obs_pct = df_model[weather_flag_col].mean() * 100
        weather_str = f"{weather_obs_pct:.1f}%"
    else:
        # Fall back: rows where temperature is not NaN as a proxy
        weather_str = (
            f"{df_model['temperature'].notna().mean()*100:.1f}%  "
            f"(non-null temperature; no explicit flag column found)"
            if "temperature" in df_model.columns else "column not found"
        )

    # ── stats on direction-filtered table ─────────────────────────────────────
    total_rows       = len(df_model)
    unique_dates     = df_model["service_date"].nunique()
    unique_trains    = df_model["train_id"].nunique()
    unique_stations  = df_model["station_id"].nunique()

    # Average stops per train instance
    # = mean number of station rows each train_id appears in
    stops_per_train  = df_model.groupby("train_id")["station_id"].count().mean()

    td = df_model["target_delay"]
    td_mean   = td.mean()
    td_median = td.median()
    td_std    = td.std()
    td_gt10   = (td > 10).mean() * 100   # fraction in percent

    # ── print results ─────────────────────────────────────────────────────────
    SEP = "-" * 52
    print(SEP)
    print(f"{'Property':<40} {'Value':>10}")
    print(SEP)
    print(f"{'Total rows':<40} {total_rows:>10,}")
    print(f"{'Unique service dates':<40} {unique_dates:>10,}")
    print(f"{'Unique train instances':<40} {unique_trains:>10,}")
    print(f"{'Unique stations':<40} {unique_stations:>10,}")
    print(f"{'Avg stops per train instance':<40} {stops_per_train:>10.1f}")
    print(f"{'Mean target_delay (min)':<40} {td_mean:>10.2f}")
    print(f"{'Median target_delay (min)':<40} {td_median:>10.2f}")
    print(f"{'Std target_delay (min)':<40} {td_std:>10.2f}")
    print(f"{'Fraction delay > 10 min':<40} {td_gt10:>9.1f}%")
    print(f"{'Weather from raw observation':<40} {weather_str:>10}")
    print(f"{'Missing direction (of all rows)':<40} {missing_dir_n:>6,} ({missing_dir_pct:.1f}%)")
    print(SEP)

    # ── LaTeX-ready lines (paste directly into the table) ─────────────────────
    print("\nLaTeX-ready values:")
    print(f"  Total rows                        & {total_rows:,} \\\\")
    print(f"  Unique service dates              & {unique_dates} \\\\")
    print(f"  Unique train instances            & {unique_trains:,} \\\\")
    print(f"  Unique stations                   & {unique_stations} \\\\")
    print(f"  Average stops per train instance  & {stops_per_train:.1f} \\\\")
    print(f"  Mean target\\_delay               & {td_mean:.2f} min \\\\")
    print(f"  Median target\\_delay             & {td_median:.2f} min \\\\")
    print(f"  Std target\\_delay                & {td_std:.2f} min \\\\")
    print(f"  Fraction with delay $> 10$ min    & {td_gt10:.1f}\\% \\\\")
    print(f"  Weather from raw observation      & {weather_str} \\\\")
    print(f"  Missing direction                 & {missing_dir_n:,} ({missing_dir_pct:.1f}\\%) \\\\")

    # ── available columns (useful for debugging) ───────────────────────────────
    print(f"\nAll columns in df_model: {list(df_model.columns)}")


if __name__ == "__main__":
    main()