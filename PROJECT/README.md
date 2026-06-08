# Rodalies delay prediction

## Setup

Run the commands below from the `timeseries-analysis/PROJECT` directory.

```bash
conda env create -f environment.yml
conda activate jupyter_env
pip install pyarrow
```

Copy `credentials.json` into the project root. Then authenticate:

```bash
python src/utils/download.py
```

The first run opens a browser OAuth flow and catches the token in `token.json`. Subsequent runs refresh authomatically.

## Download

**Everything available on Drive:**

```bash
python src/utils/download.py
```

**Specific tables:**

```bash
python src/utils/download.py --tables trains timetables
```

**Date range:**

```bash
python src/utils/download.py --start 2026_03_15 --end 2026_03_21
```

**Re-download existing files:**

```bash
python src/utils/download.py --force
```

Downloaded files are saved to `data/` (gitignored).

## Build Dataset

After downloading the raw tables, run:

```bash
python src/build_dataset.py
```

This script loads the timetable history and station metadata, filters the data to the R1 line using station IDs derived from the trains table, builds the feature table used for modeling, and prints a summary of the resulting dataset.
The main output file is timestamped as `data/processed/features_YYYYMMDD_HHMMSS.csv`.
It also creates `data/processed/unmatched_direction_stations_YYYYMMDD_HHMMSS.csv` with station names that do not currently match the official timetable naming used for direction inference.

To inspect remaining missing directions in the latest built dataset, run:

```bash
python src/analyze_missing_directions.py
```

This creates a station-level summary CSV and a bar chart in `data/processed/`.

## Build Weather Dataset

To export the raw weather parquet files into an hourly dataset for exploration, run:

```bash
python src/build_weather_dataset.py
```

This creates a timestamped file like `data/processed/weather_hourly_YYYYMMDD_HHMMSS.csv`.
The output includes:

- one row per hour
- hourly averages for `temperature`, `precipitation`, `windspeed`, and `cloudcover`
- the last observed `weathercode` in each hour
- `raw_observation_count`
- `has_raw_observation`, so you can identify hours where the source weather feed had no rows

## EDA R1
