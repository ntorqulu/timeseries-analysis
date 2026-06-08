# Rodalies delay prediction

This repository contains code and notebooks to build an hourly delay forecasting pipeline for the R1 Rodalies line. It includes data download helpers, dataset building, exploratory data analysis (EDA), baseline time-series models (SARIMA), and utilities for exporting fitted model parameters.

## Quickstart

Run the commands below from the `timeseries-analysis/PROJECT` directory to create the environment and install minimal extras:

```bash
conda env create -f environment.yml
conda activate jupyter_env
pip install pyarrow
```

Before running downloads, copy `credentials.json` (Google Drive credentials) into the project root and authenticate once:

```bash
python src/utils/download.py
```

The first run opens a browser OAuth flow and stores a refresh token in `token.json` for subsequent runs.

## Data download

Download everything from Drive with:

```bash
python src/utils/download.py
```

Or download specific tables:

```bash
python src/utils/download.py --tables trains timetables
```

Select a date range with `--start` and `--end` (format `YYYY_MM_DD`):

```bash
python src/utils/download.py --start 2026_03_15 --end 2026_03_21
```

Re-download existing files with `--force`.

Downloaded files are written under `data/` (this folder is gitignored).

## Build processed datasets

Create the main features table used by EDA and modeling:

```bash
python src/build_dataset.py
```

Outputs:

- `data/processed/features_YYYYMMDD_HHMMSS.csv` — main modeling table
- `data/processed/unmatched_direction_stations_YYYYMMDD_HHMMSS.csv` — station name mismatches

Inspect remaining missing directions:

```bash
python src/analyze_missing_directions.py
```

## Weather dataset

Convert raw weather parquet files into an hourly CSV for joins and EDA:

```bash
python src/build_weather_dataset.py
```

Outputs include hourly averages for `temperature`, `precipitation`, `windspeed`, `cloudcover`, a `weathercode`, and metadata columns like `raw_observation_count` and `has_raw_observation`.

## EDA and Notebooks

Open the EDA notebooks under `PROJECT/EDA/` for interactive exploration. Notable notebooks:

- `R1_data_exploration.ipynb` — data overview and cleaning checks
- `R1_features_latest_eda.ipynb` — feature distributions and correlations
- `R1_sarima_latest_features.ipynb` — SARIMA baseline model workflow (builds hourly series, stationarity tests, ACF/PACF, fits SARIMAX, forecasts, diagnostics)

Run notebooks using Jupyter or JupyterLab:

```bash
jupyter lab PROJECT/EDA
```

## Modeling: SARIMA baseline

The notebook `R1_sarima_latest_features.ipynb` builds hourly mean delay series for three delay definitions (`type_1_only`, `type_2_only`, `type_1_and_2`), reindexes to a complete hourly grid, interpolates overnight gaps, runs ADF/KPSS stationarity tests, inspects ACF/PACF up to lag 168, and fits a baseline SARIMAX specification.

Default fitting in the notebook uses SARIMA(1,0,1)(2,0,1,24) with `enforce_stationarity=False` and `enforce_invertibility=False`. The notebook also evaluates a seasonal-differenced alternative for `type_1_only` when diagnostics indicate a seasonal near-unit root.

Fitted parameters and a small JSON export are saved to `data/models/sarima_params.json` for downstream use.

## Reproducing the forecast

1. Build or locate the latest `data/processed/features_*.csv` file.
2. Open and run `PROJECT/EDA/R1_sarima_latest_features.ipynb` end-to-end.

The notebook keeps the last 72 hours as a holdout and reports MAE/RMSE versus a seasonal-naive (lag-168) baseline. Residual diagnostics and Ljung–Box tests are computed.
