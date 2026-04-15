# Rodalies R2 Analysis Toolkit

This folder contains a small analysis toolkit for studying Rodalies delay data,
with a current focus on the core `R2` corridor.

The workflow is split into three Python scripts:

- `explore_rodalies.py`
- `delay_timeseries_analysis.py`
- `forecast_r2_delays.py`

The idea is:

1. explore the raw parquet files and build maps/diagnostics
2. construct clean delay time series
3. train a baseline forecasting model for stop-level delays


## Data Layout

The scripts expect the parquet files to be stored locally like this:

```text
data/
  lines/
    lines.parquet
  stations/
    stations.parquet
  journeys/
    journeys_YYYY_MM_DD.parquet
  timetables/
    timetables_YYYY_MM_DD.parquet
  trains/
    trains_YYYY_MM_DD.parquet
  weather/
    weather_YYYY_MM_DD.parquet
```

In this project, the typical data root is:

```bash
/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data
```


## Dependencies

These scripts use:

- `pandas`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `folium`
- `scikit-learn`

Typical interpreter on this machine:

```bash
/home/ryaan/anaconda3/bin/python3
```


## Script 1: `explore_rodalies.py`

Purpose:

- inspect dataset structure
- summarize schemas and file sizes
- explore delays, stations, weather, and correlations
- generate maps and exploratory plots

Main commands:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" inventory
```

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" profile
```

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" correlate --line-id R2
```

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" plot --line-id R2 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/plots"
```

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" route-map --line-id R2 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/plots"
```

Open only the segment maps:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" open-maps --line-id R2 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/plots" --kind segment
```

Open only the time-of-day segment maps:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" open-maps --line-id R2 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/plots" --kind time-segment
```

Important notes:

- the segment maps are now restricted to the official core `R2` corridor
- inbound and outbound maps are generated separately
- time-of-day submaps are available for:
  - `early_morning`
  - `morning_peak`
  - `midday`
  - `evening_peak`
  - `late_evening`


## Script 2: `delay_timeseries_analysis.py`

Purpose:

- turn stop-level delay events into evenly spaced time series
- compare line-level, direction-level, and station-level delay series
- inspect persistence, periodicity, and station behavior

What “15min”, “30min”, and “1h” mean:

- `15min`: one series value every 15 minutes
- `30min`: one series value every 30 minutes
- `1h`: one series value every hour

Each bucket contains aggregated delay statistics such as:

- average delay
- median delay
- number of observations

Useful commands:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/delay_timeseries_analysis.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --freq 15min profile
```

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/delay_timeseries_analysis.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --freq 15min --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/timeseries_outputs" plot
```

To follow specific stations instead of auto-selecting the top stations:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/delay_timeseries_analysis.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --freq 30min --stations "Barcelona-Sants,El Prat de Llobregat,Granollers Centre" plot
```

Outputs include:

- line-level series CSV
- direction-level series CSV
- station-level series CSV
- line-level delay plot
- direction comparison plot
- selected station comparison plot
- hourly profile plot
- weekday-hour heatmap
- autocorrelation plot


## Script 3: `forecast_r2_delays.py`

Purpose:

- build a supervised dataset at the stop-event level
- train a baseline model to predict arrival delay at each stop
- evaluate forecasting quality overall and by station
- produce a simple “next train at each station” scenario forecast

How the forecasting table is built:

- one row = one observed stop event for one train at one station
- target = `target_delay_minutes` from the stop-level delay data
- features include:
  - lagged delays from the same train
  - previous visited stations
  - route progress
  - station and direction labels
  - weekday and day type
  - weather
  - recent line, direction, and station delay context

Model:

- baseline `RandomForestRegressor`
- categorical features are one-hot encoded
- numeric features are median-imputed
- categorical features are mode-imputed

Important limitation:

- the “next train at each station” output is a scenario forecast based on the
  latest observed conditions in the historical data
- it is not a live online prediction feed

Useful commands:

Inspect the modeling table:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/forecast_r2_delays.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 profile
```

Train the baseline model and save outputs:

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/forecast_r2_delays.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --test-days 5 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/forecast_outputs" train
```

Outputs include:

- holdout predictions CSV
- overall metrics CSV
- station error CSV
- feature importance CSV
- next-station scenario forecast CSV
- actual-vs-predicted scatter plot
- station MAE bar plot
- feature importance bar plot


## Recommended Workflow

If you are starting from scratch, a practical order is:

1. inspect the data

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" profile
```

2. build maps and segment diagnostics

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/explore_rodalies.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" plot --line-id R2 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/plots"
```

3. study the delay series

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/delay_timeseries_analysis.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --freq 15min profile
```

4. train the forecasting baseline

```bash
/home/ryaan/anaconda3/bin/python3 "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/forecast_r2_delays.py" --data-root "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/data" --line-id R2 --test-days 5 --output-dir "/home/ryaan/Documents/Time Series Project/timeseries-analysis/R2 analysis/forecast_outputs" train
```


## Interpretation Notes

- negative delay values mean the train was early
- `R2` maps in this toolkit are restricted to the official core `R2` corridor
- `R2 Nord` and `R2 Sud` are intentionally excluded from the current segment logic
- for stop-level analysis, timetable-based delay is usually more reliable than the raw `trains` feed


## Outputs

Typical output folders:

- `plots/`
- `timeseries_outputs/`
- `forecast_outputs/`

These folders are created automatically when needed.
