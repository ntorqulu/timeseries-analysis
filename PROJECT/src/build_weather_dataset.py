import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure the src directory is on sys.path so that `from utils.loader import ...`
# works regardless of the working directory from which this script is invoked.
_src = str(Path(__file__).resolve().parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from utils.loader import load_weather


def build_hourly_weather_dataset():
    weather_df = load_weather()
    if weather_df.empty:
        raise FileNotFoundError("No weather rows found in data/dynamic/weather.")

    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    weather_df["hour_trunc"] = weather_df["timestamp"].dt.floor("h")

    hourly = (
        weather_df.groupby("hour_trunc", as_index=False)
        .agg(
            raw_observation_count=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            temperature=("temperature", "mean"),
            precipitation=("precipitation", "mean"),
            windspeed=("windspeed", "mean"),
            weathercode=("weathercode", "last"),
            cloudcover=("cloudcover", "mean"),
        )
    )

    full_hour_index = pd.DataFrame(
        {
            "hour_trunc": pd.date_range(
                hourly["hour_trunc"].min(),
                hourly["hour_trunc"].max(),
                freq="h",
            )
        }
    )

    hourly = full_hour_index.merge(hourly, on="hour_trunc", how="left")
    hourly["has_raw_observation"] = hourly["raw_observation_count"].notna()
    hourly["raw_observation_count"] = hourly["raw_observation_count"].fillna(0).astype(int)

    # Forward-fill weather values so there are no NaN gaps in the output CSV.
    # Only ffill() is used (no bfill()) to avoid look-ahead leaking future
    # observations into past hours.
    fill_cols = ["temperature", "precipitation", "windspeed", "weathercode", "cloudcover"]
    hourly[fill_cols] = hourly[fill_cols].ffill()
    hourly["weathercode"] = hourly["weathercode"].astype("Int64")

    return hourly


def save_weather_dataset(hourly_weather_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_weather_df.to_csv(output_path, index=False)
    print(f"Weather dataset saved to: {output_path}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / "processed" / f"weather_hourly_{timestamp}.csv"

    hourly_weather_df = build_hourly_weather_dataset()
    save_weather_dataset(hourly_weather_df, output_path)

    print("\nShape:")
    print(hourly_weather_df.shape)

    print("\nColumns:")
    print(hourly_weather_df.columns.tolist())

    print("\nFirst rows:")
    print(hourly_weather_df.head())

    print("\nMissing values:")
    print(hourly_weather_df.isna().sum())

    print("\nHours without raw weather observations:")
    print((~hourly_weather_df["has_raw_observation"]).sum())