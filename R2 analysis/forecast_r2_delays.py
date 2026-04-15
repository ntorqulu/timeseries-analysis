#!/usr/bin/env python3
from __future__ import annotations

"""Train and evaluate a baseline forecaster for stop-level R2 delay prediction.

This script treats each observed stop event as one supervised-learning example.
The model predicts the arrival delay at that stop using only information that
would plausibly be available before the train reaches the station:

- recent delay history of the same train
- station and direction identifiers
- time-of-day and day-type context
- recent line / station delay context
- basic weather variables

It is intentionally a baseline: the goal is to create a clean, inspectable
dataset and a runnable forecasting workflow before trying more advanced models.
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib, seaborn, and scikit-learn."
    ) from exc

import delay_timeseries_analysis as delay_ts
import explore_rodalies as rod


PLOT_STYLE = "whitegrid"
RANDOM_STATE = 42
DEFAULT_TEST_DAYS = 5


def restrict_to_core_route(frame: pd.DataFrame, data_root: Path, line_id: str) -> pd.DataFrame:
    """Keep only the official stations for the requested line when available."""

    route = rod.official_route_station_order(data_root, line_id)
    if route.empty:
        return frame.copy()
    valid = route[["station_id", "route_index"]].copy()
    valid["station_id"] = valid["station_id"].astype("string")
    working = frame.copy()
    working["station_id"] = working["station_id"].astype("string")
    return working.merge(valid, on="station_id", how="inner")


def build_model_frame(data_root: Path, line_id: str) -> pd.DataFrame:
    """Create a stop-level modeling table with lagged and contextual features."""

    # Start from the cleaned stop-level frame that already contains weather,
    # delay fields, station names, weekday labels, and day-type labels.
    frame = delay_ts.prepare_timeseries_frame(data_root, line_id).copy()
    frame = restrict_to_core_route(frame, data_root, line_id)
    if frame.empty:
        raise SystemExit(f"No rows available for line '{line_id}' after route filtering")

    frame = frame.sort_values(["train_id", "timestamp_minute", "stop_sequence"]).copy()
    frame["service_date"] = frame["timestamp_minute"].dt.strftime("%Y-%m-%d")
    frame["target_delay_minutes"] = frame["delay_minutes"]
    frame = frame.dropna(subset=["target_delay_minutes"]).copy()
    frame["weekday_name"] = frame["timestamp_minute"].dt.day_name()
    frame["hour"] = frame["timestamp_minute"].dt.hour
    frame["minute"] = frame["timestamp_minute"].dt.minute
    frame["bucket_15min"] = frame["timestamp_minute"].dt.floor("15min")

    # Lag features from the same train capture delay propagation along the route.
    trip_keys = ["train_id", "service_date"]
    for lag in (1, 2, 3):
        frame[f"prev_delay_{lag}"] = frame.groupby(trip_keys, dropna=False)["target_delay_minutes"].shift(lag)
        frame[f"prev_station_{lag}"] = frame.groupby(trip_keys, dropna=False)["station_name"].shift(lag)
        frame[f"prev_time_gap_min_{lag}"] = (
            frame["timestamp_minute"] - frame.groupby(trip_keys, dropna=False)["timestamp_minute"].shift(lag)
        ).dt.total_seconds() / 60.0

    # Route progress makes the model aware of where the train is on the corridor.
    max_route_index = max(1, int(frame["route_index"].max()))
    frame["route_progress"] = frame["route_index"] / max_route_index

    # Recent delay context is built from the previous 15-minute bucket only.
    # This avoids leaking the current event into its own predictors.
    line_context = (
        frame.groupby("bucket_15min", dropna=False)["target_delay_minutes"]
        .mean()
        .shift(1)
        .rename("line_recent_delay")
        .reset_index()
    )
    direction_context = (
        frame.groupby(["direction", "bucket_15min"], dropna=False)["target_delay_minutes"]
        .mean()
        .groupby(level=0)
        .shift(1)
        .rename("direction_recent_delay")
        .reset_index()
    )
    station_context = (
        frame.groupby(["station_name", "bucket_15min"], dropna=False)["target_delay_minutes"]
        .mean()
        .groupby(level=0)
        .shift(1)
        .rename("station_recent_delay")
        .reset_index()
    )

    frame = frame.merge(line_context, on="bucket_15min", how="left")
    frame = frame.merge(direction_context, on=["direction", "bucket_15min"], how="left")
    frame = frame.merge(station_context, on=["station_name", "bucket_15min"], how="left")

    # Keep only the features we want to expose to the model and to the user.
    selected_columns = [
        "train_id",
        "service_date",
        "timestamp_minute",
        "station_id",
        "station_name",
        "direction",
        "day_type",
        "weekday_name",
        "hour",
        "minute",
        "stop_sequence",
        "route_index",
        "route_progress",
        "temperature",
        "precipitation",
        "windspeed",
        "cloudcover",
        "weathercode",
        "prev_delay_1",
        "prev_delay_2",
        "prev_delay_3",
        "prev_station_1",
        "prev_station_2",
        "prev_station_3",
        "prev_time_gap_min_1",
        "prev_time_gap_min_2",
        "prev_time_gap_min_3",
        "line_recent_delay",
        "direction_recent_delay",
        "station_recent_delay",
        "target_delay_minutes",
    ]
    return frame[selected_columns].reset_index(drop=True)


def split_train_test(frame: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the most recent days as test data to respect time ordering."""

    ordered_days = sorted(frame["service_date"].dropna().unique())
    if len(ordered_days) <= 1:
        raise SystemExit("Not enough distinct days to create a train/test split")
    holdout_days = ordered_days[-min(test_days, max(1, len(ordered_days) // 3)) :]
    train = frame[~frame["service_date"].isin(holdout_days)].copy()
    test = frame[frame["service_date"].isin(holdout_days)].copy()
    if train.empty or test.empty:
        raise SystemExit("Train/test split produced an empty partition")
    return train, test


def feature_columns() -> tuple[list[str], list[str]]:
    """List numeric and categorical features used by the baseline model."""

    numeric = [
        "hour",
        "minute",
        "stop_sequence",
        "route_index",
        "route_progress",
        "temperature",
        "precipitation",
        "windspeed",
        "cloudcover",
        "weathercode",
        "prev_delay_1",
        "prev_delay_2",
        "prev_delay_3",
        "prev_time_gap_min_1",
        "prev_time_gap_min_2",
        "prev_time_gap_min_3",
        "line_recent_delay",
        "direction_recent_delay",
        "station_recent_delay",
    ]
    categorical = [
        "station_name",
        "direction",
        "day_type",
        "weekday_name",
        "prev_station_1",
        "prev_station_2",
        "prev_station_3",
    ]
    return numeric, categorical


def build_pipeline() -> Pipeline:
    """Create the preprocessing + model pipeline.

    Random forest is a good first baseline here because it handles nonlinear
    feature interactions and mixed numeric/categorical inputs once we add
    one-hot encoding.
    """

    numeric_features, categorical_features = feature_columns()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """Train the baseline model and return predictions on the holdout set."""

    numeric_features, categorical_features = feature_columns()
    features = numeric_features + categorical_features
    pipeline = build_pipeline()
    pipeline.fit(train[features], train["target_delay_minutes"])

    scored = test.copy()
    scored["predicted_delay_minutes"] = pipeline.predict(test[features])
    scored["absolute_error"] = (scored["predicted_delay_minutes"] - scored["target_delay_minutes"]).abs()
    scored["residual"] = scored["predicted_delay_minutes"] - scored["target_delay_minutes"]
    return pipeline, scored


def metrics_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Compute overall and per-direction summary metrics."""

    rows = [
        {
            "scope": "overall",
            "mae": mean_absolute_error(scored["target_delay_minutes"], scored["predicted_delay_minutes"]),
            "rmse": mean_squared_error(
                scored["target_delay_minutes"], scored["predicted_delay_minutes"], squared=False
            ),
            "rows": len(scored),
        }
    ]
    for direction, part in scored.groupby("direction", dropna=False):
        if pd.isna(direction) or part.empty:
            continue
        rows.append(
            {
                "scope": f"direction:{direction}",
                "mae": mean_absolute_error(part["target_delay_minutes"], part["predicted_delay_minutes"]),
                "rmse": mean_squared_error(
                    part["target_delay_minutes"], part["predicted_delay_minutes"], squared=False
                ),
                "rows": len(part),
            }
        )
    return pd.DataFrame(rows)


def station_error_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE by station to see where the baseline struggles most."""

    return (
        scored.groupby("station_name", dropna=False)
        .agg(
            mae=("absolute_error", "mean"),
            rmse=("residual", lambda s: mean_squared_error([0] * len(s), s, squared=False)),
            rows=("absolute_error", "size"),
        )
        .reset_index()
        .sort_values(["mae", "rows"], ascending=[False, False])
    )


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Recover feature importances after preprocessing."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def latest_station_forecast(frame: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Forecast the 'next-train' delay per station using the latest observed context.

    This is not a live operational feed forecast. It is a scenario forecast:
    if the most recent observed conditions repeated for the next train reaching
    each station, this is the model's expected delay.
    """

    numeric_features, categorical_features = feature_columns()
    features = numeric_features + categorical_features
    latest = (
        frame.dropna(subset=["station_name"])
        .sort_values("timestamp_minute")
        .groupby("station_name", dropna=False)
        .tail(1)
        .copy()
    )
    latest["predicted_next_delay_minutes"] = pipeline.predict(latest[features])
    return latest[
        [
            "station_name",
            "timestamp_minute",
            "direction",
            "line_recent_delay",
            "station_recent_delay",
            "predicted_next_delay_minutes",
        ]
    ].sort_values("predicted_next_delay_minutes", ascending=False)


def save_scatter_plot(scored: pd.DataFrame, output_dir: Path, line_id: str) -> Path:
    """Save actual-vs-predicted plot for the holdout set."""

    sns.set_theme(style=PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=scored.sample(min(len(scored), 5000), random_state=RANDOM_STATE),
        x="target_delay_minutes",
        y="predicted_delay_minutes",
        hue="direction",
        alpha=0.45,
        ax=ax,
    )
    limits = [
        min(scored["target_delay_minutes"].min(), scored["predicted_delay_minutes"].min()),
        max(scored["target_delay_minutes"].max(), scored["predicted_delay_minutes"].max()),
    ]
    ax.plot(limits, limits, linestyle="--", color="#444444")
    ax.set_title(f"{line_id} holdout: actual vs predicted delay")
    ax.set_xlabel("Actual delay (minutes)")
    ax.set_ylabel("Predicted delay (minutes)")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_forecast_actual_vs_predicted.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_station_mae_plot(station_errors: pd.DataFrame, output_dir: Path, line_id: str) -> Path:
    """Save a station-level MAE ranking for the baseline model."""

    sns.set_theme(style=PLOT_STYLE)
    plot_frame = station_errors.head(12).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_frame, x="mae", y="station_name", ax=ax, color="#b91c1c")
    ax.set_title(f"{line_id} stations with highest forecasting MAE")
    ax.set_xlabel("MAE (minutes)")
    ax.set_ylabel("Station")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_forecast_station_mae.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_feature_importance_plot(importances: pd.DataFrame, output_dir: Path, line_id: str) -> Path:
    """Save the top feature importances from the random-forest baseline."""

    sns.set_theme(style=PLOT_STYLE)
    plot_frame = importances.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_frame, x="importance", y="feature", ax=ax, color="#1d4ed8")
    ax.set_title(f"{line_id} top forecasting features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_forecast_feature_importance.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def profile_report(data_root: Path, line_id: str, test_days: int) -> None:
    """Print dataset construction diagnostics before training."""

    frame = build_model_frame(data_root, line_id)
    train, test = split_train_test(frame, test_days)
    print(f"Line: {line_id}")
    print(f"Rows available for modeling: {len(frame):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Train dates: {train['service_date'].min()} -> {train['service_date'].max()}")
    print(f"Test dates: {test['service_date'].min()} -> {test['service_date'].max()}")
    print("\nSample rows:")
    print(
        frame[
            [
                "timestamp_minute",
                "station_name",
                "direction",
                "prev_delay_1",
                "line_recent_delay",
                "station_recent_delay",
                "target_delay_minutes",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


def train_and_report(data_root: Path, line_id: str, test_days: int, output_dir: Path) -> None:
    """Train the baseline forecaster, print metrics, and save artifacts."""

    frame = build_model_frame(data_root, line_id)
    train, test = split_train_test(frame, test_days)
    pipeline, scored = fit_and_predict(train, test)
    metrics = metrics_table(scored)
    station_errors = station_error_table(scored)
    importances = extract_feature_importance(pipeline)
    next_station = latest_station_forecast(frame, pipeline)

    print("Metrics:")
    print(metrics.to_string(index=False))
    print("\nStations with highest MAE:")
    print(station_errors.head(12).to_string(index=False))
    print("\nTop feature importances:")
    print(importances.head(15).to_string(index=False))
    print("\nScenario forecast for the next train at each station:")
    print(next_station.head(15).to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        output_dir / f"{line_id.lower()}_forecast_holdout_predictions.csv",
        output_dir / f"{line_id.lower()}_forecast_metrics.csv",
        output_dir / f"{line_id.lower()}_forecast_station_errors.csv",
        output_dir / f"{line_id.lower()}_forecast_feature_importance.csv",
        output_dir / f"{line_id.lower()}_forecast_next_station_predictions.csv",
    ]
    scored.to_csv(csv_paths[0], index=False)
    metrics.to_csv(csv_paths[1], index=False)
    station_errors.to_csv(csv_paths[2], index=False)
    importances.to_csv(csv_paths[3], index=False)
    next_station.to_csv(csv_paths[4], index=False)

    plot_paths = [
        save_scatter_plot(scored, output_dir, line_id),
        save_station_mae_plot(station_errors, output_dir, line_id),
        save_feature_importance_plot(importances, output_dir, line_id),
    ]

    print("\nSaved outputs:")
    for path in csv_paths + plot_paths:
        print(path)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(description="Forecast stop-level R2 delays with a baseline model.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing lines/, stations/, journeys/, timetables/, trains/, weather/",
    )
    parser.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help="Number of most recent service days reserved for the holdout test set",
    )
    parser.add_argument(
        "--output-dir",
        default="forecast_outputs",
        help="Directory where forecasting outputs will be saved",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("profile", help="Inspect the supervised forecasting dataset")
    subparsers.add_parser("train", help="Train the baseline model and save evaluation outputs")
    return parser


def main() -> None:
    """Parse arguments and run the requested command."""

    parser = build_parser()
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.command == "profile":
        profile_report(data_root, args.line_id, args.test_days)
    elif args.command == "train":
        train_and_report(data_root, args.line_id, args.test_days, output_dir)


if __name__ == "__main__":
    main()
