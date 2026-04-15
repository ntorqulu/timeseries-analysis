#!/usr/bin/env python3
from __future__ import annotations

"""Construct and diagnose evenly spaced delay time series for one Rodalies line."""

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires matplotlib and seaborn.") from exc

import explore_rodalies as rod


PLOT_STYLE = "whitegrid"
DEFAULT_TOP_STATIONS = 3


def choose_delay_column(frame: pd.DataFrame) -> str:
    if "delay_minutes" in frame.columns and frame["delay_minutes"].notna().any():
        return "delay_minutes"
    if "arrival_delay_minutes" in frame.columns and frame["arrival_delay_minutes"].notna().any():
        return "arrival_delay_minutes"
    return "departure_delay_minutes"


def build_direction_lookup(stops: pd.DataFrame, data_root: Path, line_id: str) -> pd.DataFrame:
    route = rod.official_route_station_order(data_root, line_id).copy()
    if route.empty:
        return pd.DataFrame(columns=["train_id", "service_date", "direction"])

    route["station_id"] = route["station_id"].astype("string")
    route_lookup = route.set_index("station_id")["route_index"]

    ordered = stops.sort_values(["train_id", "timestamp_minute", "stop_sequence"]).copy()
    ordered["station_id"] = ordered["station_id"].astype("string")
    ordered = ordered[ordered["station_id"].isin(route_lookup.index)].copy()
    if ordered.empty:
        return pd.DataFrame(columns=["train_id", "service_date", "direction"])

    ordered["route_index"] = ordered["station_id"].map(route_lookup)
    ordered["service_date"] = ordered["timestamp_minute"].dt.strftime("%Y-%m-%d")
    summary = (
        ordered.groupby(["train_id", "service_date"], dropna=False)
        .agg(
            first_route_index=("route_index", "first"),
            last_route_index=("route_index", "last"),
            records=("station_id", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["records"] >= 2].copy()
    summary["direction"] = pd.NA
    summary.loc[summary["last_route_index"] > summary["first_route_index"], "direction"] = "inbound"
    summary.loc[summary["last_route_index"] < summary["first_route_index"], "direction"] = "outbound"
    return summary[["train_id", "service_date", "direction"]]


def prepare_timeseries_frame(data_root: Path, line_id: str) -> pd.DataFrame:
    stops = rod.build_stop_delay_frame(data_root, line_id).copy()
    if stops.empty:
        raise SystemExit(f"No stop-level rows found for {line_id}")

    stops["service_date"] = stops["timestamp_minute"].dt.strftime("%Y-%m-%d")
    direction_lookup = build_direction_lookup(stops, data_root, line_id)
    if not direction_lookup.empty:
        stops = stops.merge(direction_lookup, on=["train_id", "service_date"], how="left")
    else:
        stops["direction"] = pd.NA
    return stops


def aggregate_series(frame: pd.DataFrame, freq: str, group_cols: list[str]) -> pd.DataFrame:
    delay_col = choose_delay_column(frame)
    working = frame.copy()
    working["bucket"] = working["timestamp_minute"].dt.floor(freq)
    grouped = (
        working.groupby(group_cols + ["bucket"], dropna=False)
        .agg(
            avg_delay_minutes=(delay_col, "mean"),
            median_delay_minutes=(delay_col, "median"),
            records=(delay_col, "size"),
        )
        .reset_index()
        .sort_values("bucket")
    )
    return grouped


def top_station_names(frame: pd.DataFrame, n: int) -> list[str]:
    counts = (
        frame.groupby("station_name", dropna=False)["station_id"]
        .size()
        .sort_values(ascending=False)
    )
    return counts.head(n).index.astype(str).tolist()


def add_missing_bucket_rows(frame: pd.DataFrame, freq: str, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not group_cols:
        part = frame.sort_values("bucket").copy()
        full_index = pd.date_range(part["bucket"].min(), part["bucket"].max(), freq=freq)
        return part.set_index("bucket").reindex(full_index).rename_axis("bucket").reset_index()
    filled_frames: list[pd.DataFrame] = []
    for keys, part in frame.groupby(group_cols, dropna=False):
        part = part.sort_values("bucket").copy()
        full_index = pd.date_range(part["bucket"].min(), part["bucket"].max(), freq=freq)
        expanded = part.set_index("bucket").reindex(full_index).rename_axis("bucket").reset_index()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys):
            expanded[col] = value
        filled_frames.append(expanded)
    return pd.concat(filled_frames, ignore_index=True)


def print_series_summary(frame: pd.DataFrame, label: str) -> None:
    print(f"\n## {label}")
    if frame.empty:
        print("No rows.")
        return
    print(frame.head(12).to_string(index=False))
    print("\nSummary:")
    print(
        frame["avg_delay_minutes"]
        .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .to_string()
    )


def lag_autocorrelation(frame: pd.DataFrame, max_lag: int = 16) -> pd.DataFrame:
    series = frame.sort_values("bucket")["avg_delay_minutes"].dropna()
    rows = []
    for lag in range(1, max_lag + 1):
        rows.append({"lag": lag, "autocorr": series.autocorr(lag=lag)})
    return pd.DataFrame(rows)


def save_line_series_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame["bucket"], frame["avg_delay_minutes"], color="#b91c1c", linewidth=1.5)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title(f"{line_id} average delay series ({freq})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average delay (minutes)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_delay_series_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_direction_series_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = {"inbound": "#1d4ed8", "outbound": "#dc2626"}
    for direction, part in frame.groupby("direction", dropna=False):
        if pd.isna(direction):
            continue
        ax.plot(
            part["bucket"],
            part["avg_delay_minutes"],
            label=str(direction),
            linewidth=1.4,
            color=palette.get(str(direction), "#6b7280"),
        )
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title(f"{line_id} delay series by direction ({freq})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average delay (minutes)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_delay_series_by_direction_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_station_series_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    for station_name, part in frame.groupby("station_name", dropna=False):
        ax.plot(part["bucket"], part["avg_delay_minutes"], label=str(station_name), linewidth=1.3)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title(f"{line_id} delay series for selected stations ({freq})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average delay (minutes)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_delay_series_selected_stations_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_hourly_profile_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    profile = (
        frame.assign(hour=frame["bucket"].dt.hour)
        .groupby("hour", dropna=False)["avg_delay_minutes"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=profile, x="hour", y="avg_delay_minutes", marker="o", ax=ax, color="#7c3aed")
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title(f"{line_id} average delay by hour ({freq} series)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average delay (minutes)")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_hourly_delay_profile_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_weekday_hour_heatmap(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    profile = frame.copy()
    profile["weekday"] = pd.Categorical(
        profile["bucket"].dt.day_name(),
        categories=rod.WEEKDAY_ORDER,
        ordered=True,
    )
    profile["hour"] = profile["bucket"].dt.hour
    heatmap = (
        profile.groupby(["weekday", "hour"], dropna=False)["avg_delay_minutes"]
        .mean()
        .reset_index()
        .pivot(index="weekday", columns="hour", values="avg_delay_minutes")
        .reindex(rod.WEEKDAY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{line_id} delay heatmap by weekday and hour ({freq} series)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_delay_weekday_hour_heatmap_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_autocorr_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, freq: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    autocorr = lag_autocorrelation(frame)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=autocorr, x="lag", y="autocorr", ax=ax, color="#0f766e")
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title(f"{line_id} lag autocorrelation of delay series ({freq})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    output_path = output_dir / f"{line_id.lower()}_delay_autocorr_{freq}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def series_report(data_root: Path, line_id: str, freq: str, stations: list[str] | None, top_n: int) -> None:
    frame = prepare_timeseries_frame(data_root, line_id)

    line_series = add_missing_bucket_rows(aggregate_series(frame, freq, []), freq, [])
    direction_series = add_missing_bucket_rows(
        aggregate_series(frame.dropna(subset=["direction"]), freq, ["direction"]),
        freq,
        ["direction"],
    )

    selected_stations = stations or top_station_names(frame, top_n)
    station_frame = frame[frame["station_name"].isin(selected_stations)].copy()
    station_series = add_missing_bucket_rows(
        aggregate_series(station_frame, freq, ["station_name"]),
        freq,
        ["station_name"],
    )

    print(f"Line: {line_id}")
    print(f"Aggregation: {freq}")
    print(f"Time range: {frame['timestamp_minute'].min()} -> {frame['timestamp_minute'].max()}")
    print(f"Selected stations: {', '.join(selected_stations)}")
    print_series_summary(line_series, "Line-level series")
    print_series_summary(direction_series, "Direction-level series")
    print_series_summary(station_series, "Station-level series")
    print("\nAutocorrelation:")
    print(lag_autocorrelation(line_series).to_string(index=False))


def plot_report(data_root: Path, line_id: str, freq: str, stations: list[str] | None, top_n: int, output_dir: Path) -> None:
    frame = prepare_timeseries_frame(data_root, line_id)
    line_series = add_missing_bucket_rows(aggregate_series(frame, freq, []), freq, [])
    direction_series = add_missing_bucket_rows(
        aggregate_series(frame.dropna(subset=["direction"]), freq, ["direction"]),
        freq,
        ["direction"],
    )
    selected_stations = stations or top_station_names(frame, top_n)
    station_frame = frame[frame["station_name"].isin(selected_stations)].copy()
    station_series = add_missing_bucket_rows(
        aggregate_series(station_frame, freq, ["station_name"]),
        freq,
        ["station_name"],
    )

    outputs = [
        save_line_series_plot(line_series, output_dir, line_id, freq),
        save_direction_series_plot(direction_series, output_dir, line_id, freq),
        save_station_series_plot(station_series, output_dir, line_id, freq),
        save_hourly_profile_plot(line_series, output_dir, line_id, freq),
        save_weekday_hour_heatmap(line_series, output_dir, line_id, freq),
        save_autocorr_plot(line_series, output_dir, line_id, freq),
    ]

    csv_outputs = [
        output_dir / f"{line_id.lower()}_line_series_{freq}.csv",
        output_dir / f"{line_id.lower()}_direction_series_{freq}.csv",
        output_dir / f"{line_id.lower()}_station_series_{freq}.csv",
    ]
    line_series.to_csv(csv_outputs[0], index=False)
    direction_series.to_csv(csv_outputs[1], index=False)
    station_series.to_csv(csv_outputs[2], index=False)

    print("\nSaved outputs:")
    for path in outputs + csv_outputs:
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Study time-series characteristics of Rodalies delays.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing lines/, stations/, journeys/, timetables/, trains/, weather/",
    )
    parser.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")
    parser.add_argument(
        "--freq",
        default="15min",
        choices=["15min", "30min", "1h"],
        help="Aggregation frequency for the delay series",
    )
    parser.add_argument(
        "--stations",
        help="Comma-separated station names to follow explicitly",
    )
    parser.add_argument(
        "--top-n-stations",
        type=int,
        default=DEFAULT_TOP_STATIONS,
        help="If --stations is not set, select the top N stations by number of observations",
    )
    parser.add_argument("--output-dir", default="timeseries_outputs", help="Directory where outputs will be saved")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("profile", help="Print summaries of the constructed delay time series")
    subparsers.add_parser("plot", help="Generate plots and CSV exports for the delay time series")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stations = [item.strip() for item in args.stations.split(",") if item.strip()] if args.stations else None

    if args.command == "profile":
        series_report(data_root, args.line_id, args.freq, stations, args.top_n_stations)
    elif args.command == "plot":
        plot_report(data_root, args.line_id, args.freq, stations, args.top_n_stations, output_dir)


if __name__ == "__main__":
    main()
