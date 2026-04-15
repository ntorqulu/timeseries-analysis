#!/usr/bin/env python3
from __future__ import annotations

"""Explore Rodalies parquet datasets, build descriptive reports, and create maps/plots."""

import argparse
import html
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from branca.colormap import LinearColormap
    import folium
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires pandas, pyarrow, matplotlib, seaborn, and folium."
    ) from exc


DATASET_PATTERNS = {
    "lines": "lines/lines.parquet",
    "stations": "stations/stations.parquet",
    "journeys": "journeys/journeys_*.parquet",
    "timetables": "timetables/timetables_*.parquet",
    "trains": "trains/trains_*.parquet",
    "weather": "weather/weather_*.parquet",
}

PLOT_STYLE = "whitegrid"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CATALONIA_PUBLIC_HOLIDAYS_2026 = {
    "2026-01-01",
    "2026-01-06",
    "2026-04-03",
    "2026-04-06",
    "2026-05-01",
    "2026-06-24",
    "2026-08-15",
    "2026-09-11",
    "2026-10-12",
    "2026-12-08",
    "2026-12-25",
    "2026-12-26",
}
OFFICIAL_ROUTE_STATION_IDS = {
    "R2": [
        "71705",  # Castelldefels
        "71706",  # Gava
        "71709",  # Viladecans
        "71707",  # El Prat de Llobregat
        "71708",  # Bellvitge | Gornal
        "71801",  # Barcelona-Sants
        "71802",  # Barcelona-Passeig de Gracia
        "79009",  # Barcelona-El Clot
        "79004",  # Barcelona-Sant Andreu
        "79005",  # Montcada i Reixac
        "79011",  # La Llagosta
        "79006",  # Mollet-Sant Fost
        "79007",  # Montmelo
        "79100",  # Granollers Centre
    ]
}
TIME_OF_DAY_BUCKETS = [
    ("early_morning", 5, 8),
    ("morning_peak", 9, 11),
    ("midday", 12, 15),
    ("evening_peak", 16, 19),
    ("late_evening", 20, 23),
]


def resolve_matches(data_root: Path, dataset: str) -> list[Path]:
    pattern = DATASET_PATTERNS[dataset]
    matches = sorted(data_root.glob(pattern))
    return [path for path in matches if path.is_file()]


def newest_file(paths: Iterable[Path]) -> Path:
    try:
        return sorted(paths)[-1]
    except IndexError as exc:
        raise SystemExit("No matching files found") from exc


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def print_frame(title: str, frame: pd.DataFrame, limit: int = 10) -> None:
    print(f"\n## {title}")
    if frame.empty:
        print("No rows.")
    else:
        print(frame.head(limit).to_string(index=False))


def ensure_output_dir(path: str | Path | None) -> Path:
    output_dir = Path(path or "plots").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_corr(series_a: pd.Series, series_b: pd.Series) -> float | None:
    frame = pd.concat([series_a, series_b], axis=1).dropna()
    if len(frame) < 3:
        return None
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def add_day_type(frame: pd.DataFrame, timestamp_col: str = "timestamp_minute") -> pd.DataFrame:
    result = frame.copy()
    ts = pd.to_datetime(result[timestamp_col], errors="coerce")
    dates = ts.dt.strftime("%Y-%m-%d")
    is_weekend = ts.dt.weekday >= 5
    is_holiday = dates.isin(CATALONIA_PUBLIC_HOLIDAYS_2026)
    result["day_type"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result.loc[is_weekend | is_holiday, "day_type"] = "weekend_or_holiday"
    result.loc[~(is_weekend | is_holiday), "day_type"] = "working_day"
    return result


def normalize_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def add_time_features(frame: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    result = frame.copy()
    result[column] = normalize_timestamp(result[column])
    result["date"] = result[column].dt.date
    result["hour"] = result[column].dt.hour
    result["timestamp_15min"] = result[column].dt.floor("15min")
    result["timestamp_hour"] = result[column].dt.floor("h")
    return result


def inventory(data_root: Path) -> None:
    print(f"Data root: {data_root}\n")
    for dataset in DATASET_PATTERNS:
        matches = resolve_matches(data_root, dataset)
        print(f"{dataset:10} files={len(matches)}")
        if matches:
            print(f"  first: {matches[0].name}")
            print(f"  last : {matches[-1].name}")
        else:
            print("  no files found")


def schema_report(data_root: Path) -> None:
    for dataset in DATASET_PATTERNS:
        matches = resolve_matches(data_root, dataset)
        print(f"\n## {dataset}")
        if not matches:
            print("No files found.")
            continue
        parquet_file = pq.ParquetFile(matches[0])
        print(f"sample_file: {matches[0].name}")
        print(f"rows: {parquet_file.metadata.num_rows}")
        print(f"columns: {parquet_file.schema.names}")


def dataset_sizes(data_root: Path) -> None:
    for dataset in ("journeys", "timetables", "trains", "weather"):
        matches = resolve_matches(data_root, dataset)
        print(f"\n## {dataset}")
        if not matches:
            print("No files found.")
            continue
        rows = []
        for path in matches:
            meta = pq.ParquetFile(path).metadata
            rows.append(
                {
                    "file": path.name,
                    "rows": meta.num_rows,
                    "columns": meta.num_columns,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))


def deep_profile(data_root: Path) -> None:
    for dataset in DATASET_PATTERNS:
        matches = resolve_matches(data_root, dataset)
        print(f"\n## {dataset}")
        if not matches:
            print("No files found.")
            continue

        first = matches[0]
        sample = read_parquet(first)
        meta = pq.ParquetFile(first).metadata

        print(f"sample_file: {first.name}")
        print(f"rows: {meta.num_rows:,}")
        print(f"columns: {meta.num_columns}")
        print(f"column_names: {list(sample.columns)}")
        print("dtypes:")
        print(sample.dtypes.astype(str).to_string())

        nulls = sample.isna().sum().rename_axis("column").reset_index(name="null_count")
        print_frame("Null counts in sample file", nulls, limit=50)

        numeric_cols = sample.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            summary = sample[numeric_cols].describe().transpose().rename_axis("column").reset_index()
            print_frame("Numeric summary", summary, limit=50)

        object_cols = sample.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        for col in object_cols[:5]:
            values = (
                sample[col]
                .fillna("NA")
                .astype(str)
                .value_counts()
                .rename_axis(col)
                .reset_index(name="count")
            )
            print_frame(f"Top values for {col}", values, limit=15)


def lines_report(data_root: Path) -> None:
    matches = resolve_matches(data_root, "lines")
    if not matches:
        raise SystemExit("lines.parquet not found")
    frame = read_parquet(matches[0])
    print_frame("Lines", frame, limit=20)


def stations_report(data_root: Path) -> None:
    matches = resolve_matches(data_root, "stations")
    if not matches:
        raise SystemExit("stations.parquet not found")
    frame = read_parquet(matches[0])
    print_frame("Stations", frame, limit=20)


def load_all_daily_files(data_root: Path, dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
    matches = resolve_matches(data_root, dataset)
    if not matches:
        raise SystemExit(f"No files found for dataset '{dataset}'")

    frames = []
    for path in matches:
        frame = read_parquet(path, columns=columns)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def date_key_from_path(path: Path, prefix: str) -> str | None:
    stem = path.stem
    expected = f"{prefix}_"
    if stem.startswith(expected):
        return stem[len(expected):]
    return None


def build_daily_file_map(data_root: Path, dataset: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in resolve_matches(data_root, dataset):
        key = date_key_from_path(path, dataset)
        if key is not None:
            mapping[key] = path
    return mapping


def aggregate_train_file(path: Path, line_id: str) -> tuple[pd.DataFrame, set[str]]:
    frame = read_parquet(
        path,
        columns=[
            "train_id",
            "line_id",
            "delay_minutes",
            "timestamp_minute",
        ],
    )
    frame = frame[frame["line_id"] == line_id].copy()
    if frame.empty:
        return pd.DataFrame(), set()

    frame["timestamp_minute"] = normalize_timestamp(frame["timestamp_minute"])
    frame["delay_minutes"] = pd.to_numeric(frame["delay_minutes"], errors="coerce")

    agg = (
        frame.groupby("timestamp_minute", dropna=False)
        .agg(
            active_train_records=("train_id", "size"),
            unique_trains=("train_id", "nunique"),
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
        )
        .reset_index()
    )
    train_ids = set(frame["train_id"].dropna().astype(str).unique())
    return agg, train_ids


def aggregate_timetable_file(path: Path, train_ids: set[str]) -> pd.DataFrame:
    if not train_ids:
        return pd.DataFrame()

    frame = read_parquet(
        path,
        columns=[
            "train_id",
            "planned_arrival",
            "planned_departure",
            "actual_arrival",
            "actual_departure",
            "timestamp",
        ],
    )
    frame["train_id"] = frame["train_id"].astype("string")
    frame = frame[frame["train_id"].isin(train_ids)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["timestamp_minute"] = normalize_timestamp(frame["timestamp"]).dt.floor("min")
    for col in ("planned_arrival", "planned_departure", "actual_arrival", "actual_departure"):
        frame[col] = normalize_timestamp(frame[col])

    frame["arrival_delay_minutes"] = (
        (frame["actual_arrival"] - frame["planned_arrival"]).dt.total_seconds() / 60.0
    )
    frame["departure_delay_minutes"] = (
        (frame["actual_departure"] - frame["planned_departure"]).dt.total_seconds() / 60.0
    )

    return (
        frame.groupby("timestamp_minute", dropna=False)
        .agg(
            avg_arrival_delay_minutes=("arrival_delay_minutes", "mean"),
            avg_departure_delay_minutes=("departure_delay_minutes", "mean"),
            timetable_records=("train_id", "size"),
        )
        .reset_index()
    )


def aggregate_weather_file(path: Path) -> pd.DataFrame:
    frame = read_parquet(
        path,
        columns=[
            "timestamp_minute",
            "temperature",
            "precipitation",
            "windspeed",
            "weathercode",
            "cloudcover",
        ],
    )
    if frame.empty:
        return pd.DataFrame()

    frame["timestamp_minute"] = normalize_timestamp(frame["timestamp_minute"])
    for col in ("temperature", "precipitation", "windspeed", "weathercode", "cloudcover"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return (
        frame.groupby("timestamp_minute", dropna=False)
        .agg(
            temperature=("temperature", "mean"),
            precipitation=("precipitation", "mean"),
            windspeed=("windspeed", "mean"),
            weathercode=("weathercode", "mean"),
            cloudcover=("cloudcover", "mean"),
        )
        .reset_index()
    )


def load_station_lookup(data_root: Path) -> pd.DataFrame:
    station_files = resolve_matches(data_root, "stations")
    if not station_files:
        return pd.DataFrame(columns=["station_id", "station_name"])
    stations = read_parquet(station_files[0], columns=["station_id", "name"])
    return stations.rename(columns={"name": "station_name"})


def load_station_geo_lookup(data_root: Path) -> pd.DataFrame:
    station_files = resolve_matches(data_root, "stations")
    if not station_files:
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude"])
    stations = read_parquet(
        station_files[0],
        columns=["station_id", "name", "latitude", "longitude"],
    )
    return stations.rename(columns={"name": "station_name"})


def load_line_lookup(data_root: Path) -> pd.DataFrame:
    line_files = resolve_matches(data_root, "lines")
    if not line_files:
        return pd.DataFrame(columns=["line_id", "name", "origin", "destination"])
    return read_parquet(line_files[0], columns=["line_id", "name", "origin", "destination"])


def normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def name_match_score(left: object, right: object) -> int:
    left_norm = normalized_text(left)
    right_norm = normalized_text(right)
    if not left_norm or not right_norm:
        return 0
    if left_norm == right_norm:
        return 3
    if left_norm in right_norm or right_norm in left_norm:
        return 2
    left_tokens = set(str(left).lower().replace("-", " ").split())
    right_tokens = set(str(right).lower().replace("-", " ").split())
    if left_tokens & right_tokens:
        return 1
    return 0


def line_endpoints(data_root: Path, line_id: str) -> tuple[str | None, str | None]:
    lines = load_line_lookup(data_root)
    if lines.empty:
        return None, None
    row = lines[lines["line_id"] == line_id]
    if row.empty:
        return None, None
    record = row.iloc[0]
    return record.get("origin"), record.get("destination")


def official_route_station_order(data_root: Path, line_id: str) -> pd.DataFrame:
    station_ids = OFFICIAL_ROUTE_STATION_IDS.get(line_id)
    if not station_ids:
        return pd.DataFrame()
    station_geo = load_station_geo_lookup(data_root).copy()
    if station_geo.empty:
        return pd.DataFrame()
    station_geo["station_id"] = station_geo["station_id"].astype("string")
    route = station_geo[station_geo["station_id"].isin(station_ids)].copy()
    order_lookup = {station_id: idx for idx, station_id in enumerate(station_ids)}
    route["route_index"] = route["station_id"].map(order_lookup)
    route = route.dropna(subset=["route_index"]).copy()
    route["route_index"] = route["route_index"].astype(int)
    route = route.sort_values("route_index").reset_index(drop=True)
    return route


def build_segment_delay_frame(frame: pd.DataFrame, data_root: Path, line_id: str) -> pd.DataFrame:
    station_order = official_route_station_order(data_root, line_id)
    if station_order.empty:
        routes = build_route_direction_sequences(frame, data_root, line_id)
        if not routes:
            return pd.DataFrame()
        inbound_route = next((route for route in routes if route["direction_label"] == "inbound"), routes[0])
        station_order = inbound_route["station_order"].copy().reset_index(drop=True)
        station_order["station_id"] = station_order["station_id"].astype("string")
        station_order["route_index"] = range(len(station_order))
    route_lookup = station_order.set_index("station_id")

    ordered = frame.sort_values(["train_id", "timestamp_minute", "stop_sequence"]).copy()
    ordered = ordered.dropna(subset=["train_id", "station_id", "stop_sequence", "delay_minutes"]).copy()
    ordered["train_id"] = ordered["train_id"].astype("string")
    ordered["station_id"] = ordered["station_id"].astype("string")
    ordered = ordered[ordered["station_id"].isin(route_lookup.index)].copy()
    ordered = ordered.drop_duplicates(
        subset=["train_id", "station_id", "stop_sequence", "timestamp_minute"]
    ).copy()
    ordered["prev_station_id"] = ordered.groupby("train_id", dropna=False)["station_id"].shift(1)
    ordered["prev_station_name"] = ordered.groupby("train_id", dropna=False)["station_name"].shift(1)
    ordered["prev_stop_sequence"] = ordered.groupby("train_id", dropna=False)["stop_sequence"].shift(1)
    ordered = ordered.dropna(subset=["prev_station_id"]).copy()
    ordered = ordered[ordered["prev_station_id"].astype(str) != ordered["station_id"].astype(str)].copy()

    ordered["route_index"] = ordered["station_id"].map(route_lookup["route_index"])
    ordered["prev_route_index"] = ordered["prev_station_id"].map(route_lookup["route_index"])
    ordered = ordered.dropna(subset=["route_index", "prev_route_index"]).copy()
    ordered["route_index"] = ordered["route_index"].astype(int)
    ordered["prev_route_index"] = ordered["prev_route_index"].astype(int)
    ordered = ordered[ordered["route_index"] != ordered["prev_route_index"]].copy()
    ordered["direction"] = ordered.apply(
        lambda row: "inbound" if row["route_index"] > row["prev_route_index"] else "outbound",
        axis=1,
    )

    expanded_rows: list[dict[str, object]] = []
    for row in ordered.itertuples(index=False):
        start_idx = int(row.prev_route_index)
        end_idx = int(row.route_index)
        step = 1 if end_idx > start_idx else -1
        for idx in range(start_idx, end_idx, step):
            next_idx = idx + step
            from_row = station_order.iloc[idx]
            to_row = station_order.iloc[next_idx]
            if step < 0:
                from_row, to_row = to_row, from_row
            expanded_rows.append(
                {
                    "direction": row.direction,
                    "from_station_id": str(from_row["station_id"]),
                    "to_station_id": str(to_row["station_id"]),
                    "from_station_name": from_row["station_name"],
                    "to_station_name": to_row["station_name"],
                    "from_latitude": float(from_row["latitude"]),
                    "from_longitude": float(from_row["longitude"]),
                    "to_latitude": float(to_row["latitude"]),
                    "to_longitude": float(to_row["longitude"]),
                    "delay_minutes": float(row.delay_minutes),
                }
            )

    if not expanded_rows:
        return pd.DataFrame()

    segments = pd.DataFrame(expanded_rows)
    segments["segment"] = segments["from_station_name"].astype(str) + " -> " + segments["to_station_name"].astype(str)
    return (
        segments.groupby(
            [
                "direction",
                "from_station_id",
                "to_station_id",
                "from_station_name",
                "to_station_name",
                "from_latitude",
                "from_longitude",
                "to_latitude",
                "to_longitude",
                "segment",
            ],
            dropna=False,
        )
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .sort_values(["direction", "from_station_name", "to_station_name"])
        .reset_index(drop=True)
    )


def build_route_direction_sequences(
    frame: pd.DataFrame,
    data_root: Path,
    line_id: str,
) -> list[dict[str, object]]:
    official_route = official_route_station_order(data_root, line_id)
    if not official_route.empty:
        forward = official_route.copy().reset_index(drop=True)
        reverse = official_route.iloc[::-1].reset_index(drop=True)
        return [
            {
                "direction_key": f"{forward.iloc[0]['station_name']} -> {forward.iloc[-1]['station_name']}",
                "direction_label": "inbound",
                "display_name": f"inbound ({forward.iloc[0]['station_name']} -> {forward.iloc[-1]['station_name']})",
                "start_station": forward.iloc[0]["station_name"],
                "end_station": forward.iloc[-1]["station_name"],
                "station_order": forward,
            },
            {
                "direction_key": f"{reverse.iloc[0]['station_name']} -> {reverse.iloc[-1]['station_name']}",
                "direction_label": "outbound",
                "display_name": f"outbound ({reverse.iloc[0]['station_name']} -> {reverse.iloc[-1]['station_name']})",
                "start_station": reverse.iloc[0]["station_name"],
                "end_station": reverse.iloc[-1]["station_name"],
                "station_order": reverse,
            },
        ]

    station_geo = load_station_geo_lookup(data_root)
    if frame.empty or station_geo.empty:
        return []
    station_geo = station_geo.copy()
    station_geo["station_id"] = station_geo["station_id"].astype("string")

    ordered = frame.sort_values(["timestamp_minute", "stop_sequence"]).copy()
    ordered = ordered.dropna(subset=["station_id", "stop_sequence"]).copy()
    ordered["station_id"] = ordered["station_id"].astype("string")

    station_order = (
        ordered.groupby(["station_id", "station_name"], dropna=False)
        .agg(
            median_stop_sequence=("stop_sequence", "median"),
            first_stop_sequence=("stop_sequence", "min"),
            last_stop_sequence=("stop_sequence", "max"),
            trains=("train_id", "nunique"),
            records=("station_id", "size"),
        )
        .reset_index()
        .merge(station_geo, on=["station_id", "station_name"], how="left")
        .dropna(subset=["latitude", "longitude"])
        .sort_values(["median_stop_sequence", "first_stop_sequence", "station_name"])
        .reset_index(drop=True)
    )
    if len(station_order) < 2:
        return []

    origin, destination = line_endpoints(data_root, line_id)
    first_station = station_order.iloc[0]["station_name"]
    last_station = station_order.iloc[-1]["station_name"]
    forward_score = name_match_score(first_station, origin) + name_match_score(last_station, destination)
    reverse_score = name_match_score(first_station, destination) + name_match_score(last_station, origin)
    if reverse_score > forward_score:
        station_order = station_order.iloc[::-1].reset_index(drop=True)

    forward = station_order.copy().reset_index(drop=True)
    reverse = station_order.iloc[::-1].reset_index(drop=True)
    return [
        {
            "direction_key": f"{forward.iloc[0]['station_name']} -> {forward.iloc[-1]['station_name']}",
            "direction_label": "inbound",
            "display_name": f"inbound ({forward.iloc[0]['station_name']} -> {forward.iloc[-1]['station_name']})",
            "start_station": forward.iloc[0]["station_name"],
            "end_station": forward.iloc[-1]["station_name"],
            "station_order": forward,
        },
        {
            "direction_key": f"{reverse.iloc[0]['station_name']} -> {reverse.iloc[-1]['station_name']}",
            "direction_label": "outbound",
            "display_name": f"outbound ({reverse.iloc[0]['station_name']} -> {reverse.iloc[-1]['station_name']})",
            "start_station": reverse.iloc[0]["station_name"],
            "end_station": reverse.iloc[-1]["station_name"],
            "station_order": reverse,
        },
    ]


def add_weather_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    max_precip = pd.to_numeric(result.get("precipitation"), errors="coerce").max()
    if pd.isna(max_precip) or max_precip <= 0:
        result["precip_bucket"] = "dry"
    elif max_precip <= 1:
        result["precip_bucket"] = pd.cut(
            pd.to_numeric(result["precipitation"], errors="coerce"),
            bins=[-0.001, 0, max_precip + 1e-6],
            labels=["dry", "rain"],
            include_lowest=True,
        ).astype("string").fillna("unknown")
    else:
        result["precip_bucket"] = pd.cut(
            pd.to_numeric(result["precipitation"], errors="coerce"),
            bins=[-0.001, 0, 1, max_precip + 1e-6],
            labels=["dry", "light rain", "heavier rain"],
            include_lowest=True,
        ).astype("string").fillna("unknown")

    result["cloud_bucket"] = pd.cut(
        pd.to_numeric(result.get("cloudcover"), errors="coerce"),
        bins=[-0.001, 25, 50, 75, 100],
        labels=["clear", "partly cloudy", "cloudy", "overcast"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    return result


def summarize_by_day_type(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        frame.groupby(["day_type", group_col], dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
    )


def r2_report(data_root: Path, day: str | None = None) -> None:
    train_files = resolve_matches(data_root, "trains")
    if not train_files:
        raise SystemExit("No train parquet files found")

    if day:
        selected = data_root / "trains" / f"trains_{day}.parquet"
        if not selected.exists():
            raise SystemExit(f"Train file not found for day {day}: {selected}")
    else:
        selected = newest_file(train_files)

    train_columns = [
        "train_id",
        "line_id",
        "current_station_id",
        "next_station_id",
        "platform",
        "status",
        "delay_minutes",
        "observations",
        "timestamp",
    ]
    trains = read_parquet(selected, columns=train_columns)
    r2 = trains[trains["line_id"] == "R2"].copy()

    print(f"Using train file: {selected.name}")
    print(f"Total rows: {len(trains):,}")
    print(f"R2 rows: {len(r2):,}")

    if r2.empty:
        return

    r2 = add_time_features(r2)

    print_frame("R2 sample rows", r2.sort_values("timestamp"), limit=15)
    print_frame(
        "R2 status counts",
        r2["status"].fillna("NA").value_counts().rename_axis("status").reset_index(name="count"),
        limit=20,
    )
    print_frame(
        "R2 delay summary",
        r2["delay_minutes"].describe().to_frame(name="delay_minutes").reset_index(),
        limit=20,
    )
    print_frame(
        "R2 hourly average delay",
        r2.groupby("hour", dropna=False)["delay_minutes"].mean().reset_index(name="avg_delay_minutes"),
        limit=24,
    )

    if resolve_matches(data_root, "stations"):
        stations = read_parquet(
            resolve_matches(data_root, "stations")[0],
            columns=["station_id", "name"],
        )
        station_names = stations.rename(
            columns={"station_id": "current_station_id", "name": "current_station_name"}
        )
        merged = r2.merge(station_names, on="current_station_id", how="left")
        top_stations = (
            merged["current_station_name"]
            .fillna("UNKNOWN")
            .value_counts()
            .rename_axis("current_station_name")
            .reset_index(name="count")
        )
        print_frame("R2 most common current stations", top_stations, limit=15)


def build_analysis_frame(data_root: Path, line_id: str) -> pd.DataFrame:
    train_map = build_daily_file_map(data_root, "trains")
    timetable_map = build_daily_file_map(data_root, "timetables")
    weather_map = build_daily_file_map(data_root, "weather")

    if not train_map:
        raise SystemExit("No train parquet files found")

    daily_frames: list[pd.DataFrame] = []
    for day_key, train_path in sorted(train_map.items()):
        train_agg, train_ids = aggregate_train_file(train_path, line_id)
        if train_agg.empty:
            continue

        merged = train_agg.copy()
        timetable_path = timetable_map.get(day_key)
        if timetable_path is not None:
            timetable_agg = aggregate_timetable_file(timetable_path, train_ids)
            if not timetable_agg.empty:
                merged = merged.merge(timetable_agg, on="timestamp_minute", how="left")

        weather_path = weather_map.get(day_key)
        if weather_path is not None:
            weather_agg = aggregate_weather_file(weather_path)
            if not weather_agg.empty:
                merged = merged.merge(weather_agg, on="timestamp_minute", how="left")

        merged["source_day"] = day_key
        daily_frames.append(merged)

    if not daily_frames:
        raise SystemExit(f"No train rows found for line '{line_id}'")

    merged = pd.concat(daily_frames, ignore_index=True)
    merged["date"] = merged["timestamp_minute"].dt.date
    merged["hour"] = merged["timestamp_minute"].dt.hour
    return merged.sort_values("timestamp_minute").reset_index(drop=True)


def build_stop_delay_frame(data_root: Path, line_id: str) -> pd.DataFrame:
    train_map = build_daily_file_map(data_root, "trains")
    timetable_map = build_daily_file_map(data_root, "timetables")
    weather_map = build_daily_file_map(data_root, "weather")
    station_lookup = load_station_lookup(data_root)

    frames: list[pd.DataFrame] = []
    for day_key, train_path in sorted(train_map.items()):
        timetable_path = timetable_map.get(day_key)
        if timetable_path is None:
            continue

        trains = read_parquet(train_path, columns=["train_id", "line_id"])
        trains = trains[trains["line_id"] == line_id].copy()
        if trains.empty:
            continue
        trains["train_id"] = trains["train_id"].astype("string")
        train_ids = set(trains["train_id"].dropna().unique())

        timetables = read_parquet(
            timetable_path,
            columns=[
                "train_id",
                "station_id",
                "planned_arrival",
                "planned_departure",
                "actual_arrival",
                "actual_departure",
                "timestamp",
                "stop_sequence",
            ],
        )
        timetables["train_id"] = timetables["train_id"].astype("string")
        timetables = timetables[timetables["train_id"].isin(train_ids)].copy()
        if timetables.empty:
            continue

        timetables["timestamp_minute"] = normalize_timestamp(timetables["timestamp"]).dt.floor("min")
        for col in ("planned_arrival", "planned_departure", "actual_arrival", "actual_departure"):
            timetables[col] = normalize_timestamp(timetables[col])

        timetables["arrival_delay_minutes"] = (
            (timetables["actual_arrival"] - timetables["planned_arrival"]).dt.total_seconds() / 60.0
        )
        timetables["departure_delay_minutes"] = (
            (timetables["actual_departure"] - timetables["planned_departure"]).dt.total_seconds() / 60.0
        )

        day_frame = timetables[
            [
                "train_id",
                "station_id",
                "timestamp_minute",
                "stop_sequence",
                "arrival_delay_minutes",
                "departure_delay_minutes",
            ]
        ].copy()

        weather_path = weather_map.get(day_key)
        if weather_path is not None:
            weather = aggregate_weather_file(weather_path)
            day_frame = day_frame.merge(weather, on="timestamp_minute", how="left")

        frames.append(day_frame)

    if not frames:
        raise SystemExit(f"No stop-level delay rows found for line '{line_id}'")

    result = pd.concat(frames, ignore_index=True)
    if not station_lookup.empty:
        result = result.merge(station_lookup, on="station_id", how="left")
    result["station_name"] = result.get("station_name", pd.Series(dtype="string")).fillna("UNKNOWN")
    result["delay_minutes"] = result["arrival_delay_minutes"]
    if result["delay_minutes"].isna().all():
        result["delay_minutes"] = result["departure_delay_minutes"]
    result = add_weather_buckets(result)
    result["weekday"] = result["timestamp_minute"].dt.day_name()
    result["weekday"] = pd.Categorical(result["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    result["hour"] = result["timestamp_minute"].dt.hour
    result["date"] = result["timestamp_minute"].dt.date
    result = add_day_type(result, "timestamp_minute")
    return result.sort_values("timestamp_minute").reset_index(drop=True)


def correlation_report(data_root: Path, line_id: str) -> None:
    analysis = build_analysis_frame(data_root, line_id)
    stops = build_stop_delay_frame(data_root, line_id)
    print(f"Built analysis frames for line {line_id}")
    print(f"Aggregated rows: {len(analysis):,}")
    print(f"Stop-level rows: {len(stops):,}")
    print(f"Time range: {stops['timestamp_minute'].min()} -> {stops['timestamp_minute'].max()}")

    print_frame(
        "Stop-level sample",
        stops[
            [
                "timestamp_minute",
                "station_name",
                "delay_minutes",
                "precipitation",
                "temperature",
                "windspeed",
                "precip_bucket",
                "cloud_bucket",
            ]
        ],
        limit=15,
    )

    delay_target = choose_delay_column(analysis)
    weather_features = ["temperature", "precipitation", "windspeed", "weathercode", "cloudcover"]
    rows = []
    for feature in weather_features:
        if feature in analysis.columns:
            rows.append(
                {
                    "feature": feature,
                    "corr_with_delay": safe_corr(analysis[delay_target], analysis[feature]),
                }
            )
    delay_corr = pd.DataFrame(rows).sort_values("corr_with_delay", key=lambda s: s.abs(), ascending=False)
    print_frame("Weather correlations with delay only", delay_corr, limit=20)

    station_summary = (
        stops.groupby("station_name", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .sort_values(["avg_delay_minutes", "records"], ascending=[False, False])
    )
    print_frame("Stations with highest average delay", station_summary, limit=20)

    precip_summary = (
        stops.groupby("precip_bucket", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .sort_values("avg_delay_minutes", ascending=False)
    )
    print_frame("Delay by precipitation condition", precip_summary, limit=10)

    cloud_summary = (
        stops.groupby("cloud_bucket", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .sort_values("avg_delay_minutes", ascending=False)
    )
    print_frame("Delay by cloud condition", cloud_summary, limit=10)

    hourly = (
        stops.groupby("hour", dropna=False)
        .agg(avg_delay_minutes=("delay_minutes", "mean"), records=("delay_minutes", "size"))
        .reset_index()
    )
    print_frame("Average delay by hour of day", hourly, limit=24)

    weekday = (
        stops.groupby("weekday", dropna=False)
        .agg(avg_delay_minutes=("delay_minutes", "mean"), records=("delay_minutes", "size"))
        .reset_index()
    )
    print_frame("Average delay by weekday", weekday, limit=7)

    print("\nNote: negative delay values mean trains were early.")


def diagnostics_report(data_root: Path, line_id: str) -> None:
    analysis = build_analysis_frame(data_root, line_id)
    stops = build_stop_delay_frame(data_root, line_id)

    print(f"Diagnostics for {line_id}")
    print(f"Time range: {stops['timestamp_minute'].min()} -> {stops['timestamp_minute'].max()}")
    print(f"Recorded days: {stops['date'].nunique()}")

    missing = (
        stops.groupby("date", dropna=False)
        .agg(
            rows=("delay_minutes", "size"),
            missing_delay=("delay_minutes", lambda s: s.isna().sum()),
            missing_precip=("precipitation", lambda s: s.isna().sum()),
            missing_temp=("temperature", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    print_frame("Missing-data patterns by date", missing.sort_values("date"), limit=40)

    route_direction = (
        stops.groupby("train_id", dropna=False)
        .agg(
            first_station=("station_name", "first"),
            last_station=("station_name", "last"),
            avg_delay_minutes=("delay_minutes", "mean"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
    )
    route_direction["direction"] = route_direction["first_station"].astype(str) + " -> " + route_direction["last_station"].astype(str)
    direction_summary = (
        route_direction.groupby("direction", dropna=False)
        .agg(
            avg_delay_minutes=("avg_delay_minutes", "mean"),
            trains=("train_id", "nunique"),
            records=("records", "sum"),
        )
        .reset_index()
        .sort_values("avg_delay_minutes", ascending=False)
    )
    print_frame("Direction effects", direction_summary, limit=20)

    segments = stops.sort_values(["train_id", "timestamp_minute", "stop_sequence"]).copy()
    segments["prev_station_name"] = segments.groupby("train_id", dropna=False)["station_name"].shift(1)
    segments["segment"] = segments["prev_station_name"].astype("string") + " -> " + segments["station_name"].astype("string")
    segment_summary = (
        segments.dropna(subset=["prev_station_name"])
        .groupby("segment", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .sort_values(["avg_delay_minutes", "records"], ascending=[False, False])
    )
    print_frame("Segments with highest average delay", segment_summary, limit=20)

    service_window = stops.copy()
    service_window["service_window"] = pd.cut(
        service_window["hour"],
        bins=[-1, 6, 9, 16, 20, 24],
        labels=["night_0_6", "morning_peak", "midday", "evening_peak", "late_evening"],
    ).astype("string")
    service_summary = (
        service_window.groupby("service_window", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
    )
    print_frame("First/last service windows", service_summary, limit=20)

    holiday_only = stops[stops["date"].astype(str).isin(CATALONIA_PUBLIC_HOLIDAYS_2026)].copy()
    if not holiday_only.empty:
        holiday_summary = (
            holiday_only.groupby("date", dropna=False)
            .agg(
                avg_delay_minutes=("delay_minutes", "mean"),
                records=("delay_minutes", "size"),
            )
            .reset_index()
        )
        print_frame("Catalonia holiday anomalies", holiday_summary, limit=20)

    consistency = analysis.copy()
    if "avg_arrival_delay_minutes" in consistency.columns:
        consistency["train_minus_arrival_delay"] = consistency["avg_delay_minutes"] - consistency["avg_arrival_delay_minutes"]
        consistency_summary = consistency["train_minus_arrival_delay"].describe().to_frame(name="train_minus_arrival_delay").reset_index()
        print_frame("Train vs timetable delay consistency", consistency_summary, limit=20)

    day_type_station = summarize_by_day_type(stops, "station_name").sort_values(
        ["day_type", "avg_delay_minutes"], ascending=[True, False]
    )
    print_frame("Station delay split by day type", day_type_station, limit=30)


def save_time_series_plot(frame: pd.DataFrame, output_dir: Path, line_id: str) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(frame["timestamp_minute"], frame["avg_delay_minutes"], label="Average train delay", color="#d1495b")
    if "avg_arrival_delay_minutes" in frame.columns:
        ax1.plot(frame["timestamp_minute"], frame["avg_arrival_delay_minutes"], label="Average arrival delay", color="#edae49")
    ax1.axhline(0, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax1.set_ylabel("Delay (minutes)")
    ax1.set_xlabel("Time")

    ax2 = ax1.twinx()
    ax2.plot(frame["timestamp_minute"], frame["precipitation"], label="Precipitation", color="#00798c", alpha=0.8)
    ax2.plot(frame["timestamp_minute"], frame["temperature"], label="Temperature", color="#30638e", alpha=0.8)
    ax2.set_ylabel("Weather values")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title(f"{line_id} delays and weather over time")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_delay_weather_timeseries.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def choose_delay_column(frame: pd.DataFrame) -> str:
    if "avg_arrival_delay_minutes" in frame.columns and frame["avg_arrival_delay_minutes"].notna().any():
        return "avg_arrival_delay_minutes"
    if "avg_departure_delay_minutes" in frame.columns and frame["avg_departure_delay_minutes"].notna().any():
        return "avg_departure_delay_minutes"
    return "avg_delay_minutes"


def filter_day_type(frame: pd.DataFrame, day_type: str | None) -> pd.DataFrame:
    if day_type is None:
        return frame.copy()
    return frame[frame["day_type"] == day_type].copy()


def plot_name_suffix(day_type: str | None) -> str:
    return "" if day_type is None else f"_{day_type}"


def plot_title_suffix(day_type: str | None) -> str:
    if day_type is None:
        return ""
    return f" ({day_type.replace('_', ' ')})"


def time_bucket_title(bucket_name: str) -> str:
    return bucket_name.replace("_", " ")


def filter_time_bucket(frame: pd.DataFrame, bucket_name: str | None) -> pd.DataFrame:
    if bucket_name is None:
        return frame.copy()
    for name, start_hour, end_hour in TIME_OF_DAY_BUCKETS:
        if name == bucket_name:
            return frame[frame["hour"].between(start_hour, end_hour)].copy()
    raise SystemExit(f"Unknown time bucket '{bucket_name}'")


def time_bucket_name_suffix(bucket_name: str | None) -> str:
    return "" if bucket_name is None else f"_{bucket_name}"


def save_station_delay_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    plot_frame = (
        frame.groupby("station_name", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
    )
    plot_frame = plot_frame.sort_values("avg_delay_minutes", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_frame, x="avg_delay_minutes", y="station_name", ax=ax, color="#d1495b")
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Average delay (minutes)")
    ax.set_ylabel("Station")
    ax.set_title(f"{line_id} stations with highest average delay{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_station_delay_bar{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_weather_delay_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    plot_frame = frame.dropna(subset=["delay_minutes", "precip_bucket"]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=plot_frame, x="precip_bucket", y="delay_minutes", ax=ax)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylabel("Arrival delay (minutes)")
    ax.set_xlabel("Precipitation condition")
    ax.set_title(f"{line_id} delay by precipitation condition{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_delay_by_precip_condition{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_periodicity_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    plot_frame = (
        frame.groupby(["weekday", "hour"], dropna=False)
        .agg(avg_delay_minutes=("delay_minutes", "mean"))
        .reset_index()
    )
    heatmap = plot_frame.pivot(index="weekday", columns="hour", values="avg_delay_minutes").reindex(WEEKDAY_ORDER)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title(f"{line_id} average delay by weekday and hour{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_periodicity_heatmap{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_station_hour_heatmap(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    plot_frame = (
        frame.groupby(["station_name", "hour"], dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
    )

    station_order = (
        plot_frame.groupby("station_name", dropna=False)["avg_delay_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    plot_frame = plot_frame[plot_frame["station_name"].isin(station_order)].copy()
    heatmap = (
        plot_frame.pivot(index="station_name", columns="hour", values="avg_delay_minutes")
        .reindex(station_order)
    )

    fig_height = max(8, min(22, 0.35 * len(station_order)))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Station")
    ax.set_title(f"{line_id} average delay by station and hour{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_station_hour_heatmap{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_hourly_pattern_plot(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    plot_frame = (
        frame.groupby("hour", dropna=False)
        .agg(avg_delay_minutes=("delay_minutes", "mean"), records=("delay_minutes", "size"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=plot_frame, x="hour", y="avg_delay_minutes", marker="o", ax=ax, color="#d1495b")
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylabel("Average delay (minutes)")
    ax.set_xlabel("Hour of day")
    ax.set_title(f"{line_id} average delay by hour of day{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_hourly_delay_pattern{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_delay_driver_heatmap(frame: pd.DataFrame, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    sns.set_theme(style=PLOT_STYLE)
    frame = filter_day_type(frame, day_type)
    driver_rows = []
    for feature in ["temperature", "precipitation", "windspeed", "cloudcover"]:
        value = safe_corr(frame["delay_minutes"], pd.to_numeric(frame[feature], errors="coerce"))
        if value is not None:
            driver_rows.append({"driver": feature, "corr_with_delay": value})
    heatmap = pd.DataFrame(driver_rows).set_index("driver")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"{line_id} weather drivers vs delay{plot_title_suffix(day_type)}")
    fig.tight_layout()

    output_path = output_dir / f"{line_id.lower()}_delay_driver_heatmap{plot_name_suffix(day_type)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def delay_color(value: float) -> str:
    if pd.isna(value):
        return "gray"
    if value < -1:
        return "green"
    if value < 1:
        return "blue"
    if value < 3:
        return "orange"
    return "red"


def save_station_delay_map(frame: pd.DataFrame, data_root: Path, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    frame = filter_day_type(frame, day_type)
    station_geo = load_station_geo_lookup(data_root)
    plot_frame = (
        frame.groupby("station_id", dropna=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            records=("delay_minutes", "size"),
        )
        .reset_index()
        .merge(station_geo, on="station_id", how="left")
        .dropna(subset=["latitude", "longitude"])
    )

    center_lat = float(plot_frame["latitude"].mean())
    center_lon = float(plot_frame["longitude"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

    for row in plot_frame.itertuples(index=False):
        popup = (
            f"<b>{row.station_name}</b><br>"
            f"Average delay: {row.avg_delay_minutes:.2f} min<br>"
            f"Median delay: {row.median_delay_minutes:.2f} min<br>"
            f"Records: {int(row.records)}"
        )
        radius = 5 + min(15, (row.records ** 0.5) / 8)
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=delay_color(row.avg_delay_minutes),
            fill=True,
            fill_opacity=0.75,
            popup=popup,
            tooltip=row.station_name,
        ).add_to(fmap)

    title = f"{line_id} station delay map{plot_title_suffix(day_type)}"
    fmap.get_root().html.add_child(folium.Element(f"<h3 align='center'>{title}</h3>"))
    output_path = output_dir / f"{line_id.lower()}_station_delay_map{plot_name_suffix(day_type)}.html"
    fmap.save(str(output_path))
    return output_path


def save_segment_delay_map(frame: pd.DataFrame, data_root: Path, output_dir: Path, line_id: str, day_type: str | None = None) -> Path:
    frame = filter_day_type(frame, day_type)
    plot_frame = build_segment_delay_frame(frame, data_root, line_id)
    return save_segment_delay_map_from_frame(plot_frame, output_dir, line_id, day_type, direction_name=None)


def save_segment_delay_map_from_frame(
    plot_frame: pd.DataFrame,
    output_dir: Path,
    line_id: str,
    day_type: str | None = None,
    direction_name: str | None = None,
    time_bucket: str | None = None,
) -> Path:
    if direction_name is not None:
        plot_frame = plot_frame[plot_frame["direction"] == direction_name].copy()
    if plot_frame.empty:
        suffix = plot_name_suffix(day_type)
        direction_suffix = "" if direction_name is None else "_direction"
        time_suffix = time_bucket_name_suffix(time_bucket)
        output_path = output_dir / f"{line_id.lower()}_segment_delay_map{direction_suffix}{suffix}{time_suffix}.html"
        empty_map = folium.Map(location=[41.4, 2.1], zoom_start=8, tiles="CartoDB positron")
        empty_map.get_root().html.add_child(folium.Element("<h3 align='center'>No segment data available</h3>"))
        empty_map.save(str(output_path))
        return output_path

    center_lat = float(
        pd.concat([plot_frame["from_latitude"], plot_frame["to_latitude"]], ignore_index=True).mean()
    )
    center_lon = float(
        pd.concat([plot_frame["from_longitude"], plot_frame["to_longitude"]], ignore_index=True).mean()
    )
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
    low = float(plot_frame["avg_delay_minutes"].quantile(0.05))
    high = float(plot_frame["avg_delay_minutes"].quantile(0.95))
    if low == high:
        low -= 1
        high += 1
    colormap = LinearColormap(
        colors=["#2563eb", "#93c5fd", "#fde68a", "#f97316", "#b91c1c"],
        vmin=low,
        vmax=high,
        caption="Average delay (minutes)",
    )

    station_points = pd.concat(
        [
            plot_frame[["from_station_name", "from_latitude", "from_longitude"]].rename(
                columns={
                    "from_station_name": "station_name",
                    "from_latitude": "latitude",
                    "from_longitude": "longitude",
                }
            ),
            plot_frame[["to_station_name", "to_latitude", "to_longitude"]].rename(
                columns={
                    "to_station_name": "station_name",
                    "to_latitude": "latitude",
                    "to_longitude": "longitude",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    for row in plot_frame.itertuples(index=False):
        popup = (
            f"<b>{row.segment}</b><br>"
            f"Direction: {row.direction}<br>"
            f"Average delay: {row.avg_delay_minutes:.2f} min<br>"
            f"Median delay: {row.median_delay_minutes:.2f} min<br>"
            f"Records: {int(row.records)}"
        )
        weight = 2 + min(4, (row.records ** 0.5) / 10)
        folium.PolyLine(
            locations=[(row.from_latitude, row.from_longitude), (row.to_latitude, row.to_longitude)],
            color=colormap(row.avg_delay_minutes),
            weight=weight,
            opacity=0.8,
            popup=popup,
            tooltip=row.segment,
        ).add_to(fmap)

    for row in station_points.itertuples(index=False):
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=2.5,
            color="#374151",
            weight=1,
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            tooltip=row.station_name,
        ).add_to(fmap)

    direction_title = "" if direction_name is None else f" [{direction_name}]"
    time_title = "" if time_bucket is None else f" [{time_bucket_title(time_bucket)}]"
    title = f"{line_id} segment delay map{direction_title}{plot_title_suffix(day_type)}{time_title}"
    fmap.get_root().html.add_child(folium.Element(f"<h3 align='center'>{title}</h3>"))
    legend_html = """
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: white;
        border: 1px solid #bbb;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        line-height: 1.4;
    ">
      Darker warm colors mean higher delay.<br>
      Small white dots mark station positions.<br>
      Segment geometry follows the canonical route order.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    colormap.add_to(fmap)
    direction_suffix = "" if direction_name is None else f"_{direction_name.replace(' ', '_').replace('->', 'to').replace('/', '_')}"
    output_path = output_dir / f"{line_id.lower()}_segment_delay_map{direction_suffix}{plot_name_suffix(day_type)}{time_bucket_name_suffix(time_bucket)}.html"
    fmap.save(str(output_path))
    return output_path


def add_map_title_and_legend(fmap: folium.Map, title: str, legend_html: str | None = None) -> None:
    fmap.get_root().html.add_child(folium.Element(f"<h3 align='center'>{title}</h3>"))
    if legend_html:
        fmap.get_root().html.add_child(folium.Element(legend_html))


def route_color(direction_label: str) -> str:
    if direction_label == "inbound":
        return "#1d4ed8"
    if direction_label == "outbound":
        return "#dc2626"
    return "#475569"


def add_named_station_markers(
    fmap: folium.Map,
    station_order: pd.DataFrame,
    color: str,
    layer_name: str,
) -> None:
    feature_group = folium.FeatureGroup(name=layer_name, show=True)
    for index, row in enumerate(station_order.itertuples(index=False), start=1):
        median_stop_sequence = getattr(row, "median_stop_sequence", None)
        trains = getattr(row, "trains", None)
        records = getattr(row, "records", None)
        extra_lines: list[str] = []
        if median_stop_sequence is not None and not pd.isna(median_stop_sequence):
            extra_lines.append(f"Median stop sequence: {float(median_stop_sequence):.1f}")
        if trains is not None and not pd.isna(trains):
            extra_lines.append(f"Trains: {int(trains)}")
        if records is not None and not pd.isna(records):
            extra_lines.append(f"Records: {int(records)}")
        label_html = (
            "<div style=\""
            "font-size: 11px; font-weight: 600; color: #111827; "
            "background: rgba(255,255,255,0.9); border: 1px solid #d1d5db; "
            "border-radius: 4px; padding: 1px 4px; white-space: nowrap;\">"
            f"{index}. {html.escape(str(row.station_name))}"
            "</div>"
        )
        popup = f"<b>{html.escape(str(row.station_name))}</b><br>Order: {index}"
        if extra_lines:
            popup += "<br>" + "<br>".join(extra_lines)
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=4,
            color=color,
            weight=2,
            fill=True,
            fill_color="white",
            fill_opacity=0.95,
            tooltip=f"{index}. {row.station_name}",
            popup=popup,
        ).add_to(feature_group)
        folium.Marker(
            location=[row.latitude, row.longitude],
            icon=folium.DivIcon(html=label_html),
        ).add_to(feature_group)
    feature_group.add_to(fmap)


def save_route_order_maps(
    frame: pd.DataFrame,
    data_root: Path,
    output_dir: Path,
    line_id: str,
) -> list[Path]:
    routes = build_route_direction_sequences(frame, data_root, line_id)
    if not routes:
        output_path = output_dir / f"{line_id.lower()}_route_order_map.html"
        empty_map = folium.Map(location=[41.4, 2.1], zoom_start=8, tiles="CartoDB positron")
        add_map_title_and_legend(empty_map, f"{line_id} route order map", "<div></div>")
        empty_map.save(str(output_path))
        return [output_path]

    all_lats = pd.concat([route["station_order"]["latitude"] for route in routes], ignore_index=True)
    all_lons = pd.concat([route["station_order"]["longitude"] for route in routes], ignore_index=True)
    center_lat = float(all_lats.mean())
    center_lon = float(all_lons.mean())

    combined_map = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
    outputs: list[Path] = []
    legend_rows: list[str] = []

    for route in routes:
        direction_label = str(route["direction_label"])
        display_name = str(route["display_name"])
        station_order = route["station_order"]
        color = route_color(direction_label)
        dash = None if direction_label == "inbound" else "8 6"
        points = list(zip(station_order["latitude"], station_order["longitude"]))

        folium.PolyLine(
            locations=points,
            color=color,
            weight=4,
            opacity=0.8,
            dash_array=dash,
            tooltip=display_name,
            popup=display_name,
        ).add_to(combined_map)
        add_named_station_markers(
            combined_map,
            station_order,
            color,
            f"{direction_label} stations",
        )

        single_map = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
        folium.PolyLine(
            locations=points,
            color=color,
            weight=5,
            opacity=0.85,
            dash_array=dash,
            tooltip=display_name,
            popup=display_name,
        ).add_to(single_map)
        add_named_station_markers(single_map, station_order, color, f"{direction_label} stations")
        add_map_title_and_legend(
            single_map,
            f"{line_id} route order map [{display_name}]",
            """
            <div style="
                position: fixed;
                bottom: 18px;
                left: 18px;
                z-index: 9999;
                background: white;
                border: 1px solid #bbb;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 12px;
                line-height: 1.4;
            ">
              Station labels are shown in route order.
            </div>
            """,
        )
        folium.LayerControl(collapsed=False).add_to(single_map)
        single_output = output_dir / f"{line_id.lower()}_route_order_map_{direction_label}.html"
        single_map.save(str(single_output))
        outputs.append(single_output)

        legend_rows.append(
            f"<span style=\"color:{color}; font-weight:700;\">{html.escape(direction_label)}</span>: "
            f"{html.escape(str(route['start_station']))} -> {html.escape(str(route['end_station']))}"
            + (" (dashed)" if dash else "")
        )

    legend_html = """
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: white;
        border: 1px solid #bbb;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        line-height: 1.4;
    ">
      <b>Direction legend</b><br>
      %s<br>
      Station labels are shown in route order.
    </div>
    """ % "<br>".join(legend_rows)
    add_map_title_and_legend(combined_map, f"{line_id} route order map", legend_html)
    folium.LayerControl(collapsed=False).add_to(combined_map)
    combined_output = output_dir / f"{line_id.lower()}_route_order_map.html"
    combined_map.save(str(combined_output))
    outputs.insert(0, combined_output)
    return outputs


def route_map_report(data_root: Path, line_id: str, output_dir: Path) -> None:
    stops = build_stop_delay_frame(data_root, line_id)
    outputs = save_route_order_maps(stops, data_root, output_dir, line_id)
    print("\nSaved route maps:")
    for path in outputs:
        print(path)


def matching_map_files(output_dir: Path, line_id: str, kind: str) -> list[Path]:
    patterns = {
        "all": [f"{line_id.lower()}*.html"],
        "segment": [f"{line_id.lower()}_segment_delay_map*.html"],
        "station": [f"{line_id.lower()}_station_delay_map*.html"],
        "route": [f"{line_id.lower()}_route_order_map*.html"],
        "time-segment": [f"{line_id.lower()}_segment_delay_map*early_morning.html",
                         f"{line_id.lower()}_segment_delay_map*morning_peak.html",
                         f"{line_id.lower()}_segment_delay_map*midday.html",
                         f"{line_id.lower()}_segment_delay_map*evening_peak.html",
                         f"{line_id.lower()}_segment_delay_map*late_evening.html"],
    }
    files: list[Path] = []
    for pattern in patterns[kind]:
        files.extend(sorted(output_dir.glob(pattern)))
    deduped = sorted({path.resolve() for path in files})
    return [Path(path) for path in deduped]


def open_maps_report(output_dir: Path, line_id: str, kind: str) -> None:
    html_files = matching_map_files(output_dir, line_id, kind)
    if not html_files:
        raise SystemExit(f"No HTML maps found for line '{line_id}' in {output_dir} for kind '{kind}'")

    print("Opening maps:")
    for path in html_files:
        print(path)
        subprocess.Popen(["xdg-open", str(path)])


def plot_report(data_root: Path, line_id: str, output_dir: Path) -> None:
    analysis = build_analysis_frame(data_root, line_id)
    stops = build_stop_delay_frame(data_root, line_id)
    segment_frame = build_segment_delay_frame(stops, data_root, line_id)
    route_outputs = save_route_order_maps(stops, data_root, output_dir, line_id)
    working_segment_frame = build_segment_delay_frame(filter_day_type(stops, "working_day"), data_root, line_id)
    weekend_segment_frame = build_segment_delay_frame(filter_day_type(stops, "weekend_or_holiday"), data_root, line_id)

    outputs = [
        save_time_series_plot(analysis, output_dir, line_id),
        save_hourly_pattern_plot(stops, output_dir, line_id),
        save_hourly_pattern_plot(stops, output_dir, line_id, "working_day"),
        save_hourly_pattern_plot(stops, output_dir, line_id, "weekend_or_holiday"),
        save_periodicity_plot(stops, output_dir, line_id),
        save_periodicity_plot(stops, output_dir, line_id, "working_day"),
        save_periodicity_plot(stops, output_dir, line_id, "weekend_or_holiday"),
        save_station_hour_heatmap(stops, output_dir, line_id),
        save_station_hour_heatmap(stops, output_dir, line_id, "working_day"),
        save_station_hour_heatmap(stops, output_dir, line_id, "weekend_or_holiday"),
        save_station_delay_plot(stops, output_dir, line_id),
        save_station_delay_plot(stops, output_dir, line_id, "working_day"),
        save_station_delay_plot(stops, output_dir, line_id, "weekend_or_holiday"),
        save_weather_delay_plot(stops, output_dir, line_id),
        save_weather_delay_plot(stops, output_dir, line_id, "working_day"),
        save_weather_delay_plot(stops, output_dir, line_id, "weekend_or_holiday"),
        save_delay_driver_heatmap(stops, output_dir, line_id),
        save_delay_driver_heatmap(stops, output_dir, line_id, "working_day"),
        save_delay_driver_heatmap(stops, output_dir, line_id, "weekend_or_holiday"),
        save_station_delay_map(stops, data_root, output_dir, line_id),
        save_station_delay_map(stops, data_root, output_dir, line_id, "working_day"),
        save_station_delay_map(stops, data_root, output_dir, line_id, "weekend_or_holiday"),
        save_segment_delay_map(stops, data_root, output_dir, line_id),
        save_segment_delay_map(stops, data_root, output_dir, line_id, "working_day"),
        save_segment_delay_map(stops, data_root, output_dir, line_id, "weekend_or_holiday"),
    ]
    outputs.extend(route_outputs)
    for direction_name in ["inbound", "outbound"]:
        outputs.append(save_segment_delay_map_from_frame(segment_frame, output_dir, line_id, None, direction_name))
        outputs.append(
            save_segment_delay_map_from_frame(
                working_segment_frame,
                output_dir,
                line_id,
                "working_day",
                direction_name,
            )
        )
        outputs.append(
            save_segment_delay_map_from_frame(
                weekend_segment_frame,
                output_dir,
                line_id,
                "weekend_or_holiday",
                direction_name,
            )
        )
    for bucket_name, _, _ in TIME_OF_DAY_BUCKETS:
        bucket_stops = filter_time_bucket(stops, bucket_name)
        bucket_segment_frame = build_segment_delay_frame(bucket_stops, data_root, line_id)
        for direction_name in ["inbound", "outbound"]:
            outputs.append(
                save_segment_delay_map_from_frame(
                    bucket_segment_frame,
                    output_dir,
                    line_id,
                    None,
                    direction_name,
                    bucket_name,
                )
            )
    print("\nSaved plots:")
    for path in outputs:
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore the Rodalies parquet datasets.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing lines/, stations/, journeys/, timetables/, trains/, weather/",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory", help="Show which files are available")
    subparsers.add_parser("schema", help="Show schema from one sample file per dataset")
    subparsers.add_parser("sizes", help="Show row counts and file sizes for daily datasets")
    subparsers.add_parser("profile", help="Inspect structure, dtypes, nulls, and sample distributions")
    subparsers.add_parser("lines", help="Print the line catalog")
    subparsers.add_parser("stations", help="Print the first stations rows")

    r2 = subparsers.add_parser("r2", help="Quick exploration for the R2 line")
    r2.add_argument("--day", help="Day in YYYY_MM_DD format, for example 2026_04_09")

    correlate = subparsers.add_parser(
        "correlate",
        help="Join trains, timetables, and weather over time and print correlations",
    )
    correlate.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Run route, segment, holiday, consistency, and missing-data diagnostics",
    )
    diagnose.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")

    route_map = subparsers.add_parser(
        "route-map",
        help="Generate pure station-order folium route maps with direction labels",
    )
    route_map.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")
    route_map.add_argument("--output-dir", default="plots", help="Directory where route maps will be saved")

    plot = subparsers.add_parser(
        "plot",
        help="Generate time-series, heatmaps, and folium maps for one line, including route-order maps",
    )
    plot.add_argument("--line-id", default="R2", help="Line ID to analyze, default: R2")
    plot.add_argument("--output-dir", default="plots", help="Directory where plots will be saved")

    open_maps = subparsers.add_parser(
        "open-maps",
        help="Open generated HTML maps for one line in the default browser",
    )
    open_maps.add_argument("--line-id", default="R2", help="Line ID to open, default: R2")
    open_maps.add_argument("--output-dir", default="plots", help="Directory where HTML maps were saved")
    open_maps.add_argument(
        "--kind",
        default="all",
        choices=["all", "segment", "station", "route", "time-segment"],
        help="Which family of HTML maps to open",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()

    if args.command == "inventory":
        inventory(data_root)
    elif args.command == "schema":
        schema_report(data_root)
    elif args.command == "sizes":
        dataset_sizes(data_root)
    elif args.command == "profile":
        deep_profile(data_root)
    elif args.command == "lines":
        lines_report(data_root)
    elif args.command == "stations":
        stations_report(data_root)
    elif args.command == "r2":
        r2_report(data_root, args.day)
    elif args.command == "correlate":
        correlation_report(data_root, args.line_id)
    elif args.command == "diagnose":
        diagnostics_report(data_root, args.line_id)
    elif args.command == "route-map":
        route_map_report(data_root, args.line_id, ensure_output_dir(args.output_dir))
    elif args.command == "plot":
        plot_report(data_root, args.line_id, ensure_output_dir(args.output_dir))
    elif args.command == "open-maps":
        open_maps_report(ensure_output_dir(args.output_dir), args.line_id, args.kind)


if __name__ == "__main__":
    main()
