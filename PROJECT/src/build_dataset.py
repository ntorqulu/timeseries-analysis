import os
import glob
import re
import unicodedata
from difflib import get_close_matches
from datetime import datetime
import pandas as pd
import numpy as np
from loader import get_r1_station_ids


def _normalize_station_name(name):
    if pd.isna(name):
        return np.nan

    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = normalized.replace("|", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace(".", " ")
    normalized = re.sub(r"\bst\b", "sant", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    aliases = {
        "barcelona el clot": "barcelona el clot arago",
        "barcelona placa de catalunya": "barcelona placa catalunya",
    }
    return aliases.get(normalized, normalized)


def _load_station_metadata(data_dir):
    stations_path = os.path.join(data_dir, "static", "stations.parquet")
    stations_df = pd.read_parquet(stations_path, columns=["station_id", "name"])
    stations_df = stations_df.rename(columns={"name": "station_name"})
    stations_df["station_id"] = stations_df["station_id"].astype(str)
    stations_df["station_name_normalized"] = stations_df["station_name"].map(_normalize_station_name)
    return stations_df.drop_duplicates(subset=["station_id"])


def _load_holiday_calendar(project_dir):
    holidays_path = os.path.join(
        project_dir,
        "EDA",
        "Calendario_Laboral_Catalunya",
        "Calendari_laboral_de_Catalunya_2026.csv",
    )
    holidays_df = pd.read_csv(holidays_path)
    holidays_df["service_date"] = pd.to_datetime(holidays_df["date"], errors="coerce").dt.date
    return holidays_df[["service_date", "is_holiday"]].drop_duplicates()


def _get_direction_station_orders(project_dir):
    timetable_dir = os.path.join(project_dir, "EDA", "official_timetables")

    direction1_df = pd.read_csv(os.path.join(timetable_dir, "R1_direction1_schedules.csv"), nrows=0)
    direction2_df = pd.read_csv(os.path.join(timetable_dir, "R1_direction2_schedules.csv"), nrows=0)

    direction1_order = [col for col in direction1_df.columns if col != "Day_Type"]
    direction2_order = [col for col in direction2_df.columns if col != "Day_Type"]

    return (
        {_normalize_station_name(station): idx for idx, station in enumerate(direction1_order)},
        {_normalize_station_name(station): idx for idx, station in enumerate(direction2_order)},
    )


def _get_official_direction_stations(project_dir):
    timetable_dir = os.path.join(project_dir, "EDA", "official_timetables")

    direction1_df = pd.read_csv(os.path.join(timetable_dir, "R1_direction1_schedules.csv"), nrows=0)
    direction2_df = pd.read_csv(os.path.join(timetable_dir, "R1_direction2_schedules.csv"), nrows=0)

    direction1_order = [col for col in direction1_df.columns if col != "Day_Type"]
    direction2_order = [col for col in direction2_df.columns if col != "Day_Type"]

    return sorted(set(direction1_order) | set(direction2_order))


def _infer_directions(tt_df, direction1_order, direction2_order):
    train_direction = {}

    for train_id, group in tt_df.groupby("train_id"):
        station_names = [
            station
            for station in group["station_name_normalized"].tolist()
            if pd.notna(station) and station in direction1_order and station in direction2_order
        ]

        if len(station_names) < 2:
            train_direction[train_id] = np.nan
            continue

        first_station = station_names[0]
        last_station = station_names[-1]

        dir1_delta = direction1_order[last_station] - direction1_order[first_station]
        dir2_delta = direction2_order[last_station] - direction2_order[first_station]

        if dir1_delta > 0 and dir2_delta < 0:
            train_direction[train_id] = "direction1"
        elif dir2_delta > 0 and dir1_delta < 0:
            train_direction[train_id] = "direction2"
        else:
            train_direction[train_id] = np.nan

    tt_df["direction"] = tt_df["train_id"].map(train_direction)
    return tt_df


def _assign_timetable_stop_sequence(tt_df, direction1_order, direction2_order):
    direction_orders = {
        "direction1": direction1_order,
        "direction2": direction2_order,
    }

    def lookup_sequence(row):
        order = direction_orders.get(row["direction"])
        if order is None:
            return np.nan
        return order.get(row["station_name_normalized"], np.nan)

    tt_df["stop_sequence"] = tt_df.apply(lookup_sequence, axis=1).astype("Float64")
    return tt_df


def _aggregate_timetable_file(file_path, r1_station_ids):
    df = pd.read_parquet(
        file_path,
        columns=[
            "train_id",
            "station_id",
            "planned_arrival",
            "planned_departure",
            "actual_arrival",
            "actual_departure",
            "timestamp",
        ],
    )

    for col in [
        "planned_arrival",
        "planned_departure",
        "actual_arrival",
        "actual_departure",
        "timestamp",
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["train_id"] = df["train_id"].astype(str)
    df["station_id"] = df["station_id"].astype(str)
    if r1_station_ids:
        df = df[df["station_id"].isin(r1_station_ids)]
    df = df.sort_values(by=["train_id", "station_id", "timestamp"])

    return (
        df.groupby(["train_id", "station_id"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "first"),
            last_timestamp=("timestamp", "last"),
            first_planned_arrival=("planned_arrival", "first"),
            last_planned_arrival=("planned_arrival", "last"),
            first_planned_departure=("planned_departure", "first"),
            last_planned_departure=("planned_departure", "last"),
            last_actual_arrival=("actual_arrival", "last"),
            last_actual_departure=("actual_departure", "last"),
        )
    )


def _aggregate_timetable_history(tt_files, r1_station_ids):
    aggregated_files = []

    for idx, file_path in enumerate(sorted(tt_files), start=1):
        print(f"Aggregating timetable file {idx}/{len(tt_files)}: {os.path.basename(file_path)}")
        aggregated = _aggregate_timetable_file(file_path, r1_station_ids)
        if not aggregated.empty:
            aggregated_files.append(aggregated)

    if not aggregated_files:
        raise FileNotFoundError("No R1 timetable parquet rows found in data/dynamic/timetables.")

    tt_partial = pd.concat(aggregated_files, ignore_index=True)
    tt_partial = tt_partial.sort_values(
        by=["train_id", "station_id", "first_timestamp", "last_timestamp"]
    )

    tt_df = (
        tt_partial.groupby(["train_id", "station_id"], as_index=False)
        .agg(
            first_planned_arrival=("first_planned_arrival", "first"),
            last_planned_arrival=("last_planned_arrival", "last"),
            first_planned_departure=("first_planned_departure", "first"),
            last_planned_departure=("last_planned_departure", "last"),
            last_actual_arrival=("last_actual_arrival", "last"),
            last_actual_departure=("last_actual_departure", "last"),
        )
    )

    return tt_df


def save_features_csv(tt_df, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tt_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")


def save_unmatched_station_mapping_csv(tt_df, official_station_names, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    official_name_map = {
        _normalize_station_name(station): station for station in official_station_names
    }
    official_normalized = sorted(official_name_map.keys())

    station_report = (
        tt_df.groupby(["station_name", "station_name_normalized"], dropna=False)
        .agg(
            row_count=("station_id", "size"),
            train_count=("train_id", "nunique"),
            missing_direction_rows=("direction", lambda s: s.isna().sum()),
        )
        .reset_index()
    )

    station_report["matches_timetable"] = station_report["station_name_normalized"].isin(official_name_map)
    unmatched = station_report[~station_report["matches_timetable"]].copy()
    unmatched["suggested_timetable_station"] = unmatched["station_name_normalized"].map(
        lambda name: official_name_map.get(get_close_matches(name, official_normalized, n=1, cutoff=0.6)[0])
        if pd.notna(name) and get_close_matches(name, official_normalized, n=1, cutoff=0.6)
        else np.nan
    )
    unmatched = unmatched.sort_values(
        by=["missing_direction_rows", "row_count", "train_count"],
        ascending=[False, False, False],
    )
    unmatched.to_csv(output_path, index=False)
    print(f"Unmatched station mapping report saved to: {output_path}")


def build_features(data_dir):
    print("Loading datasets...")
    # Read timetable parquet with:
    #   - train_id
    #   - station_id
    #   - planned_arrival
    #   - planned_departure
    #   - actual_arrival
    #   - actual_departure
    #   - timestamp (NEW: I am keeping it initially so we can calculate the "lie" on the delay)
    #   - then derive stop_sequence from the official R1 timetables after direction inference
    # Convert them to datetime columns
    # Sort by train_id, station_id, timestamp
    # Then aggregate per train_id + station_id:
    #   - first "planned arrival"
    #   - last "planned arrival"
    #   - last "actual arrival"
    #   - first "planned departure"
    #   - last "planned departure"
    #   - last "actual departure"
    #   - keep only one representative stop_sequence
    # Computated fields to add:
    #   - delays (in minutes:
    #       - delay_type_1 = last_actual_arrival - first_planned_arrival (should coincide with delay declared by Renfe)
    #       - delay_type_2 = last_actual_arrival - last_planned_arrival (should be the whole delay discarding potential adjustments by Renfe)
    #   - "lie" on the delay (schedule adjustment due to delay):
    #       - schedule_adjustment_minutes = last_planned_arrival - first_planned_arrival
    #       - delay_masking_minutes = delay_type_1 - delay_type_2 (the higher this is, the more Renfe adjusts the schedule due to delay)
    #   - delay with respect to official timetable (TODO later, to decide the matching strategy)
    #       - delay_from_timetable = last_actual_arrival - official_scheduled_arrival
    #   - workday / weekend / holiday / holiday&weekend for Catalunya
    #   - direction (from R1_direction1_schedules.csv, R1_direction2_schedules.csv)

    tt_files = glob.glob(os.path.join(data_dir, "dynamic", "timetables", "*.parquet"))
    r1_station_ids = get_r1_station_ids()
    tt_df = _aggregate_timetable_history(tt_files, r1_station_ids)
    print(f"Timetable history aggregated for R1: {len(tt_df)} train/station rows")

    tt_df["planned_arrival_dt"] = tt_df["last_planned_arrival"]
    tt_df["actual_arrival_dt"] = tt_df["last_actual_arrival"]

    tt_df["delay_type_1"] = (
        tt_df["last_actual_arrival"] - tt_df["first_planned_arrival"]
    ).dt.total_seconds() / 60.0
    tt_df["delay_type_2"] = (
        tt_df["last_actual_arrival"] - tt_df["last_planned_arrival"]
    ).dt.total_seconds() / 60.0
    tt_df["delay_masking_minutes"] = tt_df["delay_type_1"] - tt_df["delay_type_2"] # the higher this is, the more Renfe adjusts the schedule due to delay

    # 1. Temporal bounds and null checks (data cleaning)
    tt_df["target_delay"] = tt_df["delay_type_2"]
    tt_df = tt_df.dropna(subset=["planned_arrival_dt", "actual_arrival_dt", "target_delay"])
    tt_df = tt_df[(tt_df["target_delay"] >= -60) & (tt_df["target_delay"] <= 300)]

    # 2. Extract base temporal features
    tt_df["hour"] = tt_df["planned_arrival_dt"].dt.hour
    tt_df["day_of_week"] = tt_df["planned_arrival_dt"].dt.dayofweek
    tt_df["hour_trunc"] = tt_df["planned_arrival_dt"].dt.floor("h")
    tt_df["service_date"] = tt_df["planned_arrival_dt"].dt.date
    tt_df["is_weekend"] = tt_df["day_of_week"] >= 5
    tt_df["hour"] = tt_df["hour"].astype("int8")
    tt_df["day_of_week"] = tt_df["day_of_week"].astype("int8")
    tt_df["is_weekend"] = tt_df["is_weekend"].astype("int8")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))

    stations_df = _load_station_metadata(data_dir)
    tt_df = tt_df.merge(stations_df, on="station_id", how="left")

    holidays_df = _load_holiday_calendar(project_dir)
    tt_df = tt_df.merge(holidays_df, on="service_date", how="left")
    tt_df["is_holiday"] = tt_df["is_holiday"].fillna(0).astype(int)
    tt_df["day_type"] = np.where(
        (tt_df["is_holiday"] == 1) & tt_df["is_weekend"],
        "holiday&weekend",
        np.where(
            tt_df["is_holiday"] == 1,
            "holiday",
            np.where(tt_df["is_weekend"], "weekend", "workday"),
        ),
    )

    direction1_order, direction2_order = _get_direction_station_orders(project_dir)

    # 3. Create seq/lag features for the previous station delay
    # sort chronologically to get valid sequence per train journey
    tt_df = tt_df.sort_values(by=["train_id", "planned_arrival_dt"])
    tt_df["prev_station_delay"] = (
        tt_df.groupby("train_id")["target_delay"].shift(1).fillna(0)
    )  # fill first station delay with 0

    tt_df = _infer_directions(tt_df, direction1_order, direction2_order)
    tt_df = _assign_timetable_stop_sequence(tt_df, direction1_order, direction2_order)
    tt_df["day_type"] = tt_df["day_type"].astype("category")
    tt_df["direction"] = tt_df["direction"].astype("category")

    # 4. Integrate line maps -> TODO

    # 5. Delay from official timetable
    # TODO

    return tt_df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(data_dir, "processed", f"features_{timestamp}.csv")
    station_mapping_output_path = os.path.join(
        data_dir,
        "processed",
        f"unmatched_direction_stations_{timestamp}.csv",
    )
    tt_df = build_features(data_dir)
    save_features_csv(tt_df, output_path)
    official_station_names = _get_official_direction_stations(project_dir)
    save_unmatched_station_mapping_csv(tt_df, official_station_names, station_mapping_output_path)

    print("\nShape:")
    print(tt_df.shape)

    print("\nColumns:")
    print(tt_df.columns.tolist())

    print("\nFirst rows:")
    print(tt_df.head())

    print("\nMissing values:")
    print(tt_df.isna().sum())

    print("\nTarget delay summary:")
    print(tt_df["target_delay"].describe())
