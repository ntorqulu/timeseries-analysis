"""
loader.py — Load parquet files from disk and filter to R1 line data.

All public functions return clean DataFrames ready for EDA / modelling.
Call download_all() first if data isn't on disk yet.

Example
-------
    from src.loader import load_static, load_trains, load_timetables

    stations, lines = load_static()
    trains      = load_trains()
    timetables  = load_timetables()
    journeys    = load_journeys()
    weather     = load_weather()
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.config import (
    STATIC_DIR,
    DYNAMIC_DIR,
    DATE_FMT,
    R1_LINE_ID,
)

log = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning(f"File not found: {path}")
        return None
    return pd.read_parquet(path)


def _list_dynamic_files(table: str) -> list[Path]:
    folder = DYNAMIC_DIR / table
    return sorted(folder.glob(f"{table}_*.parquet")) if folder.exists() else []


def _date_filter(files: list[Path], start_date: str | None, end_date: str | None) -> list[Path]:
    if not start_date and not end_date:
        return files
    sd = datetime.strptime(start_date, DATE_FMT) if start_date else datetime(2000, 1, 1)
    ed = datetime.strptime(end_date, DATE_FMT) if end_date else datetime.now()
    filtered = []
    for f in files:
        stem = f.stem                           # "trains_2026_03_14"
        token = "_".join(stem.split("_")[1:])   # "2026_03_14"
        try:
            dt = datetime.strptime(token, DATE_FMT)
        except ValueError:
            continue
        if sd <= dt <= ed:
            filtered.append(f)
    return filtered


def _concat(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── R1 station cache ──────────────────────────────────────────────────────────

_r1_station_ids: set[str] | None = None


def get_r1_station_ids() -> set[str]:
    """
    Return the set of station IDs that belong to the R1 line.

    The lines.parquet table only contains metadata (name, origin, destination,
    stations_count) with no per-station IDs, and stations.parquet has no
    line_id column. So we derive R1 stations from the trains dynamic table:
    sample up to 3 recent files, filter to line_id == R1, and collect all
    station IDs seen in current_station_id and next_station_id.
    """
    global _r1_station_ids
    if _r1_station_ids is not None:
        return _r1_station_ids

    trains_dir = DYNAMIC_DIR / "trains"
    files = sorted(trains_dir.glob("trains_*.parquet")) if trains_dir.exists() else []

    if not files:
        log.warning("No trains files found — cannot derive R1 station IDs")
        _r1_station_ids = set()
        return _r1_station_ids

    # Sample up to 3 recent files to get a complete picture of all R1 stops
    sample = files[-3:]
    frames = [
        pd.read_parquet(f, columns=["line_id", "current_station_id", "next_station_id"])
        for f in sample
    ]
    df = pd.concat(frames, ignore_index=True)

    if "line_id" not in df.columns:
        log.warning("trains table has no line_id column — cannot filter R1 stations")
        _r1_station_ids = set()
        return _r1_station_ids

    r1 = df[df["line_id"].astype(str).str.upper() == R1_LINE_ID.upper()]
    ids = (
        set(r1["current_station_id"].astype(str).dropna())
        | set(r1["next_station_id"].astype(str).dropna())
    )
    ids.discard("nan")
    ids.discard("<NA>")
    ids.discard("")

    _r1_station_ids = ids
    log.info(f"R1 station IDs resolved from trains data: {len(ids)} stations")
    return _r1_station_ids


def reset_r1_station_cache() -> None:
    """Force re-derivation of R1 station IDs on next call (useful in notebooks)."""
    global _r1_station_ids
    _r1_station_ids = None


# ── Public loaders ────────────────────────────────────────────────────────────

def load_static() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stations and lines static tables.

    stations is filtered to R1 using station IDs derived from the trains table.

    Returns
    -------
    stations : DataFrame  (R1 stations only)
    lines    : DataFrame  (R1 line row only)
    """
    stations = _read_parquet(STATIC_DIR / "stations.parquet") or pd.DataFrame()
    lines_df = _read_parquet(STATIC_DIR / "lines.parquet") or pd.DataFrame()

    if not lines_df.empty and "line_id" in lines_df.columns:
        r1_lines = lines_df[lines_df["line_id"].str.upper() == R1_LINE_ID.upper()]
    else:
        r1_lines = lines_df

    r1_ids = get_r1_station_ids()
    if r1_ids and not stations.empty:
        stations = stations[stations["station_id"].astype(str).isin(r1_ids)].copy()

    log.info(f"Loaded {len(stations)} R1 stations, {len(r1_lines)} R1 line rows")
    return stations, r1_lines


def load_trains(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load train observation data, filtered to R1 line.

    Parameters
    ----------
    start_date / end_date : "YYYY_MM_DD" inclusive range; None -> all on disk
    """
    files = _list_dynamic_files("trains")
    files = _date_filter(files, start_date, end_date)
    if not files:
        log.warning("No train files found for the requested range")
        return pd.DataFrame()

    df = _concat(files)
    if df.empty:
        return df

    if "line_id" in df.columns:
        df = df[df["line_id"].astype(str).str.upper() == R1_LINE_ID.upper()].copy()
    else:
        r1_ids = get_r1_station_ids()
        if r1_ids:
            mask = (
                df["current_station_id"].astype(str).isin(r1_ids)
                | df["next_station_id"].astype(str).isin(r1_ids)
            )
            df = df[mask].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Loaded {len(df)} R1 train rows from {len(files)} file(s)")
    return df


def load_timetables(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load timetable rows for R1 stations."""
    files = _list_dynamic_files("timetables")
    files = _date_filter(files, start_date, end_date)
    if not files:
        log.warning("No timetable files found for the requested range")
        return pd.DataFrame()

    df = _concat(files)
    if df.empty:
        return df

    r1_ids = get_r1_station_ids()
    if r1_ids and "station_id" in df.columns:
        df = df[df["station_id"].astype(str).isin(r1_ids)].copy()

    for col in ["planned_departure", "planned_arrival", "actual_departure", "actual_arrival", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "planned_departure" in df.columns:
        df = df.sort_values("planned_departure").reset_index(drop=True)

    log.info(f"Loaded {len(df)} R1 timetable rows from {len(files)} file(s)")
    return df


def load_journeys(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load journey rows where both origin and destination are R1 stations."""
    files = _list_dynamic_files("journeys")
    files = _date_filter(files, start_date, end_date)
    if not files:
        log.warning("No journey files found for the requested range")
        return pd.DataFrame()

    df = _concat(files)
    if df.empty:
        return df

    r1_ids = get_r1_station_ids()
    if r1_ids:
        mask = (
            df["origin_station_id"].astype(str).isin(r1_ids)
            & df["destination_station_id"].astype(str).isin(r1_ids)
        )
        df = df[mask].copy()

    for col in ["departure_time", "arrival_time", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "duration" in df.columns:
        def _parse_duration(s: pd.Series) -> pd.Series:
            try:
                return pd.to_timedelta(s, errors="coerce")
            except Exception:
                return pd.to_timedelta(pd.to_numeric(s, errors="coerce"), unit="m")
        df["duration"] = _parse_duration(df["duration"])
        df["duration_minutes"] = df["duration"].dt.total_seconds() / 60

    if "departure_time" in df.columns:
        df = df.sort_values("departure_time").reset_index(drop=True)

    log.info(f"Loaded {len(df)} R1 journey rows from {len(files)} file(s)")
    return df


def load_weather(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load weather rows (Barcelona-wide, same for all R1 trains)."""
    files = _list_dynamic_files("weather")
    files = _date_filter(files, start_date, end_date)
    if not files:
        log.warning("No weather files found for the requested range")
        return pd.DataFrame()

    df = _concat(files)
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    log.info(f"Loaded {len(df)} weather rows from {len(files)} file(s)")
    return df


def load_all(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience loader — return all tables as a dict.

    Returns
    -------
    {
        "stations":   DataFrame,
        "lines":      DataFrame,
        "trains":     DataFrame,
        "timetables": DataFrame,
        "journeys":   DataFrame,
        "weather":    DataFrame,
    }
    """
    stations, lines = load_static()
    return {
        "stations":   stations,
        "lines":      lines,
        "trains":     load_trains(start_date, end_date),
        "timetables": load_timetables(start_date, end_date),
        "journeys":   load_journeys(start_date, end_date),
        "weather":    load_weather(start_date, end_date),
    }