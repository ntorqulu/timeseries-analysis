#!/usr/bin/env python3
"""
generate_static_app.py
=======================
Streams raw daily parquets from data/dynamic/{timetables,weather}/
one file at a time — peak RAM is one timetable file (~5 MB) instead
of the full history concatenated in memory.

Project layout (run from anywhere — paths resolve from this script):

  PROJECT/
  ├── app/
  │   └── generate_static_app.py
  ├── data/
  │   ├── dynamic/
  │   │   ├── timetables/   timetables_YYYY_MM_DD.parquet  (one per day)
  │   │   ├── trains/       trains_YYYY_MM_DD.parquet
  │   │   ├── journeys/     journeys_YYYY_MM_DD.parquet
  │   │   └── weather/      weather_YYYY_MM_DD.parquet
  │   ├── static/
  │   │   ├── stations.parquet
  │   │   ├── lines.parquet
  │   │   └── r1_station_mapping.csv
  │   └── models/
  │       ├── sarima_params.json
  │       ├── sarimax_params.json
  │       ├── garch_params.json
  │       └── models_metadata.json
  └── docs/
      └── index.html

Usage:
  python generate_static_app.py
  python generate_static_app.py --date-from 2026-03-01 --date-to 2026-05-31
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── PATH DEFAULTS — always relative to this script file, never to cwd ─────────
_HERE = Path(__file__).resolve().parent   # PROJECT/app/
_ROOT = _HERE.parent                      # PROJECT/

DEFAULT_DYNAMIC_DIR = _ROOT / "data" / "dynamic"
DEFAULT_STATIC_DIR  = _ROOT / "data" / "static"
DEFAULT_MODELS_DIR  = _ROOT / "data" / "models"
DEFAULT_OUT         = _ROOT / "docs"  / "index.html"


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate R1 delay intelligence HTML app")
    p.add_argument("--dynamic-dir", default=DEFAULT_DYNAMIC_DIR, type=Path)
    p.add_argument("--static-dir",  default=DEFAULT_STATIC_DIR,  type=Path)
    p.add_argument("--models-dir",  default=DEFAULT_MODELS_DIR,  type=Path)
    p.add_argument("--out",         default=DEFAULT_OUT,          type=Path)
    p.add_argument("--date-from",   default=None,
                   help="first date YYYY-MM-DD (default: all files)")
    p.add_argument("--date-to",     default=None,
                   help="last date  YYYY-MM-DD (default: all files)")
    args = p.parse_args()
    # Resolve so the script works from any working directory
    args.dynamic_dir = args.dynamic_dir.resolve()
    args.static_dir  = args.static_dir.resolve()
    args.models_dir  = args.models_dir.resolve()
    args.out         = args.out.resolve()
    return args


# ── FILE DISCOVERY ─────────────────────────────────────────────────────────────
def _date_from_stem(stem: str):
    """Extract the first YYYY-MM-DD found in a filename stem (handles _ or -)."""
    stem = stem.replace("_", "-")
    for i in range(len(stem) - 9):
        try:
            return pd.Timestamp(stem[i:i+10]).date()
        except Exception:
            continue
    return None


def glob_parquets(folder: Path, date_from=None, date_to=None) -> list[Path]:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets in {folder}")
    if date_from is None and date_to is None:
        return files
    df = pd.Timestamp(date_from).date() if date_from else None
    dt = pd.Timestamp(date_to).date()   if date_to   else None
    return [
        f for f in files
        if (d := _date_from_stem(f.stem)) is not None
        and (df is None or d >= df)
        and (dt is None or d <= dt)
    ]


# ── WEATHER CACHE — all weather files are tiny (46 rows each) ─────────────────
def load_weather_lookup(wx_paths: list[Path]) -> pd.Series:
    """Return a Series mapping hourly timestamp → mean temperature."""
    if not wx_paths:
        return pd.Series(dtype=float)
    parts = []
    for p in wx_paths:
        df = pd.read_parquet(p, columns=["timestamp_minute", "temperature"])
        df["ts"] = pd.to_datetime(df["timestamp_minute"], errors="coerce").dt.floor("h")
        parts.append(df[["ts", "temperature"]])
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("ts")["temperature"].mean()


# ── STREAMING AGGREGATION ──────────────────────────────────────────────────────
# Process ONE timetable file at a time.  Peak RAM ≈ one file ≈ 5 MB.

TIMETABLE_COLS = ["station_id", "planned_arrival", "actual_arrival"]


def _agg_one_timetable(path: Path, r1_ids: set):
    """
    Read one daily timetable parquet (3 columns only) and return two
    small aggregate DataFrames — raw data is discarded before returning.

    Returns
    -------
    hourly_agg : DataFrame  columns [delay_sum, n]  indexed by hour_trunc
    stn_agg    : DataFrame  columns [date_str, station_id, delay_sum, n]
    """
    tt = pd.read_parquet(path, columns=TIMETABLE_COLS)

    # Filter to R1 early — smallest possible working set
    tt = tt[tt["station_id"].isin(r1_ids)].copy()
    if tt.empty:
        return None, None

    tt["planned_arrival"] = pd.to_datetime(tt["planned_arrival"], errors="coerce")
    tt["actual_arrival"]  = pd.to_datetime(tt["actual_arrival"],  errors="coerce")
    tt.dropna(subset=["planned_arrival", "actual_arrival"], inplace=True)

    tt["delay_min"]  = (tt["actual_arrival"] - tt["planned_arrival"]).dt.total_seconds() / 60
    tt["hour_trunc"] = tt["planned_arrival"].dt.floor("h")
    tt["date_str"]   = tt["planned_arrival"].dt.date.astype(str)

    # Drop rows where the arrival date does not match the file date.
    # Overnight trains can have planned_arrival spilling into the adjacent
    # calendar day, producing ghost dates in the daily summary.
    file_date = str(_date_from_stem(path.stem))
    if file_date:
        tt = tt[tt["date_str"] == file_date]
    if tt.empty:
        return None, None

    hourly_agg = (
        tt.groupby("hour_trunc")["delay_min"]
        .agg(delay_sum="sum", n="count")
    )
    stn_agg = (
        tt.groupby(["date_str", "station_id"])["delay_min"]
        .agg(delay_sum="sum", n="count")
        .reset_index()
    )
    return hourly_agg, stn_agg


def build_datasets(tt_paths: list[Path], wx_lookup: pd.Series,
                   r1_ids: set, r1_station_map: pd.DataFrame):
    """
    Stream through timetable files one by one, accumulate sum/count,
    then compute weighted means.  Peak RAM ≈ one file + small accumulators.
    """
    hourly_acc = []   # list of tiny DataFrames → concat once at end
    stn_acc    = []

    for i, path in enumerate(tt_paths, 1):
        print(f"   [{i:3d}/{len(tt_paths)}] {path.name}", end="\r", flush=True)
        h, s = _agg_one_timetable(path, r1_ids)
        if h is not None:
            hourly_acc.append(h)
            stn_acc.append(s)

    print()  # newline after the \r progress line

    if not hourly_acc:
        raise ValueError("No R1 timetable data found in any file.")

    # ── Hourly dataset ────────────────────────────────────────────────────────
    hourly = (
        pd.concat(hourly_acc)
        .groupby(level=0)
        .sum()
    )
    hourly["target_delay"] = (hourly["delay_sum"] / hourly["n"]).round(3)
    hourly["delay_type_1"] = hourly["target_delay"]
    hourly["delay_type_2"] = hourly["target_delay"]
    hourly.drop(columns=["delay_sum", "n"], inplace=True)

    # Join temperature from weather lookup
    hourly["temperature"] = hourly.index.map(wx_lookup)
    hourly["temperature"] = hourly["temperature"].ffill().bfill()

    hourly["hour"]        = hourly.index.hour
    hourly["day_of_week"] = hourly.index.dayofweek
    hourly["is_weekend"]  = (hourly.index.dayofweek >= 5).astype(int)
    hourly["is_holiday"]  = 0
    hourly["date"]        = hourly.index.date.astype(str)
    hourly["day_name"]    = hourly.index.day_name()
    hourly["delay_category"] = pd.cut(
        hourly["target_delay"],
        bins=[-np.inf, 0, 2, 5, 10, np.inf],
        labels=["early", "on_time", "slight", "moderate", "severe"],
    ).astype(str)

    # ── Daily summary ─────────────────────────────────────────────────────────
    daily = (
        hourly.groupby("date")
        .agg(
            avg_delay =("target_delay", "mean"),
            max_delay =("target_delay", "max"),
            min_delay =("target_delay", "min"),
            std_delay =("target_delay", "std"),
            n_hours   =("target_delay", "count"),
            avg_temp  =("temperature",  "mean"),
            is_weekend=("is_weekend",   "first"),
        )
        .round(3)
    )

    # ── Station×day delays ────────────────────────────────────────────────────
    stn_all = (
        pd.concat(stn_acc)
        .groupby(["date_str", "station_id"])
        .sum()
        .reset_index()
    )
    stn_all["avg_delay_min"] = (stn_all["delay_sum"] / stn_all["n"]).round(2)

    stn_map = r1_station_map[["station_id", "name"]].copy()
    stn_map["station_id"] = stn_map["station_id"].astype(str)
    stn_all = stn_all.merge(stn_map, on="station_id", how="left")
    stn_all = stn_all.rename(columns={"name": "station_name", "date_str": "date"})
    stn_all["is_weekend"] = stn_all["date"].apply(
        lambda d: int(pd.Timestamp(d).dayofweek >= 5)
    )
    station_delays = stn_all[["date", "station_id", "station_name", "avg_delay_min", "is_weekend"]]

    return hourly, daily, station_delays


# ── SERIALISE FOR JS ───────────────────────────────────────────────────────────
def hourly_to_records(hourly: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, row in hourly.iterrows():
        if not (5 <= int(row["hour"]) <= 23):
            continue
        rows.append({
            "ts":   ts.strftime("%Y-%m-%dT%H:%M"),
            "td":   round(float(row["target_delay"]), 2),
            "d1":   round(float(row["delay_type_1"]), 2),
            "d2":   round(float(row["delay_type_2"]), 2),
            "temp": round(float(row["temperature"]), 1) if pd.notna(row["temperature"]) else None,
            "h":    int(row["hour"]),
            "dow":  int(row["day_of_week"]),
            "we":   int(row["is_weekend"]),
            "cat":  row["delay_category"],
            "date": str(row["date"]),
        })
    return rows


def daily_to_records(daily: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": str(date),
            "avg":  round(float(row["avg_delay"]),  2),
            "max":  round(float(row["max_delay"]),  2),
            "min":  round(float(row["min_delay"]),  2),
            "std":  round(float(row["std_delay"]),  2) if pd.notna(row["std_delay"]) else 0,
            "temp": round(float(row["avg_temp"]),   1) if pd.notna(row["avg_temp"])  else None,
            "we":   int(row["is_weekend"]),
        }
        for date, row in daily.iterrows()
    ]


def station_to_records(station_delays: pd.DataFrame) -> dict:
    agg = (
        station_delays.groupby("station_name")["avg_delay_min"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .reset_index()
    )
    # Replace NaN std (happens when a station has only one observation) with 0
    agg["std"] = agg["std"].fillna(0)
    agg = agg.to_dict(orient="records")
    top10 = (
        station_delays.groupby("station_name")["avg_delay_min"]
        .std()
        .nlargest(10)
        .index.tolist()
    )
    daily_rows = [
        {"date": r["date"], "stn": r["station_name"],
         "delay": round(float(r["avg_delay_min"]), 2)}
        for _, r in station_delays[station_delays["station_name"].isin(top10)].iterrows()
    ]
    return {"agg": agg, "daily": daily_rows}



def build_html(hourly_js, daily_js, station_js, sarima_js, sarimax_js, garch_js, meta_js):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>R1 Rodalies — Delay Intelligence</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0a0c10;--surface:#111318;--surface2:#191c23;--border:rgba(255,255,255,0.07);--accent:#e8ff47;--accent2:#47c8ff;--accent3:#ff6b47;--text:#e8eaf0;--muted:#5a5f6e;--on-time:#4ade80;--slight:#facc15;--moderate:#fb923c;--severe:#f87171;--early:#818cf8}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;font-size:13px;min-height:100vh;overflow-x:hidden}}
header{{padding:22px 28px 16px;border-bottom:1px solid var(--border);display:flex;align-items:flex-end;gap:24px;position:sticky;top:0;background:rgba(10,12,16,0.95);backdrop-filter:blur(12px);z-index:100}}
.logo-badge{{background:var(--accent);color:#0a0c10;font-family:'Syne',sans-serif;font-weight:800;font-size:18px;padding:3px 11px}}
.logo-label{{font-family:'Syne',sans-serif;font-size:15px;font-weight:600;color:var(--text)}}
.logo-sub{{color:var(--muted);font-size:10px;letter-spacing:0.08em}}
.header-meta{{margin-left:auto;text-align:right;color:var(--muted);font-size:11px;line-height:1.9}}
.header-meta span{{color:var(--accent)}}
.shell{{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 65px)}}
aside{{border-right:1px solid var(--border);padding:18px 14px;display:flex;flex-direction:column;gap:4px}}
.filter-group{{margin-bottom:16px}}
.filter-label{{font-size:9px;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:7px;display:block}}
.nav-btn{{width:100%;text-align:left;background:transparent;border:1px solid transparent;color:var(--muted);padding:8px 11px;cursor:pointer;font-family:'DM Mono',monospace;font-size:11px;letter-spacing:0.03em;transition:all 0.15s;display:flex;align-items:center;gap:9px}}
.nav-btn:hover{{color:var(--text);border-color:var(--border)}}
.nav-btn.active{{color:var(--accent);border-color:var(--accent);background:rgba(232,255,71,0.05)}}
.nav-btn .icon{{font-size:13px;width:16px;text-align:center}}
select,input[type=range]{{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:6px 9px;font-family:'DM Mono',monospace;font-size:11px;cursor:pointer;outline:none;margin-bottom:5px}}
select:focus{{border-color:var(--accent)}}
.range-row{{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-top:1px}}
.filter-pill-row{{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px}}
.pill{{padding:3px 8px;border:1px solid var(--border);color:var(--muted);font-size:10px;cursor:pointer;transition:all 0.12s;background:transparent;font-family:'DM Mono',monospace}}
.pill:hover{{border-color:var(--accent);color:var(--accent)}}
.pill.active{{background:var(--accent);color:#0a0c10;border-color:var(--accent);font-weight:500}}
main{{padding:20px 24px;overflow-y:auto}}
.tab-pane{{display:none}}
.tab-pane.active{{display:block}}
.kpi-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}}
.kpi{{background:var(--surface);border:1px solid var(--border);padding:14px 16px;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kpi-color,var(--accent))}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:28px;font-weight:700;color:var(--kpi-color,var(--accent));line-height:1;margin-bottom:3px}}
.kpi-lbl{{color:var(--muted);font-size:9px;letter-spacing:0.1em;text-transform:uppercase}}
.kpi-sub{{color:var(--muted);font-size:10px;margin-top:5px}}
.panels{{display:grid;gap:12px}}
.panels-2{{grid-template-columns:1fr 1fr}}
.panels-3{{grid-template-columns:2fr 1fr}}
.panel{{background:var(--surface);border:1px solid var(--border);padding:16px 18px}}
.panel-title{{font-family:'Syne',sans-serif;font-size:12px;font-weight:600;letter-spacing:0.02em;margin-bottom:14px;display:flex;align-items:center;gap:7px}}
.panel-title .badge{{font-size:8px;letter-spacing:0.1em;padding:2px 5px;background:rgba(232,255,71,0.1);color:var(--accent);font-family:'DM Mono',monospace;font-weight:400}}
svg text{{font-family:'DM Mono',monospace;fill:var(--muted)}}
.pred-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px}}
.pred-card{{background:var(--surface2);border:1px solid var(--border);padding:14px}}
.pred-model{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:9px}}
.pred-value{{font-family:'Syne',sans-serif;font-size:34px;font-weight:800;line-height:1;margin-bottom:3px}}
.pred-unit{{font-size:11px;color:var(--muted)}}
.pred-metrics{{font-size:10px;color:var(--muted);margin-top:7px;line-height:1.8}}
.pred-metrics span{{color:var(--text)}}
.pred-inputs{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:18px}}
.input-group{{display:flex;flex-direction:column;gap:4px}}
.input-label{{font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted)}}
.input-group select,.input-group input{{background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:7px 9px;font-family:'DM Mono',monospace;font-size:11px;width:100%;outline:none}}
.input-group select:focus,.input-group input:focus{{border-color:var(--accent)}}
.run-btn{{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;padding:11px;background:var(--accent);color:#0a0c10;border:none;font-family:'Syne',sans-serif;font-size:13px;font-weight:700;cursor:pointer;letter-spacing:0.05em;margin-bottom:18px}}
.run-btn:hover{{opacity:0.88}}
.section-title{{font-family:'Syne',sans-serif;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;margin-top:3px;padding-bottom:7px;border-bottom:1px solid var(--border)}}
.model-info-row{{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}}
.model-badge{{padding:3px 9px;border:1px solid var(--border);font-size:9px;letter-spacing:0.05em;color:var(--muted)}}
.model-badge span{{color:var(--text);margin-left:3px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:9px}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted)}}
.legend-dot{{width:7px;height:7px;border-radius:50%}}
.tooltip{{position:fixed;background:var(--surface2);border:1px solid var(--border);padding:7px 11px;font-size:10px;pointer-events:none;opacity:0;transition:opacity 0.1s;z-index:999;line-height:1.7}}
.tooltip.show{{opacity:1}}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<header>
  <div style="display:flex;align-items:center;gap:11px">
    <div class="logo-badge">R1</div>
    <div>
      <div class="logo-label">Rodalies Delay Intelligence</div>
      <div class="logo-sub">MOLINS DE REI · MAÇANET-MASSANES · 27 STATIONS</div>
    </div>
  </div>
  <div class="header-meta">
    <div>Data: <span id="data-range-start">—</span> → <span id="data-range-end">—</span></div>
    <div>Models: <span>SARIMA</span> · <span>SARIMAX</span> · <span>GARCH</span></div>
  </div>
</header>
<div class="shell">
<aside>
  <div class="filter-group">
    <span class="filter-label">Navigation</span>
    <button class="nav-btn active" onclick="switchTab('overview')" id="nav-overview"><span class="icon">◈</span>Overview</button>
    <button class="nav-btn" onclick="switchTab('timeseries')" id="nav-timeseries"><span class="icon">◡</span>Time Series</button>
    <button class="nav-btn" onclick="switchTab('stations')" id="nav-stations"><span class="icon">◎</span>Stations</button>
    <button class="nav-btn" onclick="switchTab('prediction')" id="nav-prediction"><span class="icon">◇</span>Prediction</button>
  </div>
  <div class="filter-group">
    <span class="filter-label">Month</span>
    <select id="filter-month" onchange="applyFilters()"><option value="all">All period</option></select>
  </div>
  <div class="filter-group">
    <span class="filter-label">Day Type</span>
    <div class="filter-pill-row">
      <button class="pill active" data-daytype="all" onclick="toggleDayType(this)">All</button>
      <button class="pill" data-daytype="workday" onclick="toggleDayType(this)">Workday</button>
      <button class="pill" data-daytype="weekend" onclick="toggleDayType(this)">Weekend</button>
    </div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Hour Range</span>
    <input type="range" id="filter-hour-min" min="5" max="23" value="5" oninput="onHourChange()">
    <input type="range" id="filter-hour-max" min="5" max="23" value="23" oninput="onHourChange()">
    <div class="range-row"><span id="hour-min-lbl">05h</span><span id="hour-max-lbl">23h</span></div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Delay Target</span>
    <div class="filter-pill-row">
      <button class="pill active" data-target="td" onclick="toggleTarget(this)">1&amp;2</button>
      <button class="pill" data-target="d1" onclick="toggleTarget(this)">Type 1</button>
      <button class="pill" data-target="d2" onclick="toggleTarget(this)">Type 2</button>
    </div>
  </div>
</aside>
<main>
  <div class="tab-pane active" id="tab-overview">
    <div class="kpi-strip" id="kpi-strip"></div>
    <div class="panels panels-3" style="margin-bottom:12px">
      <div class="panel"><div class="panel-title">Daily Average Delay <span class="badge">FULL PERIOD</span></div><svg id="chart-daily" width="100%" height="200"></svg></div>
      <div class="panel"><div class="panel-title">Delay Distribution</div><svg id="chart-donut" width="100%" height="200"></svg><div class="legend" id="legend-donut"></div></div>
    </div>
    <div class="panels panels-2">
      <div class="panel"><div class="panel-title">Avg Delay by Hour <span class="badge">INTRADAY</span></div><svg id="chart-hourly" width="100%" height="170"></svg></div>
      <div class="panel"><div class="panel-title">Avg Delay by Day of Week</div><svg id="chart-dow" width="100%" height="170"></svg></div>
    </div>
  </div>
  <div class="tab-pane" id="tab-timeseries">
    <div class="panel" style="margin-bottom:12px"><div class="panel-title">Hourly Delay Series <span class="badge">ALL TARGETS</span></div><svg id="chart-ts" width="100%" height="250"></svg><div class="legend" id="legend-ts"></div></div>
    <div class="panels panels-2">
      <div class="panel"><div class="panel-title">7-Day Rolling Average</div><svg id="chart-rolling" width="100%" height="190"></svg></div>
      <div class="panel"><div class="panel-title">Temperature vs Delay</div><svg id="chart-scatter" width="100%" height="190"></svg></div>
    </div>
  </div>
  <div class="tab-pane" id="tab-stations">
    <div class="panel" style="margin-bottom:12px"><div class="panel-title">Stations Ranked by Average Delay</div><svg id="chart-stations-bar" width="100%" height="340"></svg></div>
    <div class="panel"><div class="panel-title">Delay Heatmap — Top Stations × Date <span class="badge">HOVER</span></div><div id="heatmap-container"></div></div>
  </div>
  <div class="tab-pane" id="tab-prediction">
    <div class="section-title">Configure Prediction Scenario</div>
    <div class="pred-inputs">
      <div class="input-group"><label class="input-label">Day Type</label><select id="pred-daytype"><option value="workday">Workday</option><option value="weekend">Weekend</option></select></div>
      <div class="input-group"><label class="input-label">Hour of Day</label><select id="pred-hour"></select></div>
      <div class="input-group"><label class="input-label">Temperature (°C)</label><input type="number" id="pred-temp" value="18" min="-5" max="40" step="0.5"></div>
      <div class="input-group"><label class="input-label">Horizon (hours)</label><select id="pred-horizon"><option value="24">24 h</option><option value="48">48 h</option><option value="72" selected>72 h</option></select></div>
    </div>
    <button class="run-btn" onclick="runPrediction()">▶ RUN ALL MODELS</button>
    <div class="model-info-row" id="model-info-row"></div>
    <div class="pred-grid" id="pred-grid"></div>
    <div class="panel"><div class="panel-title">Forecast Horizon — All Models <span class="badge" id="horizon-badge">72H</span></div><svg id="chart-forecast" width="100%" height="270"></svg><div class="legend" id="legend-forecast"></div></div>
  </div>
</main>
</div>
<script>
// ── EMBEDDED DATA (auto-generated by generate_r1_app.py) ─────────────────────
const HOURLY_RAW  = {hourly_js};
const DAILY_RAW   = {daily_js};
const STATION_RAW = {station_js};
const SARIMA_CFG  = {sarima_js};
const SARIMAX_CFG = {sarimax_js};
const GARCH_CFG   = {garch_js};
const META        = {meta_js};

// ── STATE ─────────────────────────────────────────────────────────────────────
let state = {{tab:'overview', month:'all', daytype:'all', hourMin:5, hourMax:23, target:'td'}};

// ── FILTERS ───────────────────────────────────────────────────────────────────
function filtered() {{
  return HOURLY_RAW.filter(d => {{
    if (state.month !== 'all' && !d.date.startsWith(state.month)) return false;
    if (state.daytype === 'workday' && d.we === 1) return false;
    if (state.daytype === 'weekend' && d.we === 0) return false;
    if (d.h < state.hourMin || d.h > state.hourMax) return false;
    return true;
  }});
}}
function filteredDaily() {{
  return DAILY_RAW.filter(d => {{
    if (state.month !== 'all' && !d.date.startsWith(state.month)) return false;
    if (state.daytype === 'workday' && d.we > 0.5) return false;
    if (state.daytype === 'weekend' && d.we <= 0.5) return false;
    return true;
  }});
}}

// ── HELPERS ───────────────────────────────────────────────────────────────────
const delayColor = v => v < 0 ? 'var(--early)' : v < 2 ? 'var(--on-time)' : v < 5 ? 'var(--slight)' : v < 10 ? 'var(--moderate)' : 'var(--severe)';
const catColor   = c => ({{early:'var(--early)',on_time:'var(--on-time)',slight:'var(--slight)',moderate:'var(--moderate)',severe:'var(--severe)'}})[c] || 'var(--muted)';
const fmt1 = v => (typeof v === 'number' ? v.toFixed(1) : '—');
const tip = document.getElementById('tooltip');
function showTip(html, e) {{ tip.innerHTML = html; tip.classList.add('show'); moveTip(e); }}
function moveTip(e) {{ tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY-10)+'px'; }}
function hideTip() {{ tip.classList.remove('show'); }}
function svgW(id) {{ const el = document.getElementById(id); return el ? el.getBoundingClientRect().width || 600 : 600; }}
function clearSvg(id) {{ const el = document.getElementById(id); if (el) el.innerHTML = ''; return el; }}
function noData(id, msg='No data') {{ const el = document.getElementById(id); if (el && el.tagName==='svg') {{ const w=svgW(id),h=+el.getAttribute('height')||200; el.innerHTML=`<text x="${{w/2}}" y="${{h/2}}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="var(--muted)">${{msg}}</text>`; }} }}

// ── KPI STRIP ─────────────────────────────────────────────────────────────────
function renderKPIs() {{
  const data = filtered();
  if (!data.length) {{ document.getElementById('kpi-strip').innerHTML = ''; return; }}
  const vals = data.map(d => d[state.target]);
  const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
  const mx  = Math.max(...vals);
  const pct_d = vals.filter(v=>v>=5).length/vals.length*100;
  const pct_o = vals.filter(v=>v<2&&v>=-0.5).length/vals.length*100;
  const kpis = [
    {{val:fmt1(avg)+' min', lbl:'Avg Delay',    sub:`${{data.length.toLocaleString()}} obs`,     color:delayColor(avg)}},
    {{val:fmt1(mx)+' min',  lbl:'Max Delay',    sub:'in filtered window',                        color:'var(--severe)'}},
    {{val:fmt1(pct_d)+'%',  lbl:'Severe Rate',  sub:'≥ 5 min threshold',                         color:'var(--moderate)'}},
    {{val:fmt1(pct_o)+'%',  lbl:'On-Time Rate', sub:'< 2 min',                                   color:'var(--on-time)'}},
  ];
  document.getElementById('kpi-strip').innerHTML = kpis.map(k =>
    `<div class="kpi" style="--kpi-color:${{k.color}}"><div class="kpi-val">${{k.val}}</div><div class="kpi-lbl">${{k.lbl}}</div><div class="kpi-sub">${{k.sub}}</div></div>`
  ).join('');
}}

// ── DAILY LINE ────────────────────────────────────────────────────────────────
function renderDaily() {{
  const el = clearSvg('chart-daily');
  const daily = filteredDaily();
  if (!daily.length) {{ noData('chart-daily'); return; }}
  const w=svgW('chart-daily'), h=200, m={{t:8,r:10,b:28,l:38}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const x = d3.scaleTime().domain(d3.extent(daily, d=>new Date(d.date))).range([0,W]);
  const y = d3.scaleLinear().domain([d3.min(daily,d=>d.min)-0.5, d3.max(daily,d=>d.max)+0.5]).range([H,0]);
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  const area = d3.area().x(d=>x(new Date(d.date))).y0(d=>y(d.min)).y1(d=>y(d.max)).curve(d3.curveCatmullRom);
  g.append('path').datum(daily).attr('d',area).attr('fill','rgba(232,255,71,0.06)');
  const line = d3.line().x(d=>x(new Date(d.date))).y(d=>y(d.avg)).curve(d3.curveCatmullRom);
  g.append('path').datum(daily).attr('d',line).attr('stroke','var(--accent)').attr('stroke-width',1.5).attr('fill','none');
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%b %d'))).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
  const bisect = d3.bisector(d=>new Date(d.date)).left;
  svg.append('rect').attr('fill','none').attr('pointer-events','all').attr('x',m.l).attr('y',m.t).attr('width',W).attr('height',H)
    .on('mousemove', function(e) {{ const [mx]=d3.pointer(e,this); const xv=x.invert(mx-m.l); const i=bisect(daily,xv,1); const d=daily[Math.min(i,daily.length-1)]; showTip(`${{d.date}}<br>avg: <b>${{fmt1(d.avg)}} min</b><br>max: ${{fmt1(d.max)}} · temp: ${{d.temp}}°C`, e); }})
    .on('mouseleave', hideTip);
}}

// ── DONUT ─────────────────────────────────────────────────────────────────────
function renderDonut() {{
  const el = clearSvg('chart-donut');
  const data = filtered();
  if (!data.length) {{ noData('chart-donut'); return; }}
  const cats=['early','on_time','slight','moderate','severe'];
  const labels=['Early','On-time','Slight','Moderate','Severe'];
  const counts = cats.map(c => data.filter(d=>d.cat===c).length);
  const total = counts.reduce((a,b)=>a+b,0);
  const w=svgW('chart-donut'), h=200, R=Math.min(w,h)/2-8;
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{w/2}},${{h/2}})`);
  const pie = d3.pie().sort(null);
  const arc = d3.arc().innerRadius(R*0.55).outerRadius(R);
  g.selectAll('path').data(pie(counts)).join('path')
    .attr('d',arc).attr('fill',(d,i)=>catColor(cats[i])).attr('stroke','var(--bg)').attr('stroke-width',2)
    .on('mouseover',(e,d)=>showTip(`${{labels[d.index]}}: <b>${{d.value}}</b> (${{(d.value/total*100).toFixed(1)}}%)`,e))
    .on('mousemove',moveTip).on('mouseleave',hideTip);
  const avg = data.map(d=>d[state.target]).reduce((a,b)=>a+b,0)/data.length;
  g.append('text').attr('text-anchor','middle').attr('dy','-0.2em').attr('font-size','19').attr('fill','var(--text)').attr('font-family','Syne,sans-serif').attr('font-weight','700').text(fmt1(avg));
  g.append('text').attr('text-anchor','middle').attr('dy','1.2em').attr('font-size','9').attr('fill','var(--muted)').text('avg min');
  document.getElementById('legend-donut').innerHTML = cats.map((c,i) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${{catColor(c)}}"></div>${{labels[i]}}&nbsp;<span style="color:var(--text)">${{(counts[i]/total*100).toFixed(0)}}%</span></div>`
  ).join('');
}}

// ── HOURLY BARS ───────────────────────────────────────────────────────────────
function renderHourly() {{
  const el = clearSvg('chart-hourly');
  const data = filtered();
  if (!data.length) {{ noData('chart-hourly'); return; }}
  const byHour = d3.rollup(data, v=>d3.mean(v,d=>d[state.target]), d=>d.h);
  const hours  = Array.from(byHour,([h,v])=>({{h,v}})).sort((a,b)=>a.h-b.h);
  const w=svgW('chart-hourly'), h=170, m={{t:8,r:8,b:22,l:36}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const x = d3.scaleBand().domain(hours.map(d=>d.h)).range([0,W]).padding(0.15);
  const y = d3.scaleLinear().domain([0,d3.max(hours,d=>d.v)*1.1]).range([H,0]);
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  g.selectAll('rect').data(hours).join('rect')
    .attr('x',d=>x(d.h)).attr('y',d=>y(d.v)).attr('width',x.bandwidth()).attr('height',d=>H-y(d.v)).attr('fill',d=>delayColor(d.v))
    .on('mouseover',(e,d)=>showTip(`Hour ${{d.h}}h<br>avg: <b>${{fmt1(d.v)}} min</b>`,e)).on('mousemove',moveTip).on('mouseleave',hideTip);
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).tickFormat(d=>`${{d}}h`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','9');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','9');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
}}

// ── DOW BARS ──────────────────────────────────────────────────────────────────
function renderDow() {{
  const el = clearSvg('chart-dow');
  const data = filtered();
  if (!data.length) {{ noData('chart-dow'); return; }}
  const DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const byDow = d3.rollup(data, v=>d3.mean(v,d=>d[state.target]), d=>d.dow);
  const dows  = DOW.map((name,i)=>({{name, v:byDow.get(i)||0}}));
  const w=svgW('chart-dow'), h=170, m={{t:8,r:8,b:22,l:36}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const x = d3.scaleBand().domain(DOW).range([0,W]).padding(0.2);
  const y = d3.scaleLinear().domain([0,d3.max(dows,d=>d.v)*1.1]).range([H,0]);
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  g.selectAll('rect').data(dows).join('rect')
    .attr('x',d=>x(d.name)).attr('y',d=>y(d.v)).attr('width',x.bandwidth()).attr('height',d=>H-y(d.v))
    .attr('fill',(_,i)=>i>=5?'var(--accent2)':'rgba(232,255,71,0.55)')
    .on('mouseover',(e,d)=>showTip(`${{d.name}}<br>avg: <b>${{fmt1(d.v)}} min</b>`,e)).on('mousemove',moveTip).on('mouseleave',hideTip);
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','9');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
}}

// ── TIME SERIES ───────────────────────────────────────────────────────────────
function renderTS() {{
  const el = clearSvg('chart-ts');
  const data = filtered().sort((a,b)=>a.ts<b.ts?-1:1);
  if (!data.length) {{ noData('chart-ts'); return; }}
  const w=svgW('chart-ts'), h=250, m={{t:8,r:14,b:28,l:40}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const parseTs = d => new Date(d.ts);
  const x = d3.scaleTime().domain(d3.extent(data,parseTs)).range([0,W]);
  const allV = [...data.map(d=>d.td),...data.map(d=>d.d1),...data.map(d=>d.d2)];
  const y = d3.scaleLinear().domain([d3.min(allV)-0.5,d3.max(allV)+0.5]).range([H,0]);
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  const mkLine=(key,color,op=1)=>{{const line=d3.line().x(d=>x(parseTs(d))).y(d=>y(d[key])).curve(d3.curveCatmullRom.alpha(0.5));g.append('path').datum(data).attr('d',line).attr('stroke',color).attr('stroke-width',1).attr('fill','none').attr('opacity',op);}};
  mkLine('d1','rgba(255,107,71,0.4)');mkLine('d2','rgba(71,200,255,0.4)');mkLine('td','var(--accent)',0.9);
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%b %d'))).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
  document.getElementById('legend-ts').innerHTML=[['var(--accent)','Type 1&2 mean'],['var(--accent3)','Type 1'],['var(--accent2)','Type 2']].map(([c,l])=>`<div class="legend-item"><div class="legend-dot" style="background:${{c}}"></div>${{l}}</div>`).join('');
}}

// ── ROLLING ───────────────────────────────────────────────────────────────────
function renderRolling() {{
  const el = clearSvg('chart-rolling');
  const daily = filteredDaily();
  if (daily.length < 7) {{ noData('chart-rolling'); return; }}
  const rolled = daily.slice(6).map((d,i)=>({{date:d.date, avg:d3.mean(daily.slice(i,i+7),v=>v.avg)}}));
  const w=svgW('chart-rolling'), h=190, m={{t:8,r:8,b:24,l:36}};
  const Ww=w-m.l-m.r, H=h-m.t-m.b;
  const x = d3.scaleTime().domain(d3.extent(rolled,d=>new Date(d.date))).range([0,Ww]);
  const y = d3.scaleLinear().domain([d3.min(rolled,d=>d.avg)-0.3,d3.max(rolled,d=>d.avg)+0.3]).range([H,0]);
  const svg = d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g = svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  const area=d3.area().x(d=>x(new Date(d.date))).y0(H).y1(d=>y(d.avg)).curve(d3.curveCatmullRom);
  g.append('path').datum(rolled).attr('d',area).attr('fill','rgba(232,255,71,0.04)');
  const line=d3.line().x(d=>x(new Date(d.date))).y(d=>y(d.avg)).curve(d3.curveCatmullRom);
  g.append('path').datum(rolled).attr('d',line).attr('stroke','var(--accent)').attr('stroke-width',1.5).attr('fill','none');
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(4).tickFormat(d3.timeFormat('%b %d'))).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
}}

// ── SCATTER ───────────────────────────────────────────────────────────────────
function renderScatter() {{
  const el = clearSvg('chart-scatter');
  const data = filtered();
  if (!data.length) {{ noData('chart-scatter'); return; }}
  const sample = data.length > 500 ? data.filter((_,i)=>i%Math.ceil(data.length/500)===0) : data;
  const w=svgW('chart-scatter'), h=190, m={{t:8,r:8,b:24,l:36}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const x=d3.scaleLinear().domain(d3.extent(sample,d=>d.temp)).range([0,W]);
  const y=d3.scaleLinear().domain(d3.extent(sample,d=>d[state.target])).range([H,0]);
  const svg=d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g=svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  g.selectAll('circle').data(sample).join('circle')
    .attr('cx',d=>x(d.temp)).attr('cy',d=>y(d[state.target])).attr('r',2.5)
    .attr('fill',d=>delayColor(d[state.target])).attr('opacity',0.5)
    .on('mouseover',(e,d)=>showTip(`${{d.ts.slice(0,13)}}<br>temp: ${{d.temp}}°C<br>delay: <b>${{fmt1(d[state.target])}} min</b>`,e)).on('mousemove',moveTip).on('mouseleave',hideTip);
  const xv=sample.map(d=>d.temp), yv=sample.map(d=>d[state.target]), n=xv.length;
  const mx=xv.reduce((a,b)=>a+b)/n, my=yv.reduce((a,b)=>a+b)/n;
  const slope=xv.map((xi,i)=>(xi-mx)*(yv[i]-my)).reduce((a,b)=>a+b)/xv.map(xi=>(xi-mx)**2).reduce((a,b)=>a+b);
  const intercept=my-slope*mx;
  const x0=d3.min(xv), x1=d3.max(xv);
  g.append('line').attr('x1',x(x0)).attr('y1',y(slope*x0+intercept)).attr('x2',x(x1)).attr('y2',y(slope*x1+intercept)).attr('stroke','var(--accent)').attr('stroke-width',1).attr('stroke-dasharray','4,3').attr('opacity',0.7);
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(4).tickFormat(d=>`${{d}}°`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
}}

// ── STATIONS BAR ──────────────────────────────────────────────────────────────
function renderStationsBar() {{
  const el = clearSvg('chart-stations-bar');
  const agg = STATION_RAW.agg.slice().sort((a,b)=>b.mean-a.mean);
  const w=svgW('chart-stations-bar'), h=340, m={{t:8,r:70,b:8,l:200}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const x=d3.scaleLinear().domain([0,d3.max(agg,d=>d.max)*1.05]).range([0,W]);
  const y=d3.scaleBand().domain(agg.map(d=>d.station_name)).range([0,H]).padding(0.25);
  const svg=d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g=svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  g.selectAll('.ebar').data(agg).join('line').attr('x1',d=>x(d.min)).attr('x2',d=>x(d.max)).attr('y1',d=>y(d.station_name)+y.bandwidth()/2).attr('y2',d=>y(d.station_name)+y.bandwidth()/2).attr('stroke','var(--border)').attr('stroke-width',1.5);
  g.selectAll('rect').data(agg).join('rect').attr('x',0).attr('y',d=>y(d.station_name)).attr('width',d=>x(d.mean)).attr('height',y.bandwidth()).attr('fill',d=>delayColor(d.mean))
    .on('mouseover',(e,d)=>showTip(`<b>${{d.station_name}}</b><br>avg: ${{fmt1(d.mean)}} min<br>range: ${{fmt1(d.min)}}–${{fmt1(d.max)}} min`,e)).on('mousemove',moveTip).on('mouseleave',hideTip);
  g.selectAll('.val').data(agg).join('text').attr('x',d=>x(d.mean)+4).attr('y',d=>y(d.station_name)+y.bandwidth()/2).attr('dominant-baseline','middle').attr('font-size','10').attr('fill','var(--muted)').text(d=>`${{fmt1(d.mean)}}m`);
  g.append('g').call(d3.axisLeft(y).tickSize(0)).selectAll('text').attr('fill','var(--text)').attr('font-size','10').attr('dx','-5');
  g.select('.domain').remove();
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(4).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.tick line').attr('stroke','var(--border)');
}}

// ── HEATMAP ───────────────────────────────────────────────────────────────────
function renderHeatmap() {{
  const container = document.getElementById('heatmap-container');
  const daily = STATION_RAW.daily;
  const stations = [...new Set(daily.map(d=>d.stn))].sort();
  const dates    = [...new Set(daily.map(d=>d.date))].sort();
  const CELL=10, LABEL_W=195, TOP=45;
  const w=LABEL_W+dates.length*CELL, h=TOP+stations.length*CELL;
  const colorScale = d3.scaleSequential(d3.interpolateRgb('#0a0c10','#e8ff47')).domain([0,8]);
  const svg = d3.create('svg').attr('width','100%').attr('viewBox',`0 0 ${{w}} ${{h}}`);
  dates.forEach((date,di)=>{{
    if(di%7!==0)return;
    svg.append('text').attr('x',LABEL_W+di*CELL+CELL/2).attr('y',TOP-5).attr('font-size',8).attr('fill','var(--muted)').attr('text-anchor','middle').attr('transform',`rotate(-45,${{LABEL_W+di*CELL+CELL/2}},${{TOP-5}})`).text(date.slice(5));
  }});
  stations.forEach((stn,si)=>{{
    svg.append('text').attr('x',LABEL_W-5).attr('y',TOP+si*CELL+CELL/2+1).attr('text-anchor','end').attr('dominant-baseline','middle').attr('font-size',8.5).attr('fill','var(--muted)').text(stn);
    dates.forEach((date,di)=>{{
      const row=daily.find(d=>d.stn===stn&&d.date===date);
      const val=row?row.delay:null;
      const rect=svg.append('rect').attr('x',LABEL_W+di*CELL).attr('y',TOP+si*CELL).attr('width',CELL-1).attr('height',CELL-1).attr('fill',val!==null?colorScale(val):'var(--border)').attr('rx',1);
      if(val!==null)rect.on('mouseover',e=>showTip(`<b>${{stn}}</b><br>${{date}}<br>${{fmt1(val)}} min`,e)).on('mousemove',moveTip).on('mouseleave',hideTip);
    }});
  }});
  container.innerHTML='';
  container.appendChild(svg.node());
}}

// ── PREDICTION ENGINE ─────────────────────────────────────────────────────────
function sarimaCast(h) {{
  const p=SARIMA_CFG.params, phi1=p['ar.L1']||0, th1=p['ma.L1']||0;
  const PhiS1=p['ar.S.L24']||0, PhiS2=p['ar.S.L48']||0, ThS1=p['ma.S.L24']||0;
  const mu=SARIMA_CFG.train_mean, sig2=p['sigma2']||1;
  // Auto-scale if SARIMA was fitted on a different unit than the display data.
  // META.train_mean is always in the same unit as the chart (minutes).
  const scale=(META.train_mean>0&&SARIMA_CFG.train_mean>0)?META.train_mean/SARIMA_CFG.train_mean:1;
  const hist=SARIMA_CFG.train_last_values.slice(-48).map(v=>v-mu);
  const eps=new Array(48).fill(0);
  const fc=[];
  for(let i=0;i<h;i++){{
    const L1=hist.length>=1?hist[hist.length-1]:0;
    const L24=hist.length>=24?hist[hist.length-24]:0;
    const L48=hist.length>=48?hist[hist.length-48]:0;
    const eL1=eps[eps.length-1]||0, eL24=eps.length>=24?eps[eps.length-24]:0;
    const yhat=phi1*L1+PhiS1*L24+PhiS2*L48+th1*eL1+ThS1*eL24;
    fc.push((mu+yhat)*scale); hist.push(yhat); eps.push(0);
  }}
  return{{mean:fc, sigma:Math.sqrt(sig2)*scale}};
}}
function sarimaxCast(h,temp,dow,isHoliday) {{
  const p=SARIMAX_CFG.params, phi1=p['ar.L1']||0, th1=p['ma.L1']||0;
  const PhiS=p['ar.S.L24']||0, ThS=p['ma.S.L24']||0;
  const mu=SARIMAX_CFG.train_mean, sig2=p['sigma2']||1;
  const bDs=p['dow_sin']||0, bDc=p['dow_cos']||0, bH=p['is_holiday']||0, bT=p['temperature']||0;
  const dowSin=Math.sin(2*Math.PI*dow/7), dowCos=Math.cos(2*Math.PI*dow/7);
  const exogEffect=bDs*dowSin+bDc*dowCos+bH*isHoliday+bT*temp;
  const ae=SARIMAX_CFG.exog_col_means||{{}};
  const avgEffect=bDs*(ae.dow_sin||0)+bDc*(ae.dow_cos||0)+bT*(ae.temperature||15);
  const hist=SARIMAX_CFG.train_last_values.slice(-48).map(v=>v-mu-avgEffect);
  const eps=new Array(48).fill(0);
  const fc=[];
  for(let i=0;i<h;i++){{
    const L1=hist.length>=1?hist[hist.length-1]:0;
    const L24=hist.length>=24?hist[hist.length-24]:0;
    const eL1=eps[eps.length-1]||0, eL24=eps.length>=24?eps[eps.length-24]:0;
    const yhat=phi1*L1+PhiS*L24+th1*eL1+ThS*eL24+exogEffect;
    fc.push(mu+avgEffect+yhat); hist.push(yhat); eps.push(0);
  }}
  return{{mean:fc, sigma:Math.sqrt(sig2)}};
}}
function garchCast(h) {{
  const g=GARCH_CFG.params, mu=g.mu, omega=g.omega, alpha=g.alpha, beta=g.beta;
  // Auto-scale to match chart units (same ratio as SARIMA).
  const garchMean=GARCH_CFG.train_mean||mu;
  const scale=(META.train_mean>0&&garchMean>0)?META.train_mean/garchMean:1;
  let epsT2=(GARCH_CFG.last_resid||0)**2, sigT2=GARCH_CFG.last_variance||omega/(1-alpha-beta);
  const sigmas=[];
  for(let i=0;i<h;i++){{sigT2=omega+alpha*epsT2+beta*sigT2;epsT2=sigT2;sigmas.push(Math.sqrt(sigT2)*scale);}}
  return{{mean:new Array(h).fill(mu*scale), sigmas}};
}}

function runPrediction() {{
  const daytype=document.getElementById('pred-daytype').value;
  const hour=+document.getElementById('pred-hour').value;
  const temp=+document.getElementById('pred-temp').value;
  const horizon=+document.getElementById('pred-horizon').value;
  document.getElementById('horizon-badge').textContent=horizon+'H';
  const dow=daytype==='weekend'?6:2;
  const sarima_fc=sarimaCast(horizon), sarimax_fc=sarimaxCast(horizon,temp,dow,0), garch_fc=garchCast(horizon);
  const hourIdx=Math.min(Math.max(hour-5,0),horizon-1);
  document.getElementById('pred-grid').innerHTML=[
    {{model:'SARIMA (1,0,1)(2,0,1,24)',val:sarima_fc.mean[hourIdx], color:'var(--accent)',  mae:META.models.sarima.mae,  rmse:META.models.sarima.rmse,  note:'Seasonal ARIMA — captures 24h cycle'}},
    {{model:'SARIMAX + Exogenous',     val:sarimax_fc.mean[hourIdx],color:'var(--accent2)', mae:META.models.sarimax.mae, rmse:META.models.sarimax.rmse, note:'Adds temp, DoW, holiday regressors'}},
    {{model:'GARCH (1,1) Const. Mean', val:garch_fc.mean[0],        color:'rgba(255,107,71,0.9)', mae:null,rmse:null, note:`Volatility model · persistence: ${{GARCH_CFG.persistence}}`}},
  ].map(c=>`<div class="pred-card"><div class="pred-model">${{c.model}}</div><div class="pred-value" style="color:${{c.color}}">${{fmt1(c.val)}}<span class="pred-unit"> min</span></div><div style="font-size:10px;color:var(--muted);margin-top:3px">${{c.note}}</div><div class="pred-metrics">${{c.mae!==null?`MAE <span>${{c.mae}} min</span> · RMSE <span>${{c.rmse}} min</span>`:`μ: <span>${{fmt1(GARCH_CFG.params.mu)}} min</span> · σ∞: <span>${{fmt1(GARCH_CFG.unconditional_vol)}} min</span>`}}</div></div>`).join('');
  document.getElementById('model-info-row').innerHTML=[
    `Train: ${{META.train_range.start.slice(0,10)}} → ${{META.train_range.end.slice(0,10)}}`,
    `${{META.n_train_hours.toLocaleString()}} train hours`,`s=${{META.seasonal_period}}`,`Horizon: ${{horizon}}h`,
  ].map(t=>`<div class="model-badge">${{t}}</div>`).join('');
  renderForecastChart(sarima_fc,sarimax_fc,garch_fc,horizon);
}}

function renderForecastChart(sarima_fc,sarimax_fc,garch_fc,horizon) {{
  const el=clearSvg('chart-forecast');
  const w=svgW('chart-forecast'), h=270, m={{t:10,r:14,b:28,l:40}};
  const W=w-m.l-m.r, H=h-m.t-m.b;
  const xs=d3.range(horizon);
  const allV=[...sarima_fc.mean,...sarimax_fc.mean,...garch_fc.mean,...garch_fc.mean.map((v,i)=>v+1.96*garch_fc.sigmas[i]),...garch_fc.mean.map((v,i)=>v-1.96*garch_fc.sigmas[i])];
  const x=d3.scaleLinear().domain([0,horizon-1]).range([0,W]);
  const y=d3.scaleLinear().domain([d3.min(allV)-0.5,d3.max(allV)+0.5]).range([H,0]);
  const svg=d3.select(el).attr('viewBox',`0 0 ${{w}} ${{h}}`);
  const g=svg.append('g').attr('transform',`translate(${{m.l}},${{m.t}})`);
  const bandData=xs.map(i=>({{x:i,upper:garch_fc.mean[i]+1.96*garch_fc.sigmas[i],lower:garch_fc.mean[i]-1.96*garch_fc.sigmas[i]}}));
  const area=d3.area().x(d=>x(d.x)).y0(d=>y(d.lower)).y1(d=>y(d.upper));
  g.append('path').datum(bandData).attr('d',area).attr('fill','rgba(71,200,255,0.07)');
  const mkL=(vals,color,dash='')=>{{const line=d3.line().x((_,i)=>x(i)).y(v=>y(v));g.append('path').datum(vals).attr('d',line).attr('stroke',color).attr('stroke-width',1.5).attr('fill','none').attr('stroke-dasharray',dash);}};
  mkL(bandData.map(d=>d.upper),'rgba(71,200,255,0.3)','3,3');mkL(bandData.map(d=>d.lower),'rgba(71,200,255,0.3)','3,3');
  mkL(garch_fc.mean,'rgba(255,107,71,0.85)','5,3');mkL(sarima_fc.mean,'var(--accent)');mkL(sarimax_fc.mean,'var(--accent2)');
  const vLine=g.append('line').attr('stroke','var(--border)').attr('y1',0).attr('y2',H).style('opacity',0);
  g.append('rect').attr('fill','none').attr('pointer-events','all').attr('width',W).attr('height',H)
    .on('mousemove',function(e){{const[mx]=d3.pointer(e,this);const xi=Math.round(x.invert(mx));if(xi<0||xi>=horizon)return;vLine.attr('x1',x(xi)).attr('x2',x(xi)).style('opacity',1);showTip(`Hour +${{xi}}<br>SARIMA: <b>${{fmt1(sarima_fc.mean[xi])}} min</b><br>SARIMAX: <b>${{fmt1(sarimax_fc.mean[xi])}} min</b><br>GARCH μ: <b>${{fmt1(garch_fc.mean[xi])}} min</b> ±${{fmt1(1.96*garch_fc.sigmas[xi])}}`,e);}})
    .on('mouseleave',()=>{{vLine.style('opacity',0);hideTip();}});
  const xTicks=d3.range(0,horizon+1,6);
  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).tickValues(xTicks).tickFormat(d=>`+${{d}}h`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d=>`${{d}}m`)).selectAll('text').attr('fill','var(--muted)').attr('font-size','10');
  g.selectAll('.domain,.tick line').attr('stroke','var(--border)');
  document.getElementById('legend-forecast').innerHTML=[['var(--accent)','SARIMA'],['var(--accent2)','SARIMAX'],['rgba(255,107,71,0.9)','GARCH mean'],['rgba(71,200,255,0.3)','GARCH 95% CI']].map(([c,l])=>`<div class="legend-item"><div class="legend-dot" style="background:${{c}}"></div>${{l}}</div>`).join('');
}}

// ── NAVIGATION ────────────────────────────────────────────────────────────────
function switchTab(tab) {{
  state.tab=tab;
  document.querySelectorAll('.tab-pane').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el=>el.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  document.getElementById('nav-'+tab).classList.add('active');
  renderCurrentTab();
}}
function renderCurrentTab() {{
  const t=state.tab;
  if(t==='overview')      {{ renderKPIs();renderDaily();renderDonut();renderHourly();renderDow(); }}
  else if(t==='timeseries'){{ renderTS();renderRolling();renderScatter(); }}
  else if(t==='stations')  {{ renderStationsBar();renderHeatmap(); }}
  else if(t==='prediction'){{ runPrediction(); }}
}}
function applyFilters()    {{ state.month=document.getElementById('filter-month').value; renderCurrentTab(); }}
function toggleDayType(btn){{ document.querySelectorAll('[data-daytype]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');state.daytype=btn.dataset.daytype;renderCurrentTab(); }}
function toggleTarget(btn) {{ document.querySelectorAll('[data-target]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');state.target=btn.dataset.target;renderCurrentTab(); }}
function onHourChange()    {{
  state.hourMin=+document.getElementById('filter-hour-min').value;
  state.hourMax=+document.getElementById('filter-hour-max').value;
  document.getElementById('hour-min-lbl').textContent=String(state.hourMin).padStart(2,'0')+'h';
  document.getElementById('hour-max-lbl').textContent=String(state.hourMax).padStart(2,'0')+'h';
  renderCurrentTab();
}}

// ── BOOT: populate dynamic controls ──────────────────────────────────────────
(function boot() {{
  // date range header
  if (DAILY_RAW.length) {{
    document.getElementById('data-range-start').textContent = DAILY_RAW[0].date;
    document.getElementById('data-range-end').textContent   = DAILY_RAW[DAILY_RAW.length-1].date;
  }}
  // month filter — derive from data
  const months = [...new Set(DAILY_RAW.map(d => d.date.slice(0,7)))].sort();
  const sel = document.getElementById('filter-month');
  const monthNames = {{1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}};
  months.forEach(m => {{
    const [yr, mo] = m.split('-').map(Number);
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = `${{monthNames[mo]}} ${{yr}}`;
    sel.appendChild(opt);
  }});
  // hour dropdown for prediction tab
  const hsel = document.getElementById('pred-hour');
  for (let h = 5; h <= 23; h++) {{
    const o = document.createElement('option');
    o.value = h; o.textContent = String(h).padStart(2,'0') + ':00';
    if (h === 9) o.selected = true;
    hsel.appendChild(o);
  }}
  renderCurrentTab();
  window.addEventListener('resize', () => renderCurrentTab());
}})();
</script>
</body>
</html>"""



# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"📂 Dynamic dir : {args.dynamic_dir}")
    print(f"📂 Static dir  : {args.static_dir}")
    print(f"📂 Models dir  : {args.models_dir}")
    if args.date_from or args.date_to:
        print(f"   Date range  : {args.date_from or 'start'} → {args.date_to or 'end'}")

    # ── Static reference tables (tiny — load normally) ─────────────────────────
    r1_station_map = pd.read_csv(args.static_dir / "r1_station_mapping.csv")
    r1_station_map["station_id"] = r1_station_map["station_id"].astype(str)
    r1_station_ids = set(r1_station_map["station_id"])

    # ── Discover files ─────────────────────────────────────────────────────────
    kw = dict(date_from=args.date_from, date_to=args.date_to)
    tt_paths = glob_parquets(args.dynamic_dir / "timetables", **kw)
    wx_paths = glob_parquets(args.dynamic_dir / "weather",    **kw)

    print(f"\n   timetables : {len(tt_paths)} file(s)"
          f"  [{tt_paths[0].name} … {tt_paths[-1].name}]")
    print(f"   weather    : {len(wx_paths)} file(s)"
          f"  [{wx_paths[0].name} … {wx_paths[-1].name}]")

    # ── Weather: all files are tiny (~46 rows each) — load all at once ─────────
    print("\n📡 Loading weather …")
    wx_lookup = load_weather_lookup(wx_paths)
    print(f"   {len(wx_lookup)} hourly temperature readings")

    # ── Timetables: stream one file at a time ──────────────────────────────────
    print("\n🔧 Streaming timetables …")
    hourly, daily, station_delays = build_datasets(
        tt_paths, wx_lookup, r1_station_ids, r1_station_map
    )

    print(f"   hourly rows  : {len(hourly)}")
    print(f"   daily rows   : {daily.shape[0]}"
          f"  ({daily.index.min()} → {daily.index.max()})")
    print(f"   station×day  : {len(station_delays)}")

    # ── Serialise to compact JSON ──────────────────────────────────────────────
    hourly_js  = json.dumps(hourly_to_records(hourly),          separators=(",", ":"))
    daily_js   = json.dumps(daily_to_records(daily),            separators=(",", ":"))
    station_js = json.dumps(station_to_records(station_delays), separators=(",", ":"))

    # ── Load model params ──────────────────────────────────────────────────────
    def load_json(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run the corresponding Jupyter notebook first."
            )
        return path.read_text()

    print("\n📦 Loading models …")
    sarima_js  = load_json(args.models_dir / "sarima_params.json")
    sarimax_js = load_json(args.models_dir / "sarimax_params.json")
    garch_js   = load_json(args.models_dir / "garch_params.json")
    meta_js    = load_json(args.models_dir / "models_metadata.json")

    # ── Write HTML ─────────────────────────────────────────────────────────────
    print("\n🖊  Generating HTML …")
    html = build_html(hourly_js, daily_js, station_js,
                      sarima_js, sarimax_js, garch_js, meta_js)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    size_kb = args.out.stat().st_size // 1024
    print(f"\n✅ Generated : {args.out}  ({size_kb} KB)")
    print(f"   Preview   : python -m http.server 8000 --directory {args.out.parent}")
    print(f"   Deploy    : git add {args.out} && git commit -m 'update dashboard' && git push")


if __name__ == "__main__":
    main()