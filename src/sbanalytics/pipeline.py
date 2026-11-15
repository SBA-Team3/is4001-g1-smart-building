"""
raw CSVs -> data/cleaned/. One function, every floor, same rules.

    python -m sbanalytics.pipeline              # all floors (~2 min)
    python -m sbanalytics.pipeline --floor 5    # one floor (overwrites outputs with that floor only)
    python -m sbanalytics.pipeline --no-minute  # skip the large minute parquet

Outputs (all parquet unless noted):

    hourly_zone.parquet       floor-zone-hour: kWh by end use, sensor means,
                              coverage, calendar, day type, weather
    hourly_point.parquet      meter-hour, long: one row per raw column per hour
    daily_zone.parquet        floor-zone-day totals with a day_ok coverage flag
    point_day_health.parquet  per point per day, computed BEFORE cleaning
    weather_hourly.parquet    ERA5 on local time
    day_types.parquet         floor-date: workday / weekend / holiday / closure / holiday_worked
    cleaning_log.csv          every minute changed, by floor, column and rule
    build_summary.csv         per floor: coverage and raw-vs-clean energy by end use
    minute/floor{N}.parquet   cleaned minute grid (gitignored; dashboard drill-down)
"""

from __future__ import annotations

import time

import pandas as pd

from . import clean, features, io, quality, resample
from . import config as C
from . import registry as R


def _energy_by_channel(df: pd.DataFrame, floor: int) -> dict:
    out = {}
    for ch in ("ac", "light", "plug"):
        cols = [p.column for p in R.points(floor, ch, roles=R.ENERGY_ROLES)]
        out[ch] = float(df[cols].sum().sum() / 60) if cols else 0.0
    return out


def build_floor(floor: int, write_minute: bool = True, verbose: bool = True):
    t0 = time.time()
    raw = io.load_raw(floor)
    grid = io.grid_report(raw)
    health = quality.point_day_health(raw, floor)
    df, log = clean.clean_floor(raw, floor)

    if write_minute:
        C.MINUTE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.MINUTE_DIR / f"floor{floor}.parquet")

    hz = resample.to_hourly(df, floor)
    hp = resample.to_hourly_points(df, floor)

    e_raw, e_clean = _energy_by_channel(raw, floor), _energy_by_channel(df, floor)
    summary = {"floor": floor, **grid,
               **{f"{k}_kwh_raw": v for k, v in e_raw.items()},
               **{f"{k}_kwh_clean": v for k, v in e_clean.items()},
               "minutes_changed": int(log.minutes.sum()) if len(log) else 0}
    if verbose:
        print(f"floor {floor}: {grid['first']} -> {grid['last']}, "
              f"AC raw {e_raw['ac']:,.0f} / clean {e_clean['ac']:,.0f} kWh, "
              f"{summary['minutes_changed']:,} minutes changed, {time.time()-t0:.0f}s")
    return hz, hp, health, log, summary


def build(floors=C.FLOORS, write_minute: bool = True, verbose: bool = True) -> dict:
    C.DATA_CLEANED.mkdir(parents=True, exist_ok=True)
    zones, points, healths, logs, sums = [], [], [], [], []
    for f in floors:
        hz, hp, H, log, s = build_floor(f, write_minute, verbose)
        zones.append(hz); points.append(hp); healths.append(H); logs.append(log); sums.append(s)

    weather = io.load_weather()
    h = pd.concat(zones, ignore_index=True)
    h = features.add_calendar(h)
    day_types = features.detect_day_types(h)
    h = features.apply_day_types(h, day_types)
    h = h.merge(weather, on="ts", how="left")

    num = h.select_dtypes("float64").columns
    h[num] = h[num].astype("float32")
    h = h.sort_values(["floor", "zone", "ts"]).reset_index(drop=True)

    daily = resample.to_daily(h).merge(
        day_types.assign(date=pd.to_datetime(day_types.date)), on=["floor", "date"], how="left")

    out = {
        "hourly_zone": h,
        "hourly_point": pd.concat(points, ignore_index=True),
        "daily_zone": daily,
        "point_day_health": pd.concat(healths, ignore_index=True),
        "weather_hourly": weather,
        "day_types": day_types,
    }
    cats = ["zone", "column", "channel", "role"]   # concat drops mismatched categoricals
    out["hourly_point"][cats] = out["hourly_point"][cats].astype(str).astype("category")
    for name, frame in out.items():
        frame.to_parquet(C.DATA_CLEANED / f"{name}.parquet", index=False)

    pd.concat(logs, ignore_index=True).to_csv(C.DATA_CLEANED / "cleaning_log.csv", index=False)
    pd.DataFrame(sums).to_csv(C.DATA_CLEANED / "build_summary.csv", index=False)
    pd.concat([io.column_meta(f) for f in floors]).to_csv(
        C.DATA_CLEANED / "column_registry.csv", index=False)

    if verbose:
        for f in floors:
            measured = features.measure_ac_hours(h, f)
            print(f"floor {f}: measured AC hours {measured}  (registry: {R.ac_on_hours(f)})")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild data/cleaned/ from data/raw/.")
    ap.add_argument("--floor", type=int, action="append", help="one floor (repeatable); default all")
    ap.add_argument("--no-minute", action="store_true", help="skip the large minute parquet")
    a = ap.parse_args()
    build(tuple(a.floor) if a.floor else C.FLOORS, write_minute=not a.no_minute)
    print(f"done -> {C.DATA_CLEANED}")
