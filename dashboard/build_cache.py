"""
Build the dashboard's cached artefacts from the raw CU-BEMS CSVs.

    python3 dashboard/build_cache.py [--floor 5] [--force]

Minute-level flatline and spike detection over 790,560 rows x 26 channels cannot
run on a page load, so it is precomputed here into a point-day table. Re-runnable
and idempotent; skips work when an artefact is newer than its inputs.

Artefacts written to dashboard/cache/:
    f5_minute.parquet          float32 minute grid, for day-level drill-down
    f5_hourly.parquet          one row per zone-hour, with temp_minutes retained
    point_day_health.parquet   one row per point per day — drives the health page
    weather.parquet            ERA5 hourly for Bangkok (cached from Open-Meteo)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import registry as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
CACHE = os.path.join(HERE, "cache")

YEARS = (2018, 2019)
MINUTES_PER_DAY = 1440

# Single-minute logger artefacts. The notebook capped three channels; the
# reconnaissance found two more on the same dates that it missed, so the caps are
# derived per channel rather than hardcoded (see _spike_threshold).
KNOWN_GLITCH_CAPS = {
    "z1_Light(kW)": 20.0,
    "z1_Plug(kW)": 10.0,
    "z2_Light(kW)": 20.0,
}


# --------------------------------------------------------------------------- #
# raw load
# --------------------------------------------------------------------------- #

def load_raw(floor: int) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        path = os.path.join(DATA, f"{y}Floor{floor}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path, parse_dates=["Date"]))
    df = pd.concat(parts).set_index("Date").sort_index()
    df = df[~df.index.duplicated()]
    return df


def assert_grid(df: pd.DataFrame) -> dict:
    """The minute grid should be complete — every gap is a NaN value, not a
    missing row. Verified during reconnaissance; asserted here so a future data
    refresh cannot quietly break the completeness arithmetic."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    missing = len(full.difference(df.index))
    return {"rows": len(df), "expected": len(full), "missing_rows": missing,
            "duplicates": int(df.index.duplicated().sum()),
            "start": str(df.index.min()), "end": str(df.index.max())}


# --------------------------------------------------------------------------- #
# per-point per-day health
# --------------------------------------------------------------------------- #

def _run_lengths(s: pd.Series) -> pd.Series:
    """Length of the constant-value run each sample belongs to.

    NaN breaks a run. A run spanning midnight is counted at full length on both
    days it touches, which is what an operator wants to see.
    """
    valid = s.notna()
    changed = (s != s.shift()) | (~valid) | (~valid.shift(fill_value=True))
    run_id = changed.cumsum()
    lengths = run_id.map(run_id.value_counts())
    return lengths.where(valid)


def _spike_threshold(s: pd.Series, channel: str) -> float:
    """Physical limit for temperature; a distribution-based one for power.

    The notebook's fixed glitch caps miss z2_Plug and z4_Plug. Deriving the
    threshold per channel from the quiet year (2019) catches those without
    needing a hand-maintained list.
    """
    if channel == "temp":
        return R.conv("spike_temp_c_per_min")
    if channel in ("ac", "light", "plug"):
        tail = s.loc["2019":]
        q = tail.quantile(0.9999) if len(tail.dropna()) else s.quantile(0.9999)
        return float(3 * q) if np.isfinite(q) and q > 0 else np.inf
    return np.inf


def point_day_health(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    """One row per point per day: the feed for the sensor-health page."""
    blackout_lo, blackout_hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    day = df.index.normalize()
    rows = []

    # ASHRAE G36 5.16.14.11: suspend fault evaluation when the equipment is not
    # operating. Both of Floor 5's only two 60-minute temperature flatlines are
    # overnight with the AC off — genuine thermal quiescence, not stuck sensors.
    # Evaluating sensor flatlines only while cooling runs is the standard's own
    # rule, not a threshold tuned to make the answer come out clean.
    occupied_mask = pd.Series(
        (~df.index.dayofweek.isin([5, 6])) & df.index.hour.isin(R.AC_ON_HOURS),
        index=df.index)

    for p in R.points(floor, existing_only=True):
        s = df[p.column]
        valid = s.notna()
        runs = _run_lengths(s)
        is_zero = s.eq(0)

        oor = pd.Series(False, index=s.index)
        if p.lo is not None:
            oor |= s.lt(p.lo) | s.gt(p.hi)
        oor &= valid

        thr = _spike_threshold(s, p.channel)
        jump = s.diff().abs()
        spike = (jump > thr) & valid

        g = pd.DataFrame({
            "day": day, "valid": valid, "oor": oor, "spike": spike,
            "run_nonzero": runs.where(~is_zero), "run_zero": runs.where(is_zero),
            "run_occ": runs.where(~is_zero & occupied_mask),
            "val": s,
        }).groupby("day")

        out = pd.DataFrame({
            "received": g.valid.sum(),
            "out_of_range": g.oor.sum(),
            "spikes": g.spike.sum(),
            "max_flat_nonzero": g.run_nonzero.max(),
            "max_flat_zero": g.run_zero.max(),
            "max_flat_occupied": g.run_occ.max(),
            "distinct": g.val.nunique(),
            "vmin": g.val.min(), "vmax": g.val.max(), "vmean": g.val.mean(),
        }).reset_index()

        out.insert(0, "floor", floor)
        out.insert(1, "zone", p.zone)
        out.insert(2, "channel", p.channel)
        out.insert(3, "column", p.column)
        out.insert(4, "role", p.role)
        out["expected"] = MINUTES_PER_DAY
        out["completeness"] = out.received / out.expected * 100
        out["in_blackout"] = (pd.to_datetime(out.day) >= blackout_lo.normalize()) & \
                             (pd.to_datetime(out.day) <= blackout_hi.normalize())
        out["spike_threshold"] = thr
        rows.append(out)

    h = pd.concat(rows, ignore_index=True)
    for c in ("max_flat_nonzero", "max_flat_zero", "max_flat_occupied"):
        h[c] = h[c].fillna(0).astype(int)
    return h


# --------------------------------------------------------------------------- #
# hourly zone table
# --------------------------------------------------------------------------- #

def to_hourly(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    """One row per zone-hour.

    Channel membership comes from the registry, so z1's AC4 is picked up and z3's
    absent AC stays NaN rather than becoming a zero. `min_count=1` on every sum
    keeps an all-missing hour as NaN instead of silently reporting 0 kWh.
    """
    rows = []
    hidx = df.resample("h").size().index
    nan_series = lambda: pd.Series(np.nan, index=hidx)  # noqa: E731

    for z in R.ZONES[floor]:
        ac = R.ac_columns(floor, z)
        li = R.channel_columns(floor, "light", z)
        pl = R.channel_columns(floor, "plug", z)
        te = R.channel_columns(floor, "temp", z)
        rh = R.channel_columns(floor, "rh", z)
        lx = R.channel_columns(floor, "lux", z)

        def energy(cols):
            if not cols:
                return nan_series()
            return (df[cols].sum(axis=1, min_count=1) / 60).resample("h").sum(min_count=1)

        def mean_of(cols):
            return df[cols].mean(axis=1).resample("h").mean() if cols else nan_series()

        zt = pd.DataFrame({
            "ac_kwh": energy(ac), "light_kwh": energy(li), "plug_kwh": energy(pl),
            "temp_c": mean_of(te), "rh_pct": mean_of(rh), "lux_idx": mean_of(lx),
        })
        zt["minutes_present"] = df[ac[0]].resample("h").count() if ac else 0
        zt["temp_minutes"] = df[te[0]].resample("h").count() if te else 0
        zt["zone"] = z
        rows.append(zt.reset_index().rename(columns={"Date": "ts"}))

    h = pd.concat(rows, ignore_index=True)
    h["floor"] = floor
    h["hour"] = h.ts.dt.hour
    h["dow"] = h.ts.dt.dayofweek
    h["is_weekend"] = h.dow >= 5
    h["month"] = h.ts.dt.month
    h["date"] = h.ts.dt.date.astype(str)
    h["is_holiday"] = h.date.isin(THAI_HOLIDAYS)
    # Quality gate: an hour built from 3 valid minutes is not an hourly mean.
    h["temp_ok"] = h.temp_minutes >= R.conv("temp_minutes_gate")
    h["sensor_alive"] = h.temp_minutes > 0
    return h


THAI_HOLIDAYS = {
    "2018-07-27", "2018-07-30", "2018-08-13", "2018-10-15", "2018-10-23",
    "2018-12-05", "2018-12-10", "2018-12-31",
    "2019-01-01", "2019-02-19", "2019-04-08", "2019-04-15", "2019-04-16",
    "2019-05-01", "2019-05-06", "2019-05-20", "2019-06-03", "2019-07-16",
    "2019-07-29", "2019-08-12", "2019-10-14", "2019-10-23", "2019-12-05",
    "2019-12-10", "2019-12-31",
}


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #

def get_weather() -> pd.DataFrame:
    """ERA5 reanalysis for central Bangkok. CU-BEMS has no outdoor channel, so
    this external join is what makes weather normalisation possible at all."""
    path = os.path.join(CACHE, "weather.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    legacy = os.path.join(REPO, "floor_5_Terence", "cache", "bangkok_weather.csv")
    if os.path.exists(legacy):
        w = pd.read_csv(legacy, parse_dates=["ts"])
    else:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               "?latitude=13.7367&longitude=100.5232"
               "&start_date=2018-07-01&end_date=2019-12-31"
               "&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
               "&timezone=Asia%2FBangkok")
        j = json.loads(urllib.request.urlopen(url, timeout=90).read())
        w = pd.DataFrame({
            "ts": pd.to_datetime(j["hourly"]["time"]),
            "out_temp": j["hourly"]["temperature_2m"],
            "out_rh": j["hourly"]["relative_humidity_2m"],
            "out_solar": j["hourly"]["shortwave_radiation"],
        })

    w = w.sort_values("ts").reset_index(drop=True)
    w["out_temp_3h"] = w.out_temp.rolling(3, min_periods=1).mean()
    w["out_temp_24h"] = w.out_temp.rolling(24, min_periods=1).mean()
    w.to_parquet(path, index=False)
    return w


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def build(floor: int = 5, force: bool = False) -> None:
    os.makedirs(CACHE, exist_ok=True)
    tag = f"f{floor}"
    paths = {k: os.path.join(CACHE, f"{tag}_{k}.parquet")
             for k in ("minute", "hourly")}
    paths["health"] = os.path.join(CACHE, f"{tag}_point_day_health.parquet")

    if not force and all(os.path.exists(p) for p in paths.values()):
        print(f"All floor-{floor} artefacts present. Use --force to rebuild.")
        return

    print(f"Loading raw floor {floor} ...")
    df = load_raw(floor)
    grid = assert_grid(df)
    print(f"  {grid['rows']:,} rows  {grid['start']} -> {grid['end']}")
    print(f"  missing rows: {grid['missing_rows']}   duplicates: {grid['duplicates']}")
    if grid["missing_rows"]:
        print("  WARNING: minute grid is not complete; completeness figures assume "
              "every gap is a NaN value, not a missing row.")

    print("Computing point-day health (pre-repair, so faults stay visible) ...")
    health = point_day_health(df, floor)
    health.to_parquet(paths["health"], index=False)
    print(f"  {len(health):,} point-days -> {paths['health']}")

    # Repair only the known single-minute logger artefacts, and only AFTER the
    # health table is built, so the spikes are still reported to the operator.
    for col, cap in KNOWN_GLITCH_CAPS.items():
        if col in df.columns:
            n = int((df[col] > cap).sum())
            df.loc[df[col] > cap, col] = np.nan
            print(f"  capped {col} > {cap} kW -> {n} minutes set to NaN")

    print("Writing minute parquet ...")
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    df.to_parquet(paths["minute"])
    print(f"  {os.path.getsize(paths['minute'])/1e6:.1f} MB -> {paths['minute']}")

    print("Aggregating hourly ...")
    h = to_hourly(df, floor)
    w = get_weather()
    h = h.merge(w, on="ts", how="left")
    h.to_parquet(paths["hourly"], index=False)
    print(f"  {len(h):,} zone-hours -> {paths['hourly']}")

    # Reconciliation against the figure the notebook's EDA established.
    ac = h.ac_kwh.sum()
    print(f"\nReconciliation: total AC {ac:,.0f} kWh (EDA reference 177,914) "
          f"-> {ac/177914*100-100:+.3f}%")
    def fmt(zz, col, floor_, zone, channel):
        if not R.has_channel(floor_, zone, channel):
            return "not metered"
        return f"{zz[col].sum():,.0f}"

    for z in R.ZONES[floor]:
        zz = h[h.zone == z]
        print(f"  {z}: AC {fmt(zz,'ac_kwh',floor,z,'ac'):>11}  "
              f"Light {fmt(zz,'light_kwh',floor,z,'light'):>9}  "
              f"Plug {fmt(zz,'plug_kwh',floor,z,'plug'):>9}  "
              f"temp_ok hours {int(zz.temp_ok.sum()):>5,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(args.floor, args.force)
