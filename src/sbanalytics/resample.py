"""
Minute -> hourly -> daily, with the coverage kept alongside every value.

Energy is summed from what was received (kWh = sum of kW / 60), so totals
reconcile with the raw files; `kwh_ok` says whether enough minutes arrived for
the hour to be used in a model. `min_count=1` everywhere keeps an all-missing
hour as NaN rather than a silent 0 kWh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import registry as R


def to_hourly(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    """One row per zone-hour — the schema the dashboard and `energy/faults/analysis` use.

    Channel membership comes from the registry, so z1's AC4 is picked up, a dead
    meter is left out, and an absent AC stays NaN rather than becoming a zero.
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
        # Coverage of the zone's main meter(s): primary AC if any, else lighting.
        main = R.ac_columns(floor, z, primary_only=True) or li
        zt["minutes_present"] = (df[main].notna().all(axis=1).resample("h").sum()
                                 if main else 0)
        zt["temp_minutes"] = df[te[0]].resample("h").count() if te else 0
        zt["zone"] = z
        rows.append(zt.reset_index())

    h = pd.concat(rows, ignore_index=True)
    h["floor"] = floor
    h["kwh_ok"] = h.minutes_present >= C.CLEAN["hour_min_minutes"]
    # Quality gate: an hour built from 3 valid minutes is not an hourly mean.
    h["temp_ok"] = h.temp_minutes >= R.conv("temp_minutes_gate")
    h["sensor_alive"] = h.temp_minutes > 0
    return h


def to_hourly_points(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    """Long table, one row per meter-hour: kWh for power points, mean for sensors.

    Needed for unit-level questions (dead units, always-on units, peer
    comparison of the 14 small ACs on F2) that the zone table aggregates away.
    """
    out = []
    for p in R.points(floor, existing_only=True):
        s = df[p.column]
        if p.channel in ("ac", "light", "plug"):
            v = (s / 60).resample("h").sum(min_count=1)
        else:
            v = s.resample("h").mean()
        out.append(pd.DataFrame({
            "ts": v.index, "floor": floor, "zone": p.zone, "column": p.column,
            "channel": p.channel, "role": p.role, "value": v.to_numpy(dtype="float32"),
            "minutes": s.resample("h").count().to_numpy(dtype="int16")}))
    long = pd.concat(out, ignore_index=True)
    for c in ("zone", "column", "channel", "role"):
        long[c] = long[c].astype("category")
    return long


def to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Floor-zone-day totals with the number of hours that passed the coverage gate."""
    g = h.assign(date=h.ts.dt.normalize()).groupby(["floor", "zone", "date"])
    agg = g[["ac_kwh", "light_kwh", "plug_kwh"]].sum(min_count=1)
    agg["temp_c"] = g.temp_c.mean()
    agg["ok_hours"] = g.kwh_ok.sum()
    agg = agg.reset_index()
    agg["total_kwh"] = agg[["ac_kwh", "light_kwh", "plug_kwh"]].sum(axis=1, min_count=1)
    agg["day_ok"] = agg.ok_hours >= C.CLEAN["day_min_hours"]
    return agg
