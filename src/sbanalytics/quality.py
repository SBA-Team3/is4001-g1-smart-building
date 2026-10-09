"""
Per-point, per-day health — computed on the *raw* minute frame, before
cleaning, so spikes and flatlines stay visible to the operator.

Moved from dashboard/build_cache.py and generalised to every floor; scoring of
this table lives in `health.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import registry as R

MINUTES_PER_DAY = 1440


def run_lengths(s: pd.Series) -> pd.Series:
    """Length of the constant-value run each sample belongs to.

    NaN breaks a run. A run spanning midnight is counted at full length on both
    days it touches, which is what an operator wants to see.
    """
    valid = s.notna()
    changed = (s != s.shift()) | (~valid) | (~valid.shift(fill_value=True))
    run_id = changed.cumsum()
    lengths = run_id.map(run_id.value_counts())
    return lengths.where(valid)


def spike_threshold(s: pd.Series, channel: str) -> float:
    """Physical limit for temperature; a distribution-based one for power.

    Derived per channel from the quieter year (2019), so it catches spikes on
    channels nobody hand-capped.
    """
    if channel == "temp":
        return R.conv("spike_temp_c_per_min")
    if channel in ("ac", "light", "plug"):
        tail = s.loc["2019":]
        q = tail.quantile(0.9999) if len(tail.dropna()) else s.quantile(0.9999)
        return float(3 * q) if np.isfinite(q) and q > 0 else np.inf
    return np.inf


def point_day_health(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    """One row per point per day: the feed for sensor-health scoring."""
    blackout_lo, blackout_hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    day = df.index.normalize()
    rows = []

    # ASHRAE G36 5.16.14.11: suspend fault evaluation when the equipment is not
    # operating. Overnight temperature flatlines with the AC off are genuine
    # thermal quiescence, not stuck sensors.
    occupied_mask = pd.Series(
        (~df.index.dayofweek.isin([5, 6])) & df.index.hour.isin(R.ac_on_hours(floor)),
        index=df.index)

    for p in R.points(floor, existing_only=True):
        s = df[p.column].astype("float64")
        valid = s.notna()
        runs = run_lengths(s)
        is_zero = s.eq(0)

        oor = pd.Series(False, index=s.index)
        if p.lo is not None:
            oor |= s.lt(p.lo) | s.gt(p.hi)
        oor &= valid

        thr = spike_threshold(s, p.channel)
        spike = (s.diff().abs() > thr) & valid

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
        d = pd.to_datetime(out.day)
        out["in_blackout"] = (d >= blackout_lo.normalize()) & (d <= blackout_hi.normalize())
        out["spike_threshold"] = thr
        rows.append(out)

    h = pd.concat(rows, ignore_index=True)
    for c in ("max_flat_nonzero", "max_flat_zero", "max_flat_occupied"):
        h[c] = h[c].fillna(0).astype(int)
    return h
