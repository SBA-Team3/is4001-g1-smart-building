"""
Anomaly detection in three layers, for every floor.

1. data quality   — health.score_point_days on the point-day table (not here)
2. rules          — after-hours AC, always-on and dead units: things a facilities
                    manager can act on directly
3. statistical    — residuals from a weather/calendar baseline:
                    * robust z  (median / 1.4826 MAD) per floor and occupancy
                      regime, flagged at |z| > 3.5 for >= 2 consecutive hours
                    * CUSUM on daily residuals, for slow drifts a point test misses

There are no labels, so the statistical layer is validated against known
events instead: holidays and closures should be *explained* (the baseline
knows the calendar), the F3 weekend AC programmes found independently in the
F3 notebook should be *caught*, and the share of hours flagged should stay at a
few percent so the register is usable (ISA-18.2 alarm-load thinking).

Residuals are cross-fitted by quarter: each quarter is predicted by a model
that never saw it, so an anomaly cannot be absorbed into its own baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baseline as B
from . import registry as R

ROBUST_Z = 3.5          # Iglewicz & Hoaglin modified z-score cut-off
MIN_RUN_H = 2           # persistence, in the spirit of G36 AlarmDelay
CUSUM_K, CUSUM_H = 0.5, 5.0   # slack and decision interval, in robust-sd units
#: Materiality floor on the residual scale, as a share of the floor's median
#: occupied-hour load. Off-hours residuals have a near-zero MAD (the AC is off),
#: so without a floor any 1 kWh blip scores z > 3.5 and 8-16% of hours flag.
#: At 5%, a flag needs >= ~17% of a typical occupied hour — the same idea as the
#: F3 notebook's max(3.5 MAD, 5 kWh) rule, scaled to each floor.
MIN_SCALE_FRAC = 0.05

#: Weekend AC programmes the F3 notebook (eda/floor3_mik) found with its own rules.
F3_KNOWN_WEEKEND_RUNS = ["2018-09-08", "2019-04-27", "2019-09-14", "2019-09-15", "2019-09-21"]


def mad_scale(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def robust_z(x: pd.Series, min_scale: float = 0.0) -> pd.Series:
    s = max(mad_scale(x), min_scale)
    return (x - x.median()) / s if s and np.isfinite(s) else x * np.nan


# --------------------------------------------------------------------------- #
# statistical layer
# --------------------------------------------------------------------------- #

def crossfit_residuals(fh: pd.DataFrame, target: str = "total_kwh", freq: str = "QS") -> pd.DataFrame:
    """Out-of-fold TOWT predictions for every ok floor-hour."""
    d = fh[fh.ok].dropna(subset=[target, "out_temp"]).copy()
    d["fold"] = d.ts.dt.to_period(freq[0])
    out = []
    for f, g in d.groupby("floor"):
        for fold, te in g.groupby("fold"):
            tr = g[g.fold != fold]
            m = B.TOWT().fit(tr, tr[target])
            out.append(te.assign(expected=np.clip(m.predict(te), 0, None)))
    r = pd.concat(out, ignore_index=True)
    r["resid"] = r[target] - r.expected
    typical = r[r.occupied].groupby("floor")[target].median()
    r["z"] = np.nan
    for (f, occ), g in r.groupby(["floor", "occupied"]):
        r.loc[g.index, "z"] = robust_z(g.resid, MIN_SCALE_FRAC * typical.get(f, 0))
    return r.drop(columns="fold")


def flag_events(r: pd.DataFrame, z: float = ROBUST_Z, min_run: int = MIN_RUN_H) -> pd.DataFrame:
    """Runs of >= min_run consecutive hours with z beyond the cut-off, same sign."""
    rows = []
    for f, g in r.sort_values("ts").groupby("floor"):
        sign = np.where(g.z > z, 1, np.where(g.z < -z, -1, 0))
        gap = g.ts.diff() != pd.Timedelta(hours=1)
        run_id = np.cumsum((np.r_[True, sign[1:] != sign[:-1]]) | gap.to_numpy())
        g = g.assign(sign=sign, run=run_id)
        for _, e in g[g.sign != 0].groupby("run"):
            if len(e) < min_run:
                continue
            rows.append({
                "floor": f, "start": e.ts.min(), "end": e.ts.max() + pd.Timedelta(hours=1),
                "hours": len(e), "direction": "excess" if e.sign.iloc[0] > 0 else "deficit",
                "excess_kwh": e.resid.sum(), "peak_z": e.z.abs().max(),
                "day_type": e.day_type.iloc[0], "occupied_share": e.occupied.mean(),
            })
    return pd.DataFrame(rows)


def cusum(daily_resid: pd.Series, k: float = CUSUM_K, h: float = CUSUM_H) -> pd.DataFrame:
    """Two-sided tabular CUSUM on robust-standardised daily residuals.

    Resets after each alarm so a long shift produces a series of alarms, one
    per ~h/k days, rather than a single permanently-high statistic.
    """
    x = robust_z(daily_resid.dropna())
    hi = lo = 0.0
    rows = []
    for t, v in x.items():
        hi, lo = max(0.0, hi + v - k), min(0.0, lo + v + k)
        alarm = "up" if hi > h else ("down" if lo < -h else "")
        rows.append({"day": t, "z": v, "cusum_hi": hi, "cusum_lo": lo, "alarm": alarm})
        if alarm:
            hi = lo = 0.0
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# rule layer
# --------------------------------------------------------------------------- #

def after_hours_ac(fh: pd.DataFrame, min_kwh: float = 0.5, grace_h: int = 1) -> pd.DataFrame:
    """AC energy outside each floor's schedule above max(min_kwh, median + 3.5 MAD) of off-hours.

    Off-hours = weekends, holidays, closures, and workday hours more than
    `grace_h` outside the measured AC schedule — so the normal 07:00 start-up
    ramp and 17:00 wind-down are not reported as faults. Grouped into events
    per floor-day.
    """
    rows = []
    for f, g in fh[~fh.occupied & fh.ac_kwh.notna()].groupby("floor"):
        sched = R.ac_on_hours(int(f))
        grace = set(range(min(sched) - grace_h, max(sched) + grace_h + 1))
        g = g[~(g.is_workday & g.hour.isin(grace))]
        night = g[g.hour.isin(range(1, 5))].ac_kwh
        thr = max(min_kwh, night.median() + ROBUST_Z * mad_scale(night))
        hit = g[g.ac_kwh > thr]
        for day, e in hit.groupby(hit.ts.dt.normalize()):
            rows.append({"floor": f, "day": day, "hours": len(e), "day_type": e.day_type.iloc[0],
                         "first_hour": int(e.hour.min()), "last_hour": int(e.hour.max()),
                         "ac_kwh": e.ac_kwh.sum(), "excess_kwh": (e.ac_kwh - night.median()).sum(),
                         "threshold_kwh": thr})
    return pd.DataFrame(rows)


def unit_status(hp: pd.DataFrame, on_kwh: float = 0.1) -> pd.DataFrame:
    """Per AC meter: energy, share of hours running, share of *night* hours running.

    Night = 01:00-04:59. A unit running in > 80% of night hours is always-on —
    either a 24/7 space (server room) that should be confirmed, or a schedule
    fault. A unit that never runs is dead or decommissioned.
    """
    ac = hp[hp.channel == "ac"].dropna(subset=["value"])
    ac = ac.assign(on=ac.value > on_kwh, night=ac.ts.dt.hour.between(1, 4))
    g = ac.groupby(["floor", "zone", "column", "role"], observed=True)
    t = pd.DataFrame({
        "kwh": g.value.sum(),
        "hours": g.size(),
        "on_share": g.on.mean(),
        "night_on_share": ac[ac.night].groupby(["floor", "zone", "column", "role"], observed=True).on.mean(),
    }).reset_index()
    t["floor_share"] = t.kwh / t.groupby("floor").kwh.transform("sum")
    t["status"] = np.select(
        [t.kwh < 1, t.night_on_share > 0.8, t.on_share < 0.01],
        ["dead", "always-on", "rarely runs"], "scheduled")
    return t.sort_values(["floor", "kwh"], ascending=[True, False])
