"""
Descriptive KPIs that compare across floors without needing floor area.

Absolute kWh is not comparable between floors (areas are unknown), so the
benchmark uses shape ratios: how much load is left on at night, how much runs
outside the schedule, how load responds to outdoor temperature, and how often
occupied zones are overcooled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import registry as R

NIGHT_HOURS = range(1, 5)          # 01:00-04:59, the quietest hours


def floor_kpis(fh: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, g in fh[fh.ok].groupby("floor"):
        d = fd[(fd.floor == f) & fd.ok]
        occ = g[g.occupied]
        night = g[g.is_workday & g.hour.isin(NIGHT_HOURS)]
        tot = g.total_kwh.sum()
        rows.append({
            "floor": f,
            "ok_days": len(d),
            "kwh_per_day": d.total_kwh.mean(),
            "workday_kwh_per_day": d[d.is_workday].total_kwh.mean(),
            "nonworkday_kwh_per_day": d[~d.is_workday].total_kwh.mean(),
            "ac_share": g.ac_kwh.sum() / tot,
            "light_share": g.light_kwh.sum() / tot,
            "plug_share": g.plug_kwh.sum() / tot,
            "baseload_ratio": night.total_kwh.mean() / occ.total_kwh.mean(),
            "after_hours_share": g[~g.occupied].total_kwh.sum() / tot,
            "after_hours_ac_share": g[~g.occupied].ac_kwh.sum() / g.ac_kwh.sum(),
            "peak_kw": g.total_kwh.max(),             # hourly mean kW at the peak hour
            "load_factor": g.total_kwh.mean() / g.total_kwh.max(),
        })
    return pd.DataFrame(rows).set_index("floor")


def zone_comfort(h: pd.DataFrame, lo: float | None = None, hi: float | None = None,
                 rh_hi: float | None = None) -> pd.DataFrame:
    """Occupied, AC-on, sensor-valid zone-hours: overcooling and overheating shares."""
    lo = R.conv("comfort_lo") if lo is None else lo
    hi = R.conv("comfort_hi") if hi is None else hi
    rh_hi = R.conv("rh_hi") if rh_hi is None else rh_hi
    rows = []
    for (f, z), g in h.groupby(["floor", "zone"]):
        if z not in R.comfort_zones(int(f)):
            continue
        d = g[g.is_workday & g.hour.isin(R.ac_on_hours(int(f))) & g.temp_ok & (g.ac_kwh > 0.5)]
        if len(d) < 50:
            continue
        rows.append({"floor": f, "zone": z, "hours": len(d), "mean_temp_c": d.temp_c.mean(),
                     "share_below_lo": (d.temp_c < lo).mean(), "share_above_hi": (d.temp_c > hi).mean(),
                     "cold_degree_hours": (lo - d.temp_c).clip(lower=0).sum(),
                     "share_rh_above": (d.rh_pct > rh_hi).mean(),
                     "ac_kwh_in_window": d.ac_kwh.sum()})
    return pd.DataFrame(rows)


def daily_profiles(fh: pd.DataFrame, value: str = "total_kwh") -> pd.DataFrame:
    """One row per floor-day: the 24-hour profile divided by its daily mean."""
    d = fh[fh.ok].assign(day=fh.ts.dt.normalize())
    p = d.pivot_table(index=["floor", "day"], columns="hour", values=value)
    p = p[p.notna().sum(axis=1) == 24]
    p = p.div(p.mean(axis=1), axis=0)
    meta = d.groupby(["floor", "day"]).day_type.first()
    return p.join(meta)


def cluster_profiles(profiles: pd.DataFrame, k_range=range(2, 7), seed: int = 0):
    """k-means on normalised daily shapes; k chosen by silhouette score."""
    X = profiles.drop(columns="day_type").to_numpy()
    scores = {}
    for k in k_range:
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = silhouette_score(X, lab, sample_size=min(len(X), 3000), random_state=seed)
    k = max(scores, key=scores.get)
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    out = profiles[["day_type"]].copy()
    out["cluster"] = km.labels_
    centres = pd.DataFrame(km.cluster_centers_, columns=range(24))
    return out, centres, pd.Series(scores, name="silhouette")
