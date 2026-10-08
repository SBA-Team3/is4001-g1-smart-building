"""
Energy KPIs: EUI, end-use split, peak and load factor, baseload, after-hours.

One caveat is carried through every function and must stay on the charts: this
is **submetered zone energy only** — AC, lighting and plug loads inside the five
zones. It excludes central plant, lifts, risers and common areas. So the EUI
computed here is a *partial* EUI and is not directly comparable to a
whole-building benchmark without saying so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import registry as R

END_USES = ["ac_kwh", "light_kwh", "plug_kwh"]


def end_use(h: pd.DataFrame, start=None, end=None, zones=None,
            floor: int = 5) -> pd.DataFrame:
    """Per-zone energy by end use. Zones with no meter stay NaN, never 0."""
    d = D.slice_hourly(h, start, end, zones)
    g = d.groupby("zone", as_index=False)[END_USES].sum(min_count=1)
    for z in g.zone:
        for ch, col in (("ac", "ac_kwh"), ("light", "light_kwh"), ("plug", "plug_kwh")):
            if not R.has_channel(floor, z, ch):
                g.loc[g.zone == z, col] = np.nan
    g["total_kwh"] = g[END_USES].sum(axis=1, min_count=1)
    return g


def floor_totals(h: pd.DataFrame, start=None, end=None, floor: int = 5) -> dict:
    d = D.slice_hourly(h, start, end)
    tot = {c: d[c].sum() for c in END_USES}
    tot["total"] = sum(tot.values())
    tot["ac_share"] = tot["ac_kwh"] / tot["total"] * 100 if tot["total"] else np.nan
    tot["days"] = d.date.nunique()
    return tot


def eui(h: pd.DataFrame, area_m2: float, floor: int = 5) -> dict:
    """Annualised partial EUI against the Bangkok office benchmarks."""
    t = floor_totals(h, floor=floor)
    per_year = t["total"] * 365.0 / max(t["days"], 1)
    return {
        "kwh_year": per_year,
        "eui": per_year / area_m2,
        "good": R.conv("eui_good"),
        "typical": R.conv("eui_typical"),
        "area": area_m2,
        "ac_share": t["ac_share"],
    }


def floor_demand(h: pd.DataFrame, start=None, end=None) -> pd.Series:
    """Whole-floor hourly demand (kW, since a kWh in one hour is a mean kW)."""
    d = D.slice_hourly(h, start, end)
    return d.groupby("ts")[END_USES].sum(min_count=1).sum(axis=1)


def demand_stats(h: pd.DataFrame, start=None, end=None) -> dict:
    s = floor_demand(h, start, end).dropna()
    if not len(s):
        return {}
    peak = s.max()
    return {"peak_kw": peak, "peak_at": s.idxmax(), "mean_kw": s.mean(),
            "load_factor": s.mean() / peak if peak else np.nan, "n": len(s)}


def baseload(h: pd.DataFrame, start=None, end=None, floor: int = 5) -> pd.DataFrame:
    """Night-to-day load ratio per end use.

    The standard FM framing of off-hours waste. A high ratio on lighting or plug
    is the classic quick win; on this floor it is what exposes the stairwell
    lighting and z4's weekend lighting.
    """
    d = D.slice_hourly(h, start, end)
    night = d[(d.hour >= 20) | (d.hour <= 5)]
    day = d[~d.is_weekend & d.hour.isin(R.AC_ON_HOURS)]
    rows = []
    for z in sorted(d.zone.unique()):
        n, dd = night[night.zone == z], day[day.zone == z]
        for ch, col in (("ac", "ac_kwh"), ("light", "light_kwh"), ("plug", "plug_kwh")):
            if not R.has_channel(floor, z, ch):
                continue
            nm, dm = n[col].mean(), dd[col].mean()
            rows.append({"zone": z, "end_use": ch, "night_kw": nm, "day_kw": dm,
                         "ratio_pct": nm / dm * 100 if dm and dm > 0 else np.nan,
                         "night_kwh": n[col].sum(),
                         "share_of_total_pct": n[col].sum() / d[d.zone == z][col].sum() * 100
                         if d[d.zone == z][col].sum() else np.nan})
    return pd.DataFrame(rows)


def after_hours(h: pd.DataFrame, start=None, end=None, floor: int = 5) -> pd.DataFrame:
    """Energy outside the *measured* weekday schedule, per zone and end use."""
    d = D.slice_hourly(h, start, end)
    inside = (~d.is_weekend) & (~d.is_holiday) & d.hour.isin(R.AC_ON_HOURS)
    rows = []
    for z in sorted(d.zone.unique()):
        zz = d[d.zone == z]
        m = inside[d.zone == z]
        for ch, col in (("ac", "ac_kwh"), ("light", "light_kwh"), ("plug", "plug_kwh")):
            if not R.has_channel(floor, z, ch):
                continue
            tot = zz[col].sum()
            out = zz.loc[~m, col].sum()
            rows.append({"zone": z, "end_use": ch, "in_hours_kwh": zz.loc[m, col].sum(),
                         "after_hours_kwh": out, "total_kwh": tot,
                         "after_hours_pct": out / tot * 100 if tot else np.nan})
    return pd.DataFrame(rows)


def carpet(h: pd.DataFrame, value: str = "ac_kwh", start=None, end=None,
           zones=None) -> pd.DataFrame:
    """Day x hour-of-day matrix — the standard FM energy visual."""
    d = D.slice_hourly(h, start, end, zones)
    g = d.groupby(["date", "hour"], as_index=False)[value].sum(min_count=1)
    g["date"] = pd.to_datetime(g.date)
    return g


def load_duration(h: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    s = floor_demand(h, start, end).dropna().sort_values(ascending=False)
    return pd.DataFrame({"rank_pct": np.arange(1, len(s) + 1) / len(s) * 100,
                         "kw": s.values})


def daily(h: pd.DataFrame, start=None, end=None, zones=None) -> pd.DataFrame:
    d = D.slice_hourly(h, start, end, zones)
    g = d.groupby("date", as_index=False)[END_USES].sum(min_count=1)
    g["date"] = pd.to_datetime(g.date)
    g["total_kwh"] = g[END_USES].sum(axis=1, min_count=1)
    g["dow"] = g.date.dt.dayofweek
    g["is_weekend"] = g.dow >= 5
    return g


def same_weekday_baseline(h: pd.DataFrame, as_of, weeks: int = 4,
                          zones=None) -> dict:
    """Compare a day with the same weekday over recent weeks.

    Deliberately short-window and same-weekday rather than year-on-year: the
    floor ran 2-3 degC colder after the 2019 maintenance window, so a
    year-on-year comparison would be measuring a configuration change.
    """
    d = pd.Timestamp(as_of).normalize()
    g = daily(h, zones=zones).set_index("date")
    if d not in g.index:
        return {}
    prior = [d - pd.Timedelta(weeks=w) for w in range(1, weeks + 1)]
    ref = g.reindex([p for p in prior if p in g.index])
    if not len(ref):
        return {"actual": g.loc[d, "total_kwh"], "baseline": np.nan, "n": 0}
    base = ref.total_kwh.mean()
    act = g.loc[d, "total_kwh"]
    return {"actual": act, "baseline": base, "n": len(ref),
            "delta_pct": (act / base - 1) * 100 if base else np.nan}
