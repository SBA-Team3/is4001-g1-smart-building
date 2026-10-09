"""
Calendar, schedule and weather features shared by every analysis.

The calendar has four day types. Public holidays come from the committed CSV;
`closure` (a listed workday the floor did not operate) and `holiday_worked`
are detected from the floor's own load, following the F3 notebook, so an
unlisted university closure is not later reported as a "low-energy anomaly".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import io
from . import registry as R

#: Day the floor's 08-16 h energy is below this share of a typical workday -> closure.
CLOSURE_FRAC = 0.30
#: Holiday whose 08-16 h energy reaches this share of a typical workday -> worked.
HOLIDAY_WORKED_FRAC = 0.70


def add_calendar(h: pd.DataFrame, holidays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hour, weekday, month, date string, holiday flag and name, base day type."""
    holidays = io.load_holidays() if holidays is None else holidays
    names = holidays.groupby(holidays.date.dt.normalize()).name.first()
    d = h.copy()
    day = d.ts.dt.normalize()
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["is_weekend"] = d.dow >= 5
    d["month"] = d.ts.dt.month
    d["date"] = day.dt.strftime("%Y-%m-%d")
    d["holiday_name"] = day.map(names)
    d["is_holiday"] = d.holiday_name.notna()
    d["day_type"] = np.select([d.is_holiday, d.is_weekend], ["holiday", "weekend"], "workday")
    return d


def detect_day_types(h: pd.DataFrame) -> pd.DataFrame:
    """Per floor-date: refine `day_type` with closures and worked holidays.

    Typical = rolling median (20 workdays, centred) of the floor's 08-16 h
    energy. Returns floor, date, day_type, share_of_typical.
    """
    core = h[h.hour.between(8, 16)]
    tot = core[["ac_kwh", "light_kwh", "plug_kwh"]].sum(axis=1, min_count=1)
    day = (core.assign(kwh=tot)
           .groupby(["floor", "date"])
           .agg(kwh=("kwh", lambda s: s.sum(min_count=8)), day_type=("day_type", "first"))
           .reset_index())
    out = []
    for f, g in day.groupby("floor"):
        g = g.sort_values("date").copy()
        wd = g.kwh.where(g.day_type == "workday")
        typical = wd.rolling(20, min_periods=5, center=True).median().ffill().bfill()
        g["share_of_typical"] = g.kwh / typical
        closure = (g.day_type == "workday") & (g.share_of_typical < CLOSURE_FRAC)
        worked = (g.day_type == "holiday") & (g.share_of_typical >= HOLIDAY_WORKED_FRAC)
        g.loc[closure, "day_type"] = "closure"
        g.loc[worked, "day_type"] = "holiday_worked"
        out.append(g)
    return pd.concat(out, ignore_index=True)[["floor", "date", "day_type", "share_of_typical"]]


def apply_day_types(h: pd.DataFrame, day_types: pd.DataFrame) -> pd.DataFrame:
    d = h.drop(columns=["day_type"]).merge(
        day_types[["floor", "date", "day_type"]], on=["floor", "date"], how="left")
    d["is_workday"] = d.day_type.isin(["workday", "holiday_worked"])
    return d


def measure_ac_hours(h: pd.DataFrame, floor: int, on_share: float = 0.5) -> list[int]:
    """Hours in which the floor's AC is 'on' on at least `on_share` of workdays.

    'On' = floor AC kWh above night baseload + 30% of (workday P95 - baseload),
    the F3 notebook's schedule rule. Read off the meter, not assumed.
    """
    d = h[(h.floor == floor) & (h.day_type == "workday")]
    fl = d.groupby(["date", "hour"]).ac_kwh.sum(min_count=1).dropna()
    if fl.empty:
        return []
    base = fl[fl.index.get_level_values("hour").isin(range(1, 5))].median()
    thr = base + 0.3 * (fl.quantile(0.95) - base)
    share = (fl > thr).groupby(level="hour").mean()
    return sorted(int(x) for x in share[share >= on_share].index)


def add_cyclic(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["hour_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)
    return d


def degree_days(weather: pd.DataFrame, base_c: float) -> pd.Series:
    """Daily cooling degree-days from hourly outdoor temperature (mean-of-hours method)."""
    w = weather.set_index("ts").out_temp
    return ((w - base_c).clip(lower=0) / 24).resample("D").sum().rename(f"cdd_{base_c:g}")


def occupied_mask(h: pd.DataFrame) -> pd.Series:
    """Workday rows inside each floor's measured AC hours."""
    key = h.floor.astype(int) * 100 + h.hour
    allowed = {f * 100 + hr for f in h.floor.unique() for hr in R.ac_on_hours(int(f))}
    return h.is_workday & key.isin(allowed)
