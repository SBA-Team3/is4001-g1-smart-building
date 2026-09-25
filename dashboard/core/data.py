"""
Loaders for the cached artefacts. Floor-agnostic: every function takes a floor.

Only floor 5 is wired (see registry.FLOORS). Loading an unwired floor raises a
FloorNotWired the pages catch and turn into a "not wired yet" panel, rather than
an exception or — worse — an empty chart that looks like real data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import registry as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")

RECORD_START = pd.Timestamp("2018-07-01")
RECORD_END = pd.Timestamp("2019-12-31")
#: A weekday well clear of the blackout, both summer outages and the 2019-09-27
#: change-point — so the dashboard opens on a day that actually shows something.
DEFAULT_AS_OF = pd.Timestamp("2019-11-14")


class FloorNotWired(Exception):
    def __init__(self, floor: int):
        meta = R.FLOORS.get(floor, {})
        super().__init__(meta.get("note", "This floor is not wired yet."))
        self.floor = floor
        self.note = meta.get("note", "")


def _require(floor: int) -> None:
    if not R.FLOORS.get(floor, {}).get("wired"):
        raise FloorNotWired(floor)


def _path(floor: int, name: str) -> str:
    return os.path.join(CACHE, f"f{floor}_{name}.parquet")


def cache_ready(floor: int = 5) -> bool:
    return all(os.path.exists(_path(floor, n))
               for n in ("hourly", "minute", "point_day_health"))


def load_hourly(floor: int = 5) -> pd.DataFrame:
    _require(floor)
    h = pd.read_parquet(_path(floor, "hourly"))
    h["ts"] = pd.to_datetime(h.ts)
    return h


def load_health(floor: int = 5) -> pd.DataFrame:
    _require(floor)
    H = pd.read_parquet(_path(floor, "point_day_health"))
    H["day"] = pd.to_datetime(H.day)
    return H


def load_minute(floor: int = 5, start=None, end=None, columns=None) -> pd.DataFrame:
    """Minute data for drill-down. Always pass a date range — the full frame is
    790k rows and nothing on a page needs all of it at once."""
    _require(floor)
    df = pd.read_parquet(_path(floor, "minute"), columns=columns)
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df


def load_day(floor: int, day, columns=None) -> pd.DataFrame:
    d = pd.Timestamp(day).normalize()
    return load_minute(floor, d, d + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), columns)


# --------------------------------------------------------------------------- #
# as-of window helpers
# --------------------------------------------------------------------------- #

def clamp_as_of(day) -> pd.Timestamp:
    d = pd.Timestamp(day).normalize()
    return min(max(d, RECORD_START), RECORD_END)


def window(as_of, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The trailing `days`-day window ending on (and including) as_of."""
    end = pd.Timestamp(as_of).normalize()
    return end - pd.Timedelta(days=days - 1), end


def slice_hourly(h: pd.DataFrame, start=None, end=None, zones=None) -> pd.DataFrame:
    d = h
    if start is not None:
        d = d[d.ts >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.ts < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if zones:
        d = d[d.zone.isin(list(zones))]
    return d.copy()


def occupied(h: pd.DataFrame, zones=None, hours=None, require_temp=False,
             ac_on=False, exclude_holidays=True) -> pd.DataFrame:
    """Weekday, in-schedule zone-hours.

    `require_temp` uses the quality gate (>=30 valid minutes), not the blunt
    `sensor_alive`, because 11-14% of zone-hours are only partially covered and
    an hour built from 3 minutes is not an hourly mean.
    """
    hours = R.AC_ON_HOURS if hours is None else list(hours)
    d = h[~h.is_weekend & h.hour.isin(hours)]
    if exclude_holidays:
        d = d[~d.is_holiday]
    if zones:
        d = d[d.zone.isin(list(zones))]
    if require_temp:
        d = d[d.temp_ok].dropna(subset=["temp_c"])
    if ac_on:
        d = d[d.ac_kwh > 0.5]
    return d.copy()


def data_status(as_of, floor: int = 5) -> list[dict]:
    """Warnings the sidebar shows for the chosen as-of date: the blackout, a
    zone-specific outage, or proximity to a change-point."""
    d = pd.Timestamp(as_of).normalize()
    out = []

    lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    if lo.normalize() <= d <= hi.normalize():
        out.append({"level": "error", "title": "Inside the sensor blackout",
                    "detail": f"No indoor sensor data anywhere on the floor between "
                              f"{lo.date()} and {hi.date()} (planned firmware "
                              f"maintenance). Energy metering is unaffected."})

    for zone, s, e, kind, note in R.KNOWN_OUTAGES:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if s.normalize() <= d <= e.normalize():
            who = "all zones" if zone == "*" else zone
            out.append({"level": "warning", "title": f"{who}: {kind} offline",
                        "detail": f"{note} ({s.date()} to {e.date()})."})

    for c in R.CHANGE_POINTS:
        cd = pd.Timestamp(c.date)
        if abs((d - cd).days) <= 7:
            out.append({"level": "warning" if c.severity != "critical" else "error",
                        "title": f"Within 7 days of a change-point: {c.title}",
                        "detail": c.detail})
    return out


# --------------------------------------------------------------------------- #
# psychrometrics — temp + RH give us more than either alone
# --------------------------------------------------------------------------- #

def sat_vapour_pressure(t_c):
    """Magnus formula, hPa."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def dew_point(t_c, rh_pct):
    a, b = 17.67, 243.5
    rh = np.clip(rh_pct, 1e-6, 100)
    g = np.log(rh / 100.0) + a * t_c / (b + t_c)
    return b * g / (a - g)


def humidity_ratio(t_c, rh_pct, p_hpa: float = 1013.25):
    """kg water / kg dry air. ASHRAE 55 caps this at 0.012."""
    e = sat_vapour_pressure(t_c) * np.clip(rh_pct, 0, 100) / 100.0
    return 0.622 * e / np.maximum(p_hpa - e, 1e-6)


def absolute_humidity(t_c, rh_pct):
    """g/m3. Should be near-uniform across zones sharing an air path, which makes
    a zone that diverges a strong sensor-fault or separate-air-path signal."""
    e = sat_vapour_pressure(t_c) * np.clip(rh_pct, 0, 100) / 100.0
    return 2.1674 * (e * 100) / (273.15 + t_c)


def annual_factor(h: pd.DataFrame) -> float:
    return 365.0 / h.date.nunique()
