"""
Loaders for the cleaned dataset in data/cleaned/ (built by
`python -m sbanalytics.pipeline`). Floor-agnostic: every function takes a floor.

Every floor is in the cleaned data. The dashboard additionally gates its pages
on registry.FLOORS[f]["wired"]; notebooks can load any floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import registry as R

CLEANED = C.DATA_CLEANED
HOURLY = CLEANED / "hourly_zone.parquet"
HOURLY_POINTS = CLEANED / "hourly_point.parquet"
DAILY = CLEANED / "daily_zone.parquet"
HEALTH = CLEANED / "point_day_health.parquet"
WEATHER = CLEANED / "weather_hourly.parquet"
DAY_TYPES = CLEANED / "day_types.parquet"

RECORD_START = pd.Timestamp("2018-07-01")
RECORD_END = pd.Timestamp("2019-12-31")
#: A weekday well clear of the blackout, both summer outages and the 2019-09-27
#: change-point — so the dashboard opens on a day that actually shows something.
DEFAULT_AS_OF = pd.Timestamp("2019-11-14")


class FloorNotWired(Exception):
    def __init__(self, floor: int):
        meta = R.FLOORS.get(floor, {})
        super().__init__(meta.get("note", "This floor is not in the dataset."))
        self.floor = floor
        self.note = meta.get("note", "")


def _require(floor: int) -> None:
    if floor not in R.FLOORS:
        raise FloorNotWired(floor)


def _floor_filter(floor):
    if floor is None:
        return None
    floors = [floor] if np.isscalar(floor) else list(floor)
    for f in floors:
        _require(int(f))
    return [("floor", "in", [int(f) for f in floors])]


def minute_path(floor: int):
    return C.MINUTE_DIR / f"floor{floor}.parquet"


def cache_ready(floor: int = 5) -> bool:
    return HOURLY.exists() and HEALTH.exists() and minute_path(floor).exists()


def floor_of(h: pd.DataFrame, default: int = 5) -> int:
    """The single floor an hourly frame belongs to (for floor-specific lookups)."""
    if "floor" in h and len(h):
        return int(h.floor.iloc[0])
    return default


def load_hourly(floor: int | list | None = 5) -> pd.DataFrame:
    """Zone-hour table. `floor=None` loads the whole building."""
    h = pd.read_parquet(HOURLY, filters=_floor_filter(floor))
    h["ts"] = pd.to_datetime(h.ts)
    return h


def load_hourly_points(floor: int | list | None = None) -> pd.DataFrame:
    return pd.read_parquet(HOURLY_POINTS, filters=_floor_filter(floor))


def load_daily(floor: int | list | None = None) -> pd.DataFrame:
    return pd.read_parquet(DAILY, filters=_floor_filter(floor))


def load_weather() -> pd.DataFrame:
    return pd.read_parquet(WEATHER)


def load_day_types() -> pd.DataFrame:
    return pd.read_parquet(DAY_TYPES)


def load_health(floor: int | list | None = 5) -> pd.DataFrame:
    H = pd.read_parquet(HEALTH, filters=_floor_filter(floor))
    H["day"] = pd.to_datetime(H.day)
    return H


def load_minute(floor: int = 5, start=None, end=None, columns=None) -> pd.DataFrame:
    """Cleaned minute data for drill-down. Always pass a date range — the full
    frame is 790k rows and nothing on a page needs all of it at once."""
    _require(floor)
    df = pd.read_parquet(minute_path(floor), columns=columns)
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
    hours = R.ac_on_hours(floor_of(h)) if hours is None else list(hours)
    d = h[~h.is_weekend & h.hour.isin(hours)]
    if exclude_holidays:
        d = d[~d.is_holiday]
        if "day_type" in d:
            d = d[d.day_type != "closure"]
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
