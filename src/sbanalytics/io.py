"""
Readers for the raw CU-BEMS CSVs and the external inputs.

Every notebook used to locate and parse these files its own way; these are the
only readers the pipeline uses.
"""

from __future__ import annotations

import pandas as pd

from . import config as C
from . import registry as R


def load_raw(floor: int, years=C.YEARS, reindex: bool = True) -> pd.DataFrame:
    """Minute frame for one floor, indexed by tz-naive Bangkok local time.

    Dates are parsed strictly: the stray `2` that ends the truncated
    2019Floor6.csv becomes NaT and is dropped instead of being coerced into a
    timestamp. With `reindex`, the frame is put on the full record grid so a
    file that ends early (F6) shows up as NaN, not as a shorter table.
    """
    parts = []
    for y in years:
        d = pd.read_csv(C.DATA_RAW / f"{y}Floor{floor}.csv", dtype={"Date": str})
        d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        parts.append(d.dropna(subset=["Date"]))
    df = pd.concat(parts).set_index("Date").sort_index()
    df = df[~df.index.duplicated()]
    df.index.name = "ts"
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")

    expected = {p.column for p in R.points(floor, existing_only=False) if p.column}
    unknown = set(df.columns) - expected
    if unknown:
        raise ValueError(f"Floor {floor}: columns not in the registry: {sorted(unknown)}")

    if reindex:
        lo = max(pd.Timestamp(C.RECORD_START), pd.Timestamp(f"{min(years)}-01-01"))
        hi = min(pd.Timestamp(C.RECORD_END), pd.Timestamp(f"{max(years)}-12-31 23:59"))
        df = df.reindex(pd.date_range(lo, hi, freq="min", name="ts"))
    return df


def grid_report(df: pd.DataFrame) -> dict:
    """Rows actually present vs the full minute grid between first and last reading."""
    present = df.dropna(how="all")
    full = pd.date_range(present.index.min(), present.index.max(), freq="min")
    return {"rows_with_data": len(present), "first": str(present.index.min()),
            "last": str(present.index.max()),
            "all_nan_minutes_inside": len(full) - len(present)}


def load_weather() -> pd.DataFrame:
    """Hourly ERA5 (Open-Meteo) weather on Bangkok local time.

    The file is in UTC; +7 h puts it on the building's clock. Shortwave
    radiation is a mean over the *preceding* hour (labelled at hour end), so it
    is shifted back one hour to match an hour-start label; the instantaneous
    variables stay put. Column names match the dashboard's feature set.
    """
    w = pd.read_csv(C.WEATHER_CSV, skiprows=3, encoding="utf-8")
    w.columns = [c.split(" ")[0] for c in w.columns]
    ts = pd.to_datetime(w["time"]).dt.tz_localize("UTC").dt.tz_convert(C.TZ).dt.tz_localize(None)
    out = pd.DataFrame({
        "ts": ts,
        "out_temp": w["temperature_2m"].astype(float),
        "out_rh": w["relative_humidity_2m"].astype(float),
        "out_dew": w["dew_point_2m"].astype(float),
        "out_solar": w["shortwave_radiation"].astype(float).shift(-1),
        "out_cloud": w["cloud_cover"].astype(float),
    })
    out["out_temp_3h"] = out.out_temp.rolling(3, min_periods=1).mean()
    out["out_temp_24h"] = out.out_temp.rolling(24, min_periods=1).mean()
    lo, hi = pd.Timestamp(C.RECORD_START), pd.Timestamp(C.RECORD_END)
    return out[(out.ts >= lo) & (out.ts <= hi)].reset_index(drop=True)


def load_holidays() -> pd.DataFrame:
    h = pd.read_csv(C.HOLIDAYS_CSV, parse_dates=["date"])
    return h.rename(columns={"holiday_name": "name"})[["date", "name"]]


def load_academic_calendar() -> pd.DataFrame:
    """Only Semester 1 of 2019 is covered — use as a 2019 sensitivity feature only."""
    return pd.read_csv(C.ACADEMIC_CSV, parse_dates=["start_date", "end_date"])


def column_meta(floor: int) -> pd.DataFrame:
    """The registry as a column dictionary for one floor (raw column -> meaning)."""
    return pd.DataFrame([{
        "floor": p.floor, "zone": p.zone, "column": p.column, "channel": p.channel,
        "unit": p.unit, "role": p.role, "lo": p.lo, "hi": p.hi, "note": p.note,
    } for p in R.points(floor, existing_only=False) if p.column])
