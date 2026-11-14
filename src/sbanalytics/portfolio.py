"""
Building-level (portfolio) view: every floor side by side, read-only.

This is the page an FM lead opens first, so it reads the whole building
whether or not a floor is wired into the operational pages. Two rules keep it
honest:

* Floor 1 is not an office floor. It stays in building totals (its energy is
  real) but is left out of every schedule or overcooling ranking.
* Floor 6's file ends on 2019-10-22. After that date it is "not metered", so
  building totals for those days exclude it and say so.

Cross-floor structural KPIs come from data/derived/ (notebooks 01-02), so the
dashboard and the notebooks quote the same numbers.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import kpis as K
from . import registry as R


def derived(name: str) -> pd.DataFrame:
    """A cross-floor table the notebooks wrote to data/derived/."""
    return pd.read_csv(C.DATA_DERIVED / f"{name}.csv")


def office_floors() -> list[int]:
    return [f for f in sorted(R.FLOORS) if f not in R.NON_OFFICE_FLOORS]


# --------------------------------------------------------------------------- #
# window KPIs
# --------------------------------------------------------------------------- #

def drop_after_file_end(d: pd.DataFrame) -> pd.DataFrame:
    """Rows past a floor's file end are "no file", not a meter outage."""
    keep = pd.Series(True, index=d.index)
    for f in R.FILE_END:
        keep &= ~((d.floor == f) & (d.ts >= R.floor_file_end(f)))
    return d[keep]


def completeness(h: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Per floor: share of zone-hours with usable metering, and with usable
    indoor temperature on sensored zones. Hours past a floor's file end are left
    out — they are "no file", flagged separately, not a logger outage."""
    d = drop_after_file_end(D.slice_hourly(h, start, end))
    rows = []
    for f, g in d.groupby("floor"):
        sens = g[g.zone.isin(R.sensored_zones(int(f)))]
        rows.append({"floor": int(f), "meter_pct": g.kwh_ok.mean() * 100,
                     "sensor_pct": sens.temp_ok.mean() * 100 if len(sens) else np.nan})
    return pd.DataFrame(rows).set_index("floor")


def building_kpis(h: pd.DataFrame, start=None, end=None, tariff: float | None = None,
                  ef: float | None = None, lo: float | None = None, hi: float | None = None) -> dict:
    """Headline numbers for the whole building over a window."""
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    ef = R.conv("co2_kg_per_kwh") if ef is None else ef
    d = D.slice_hourly(h, start, end)
    kwh = float(d[["ac_kwh", "light_kwh", "plug_kwh"]].sum().sum())
    oc = K.overcooled_share(K.cooled_occupied(d[d.floor.isin(office_floors())]), by=(), lo=lo, hi=hi)
    comp = completeness(h, start, end)
    truncated = [f for f in R.FILE_END
                 if end is None or pd.Timestamp(end) >= R.floor_file_end(f).normalize()]
    return {
        "kwh": kwh, "thb": kwh * tariff, "tco2e": kwh * ef / 1000,
        "overcooled_pct": float(oc.share_below.iloc[0] * 100) if len(oc) else np.nan,
        "overcooled_hours": int(oc.hours.iloc[0]) if len(oc) else 0,
        "meter_pct": float(drop_after_file_end(d).kwh_ok.mean() * 100) if len(d) else np.nan,
        "sensor_pct": float(comp.sensor_pct.mean()) if len(comp) else np.nan,
        "days": d.date.nunique(), "truncated_floors": truncated,
    }


def monthly_energy(h: pd.DataFrame) -> pd.DataFrame:
    """kWh per floor per month. Months with no metered hours stay out — a floor
    whose file has ended has no bar, not a zero bar."""
    mo = (h.assign(month=h.ts.dt.to_period("M").dt.to_timestamp())
          .groupby(["floor", "month"])[["ac_kwh", "light_kwh", "plug_kwh"]].sum(min_count=1)
          .sum(axis=1, min_count=1).rename("kWh").reset_index())
    return mo.dropna(subset=["kWh"])


# --------------------------------------------------------------------------- #
# floor league
# --------------------------------------------------------------------------- #


def meter_outages(H: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Floor-level meter outage episodes from the point-day health table.

    A day counts when at least `min_share` of a floor's energy points received
    nothing; days closer together than `episode_gap_days` join one episode, and
    only episodes longer than that gap are kept, so a one-day logger blip is not
    a status note.
    """
    gap = R.conv("episode_gap_days")
    e = H[H.channel.isin(["ac", "light", "plug"])]
    day = e.assign(off=e.received == 0).groupby(["floor", "day"]).off.mean()
    day = day[day >= min_share].reset_index()
    rows = []
    for f, g in day.groupby("floor"):
        g = g.sort_values("day")
        ep = (g.day.diff().dt.days.fillna(1) > gap).cumsum()
        for _, x in g.groupby(ep):
            n = x.day.nunique()
            fe = R.floor_file_end(int(f))
            if n <= gap or (fe is not None and x.day.min() >= fe.normalize()):
                continue                       # past the file end is "no file", not an outage
            rows.append({"floor": int(f), "start": x.day.min(), "end": x.day.max(), "days": n,
                         "share_offline": x.off.mean() * 100})
    return pd.DataFrame(rows, columns=["floor", "start", "end", "days", "share_offline"])


def overcooled_by_month(h: pd.DataFrame, lo: float | None = None, hi: float | None = None) -> pd.DataFrame:
    """Floor x month overcooled share — the 'where do I send the team' view.

    Office floors with sensors only. Months with no valid cooled hours (the
    building-wide sensor gap, F6 after its file ends) come back as NaN rows so
    the heatmap shows them as gaps, not as zero.
    """
    d = K.cooled_occupied(h[h.floor.isin(office_floors())])
    d = d.assign(month=d.ts.dt.to_period("M").dt.to_timestamp())
    s = K.overcooled_share(d, by=("floor", "month"), lo=lo, hi=hi)
    months = pd.period_range(D.RECORD_START, D.RECORD_END, freq="M").to_timestamp()
    floors = [f for f in office_floors() if R.sensored_zones(f)]
    grid = pd.MultiIndex.from_product([floors, months], names=["floor", "month"]).to_frame(index=False)
    out = grid.merge(s, on=["floor", "month"], how="left")
    out["overcooled_pct"] = out.share_below * 100
    return out


# --------------------------------------------------------------------------- #
# the one-sentence "what to look at"
# --------------------------------------------------------------------------- #


