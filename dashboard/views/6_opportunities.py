"""Savings — the potential saving from two FM settings, building-wide.

The FM sets the HVAC operating schedule and a space temperature target; the page
returns the potential saving on every wired office floor (Floor 1, the lobby, is
excluded), split into the two Energy Conservation Measures:

* ECM 1 · HVAC time scheduling — metered AC outside the schedule, above each
  floor's night baseload.
* ECM 2 · Space temperature setpoint adjustment — cooled occupied hours below the
  target raised to it, converted to kWh with the thermal model's %/°C (the floor's
  own where it calibrates, the building average of those floors elsewhere).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import analysis as A
from sbanalytics import anomaly as AN
from sbanalytics import baseline as B
from sbanalytics import data as D
from sbanalytics import registry as R
import ui

c = ui.ctx()
tariff, LO, HI = float(c["tariff"]), float(c["lo"]), float(c["hi"])
WIRED = tuple(R.wired_floors())                     # office floors; F1 (lobby) is not wired
FLOORS = [f"F{f}" for f in WIRED]
ECM1 = "ECM 1 · HVAC time scheduling"
ECM2 = "ECM 2 · Space temperature setpoint adjustment"
NIGHT = range(1, 5)                                 # 01:00–04:59: the floor's baseload hours
DEF_START, DEF_END = 7, 17


# --------------------------------------------------------------------------- #
# cached data and fits (resources: shared, never mutated)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the office floors…")
def _floor_hourly() -> pd.DataFrame:
    """Floor-hour table for the wired floors, metered ok hours only."""
    fh = B.floor_hourly(D.load_hourly(list(WIRED)))
    return fh[fh.ok & fh.ac_kwh.notna()].reset_index(drop=True)


@st.cache_resource(show_spinner="Loading zone data…")
def _zone_hourly(floor: int) -> pd.DataFrame:
    return D.load_hourly(floor)


@st.cache_resource(show_spinner="Calibrating the thermal model…")
def _physics(floor: int) -> dict:
    return A.fit_physics(_zone_hourly(floor), floor=floor)


@st.cache_data(show_spinner=False)
def _sensitivity() -> tuple[dict, float]:
    """%/°C per floor where the thermal model calibrates, and their building average."""
    own = {f: float(p["pct_per_c"]) for f in WIRED if (p := _physics(f).get("pooled"))}
    return own, float(np.mean(list(own.values()))) if own else np.nan


@st.cache_data(show_spinner=False)
def _default_holidays() -> tuple[str, ...]:
    """Weekdays that are a holiday or closure on every office floor with data that day.

    The calendar is building-wide, so a one-floor closure (F2 31 Jan–1 Feb 2019, F3
    11 Oct 2018) is left off: the other floors cooled normally on those days.
    """
    dt = D.load_day_types()
    fh = _floor_hourly()
    has_data = pd.MultiIndex.from_frame(fh[["floor", "date"]].drop_duplicates())
    dt = dt[pd.MultiIndex.from_frame(dt[["floor", "date"]]).isin(has_data)]
    off = dt.day_type.isin(["holiday", "closure"]).groupby(dt.date).all()
    days = pd.to_datetime(off[off].index)
    return tuple(sorted(d.strftime("%Y-%m-%d") for d in days if d.day_of_week < 5))


def _weekdays() -> list[str]:
    days = pd.date_range(D.RECORD_START, D.RECORD_END, freq="D")
    return [d.strftime("%Y-%m-%d") for d in days if d.day_of_week < 5]


# --------------------------------------------------------------------------- #
# the two measures (cached on hashable inputs)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Sizing HVAC time scheduling…")
def _ecm1(start: int, end: int, hols: tuple[str, ...]) -> pd.DataFrame:
    """Per floor: metered AC outside the schedule, above the floor's night baseload.

    Same qualifying rule as anomaly.after_hours_ac (> max(0.5, night median + 3.5
    MAD)); the saving is the energy above the night median. On workdays, the hours the
    floor cools today (R.ac_on_hours) and the pre-cooling hour before them are never
    counted, whatever the schedule.
    """
    fh, hol = _floor_hourly(), set(hols)
    rows = []
    for f in WIRED:
        g = fh[fh.floor == f]
        n_days = g.date.nunique()
        ann = 365.0 / n_days if n_days else np.nan
        night = g[g.hour.isin(NIGHT)].ac_kwh
        med = float(night.median())
        thr = max(0.5, med + AN.ROBUST_Z * AN.mad_scale(night))
        on = R.ac_on_hours(f)
        protect = set(on) | {min(on) - 1}                           # in use + pre-cooling
        off_day = (g.dow >= 5) | g.date.isin(hol)
        out = off_day | ~((g.hour >= start) & (g.hour < end))
        out &= off_day | ~g.hour.isin(protect)
        hit = g[out & (g.ac_kwh > thr)]
        hit_off = off_day[hit.index]
        rows.append({
            "floor": f, "kwh": float((hit.ac_kwh - med).sum()) * ann,
            "baseline": g.ac_kwh.sum() * ann,
            "days_offday": hit[hit_off].date.nunique() * ann,
            "days_workday": hit[~hit_off].date.nunique() * ann,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sizing the setpoint adjustment…")
def _ecm2(start: int, end: int, hols: tuple[str, ...], target: float) -> pd.DataFrame:
    """Per floor: raise in-schedule cooled occupied hours that run below the target.

    The deficit is read on hours with a valid zone temperature, then scaled to all the
    floor's in-schedule cooled AC energy (the sensor blackout and outages leave about a
    third of cooled hours without a temperature; counting them as zero would understate).
    """
    hol = set(hols)
    own, avg = _sensitivity()
    rows = []
    for f in WIRED:
        hf = _zone_hourly(f)
        zones = [z for z in R.comfort_zones(f) if R.has_channel(f, z, "ac")]
        cooled = D.occupied(hf, zones=zones, hours=range(start, end), ac_on=True,
                            exclude_holidays=False)
        cooled = cooled[~cooled.date.isin(hol)]
        occ = cooled[cooled.temp_ok & cooled.temp_c.notna()]
        deficit = (target - occ.temp_c).clip(lower=0)
        below = deficit > 0
        cover = occ.ac_kwh.sum() / cooled.ac_kwh.sum() if len(occ) else np.nan
        # Annualise on days with AC data, as ECM 1 does: D.annual_factor counts every calendar
        # date in the file, which understates floors with gaps (F6's file ends Oct 2019).
        ann = 365.0 / max(hf.loc[hf.ac_kwh.notna(), "date"].nunique(), 1)
        pct = own.get(f, avg)
        kwh = (-(pct / 100) * float((deficit * occ.ac_kwh).sum()) / cover * ann
               if len(occ) and np.isfinite(pct) else 0.0)
        rows.append({"floor": f, "kwh": kwh, "hours": len(occ),
                     "pct_below": below.mean() * 100 if len(occ) else np.nan,
                     "mean_below": deficit[below].mean() if below.any() else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# header and controls
# --------------------------------------------------------------------------- #
ui.page_header("Savings", "Building savings",
               f"Potential saving on the office floors F{WIRED[0]}–F{WIRED[-1]} (Floor 1, the "
               "lobby, excluded) · sized on the whole record, annualised")

HOURS = [f"{h:02d}:00" for h in range(25)]
DEFAULT_HOLS = _default_holidays()
with st.container(border=True, key="card_sav_controls"):
    k1, k2, k3 = st.columns([5, 3, 4], gap="large", vertical_alignment="bottom")
    s_lbl, e_lbl = k1.select_slider(
        "HVAC operating schedule (workdays)", HOURS, value=(HOURS[DEF_START], HOURS[DEF_END]),
        key="sav_sched",
        help="Cooling is allowed from the start hour up to the end hour on workdays. Weekends "
             "are always non-operating. AC outside this window counts toward ECM 1.")
    start, end = HOURS.index(s_lbl), HOURS.index(e_lbl)
    with k2.popover(f"Holiday calendar · {len(st.session_state.get('sav_hols', DEFAULT_HOLS))} days",
                    icon=":material/event_busy:", width="stretch"):
        hols = st.multiselect(
            "AC-off weekdays", _weekdays(), default=list(DEFAULT_HOLS), key="sav_hols",
            format_func=lambda d: f"{pd.Timestamp(d):%a %d %b %Y}",
            help="Pre-filled with every weekday that is a holiday or closure on all office floors. "
                 "Remove a date to treat it as a workday; add one to switch the AC off.")
        st.caption("Weekends are always non-operating, so only weekdays are listed.")
    target = k3.slider("Space temperature target (°C)", min(20.0, LO), HI, LO, 0.5, key="sav_target",
                       help="Cooled occupied hours below this are raised to it (ECM 2). Hours "
                            f"already at or above the target are left alone. Below {LO:g} °C the "
                            "target is still under the comfort band: a cautious first step.")
    st.caption(f"Tariff {tariff:.2f} ฿/kWh (Assumptions) · comfort band {LO:g}–{HI:g} °C.")
hols_t = tuple(sorted(hols))


def _spans(hours: list[int]) -> list[tuple[int, int]]:
    """[7, 8, 16] -> [(7, 9), (16, 17)]: contiguous hour-starts as (start, end) spans."""
    out: list[list[int]] = []
    for h in sorted(hours):
        if out and h == out[-1][1]:
            out[-1][1] = h + 1
        else:
            out.append([h, h + 1])
    return [(a, b) for a, b in out]


# Guard: a schedule that cuts hours the floor cools today is not cutting waste.
cut = {f: [h for h in R.ac_on_hours(f) if not start <= h < end] for f in WIRED}
cut = {f: hs for f, hs in cut.items() if hs}
if cut:
    where = "; ".join(f"F{f} " + ", ".join(f"{a:02d}:00–{b:02d}:00" for a, b in _spans(hs))
                      for f, hs in cut.items())
    st.warning(
        f"**The schedule cuts hours the AC cools today.** {where}. Plug load shows these floors "
        "occupied in those hours (on F3–F6 it stays above night level until about 20:00), so "
        "that cooling is not waste. They are not counted as savings.",
        icon=":material/warning:")

e1, e2 = _ecm1(start, end, hols_t), _ecm2(start, end, hols_t, float(target))
own, avg = _sensitivity()


# --------------------------------------------------------------------------- #
# headline tiles
# --------------------------------------------------------------------------- #
def _baht(kwh_v: float) -> str:
    return f"฿{kwh_v * tariff:,.0f}/yr" if tariff else "set rate in Assumptions"


base_tot = float(e1.baseline.sum())
s1, s2 = float(e1.kwh.sum()), float(e2.kwh.sum())
tot = s1 + s2
k = st.columns(4)
k[0].metric("Potential savings", _baht(tot), delta=f"{tot:,.0f} kWh/yr", delta_color="off",
            delta_arrow="off", border=True,
            help="ECM 1 + ECM 2 at the settings above. They do not overlap: ECM 1 is AC outside "
                 "the schedule, ECM 2 is cooling inside it.")
k[1].metric("% of HVAC baseline", f"{tot / base_tot * 100:.1f}%" if base_tot else "—",
            delta=f"of {base_tot / 1e3:,.0f}k kWh/yr", delta_color="off", delta_arrow="off",
            border=True, help="HVAC baseline = metered AC on the office floors, annualised per floor.")
k[2].metric(ECM1, f"{s1:,.0f} kWh/yr", delta=_baht(s1), delta_color="off", delta_arrow="off",
            border=True, help="From the operating schedule and holiday calendar.")
k[3].metric(ECM2, f"{s2:,.0f} kWh/yr", delta=_baht(s2), delta_color="off", delta_arrow="off",
            border=True, help="From the space temperature target.")


# --------------------------------------------------------------------------- #
# by floor
# --------------------------------------------------------------------------- #
fl = e1.merge(e2, on="floor", suffixes=("_1", "_2"))
fl["Floor"] = "F" + fl.floor.astype(str)
fl["total"] = fl.kwh_1 + fl.kwh_2

ui.section("By floor")
use_baht = bool(tariff)
unit = "฿/yr" if use_baht else "kWh/yr"
long = fl.melt(id_vars="Floor", value_vars=["kwh_1", "kwh_2"], var_name="measure", value_name="kWh")
long["measure"] = long.measure.map({"kwh_1": ECM1, "kwh_2": ECM2})
long["value"] = long.kWh * tariff if use_baht else long.kWh
st.altair_chart(ui.chart(alt.Chart(long).mark_bar().encode(
    # Every floor labelled: overlap culling was dropping alternate floors.
    y=alt.Y("Floor:N", sort=FLOORS, title=None, scale=alt.Scale(domain=FLOORS, paddingInner=0.35),
            axis=alt.Axis(labelOverlap=False, labelPadding=6, ticks=False)),
    x=alt.X("value:Q", stack="zero", title=f"potential saving ({unit})", axis=alt.Axis(format=",.0f")),
    color=alt.Color("measure:N", title=None, sort=[ECM1, ECM2],
                    scale=alt.Scale(domain=[ECM1, ECM2], range=[ui.PRIMARY, ui.SERIES_2]),
                    legend=alt.Legend(orient="top")),
    order=alt.Order("measure:N"),
    tooltip=["Floor", "measure", alt.Tooltip("kWh:Q", title="kWh/yr", format=",.0f"),
             *([alt.Tooltip("value:Q", title="฿/yr", format=",.0f")] if use_baht else [])],
), height=alt.Step(40)), width="stretch")

tbl = pd.DataFrame({
    "Floor": fl.Floor,
    "ECM 1 kWh/yr": fl.kwh_1,
    "ECM 2 kWh/yr": fl.kwh_2,
    "Total kWh/yr": fl.total,
    "Total ฿/yr": fl.total * tariff if tariff else np.nan,
    "% of HVAC baseline": fl.total / fl.baseline * 100,
    "Out-of-hours AC": [f"~{a:.0f} weekend/holiday days, ~{b:.0f} workdays a year"
                        for a, b in zip(fl.days_offday, fl.days_workday)],
    f"Cooled hours below {target:g} °C": [
        f"{p:.0f}%, by {m:.1f} °C on average" if h else ui.DASH
        for p, m, h in zip(fl.pct_below, fl.mean_below, fl.hours)],
})
st.dataframe(
    tbl.style.format({"ECM 1 kWh/yr": "{:,.0f}", "ECM 2 kWh/yr": "{:,.0f}", "Total kWh/yr": "{:,.0f}",
                      "% of HVAC baseline": "{:.1f}%"}, na_rep=ui.DASH)
       .format({"Total ฿/yr": "฿{:,.0f}"}, subset=["Total ฿/yr"],
               na_rep="set rate in Assumptions" if not tariff else ui.DASH),
    hide_index=True, width="stretch",
    column_config={"% of HVAC baseline": st.column_config.Column(
        help="Of the floor's metered annual AC energy.")})

pooled_floors = [f"F{f}" for f in WIRED if f not in own]
st.caption(
    f"**{ECM1}:** the metered AC energy outside the schedule — weekends, the holiday calendar, "
    "and workday hours before the start or after the end — counting only hours clearly above "
    "the floor's overnight baseload, and only the energy above it, so a legitimate 24/7 unit is "
    "not called waste. On workdays, the hours a floor cools today and the pre-cooling hour "
    "before them are never counted.  \n"
    f"**{ECM2}:** each in-schedule cooled occupied hour below the target is raised to it (read "
    "on hours with a valid zone temperature, scaled to all cooled hours), and converted to kWh "
    "with the thermal model's AC-energy change per °C. "
    + (f"{', '.join(pooled_floors)} use the building average ({avg:+.1f} %/°C) of the floors "
       "where the model calibrates, which assumes they respond like those floors. "
       if pooled_floors and np.isfinite(avg) else "")
    + "The model counts envelope conduction only, so the figure is likely conservative.")
