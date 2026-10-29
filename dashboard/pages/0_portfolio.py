"""Portfolio overview — the whole building on one screen: where do I send the team?"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import portfolio as P
from sbanalytics import registry as R
import ui

c = ui.ctx()
as_of, tariff, ef, area = c["as_of"], c["tariff"], c["ef"], c["area"]
target = R.conv("overcooled_target_pct")

h = ui.building_hourly()
H = ui.building_health()


@st.cache_data(show_spinner="Ranking floors…")
def _league(area: float) -> pd.DataFrame:
    return P.floor_league(D.load_hourly(None), area, D.load_health(None))


@st.cache_resource(show_spinner="Sizing each floor's levers…")
def _levers() -> pd.DataFrame:
    return P.floor_levers(D.load_hourly(None))


@st.cache_data(show_spinner="Building the overcooling map…")
def _heat() -> pd.DataFrame:
    return P.overcooled_by_month(D.load_hourly(None))


st.title("Portfolio overview")
st.caption("CU-BEMS · Chulalongkorn University · seven-storey office, Bangkok · 11,700 m², "
           "33 zones · 1-minute data Jul 2018 – Dec 2019. Read-only: every floor, wired or not.")

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
periods = {"Month to date": (as_of.replace(day=1), as_of),
           f"Trailing {c['window']} day(s)": D.window(as_of, c["window"]),
           "Last 12 months": D.window(as_of, 365),
           "Whole record": (D.RECORD_START, D.RECORD_END)}
cc = st.columns([2, 5])
pname = cc[0].selectbox("Period", list(periods), index=0)
s, e = periods[pname]
s = max(pd.Timestamp(s), D.RECORD_START)
cc[1].caption(f"**{s.date()} – {pd.Timestamp(e).date()}** · as of {as_of.date()}")

bk = P.building_kpis(h, s, e, tariff, ef)
k = st.columns(5)
k[0].metric("Building energy", f"{bk['kwh']:,.0f} kWh",
            help="Submetered zone energy (AC + lighting + plug) on all seven floors, including "
                 "Floor 1. Excludes central plant, lifts and common areas.")
k[1].metric("Cost", f"฿{bk['thb']:,.0f}", help=f"At {tariff:.2f} ฿/kWh (sidebar).")
k[2].metric("Emissions", f"{bk['tco2e']:,.1f} tCO₂e",
            help=f"At {ef:.4f} kgCO₂e/kWh — {R.CONVENTIONS['co2_kg_per_kwh'][2]}.")
oc = bk["overcooled_pct"]
k[3].metric(f"Cooled hours below {R.conv('comfort_lo'):g} °C", f"{oc:.0f}%" if np.isfinite(oc) else "no sensor data",
            delta=f"{oc - target:+.0f} pp vs {target:.0f}% target" if np.isfinite(oc) else None,
            delta_color="inverse",
            help="Office floors F2–F7: share of cooled, occupied zone-hours below "
                 f"{R.conv('comfort_lo'):g} °C, weighted by zone AC energy (the notebooks' "
                 "definition). Floor 1 has no sensors and is not an office floor.")
k[4].metric("Data completeness", f"{bk['meter_pct']:.0f}% meters",
            delta=(f"{bk['sensor_pct']:.0f}% sensors" if np.isfinite(bk["sensor_pct"]) else "no sensors"),
            delta_color="off",
            help=f"Share of zone-hours with usable metering / usable indoor temperature. "
                 f"Target {R.conv('completeness_target'):.0f}%.")

for f in bk["truncated_floors"]:
    st.warning(f"**Floor {f}'s file ends {R.floor_file_end(f):%Y-%m-%d %H:%M}.** Building totals for "
               "days after that exclude Floor 6 — they are not metered, not zero.")
COLD = f"below {R.conv('comfort_lo'):g} °C %"
lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
if s <= hi and pd.Timestamp(e) >= lo:
    st.info(f"**This period overlaps the building-wide sensor gap ({lo.date()} – {hi.date()}).** "
            "Energy is metered throughout; the overcooled share only counts hours with live sensors.")
st.caption("**Floor 1 is not an office floor** (AC never switches off). Its energy is in the "
           "totals; it is excluded from the overcooled share and every schedule ranking.")

# --------------------------------------------------------------------------- #
# floor league
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Floor league")
st.caption("Whole record, from the shared cleaned data and the notebooks' derived tables "
           "(data/derived/). Office floors sorted by overcooled share, worst first. Floor 1 last.")

L = _league(area)
show = pd.DataFrame({
    "floor": [f"F{f}" + ("  · non-office" if f in R.NON_OFFICE_FLOORS else "")
              + (f"  · file ends {R.floor_file_end(f):%d %b %y}" if f in R.FILE_END else "")
              for f in L.floor],
    "partial EUI": L.eui,
    "AC share %": L.ac_share_pct,
    COLD: L.overcooled_pct,
    "after-hours AC %": L.after_hours_ac_pct,
    "vs baseline %": L.deviation_pct,
    "meters %": L.meter_pct,
    "sensors %": L.sensor_pct,
    "status": L.status,
})


def _grey(row):
    off = "  · non-office" in row["floor"]
    return ["color: #9b9b9b; font-style: italic" if off else ""] * len(row)


st.dataframe(
    show.style.apply(_grey, axis=1).format(
        {"partial EUI": "{:.0f}", "AC share %": "{:.0f}", COLD: "{:.0f}",
         "after-hours AC %": "{:.1f}", "vs baseline %": "{:+.1f}", "meters %": "{:.0f}",
         "sensors %": "{:.0f}"}, na_rep="—"),
    width="stretch", hide_index=True,
    column_config={
        "partial EUI": st.column_config.NumberColumn(
            "partial EUI", help="kWh/m²/yr on the assumed floor area (sidebar). Submetered zone "
                                "loads only — not comparable to a whole-building benchmark."),
        COLD: st.column_config.ProgressColumn(
            f"{COLD} (target {target:.0f})", min_value=0, max_value=100, format="%.0f"),
        "vs baseline %": st.column_config.NumberColumn(
            "vs baseline %", help="Jul–Dec 2019 actual vs the weather- and calendar-normalised "
                                  "baseline trained on Jul 2018 – Jun 2019 (notebook 02, total kWh). "
                                  "Positive = used more than expected."),
        "sensors %": st.column_config.NumberColumn(
            "sensors %", help="Includes the six-month building-wide sensor gap."),
        "status": st.column_config.TextColumn(width="large"),
    })
fails = [f"F{f}" for f in L.loc[L.baseline_meets_g14.eq(False), "floor"]]
if fails:
    st.caption(f"Baseline for {', '.join(fails)} fails ASHRAE G14 out of sample — read its "
               "'vs baseline' figure as indicative only.")

# --------------------------------------------------------------------------- #
# what to look at
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("What to look at, floor by floor")
st.caption("Generated from the data: data problems first, then the biggest lever this data can "
           "size (+1 °C lift from the two-estimator range, or metered after-hours AC), then the "
           "zone to walk to.")
lv = _levers()
for _, r in L.iterrows():
    f = int(r.floor)
    txt = P.what_to_look_at(f, r.to_dict(), lv, tariff)
    st.markdown(f"**F{f}** — {txt}" if f not in R.NON_OFFICE_FLOORS
                else f"<span style='color:#9b9b9b'>**F{f}** — {txt}</span>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# floor x month heatmap
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Where the building is overcooled, month by month")
hm = _heat().assign(floor=lambda d: "F" + d.floor.astype(str))
hm["label"] = np.where(hm.overcooled_pct.notna(), hm.overcooled_pct.round(0).astype("Int64").astype(str),
                       "")
hm["cell"] = np.where(hm.overcooled_pct.notna(), "data", "no data")
base = alt.Chart(hm).encode(
    x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(labelAngle=-45, format="%b %y")),
    y=alt.Y("floor:N", title=None, sort=[f"F{f}" for f in P.office_floors()]))
heat = alt.layer(
    base.transform_filter("datum.cell == 'no data'").mark_rect(color="#efefef"),
    base.transform_filter("datum.cell == 'data'").mark_rect().encode(
        color=alt.Color("overcooled_pct:Q", scale=ui.OVERCOOL_COLOR,
                        legend=alt.Legend(title=f"% < {R.conv('comfort_lo'):g} °C")),
        tooltip=["floor", alt.Tooltip("month:T", format="%b %Y"),
                 alt.Tooltip("overcooled_pct:Q", format=".0f", title=COLD),
                 alt.Tooltip("hours:Q", title="cooled zone-hours"),
                 alt.Tooltip("zones:Q", title="zones counted")]),
    base.mark_text(fontSize=10).encode(text="label:N"),
    alt.Chart(pd.DataFrame({"m": [as_of.replace(day=1)]})).mark_rule(
        color=ui.BAD, strokeWidth=2).encode(x="yearmonth(m):O"),
)
st.altair_chart(ui.chart(heat, height=260), width="stretch")
st.caption(
    f"Green at or below the {target:.0f}% target, red towards 100%. Grey cells are not data: "
    f"the building-wide sensor gap ({lo:%b %Y} – {hi:%b %Y}) and "
    + ", ".join(f"Floor {f} after its file ends ({R.floor_file_end(f):%d %b %Y})" for f in R.FILE_END)
    + ". Floor 1 has no indoor sensors. The step from 2018 to 2019 is real and "
    "registered as a change-point: every zone ran 2–3 °C colder after the sensors came back on "
    "2019-03-06, so the whole-record shares in the league table average two configurations.")

# --------------------------------------------------------------------------- #
# energy by floor
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Monthly energy by floor")
mo = P.monthly_energy(h)
mo["floor"] = "F" + mo.floor.astype(str)
st.altair_chart(ui.chart(alt.Chart(mo).mark_bar().encode(
    x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(labelAngle=-45, format="%b %y")),
    y=alt.Y("kWh:Q", title="kWh/month"),
    color=alt.Color("floor:N", scale=ui.FLOOR_COLOR, legend=alt.Legend(orient="top", title=None)),
    order=alt.Order("floor:N"),
    tooltip=["floor", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("kWh:Q", format=",.0f")]),
    height=280), width="stretch")
st.caption("Floor 1 is the largest single consumer and runs around the clock. Floor 6 has no bar "
           "after October 2019 because its file ends, not because it used nothing.")
