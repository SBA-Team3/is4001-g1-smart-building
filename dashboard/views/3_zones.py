"""Comfort — which zones are too cold, by how much, and where the building is
overcooled month by month."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import comfort as C
from sbanalytics import data as D
from sbanalytics import portfolio as P
from sbanalytics import registry as R
import ui

c = ui.ctx()
floor, as_of = c["floor"], c["as_of"]
s, e = c["start"], c["end"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
lo, hi = C.band(c["lo"], c["hi"])            # comfort band from Assumptions
target = R.conv("overcooled_target_pct")


@st.cache_data(show_spinner="Building the overcooling map…")
def _heat(lo: float, hi: float) -> pd.DataFrame:
    return P.overcooled_by_month(D.load_hourly(None), lo=lo, hi=hi)


ui.page_header("Comfort", f"Floor {floor} comfort", f"{c['range']} | Band {lo:g}–{hi:g} °C")
e = ui.file_end_gate(floor, s, e)
ui.floor_band(floor, as_of)

ac_on = st.toggle("Only hours with cooling running", value=True,
                  help="ASHRAE G36 suspends fault evaluation when the equipment is off. A zone "
                       "drifting warm at 18:00 with the AC scheduled off is not a comfort failure.")
zc = C.zone_comfort(h, start=s, end=e, lo=lo, hi=hi, ac_on=ac_on, floor=floor)

# --------------------------------------------------------------------------- #
left, right = st.columns([5, 6], gap="large")
with left:
    ui.md('<p class="cb-overline">League table</p>')
    if len(zc):
        show = zc.sort_values("cold_dh", ascending=False)
        st.dataframe(pd.DataFrame({
            "Zone": [ui.zone_label(floor, z) for z in show.zone],
            "Mean °C": show.mean_c, f"% < {lo:g} °C": show.cold_pct,
            "°C·h below": show.cold_dh, "AC kWh": show.ac_kwh}).style.format(
                {"Mean °C": "{:.1f}", f"% < {lo:g} °C": "{:.0f}", "°C·h below": "{:,.0f}",
                 "AC kWh": "{:,.0f}"}, na_rep=ui.DASH),
            hide_index=True, width="stretch", placeholder=ui.DASH)
    else:
        st.info("No zone-hours with valid temperature in this period.", icon=":material/info:")
with right:
    if len(zc):
        m = zc.melt(id_vars="zone", value_vars=["cold_dh", "warm_dh"], var_name="kind", value_name="dh")
        m["kind"] = m.kind.map({"cold_dh": f"overcooled (< {lo:g} °C)", "warm_dh": f"overheated (> {hi:g} °C)"})
        ui.md('<p class="cb-overline">Exceedance degree-hours</p>')
        st.altair_chart(ui.chart(alt.Chart(m).mark_bar().encode(
            y=alt.Y("zone:N", title=None, sort="-x"), x=alt.X("dh:Q", title="degree-hours"),
            color=alt.Color("kind:N", scale=ui.COLD_WARM, legend=alt.Legend(orient="top", title=None, columnPadding=20, labelLimit=200)),
            tooltip=["zone", "kind", alt.Tooltip("dh:Q", format=",.0f")]),
            height=110 + 32 * zc.zone.nunique()), width="stretch")
        ui.md(f'<p class="cb-overline">Share of cooling delivered below {lo:g} °C</p>')
        st.altair_chart(ui.chart(alt.Chart(zc).mark_bar(color=ui.PRIMARY).encode(
            y=alt.Y("zone:N", title=None, sort="-x"),
            x=alt.X("cooling_below_band_pct:Q", title="% of cooling", scale=alt.Scale(domain=[0, 100])),
            tooltip=["zone", alt.Tooltip("cooling_below_band_pct:Q", format=".1f"),
                     alt.Tooltip("ac_kwh:Q", format=",.0f")]),
            height=70 + 32 * zc.zone.nunique()), width="stretch")

# --------------------------------------------------------------------------- #
ui.section("Overcooling by floor and month",
           f"Cooled occupied hours below {lo:g} °C | office floors | whole record | target {target:.0f}%")
hm = _heat(lo, hi).assign(floor=lambda d: "F" + d.floor.astype(str))
office = [f"F{f}" for f in P.office_floors() if R.sensored_zones(f)]
st.altair_chart(ui.chart(ui.overcool_heatmap(hm, as_of, office), height=230), width="stretch")
ui.md(f'<div style="margin-top:-8px">{ui.seq_legend()}</div>')
bl0, bl1 = (pd.Timestamp(x) for x in R.BLACKOUT)
st.caption(f"'—': the building-wide sensor gap ({bl0:%b %Y} – {bl1:%b %Y}) and "
           + ", ".join(f"Floor {f} after its file ends ({R.floor_file_end(f):%d %b %Y})" for f in R.FILE_END)
           + ". White outline: the as-of month. The step from 2018 to 2019 is a registered change-point: "
             "every zone ran 2–3 °C colder after the sensors came back on 2019-03-06.")

# --------------------------------------------------------------------------- #
ui.section("Zone detail")
zones = R.ZONES[floor]
pre = st.session_state.pop("comfort_zone", None)
if pre in zones:
    st.session_state["zone_detail"] = pre
if st.session_state.get("zone_detail") not in zones:
    st.session_state.pop("zone_detail", None)
zsel = st.selectbox("Zone", zones, format_func=lambda z: ui.zone_label(floor, z), key="zone_detail")
tp = R.point_for(floor, zsel, "temp")

if not R.has_channel(floor, zsel, "ac"):
    what = "is a stairwell" if ui.is_stairwell(floor, zsel) else "has no AC meter"
    st.info(f"**{zsel} {what}.** It has no metered cooling"
            + ("" if R.has_channel(floor, zsel, "temp") else " and no environmental sensor")
            + ". It is excluded from every cooling metric rather than shown as zero.", icon=":material/info:")
elif tp and tp.role == R.ROLE_SUPPLY_TEMP:
    st.warning(f"**{zsel}'s temperature sensor sits in a supply-air path.** It measures the air being "
               "delivered, not the occupied space, so it has no valid room-comfort statistic. It is "
               "shown below as a supply-air diagnostic.", icon=":material/warning:")

zd = D.occupied(h, zones=[zsel], require_temp=R.has_channel(floor, zsel, "temp"))
zd = zd[(zd.ts >= s) & (zd.ts < e + pd.Timedelta(days=1))]
if len(zd):
    daily = zd.set_index("ts").resample("D").agg(temp=("temp_c", "mean"), ac=("ac_kwh", "sum")).reset_index()
    base = alt.Chart(daily)
    band_df = pd.DataFrame({"lo": [lo], "hi": [hi]})
    st.altair_chart(ui.chart(alt.layer(
        alt.Chart(band_df).mark_rect(opacity=0.5, color=ui.NO_DATA).encode(y="lo:Q", y2="hi:Q"),
        base.mark_line(color=ui.PRIMARY, strokeWidth=2, point=len(daily) <= 45).encode(
            x=alt.X("ts:T", title=None), y=alt.Y("temp:Q", title="mean occupied °C", scale=alt.Scale(zero=False)),
            tooltip=["ts:T", alt.Tooltip("temp:Q", format=".1f")])), height=220), width="stretch")
    st.caption(f"Grey band: the declared comfort band {lo:g}–{hi:g} °C.")
    if R.has_channel(floor, zsel, "ac"):
        st.altair_chart(ui.chart(base.mark_bar(color=ui.PRIMARY).encode(
            x=alt.X("ts:T", title=None), y=alt.Y("ac:Q", title="AC kWh/day"),
            tooltip=["ts:T", alt.Tooltip("ac:Q", format=",.0f")]), height=150), width="stretch")
else:
    st.info("No data for this zone in the selected period.", icon=":material/info:")
