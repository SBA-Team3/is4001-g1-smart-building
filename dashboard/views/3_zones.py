"""Zones — league table, comfort trade-off, and per-zone drill-down."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import comfort as C
from sbanalytics import data as D
from sbanalytics import registry as R
import ui

c = ui.ctx()
floor, as_of, win = c["floor"], c["as_of"], c["window"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
start, end = D.window(as_of, max(win, 30))

ui.page_header(
    f"Operations · Floor {floor}", "Zones & comfort",
    "Which zones are too cold, and by how much? Ranked by **exceedance degree-hours**, not exceedance hours. Hour counts "
    "hide the magnitude of the deviation — ten hours at 22.9 °C and ten hours "
    "at 18 °C score identically on hours and very differently on degree-hours.")

cc = st.columns(4)
lo = cc[0].slider("Band lower (°C)", 20.0, 25.0, float(C.band()[0]), 0.5)
hi = cc[1].slider("Band upper (°C)", 24.0, 29.0, float(C.band()[1]), 0.5)
period = cc[2].selectbox("Period", ["Trailing window", "Whole record"], index=1)
ac_on = cc[3].toggle("Only hours with cooling running", value=True,
                     help="ASHRAE G36 suspends fault evaluation when the "
                          "equipment is off. A zone drifting warm at 18:00 with "
                          "the AC scheduled off is not a comfort failure.")

s, e = (start, end) if period == "Trailing window" else (None, None)
zc = C.zone_comfort(h, start=s, end=e, lo=lo, hi=hi, ac_on=ac_on, floor=floor)

st.info(
    "**This is not 'Unmet Load Hours'.** UMLH is defined relative to a setpoint, "
    "and this building has no setpoint channel — we only ever see realised "
    "temperature, which is an *outcome* of the setpoint. So this is exceedance "
    "against a declared band (Thai field studies put thermal neutrality near "
    "26 °C; below 23 °C in a tropical air-conditioned office is overcooling). "
    "We also do not use the ASHRAE 55 adaptive comfort model, which applies only "
    "to naturally conditioned spaces, not to a mechanically cooled building."
)

if not len(zc):
    st.warning("No zone-hours with valid temperature in this period.")
    st.stop()

# --------------------------------------------------------------------------- #
st.subheader("League table")

show = zc.copy()
show["zone"] = [ui.zone_label(floor, z) for z in show.zone]
st.dataframe(
    show.rename(columns={
        "hours": "hours", "mean_c": "mean °C", "sd_c": "sd °C",
        "cold_pct": f"< {lo:g} °C  %", "warm_pct": f"> {hi:g} °C  %",
        "cold_dh": "overcool °C·h", "warm_dh": "overheat °C·h",
        "in_band_pct": "in band %", "ac_kwh": "AC kWh",
        "cooling_below_band_pct": "cooling below band %",
        "data_confidence": "data confidence"})
    [["zone", "hours", "mean °C", "sd °C", f"< {lo:g} °C  %", f"> {hi:g} °C  %",
      "overcool °C·h", "overheat °C·h", "in band %", "AC kWh",
      "cooling below band %", "data confidence"]].round(1),
    width="stretch", hide_index=True,
    column_config={"in band %": st.column_config.ProgressColumn(color=ui.ACTUAL, label=
        "in band %", min_value=0, max_value=100, format="%.0f"),
                   "AC kWh": st.column_config.NumberColumn(format="%.0f")})

left, right = st.columns(2)
with left:
    st.markdown("**Exceedance degree-hours, split by direction**")
    m = zc.melt(id_vars="zone", value_vars=["cold_dh", "warm_dh"],
                var_name="kind", value_name="dh")
    m["kind"] = m.kind.map({"cold_dh": f"overcooled (< {lo:g} °C)",
                            "warm_dh": f"overheated (> {hi:g} °C)"})
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            y=alt.Y("zone:N", title=None, sort="-x"),
            x=alt.X("dh:Q", title="degree-hours"),
            color=alt.Color("kind:N", scale=ui.COLD_WARM,
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["zone", "kind", alt.Tooltip("dh:Q", format=",.0f")],
        ).properties(height=200), width="stretch")
    st.caption("Overcooling dominates by an order of magnitude — and unlike "
               "overheating it costs energy as well as comfort.")

with right:
    st.markdown("**Where the cooling energy goes**")
    st.altair_chart(
        alt.Chart(zc).mark_bar(color=ui.ACTUAL).encode(
            y=alt.Y("zone:N", title=None, sort="-x"),
            x=alt.X("cooling_below_band_pct:Q",
                    title=f"% of cooling delivered below {lo:g} °C",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=["zone", alt.Tooltip("cooling_below_band_pct:Q", format=".1f"),
                     alt.Tooltip("ac_kwh:Q", format=",.0f")],
        ).properties(height=200), width="stretch")
    st.caption("Exposure, not recoverable waste — some of that cooling is what "
               "holds the zone where it is. Sizing the recoverable part is the "
               "Opportunities page.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("The second lever: control stability")
st.caption("Raising the setpoint moves the distribution. Tightening the control "
           "deadband narrows it. They are different work orders and only one of "
           "them is a setpoint change.")

f1, f2 = st.columns(2)
shift = f1.slider("Setpoint rise (°C)", 0.0, 4.0, 1.5, 0.25)
sd_scale = f2.slider("Deadband tightening", 0.4, 1.0, 1.0, 0.05,
                     help="1.00 leaves the spread unchanged; 0.60 is a control "
                          "loop retuned to cut swing by 40%.")

fr = C.frontier(h, lo=lo, hi=hi, sd_scale=sd_scale, floor=floor)
fl = fr.melt(id_vars=["zone", "shift"], value_vars=["pct_cold", "pct_warm"],
             var_name="band", value_name="pct")
fl["band"] = fl["band"].map({"pct_cold": f"< {lo:g} °C", "pct_warm": f"> {hi:g} °C"})

st.altair_chart(
    alt.layer(
        alt.Chart(fl).mark_line(strokeWidth=2.5).encode(
            x=alt.X("shift:Q", title="setpoint rise (°C)"),
            y=alt.Y("pct:Q", title="% of cooled hours"),
            color=alt.Color("band:N", title=None, scale=ui.COLD_WARM,
                legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("zone:N", legend=alt.Legend(orient="top")),
            tooltip=["zone", "band", alt.Tooltip("shift:Q", format=".2f"),
                     alt.Tooltip("pct:Q", format=".1f")]),
        alt.Chart(pd.DataFrame({"s": [shift]})).mark_rule(
            color=ui.INK, strokeDash=[5, 3]).encode(x="s:Q"),
    ).properties(height=300), width="stretch")

now = C.zone_comfort(h, start=s, end=e, lo=lo, hi=hi, ac_on=ac_on, floor=floor)
after = []
d = D.occupied(h, zones=R.comfort_zones(floor), require_temp=True, ac_on=ac_on)
for z, g in d.groupby("zone"):
    n = C.shift_distribution(g.temp_c, shift, sd_scale)
    after.append({"zone": z, "cold %": (n < lo).mean() * 100,
                  "warm %": (n > hi).mean() * 100, "mean °C": n.mean(),
                  "sd °C": n.std()})
st.dataframe(pd.DataFrame(after).round(1), width="stretch", hide_index=True)
st.warning(
    f"At +{shift:g} °C with the spread unchanged, roughly "
    f"**{np.mean([a['cold %'] for a in after]):.0f}%** of cooled hours are still "
    f"below {lo:g} °C. These distributions are wide (σ ≈ 1.8–2.1 °C), not merely "
    "mis-centred — so a zone that stays volatile after a setpoint change is a "
    "controls-tuning item, not a setpoint item."
)

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Zone detail")

zsel = st.selectbox("Zone", R.ZONES[floor],
                    format_func=lambda z: ui.zone_label(floor, z))
tp = R.point_for(floor, zsel, "temp")

if not R.has_channel(floor, zsel, "ac"):
    what = "is a stairwell" if ui.is_stairwell(floor, zsel) else "has no AC meter"
    st.info(f"**{zsel} {what}.** It has no metered cooling"
            + ("" if R.has_channel(floor, zsel, "temp") else " and no environmental sensor")
            + ". It is excluded from every cooling metric rather than shown as zero.")
elif tp and tp.role == R.ROLE_SUPPLY_TEMP:
    st.warning(
        f"**{zsel}'s temperature sensor sits in a supply-air path.** It measures "
        "the air being delivered, not the space anyone occupies, so it has no "
        "valid room-comfort statistic. It is shown below as a supply-air "
        "diagnostic — which is arguably more useful, since supply air temperature "
        "is what the whole ASHRAE G36 fault suite is built around."
    )

zd = D.occupied(h, zones=[zsel], require_temp=R.has_channel(floor, zsel, "temp"))
if s is not None:
    zd = zd[(zd.ts >= s) & (zd.ts <= e + pd.Timedelta(days=1))]

if len(zd):
    daily = zd.set_index("ts").resample("D").agg(
        temp=("temp_c", "mean"), ac=("ac_kwh", "sum")).reset_index()
    base = alt.Chart(daily)
    band_df = pd.DataFrame({"lo": [lo], "hi": [hi]})
    st.altair_chart(
        alt.layer(
            alt.Chart(band_df).mark_rect(opacity=0.18, color=ui.C["data/overcool-low"]).encode(
                y="lo:Q", y2="hi:Q"),
            base.mark_line(color=ui.WARM, strokeWidth=2).encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y("temp:Q", title="mean °C", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", alt.Tooltip("temp:Q", format=".1f")]),
        ).properties(height=220), width="stretch")
    st.altair_chart(
        base.mark_bar(color=ui.ACTUAL).encode(
            x=alt.X("ts:T", title=None), y=alt.Y("ac:Q", title="AC kWh/day"),
            tooltip=["ts:T", alt.Tooltip("ac:Q", format=",.0f")],
        ).properties(height=150), width="stretch")
else:
    st.info("No data for this zone in the selected period.")
