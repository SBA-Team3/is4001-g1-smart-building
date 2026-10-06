"""Energy — EUI, end use, demand, baseload and after-hours."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data as D
from core import energy as E
from core import registry as R
from core import ui

c = ui.ctx()
floor, as_of, win, tariff, area = (c["floor"], c["as_of"], c["window"],
                                   c["tariff"], c["area"])
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
st.title("Energy")

period = st.radio("Period", ["Whole record", "Trailing window"],
                  horizontal=True, index=0)
if period == "Trailing window":
    s, e = D.window(as_of, max(win, 30))
    st.caption(f"{s.date()} – {e.date()}")
else:
    s = e = None
    st.caption("1 Jul 2018 – 31 Dec 2019")

tot = E.floor_totals(h, s, e, floor)
ei = E.eui(h, area, floor)
dem = E.demand_stats(h, s, e)

k = st.columns(5)
k[0].metric("Energy", f"{tot['total']:,.0f} kWh")
k[1].metric("Cost", f"฿{tot['total'] * tariff:,.0f}")
k[2].metric("AC share", f"{tot['ac_share']:.0f}%")
k[3].metric("Peak demand", f"{dem.get('peak_kw', np.nan):,.0f} kW" if dem else "—",
            help=f"at {dem['peak_at']}" if dem else None)
k[4].metric("Load factor", f"{dem.get('load_factor', np.nan):.2f}" if dem else "—",
            help="Mean ÷ peak demand. Low means a peaky profile — here a "
                 "consequence of a tightly scheduled 9-hour cooling day.")

# --------------------------------------------------------------------------- #
st.divider()
left, right = st.columns([2, 3])

with left:
    st.subheader("Energy use intensity")
    st.metric("Partial EUI", f"{ei['eui']:.0f} kWh/m²/yr",
              help=f"{ei['kwh_year']:,.0f} kWh/yr ÷ {area:,.0f} m².")
    bench = pd.DataFrame({
        "label": ["Floor 5 (partial)", "Bangkok good practice", "Bangkok typical"],
        "value": [ei["eui"], ei["good"], ei["typical"]],
        "kind": ["measured", "benchmark", "benchmark"]})
    st.altair_chart(
        alt.Chart(bench).mark_bar().encode(
            y=alt.Y("label:N", title=None, sort=None),
            x=alt.X("value:Q", title="kWh/m²/yr"),
            color=alt.Color("kind:N", scale=alt.Scale(
                domain=["measured", "benchmark"], range=["#2e74b5", "#c9c9c9"]),
                legend=None),
            tooltip=["label", alt.Tooltip("value:Q", format=".0f")],
        ).properties(height=140), width="stretch")
    st.warning(
        "**This is a partial EUI and is not comparable to the benchmark without "
        "that caveat.** It covers submetered zone loads only — AC, lighting and "
        "plug inside the five zones. It excludes central plant, lifts, risers and "
        "common areas, and the floor area is an assumption (11,700 m² ÷ 7; the "
        "dataset paper gives no per-floor split). The benchmarks are "
        "whole-building figures from a 41-building Bangkok office study."
    )

with right:
    st.subheader("End use by zone")
    eu = E.end_use(h, s, e, floor=floor)
    m = eu.melt(id_vars="zone", value_vars=E.END_USES,
                var_name="end_use", value_name="kWh").dropna(subset=["kWh"])
    m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("zone:N", title=None),
            y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["zone", "end_use", alt.Tooltip("kWh:Q", format=",.0f")],
        ).properties(height=260), width="stretch")
    st.caption("z3 has no AC bar because it has no AC meter — it is a stairwell. "
               "A zero there would be a lie.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("When the floor uses energy")

t1, t2, t3 = st.tabs(["Carpet plot", "Load duration", "Hourly profile"])

with t1:
    val = st.selectbox("Channel", ["ac_kwh", "light_kwh", "plug_kwh"],
                       format_func=lambda v: v.replace("_kwh", "").upper())
    cp = E.carpet(h, val, s, e)
    st.altair_chart(
        alt.Chart(cp).mark_rect().encode(
            x=alt.X("yearmonthdate(date):O", title=None,
                    axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("hour:O", title="hour of day", sort="descending"),
            color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme="viridis"),
                            legend=alt.Legend(title="kWh")),
            tooltip=[alt.Tooltip("date:T"), "hour:O",
                     alt.Tooltip(f"{val}:Q", format=",.1f")],
        ).properties(height=420), width="stretch")
    st.caption("The two dark horizontal stripes across the cooling block are "
               "12:00 and 17:00 — the AC is off by schedule, on ~94% of "
               "weekdays. The notebook's assumed 08:00–18:00 occupied window "
               "averaged those two uncooled hours in with eight cooled ones.")

with t2:
    ld = E.load_duration(h, s, e)
    st.altair_chart(
        alt.Chart(ld).mark_area(opacity=0.7, color="#2e74b5").encode(
            x=alt.X("rank_pct:Q", title="% of hours at or above"),
            y=alt.Y("kw:Q", title="floor demand (kW)"),
            tooltip=[alt.Tooltip("rank_pct:Q", format=".1f"),
                     alt.Tooltip("kw:Q", format=".1f")],
        ).properties(height=300), width="stretch")

with t3:
    d = D.slice_hourly(h, s, e)
    prof = d[~d.is_weekend & ~d.is_holiday].groupby("hour", as_index=False)[
        E.END_USES].mean()
    m = prof.melt(id_vars="hour", value_vars=E.END_USES,
                  var_name="end_use", value_name="kW")
    m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("hour:O", title="hour of day"),
            y=alt.Y("kW:Q", title="mean kW per zone-hour"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["hour", "end_use", alt.Tooltip("kW:Q", format=".2f")],
        ).properties(height=280), width="stretch")
    st.caption("Measured schedule: cooling runs 07:48–11:50 and 13:03–16:31. "
               "Hours 7 and 16 are only partly covered (16% and 45% of minutes), "
               "so averaging them with hours 8–10 understates the on-load.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Off-hours energy — the quick wins")

b1, b2 = st.columns(2)
with b1:
    st.markdown("**Baseload ratio** (overnight mean ÷ occupied mean)")
    bl = E.baseload(h, s, e, floor).dropna(subset=["ratio_pct"])
    bl["point"] = bl.zone + " " + bl.end_use
    st.altair_chart(
        alt.Chart(bl.sort_values("ratio_pct", ascending=False)).mark_bar().encode(
            y=alt.Y("point:N", title=None, sort="-x"),
            x=alt.X("ratio_pct:Q", title="night as % of occupied"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]), legend=None),
            tooltip=["point", alt.Tooltip("ratio_pct:Q", format=".1f"),
                     alt.Tooltip("night_kwh:Q", format=",.0f")],
        ).properties(height=320), width="stretch")

with b2:
    st.markdown("**Energy outside the measured schedule**")
    ah = E.after_hours(h, s, e, floor).dropna(subset=["after_hours_pct"])
    ah["point"] = ah.zone + " " + ah.end_use
    st.dataframe(
        ah.sort_values("after_hours_kwh", ascending=False).rename(columns={
            "after_hours_kwh": "after-hours kWh", "total_kwh": "total kWh",
            "after_hours_pct": "after-hours %"})
        [["point", "after-hours kWh", "total kWh", "after-hours %"]].round(1),
        width="stretch", hide_index=True)

st.info(
    "**Two findings the cooling analysis missed entirely, because it excluded "
    "the stairwell and looked only at AC.** `z3_Plug` draws a flat 0.27 kW "
    "round the clock — 98% of its occupied level overnight, and 41% of its "
    "annual energy falls outside working hours. It is also a frozen meter (see "
    "Sensor health), so the right first action is to verify it at the panel "
    "rather than chase the load. Stairwell lighting starts at 06:00, nearly two "
    "hours before cooling, and runs until 19:00, two hours after it stops. z4's "
    "lighting runs at 0.48 kW through the weekend — about five times any other "
    "zone."
)
