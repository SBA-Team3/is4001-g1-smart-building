"""Sensor & data health — which points are down, stale or lying."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import health as HL
from sbanalytics import registry as R
import ui

c = ui.ctx()
floor, as_of = c["floor"], c["as_of"]
if not ui.require_wired(floor):
    st.stop()

st.title("Sensor & data health")
st.caption(
    "This page sits above the others because a rule cannot tell that its input "
    "has gone stale. A comfort or energy figure computed on a dead point looks "
    "exactly like one computed on a live point — so every other page is "
    "conditional on what is shown here."
)

H = ui.health(floor)
S = ui.scored(floor)
hourly = ui.hourly(floor)

ui.trust_band(floor, as_of)
ui.status_notes(as_of, floor)

tab_now, tab_map, tab_points, tab_cross, tab_trace = st.tabs(
    ["Today", "Point × day map", "Per-point record", "Cross-sensor checks", "Raw trace"])


# --------------------------------------------------------------------------- #
with tab_now:
    d = S[S.day == as_of.normalize()]
    if not len(d):
        st.error("No point-day records for this date.")
    else:
        view = d[["zone", "channel", "column", "role", "received", "completeness",
                  "flat_measured", "out_of_range", "spikes", "health"]].copy()
        view["status"] = np.select(
            [view.received == 0,
             d.flag_flatline.values,
             view.completeness < R.conv("completeness_target"),
             (view.out_of_range > 0) | (view.spikes > 0)],
            ["OFFLINE", "FROZEN", "PARTIAL", "BAD SAMPLES"], default="OK")
        view = view.sort_values(["health", "zone"])

        n_ok = int((view.status == "OK").sum())
        st.markdown(f"**{n_ok} of {len(view)} points OK on "
                    f"{as_of.date()}.** Sorted worst first.")
        st.dataframe(
            view.rename(columns={
                "column": "point", "received": "samples", "completeness": "complete %",
                "flat_measured": "longest flat (min)", "out_of_range": "out of range",
            }).round(1),
            width="stretch", hide_index=True,
            column_config={
                "health": st.column_config.ProgressColumn(
                    "health", min_value=0, max_value=100, format="%d"),
                "complete %": st.column_config.NumberColumn(format="%.1f"),
            })

        stale = HL.last_good(H, as_of, floor)
        stale = stale[stale.days_stale > 0]
        if len(stale):
            st.warning(
                "**Points not reporting on this date.** Last good day and how "
                "long they have been silent:")
            st.dataframe(stale.rename(columns={"column": "point"}),
                         width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
with tab_map:
    st.markdown("**Point × day health.** The single most useful data-quality "
                "visual: intermittent comms loss shows as vertical stripes, a "
                "dead point as a horizontal band.")

    ch = st.multiselect("Channels", sorted(S.channel.unique()),
                        default=["temp", "rh", "ac", "light", "plug"])
    show_blackout = st.toggle("Include the planned maintenance window", value=True,
                              help="15 Sep 2018 – 6 Mar 2019: a building-wide "
                                   "firmware update, per the dataset paper. "
                                   "Energy metering was unaffected.")

    hm = HL.heatmap_frame(H, channels=ch or None)
    if not show_blackout:
        hm = hm[~hm.in_blackout]

    if len(hm):
        st.altair_chart(
            alt.Chart(hm).mark_rect().encode(
                x=alt.X("yearmonthdate(day):O", title=None,
                        axis=alt.Axis(labelOverlap=True, tickCount=18, labels=False)),
                y=alt.Y("point:N", title=None, sort="-x"),
                color=alt.Color("health:Q", scale=alt.Scale(
                    scheme="redyellowgreen", domain=[0, 100]),
                    legend=alt.Legend(title="health")),
                tooltip=["point", alt.Tooltip("day:T"), "health:Q", "completeness:Q",
                         "received:Q", "max_flat_nonzero:Q", "spikes:Q"],
            ).properties(height=26 * hm.point.nunique() + 40),
            width="stretch")
        st.caption(
            "Red bands from mid-Sep 2018 to Mar 2019 are the planned sensor "
            "maintenance. The separate red block on z1/z2 in Aug–Sep 2019 is a "
            "real 45- and 69-day outage that the notebook never accounted for — "
            "z2 has no indoor temperature at all for August 2019, the peak "
            "cooling month.")

    st.divider()
    st.markdown("**Monthly completeness by point**")
    mm = H.assign(month=H.day.dt.to_period("M").dt.to_timestamp()) \
          .groupby(["zone", "channel", "month"], as_index=False) \
          .agg(received=("received", "sum"), expected=("expected", "sum"))
    mm["completeness"] = mm.received / mm.expected * 100
    mm["point"] = mm.zone + " " + mm.channel
    st.altair_chart(
        alt.Chart(mm[mm.channel.isin(ch)] if ch else mm).mark_rect().encode(
            x=alt.X("yearmonth(month):O", title=None),
            y=alt.Y("point:N", title=None),
            color=alt.Color("completeness:Q", scale=alt.Scale(
                scheme="redyellowgreen", domain=[0, 100]), legend=alt.Legend(title="%")),
            tooltip=["point", alt.Tooltip("month:T"),
                     alt.Tooltip("completeness:Q", format=".1f")],
        ).properties(height=22 * mm[mm.channel.isin(ch)].point.nunique() + 40 if ch else 400),
        width="stretch")


# --------------------------------------------------------------------------- #
with tab_points:
    excl = st.toggle("Exclude the planned maintenance window from the denominator",
                     value=True,
                     help="With it included every zone looks equally broken by one "
                          "known planned outage. Excluding it shows z2 is still "
                          "only ~75% complete, which is the number to act on.")
    ct = HL.completeness_table(H, exclude_blackout=excl)
    ct["verdict"] = ct.apply(HL.verdict, axis=1)

    st.dataframe(
        ct.rename(columns={
            "column": "point", "completeness": "complete %",
            "offline_days": "days offline", "flat_day_share": "days frozen %",
            "out_of_range": "out of range", "max_flat": "longest flat (min)",
        })[["zone", "channel", "point", "role", "complete %", "days offline",
            "days frozen %", "longest flat (min)", "out of range", "spikes", "verdict"]]
        .round(1),
        width="stretch", hide_index=True,
        column_config={"complete %": st.column_config.ProgressColumn(
            "complete %", min_value=0, max_value=100, format="%.1f")})

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Frozen points — ranked**")
        ss = HL.stuck_summary(H)
        ss = ss[ss.flat_day_share > 0].head(10)
        st.dataframe(
            ss.rename(columns={"column": "point", "flat_day_share": "days frozen %",
                               "longest_min": "longest run (min)",
                               "median_distinct": "distinct values/day"})
              [["point", "role", "days frozen %", "longest run (min)",
                "distinct values/day"]].round(1),
            width="stretch", hide_index=True)
        st.info(
            "**The frozen points are plug meters, not temperature sensors.** "
            "`z3_Plug` reports a constant ~0.27 kW with about 7 distinct values a "
            "day and no diurnal shape at all — that is a constant, not a "
            "measurement. `z5_Plug` and `z2_Plug` sit pinned for up to 2.6 days "
            "at a time. Three of the five plug meters are arguably dead."
        )

    with right:
        st.markdown("**Why temperature reports zero frozen days**")
        t = S[S.channel == "temp"]
        st.metric("Temperature point-days flagged frozen", int(t.flag_flatline.sum()))
        st.metric("Longest flat run while cooling runs",
                  f"{int(t.max_flat_occupied.max())} min",
                  help="Threshold is 60 min.")
        st.markdown(
            "Flatline thresholds differ by channel because the physics differs. A "
            "zone temperature cannot hold ±0.01 °C for an hour *while cooling is "
            "running*; a lighting circuit can sit at a constant load all "
            "afternoon. So sensors are judged on a 60-minute threshold evaluated "
            "**only during operating hours** — ASHRAE Guideline 36 §5.16.14.11 "
            "suspends fault evaluation when the equipment is off — and power "
            "channels on a 4-hour threshold.\n\n"
            "The two 60-minute temperature flatlines in the record are both "
            "overnight with the AC off: genuine thermal quiescence. Reporting "
            "them as faults would be manufacturing an alarm."
        )


# --------------------------------------------------------------------------- #
with tab_cross:
    st.markdown("**Cross-sensor consistency** — the strongest check available "
                "without a reference instrument.")

    left, right = st.columns(2)

    with left:
        st.markdown("##### Absolute humidity by zone")
        hc = HL.humidity_consistency(hourly)
        if len(hc):
            st.altair_chart(
                alt.Chart(hc).mark_bar().encode(
                    y=alt.Y("zone:N", title=None),
                    x=alt.X("abs_hum:Q", title="absolute humidity (g/m³)",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("zone:N", scale=ui.ZONE_COLOR, legend=None),
                    tooltip=["zone", alt.Tooltip("abs_hum:Q", format=".2f"),
                             alt.Tooltip("dew:Q", format=".2f", title="dew point °C"),
                             alt.Tooltip("temp:Q", format=".1f"),
                             alt.Tooltip("rh:Q", format=".1f")],
                ).properties(height=170), width="stretch")
            st.dataframe(
                hc.rename(columns={"abs_hum": "abs humidity g/m³", "dew": "dew point °C",
                                   "w": "humidity ratio", "temp": "temp °C",
                                   "rh": "RH %", "abs_hum_dev": "vs median"})
                  [["zone", "abs humidity g/m³", "dew point °C", "temp °C", "RH %",
                    "vs median"]].round(2),
                width="stretch", hide_index=True)
            st.success(
                "**An independent confirmation that z1 is not measuring room air.** "
                "Temperature legitimately differs between zones; moisture content "
                "largely does not, because they share an air path. z1's absolute "
                "humidity sits ~1.4 g/m³ below the floor median and its dew point "
                "~2.2 °C below — the signature of air that has just crossed a "
                "cooling coil and been dehumidified. This reaches the same verdict "
                "as the temperature argument without using temperature at all."
            )

    with right:
        st.markdown("##### Temperature offset from peer median")
        pv = HL.peer_deviation(hourly)
        if len(pv):
            tol = float(pv.tolerance.iloc[0])
            base = alt.Chart(pv[pv.reliable])
            st.altair_chart(
                alt.layer(
                    alt.Chart(pd.DataFrame({"y": [-tol], "y2": [tol]})).mark_rect(
                        opacity=0.14, color="#2e8b57").encode(y="y:Q", y2="y2:Q"),
                    base.mark_line(point=True, strokeWidth=2).encode(
                        x=alt.X("period:T", title=None),
                        y=alt.Y("deviation:Q", title="°C vs peer median"),
                        color=alt.Color("zone:N", scale=ui.ZONE_COLOR),
                        tooltip=["zone", alt.Tooltip("period:T"),
                                 alt.Tooltip("deviation:Q", format=".2f"),
                                 alt.Tooltip("n:Q", title="hours")]),
                ).properties(height=250), width="stretch")
            st.caption(
                f"Green band is the combined sensor tolerance ±{tol:.2f} °C, from "
                "√(ε₁²+ε₂²) with the dataset's stated ±0.4 °C accuracy — the "
                "ASHRAE G36 way of combining two instruments' error bands. "
                "Months with fewer than 40 valid hours are omitted.")
            st.warning(
                "**This is a stable offset, not calibration drift — and we will "
                "not call it drift.** z5 runs persistently ~1 °C above the peer "
                "median and z4 below it, but the offsets do not grow. With 18 "
                "months, no reference sensor, a 2–3 °C step change across the "
                "2019 maintenance window and two long z2 outages, true drift "
                "cannot be separated from a spatial difference. Reporting it as "
                "drift would be a claim the data cannot support."
            )


# --------------------------------------------------------------------------- #
with tab_trace:
    st.markdown("**Raw minute trace** — confirm a flagged point by eye.")
    pts = [p for p in R.points(floor) if p.column]
    labels = {p.column: f"{p.zone} {p.channel} ({p.column})" for p in pts}

    cc = st.columns([2, 1])
    col = cc[0].selectbox("Point", list(labels), format_func=labels.get,
                          index=list(labels).index("z3_Plug(kW)")
                          if "z3_Plug(kW)" in labels else 0)
    day = cc[1].date_input("Day", value=as_of.date(),
                           min_value=D.RECORD_START.date(),
                           max_value=D.RECORD_END.date(), key="trace_day")

    m = ui.day_minutes(floor, str(day), (col,))
    if not len(m) or m[col].notna().sum() == 0:
        st.error(f"{labels[col]} reported nothing on {day}.")
    else:
        t = m.reset_index().rename(columns={m.index.name or "index": "ts"})
        st.altair_chart(
            alt.Chart(t).mark_line(strokeWidth=1.3, color="#2e74b5").encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y(f"{col}:Q", title=col, scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("ts:T"), alt.Tooltip(f"{col}:Q", format=".3f")],
            ).properties(height=260), width="stretch")

        row = S[(S.column == col) & (S.day == pd.Timestamp(day))]
        if len(row):
            r = row.iloc[0]
            k = st.columns(5)
            k[0].metric("Samples", f"{int(r.received)} / 1440")
            k[1].metric("Distinct values", int(r.distinct))
            k[2].metric("Longest flat run", f"{int(r.max_flat_nonzero)} min")
            k[3].metric("Out of range", int(r.out_of_range))
            k[4].metric("Spikes", int(r.spikes))
