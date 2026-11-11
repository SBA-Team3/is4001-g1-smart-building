"""Sensors — which points are down, stale or lying, worst first."""

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
start, end = c["start"], c["end"]
if not ui.require_wired(floor):
    st.stop()

ui.page_header("Sensors", f"Floor {floor} sensors", f"Status on {ui.day_txt(as_of)} | history {c['range']}")
end = ui.file_end_gate(floor, start, end)

H = ui.health(floor)
S = ui.scored(floor)
hourly = ui.hourly(floor)

ps = ui.health_caption(floor, as_of)
ui.status_notes(as_of, floor)

# --------------------------------------------------------------------------- #
# not OK first
# --------------------------------------------------------------------------- #
bad = ps[ps.status != "OK"]
ok = ps[ps.status == "OK"]


def _tbl(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Point": d.point, "Zone": d.zone, "Channel": d.channel,
                         "Status": [ui.point_glyph(s) for s in d.status],
                         "Last good day": pd.to_datetime(d.last_good_day).dt.date,
                         "Days stale": d.days_stale})


STATUS_HELP = ("○ OFFLINE no samples · ● FROZEN flatline · ▲ BAD SAMPLES out of range or spikes — "
               "these three are down. ◐ PARTIAL is below the completeness target: degraded, not down.")
# Map (5/12) beside the Not OK table (7/12): both stay near the fold at 1440.
mcol, tcol = st.columns([5, 7], gap="large")
with mcol:
    ui.md('<p class="cb-overline">Zones</p>')
    meters = ps[ps.channel.isin(["ac", "light", "plug"]) & ps.is_down]
    n_down = meters.groupby("zone").size().to_dict()
    tips = {z: "; ".join(f"{r.point} {r.channel} {r.status}" for r in g.itertuples())
            for z, g in ps.groupby("zone")}
    ui.zone_map(floor, as_of=as_of, neutral=True, height=320, zone_tips=tips,
                sublabel={z: f"● {n} meter{'s' if n != 1 else ''} down" for z, n in n_down.items()})
    st.caption("Schematic, not to scale.")
with tcol:
    ui.section(f"Not OK ({len(bad)})")
    if len(bad):
        st.dataframe(ui.glyph_style(_tbl(bad), ["Status"], {"Days stale": "{:.0f}"}),
                     column_config={"Status": st.column_config.Column(help=STATUS_HELP)},
                     hide_index=True, width="stretch", placeholder=ui.DASH,
                     height="auto" if len(bad) <= 12 else 420)
    else:
        st.caption(f"✓ All {len(ps)} points OK on {ui.day_txt(as_of)}.")
with st.expander(f"OK points ({len(ok)})", icon=":material/check:"):
    st.dataframe(ui.glyph_style(_tbl(ok), ["Status"], {"Days stale": "{:.0f}"}),
                 hide_index=True, width="stretch", placeholder=ui.DASH)

tab_map, tab_points, tab_cross, tab_trace = st.tabs(
    ["Point × day map", "Per-point record", "Cross-sensor checks", "Raw trace"])

# --------------------------------------------------------------------------- #
with tab_map:
    ch = st.multiselect("Channels", sorted(S.channel.unique()),
                        default=[x for x in ["temp", "rh", "ac", "light", "plug"] if x in set(S.channel)])
    hm = S[(S.day >= start) & (S.day <= end)]
    if ch:
        hm = hm[hm.channel.isin(ch)]
    if len(hm):
        hm = hm.assign(point=hm.zone + " " + hm.channel, status=np.select(
            [hm.received == 0, hm.flag_flatline.values,
             hm.completeness < R.conv("completeness_target"),
             (hm.out_of_range > 0) | (hm.spikes > 0)],
            ["OFFLINE", "FROZEN", "PARTIAL", "BAD SAMPLES"], default="OK"))
        hm["mark"] = hm.status.map(ui.POINT_STATUS_MARK)
        n_days = hm.day.nunique()
        base = alt.Chart(hm).encode(
            x=alt.X("yearmonthdate(day):O", title=None,
                    axis=alt.Axis(labels=n_days <= 45, labelAngle=0, format="%-d", labelOverlap=True)),
            y=alt.Y("point:N", title=None, sort="ascending", axis=alt.Axis(minExtent=64)))
        layers = [base.mark_rect(strokeWidth=1.5 if n_days <= 90 else 0).encode(
            fill=alt.Fill("status:N", scale=ui.POINT_STATUS_COLOR,
                          legend=alt.Legend(orient="top", title=None, symbolType="square",
                                            symbolStrokeWidth=1.5, columnPadding=24, labelLimit=160)),
            stroke=alt.Stroke("status:N", scale=ui.POINT_STATUS_STROKE, legend=None),
            tooltip=["point", alt.Tooltip("day:T"), "status", alt.Tooltip("health:Q", format=".0f"),
                     alt.Tooltip("completeness:Q", format=".0f"), "received:Q",
                     "max_flat_nonzero:Q", "spikes:Q"])]
        if n_days <= 45:
            layers.append(base.mark_text(fontSize=10).encode(
                text="mark:N", color=alt.Color("status:N", legend=None, scale=alt.Scale(
                    domain=ui.POINT_STATUS, range=[ui.INK, ui.C["status/warning-text"],
                                                   ui.C["text/on-accent"], ui.C["text/on-accent"], ui.INK_2]))))
        st.altair_chart(ui.chart(alt.layer(*layers), height=22 * hm.point.nunique() + 40), width="stretch")
        st.caption("◐ PARTIAL (dark amber, amber outline) is degraded; ▲ BAD SAMPLES (solid amber) and "
                   "● FROZEN (red) are down; '—' on grey with a ring is OFFLINE (no samples). Glyphs show "
                   "for periods up to 45 days. The building-wide sensor gap "
                   "(Sep 2018 – Mar 2019) shows as a block of them on the environmental channels."
                   + (" The separate block on z1/z2 in Aug–Sep 2019 is a real 45- and 69-day outage."
                      if floor == 5 else ""))
    else:
        st.info("No point-days in this period.", icon=":material/info:")

# --------------------------------------------------------------------------- #
with tab_points:
    excl = st.toggle("Exclude the planned maintenance window from the denominator", value=True,
                     help="With it included every zone looks equally broken by one known planned "
                          "outage. Excluding it shows which points are still incomplete.")
    ct = HL.completeness_table(H, exclude_blackout=excl)
    ct["verdict"] = ct.apply(HL.verdict, axis=1)
    st.dataframe(
        ct.rename(columns={
            "column": "point", "completeness": "complete %",
            "offline_days": "days offline", "flat_day_share": "days frozen %",
            "out_of_range": "out of range", "max_flat": "longest flat (min)",
        })[["zone", "channel", "point", "role", "complete %", "days offline",
            "days frozen %", "longest flat (min)", "out of range", "spikes", "verdict"]].round(1),
        width="stretch", hide_index=True,
        column_config={"complete %": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Whole record. Health is a number here on purpose: a bar would read as data.")

    ui.md('<p class="cb-overline" style="margin-top:12px">Frozen points</p>')
    ss = HL.stuck_summary(H)
    ss = ss[ss.flat_day_share > 0].head(10)
    st.dataframe(
        ss.rename(columns={"column": "point", "flat_day_share": "days frozen %",
                           "longest_min": "longest run (min)", "median_distinct": "distinct values/day"})
          [["point", "role", "days frozen %", "longest run (min)", "distinct values/day"]].round(1),
        width="stretch", hide_index=True)
    if floor == 5:
        st.caption("On this floor the frozen points are plug meters, not temperature sensors: z3_Plug "
                   "reports a constant ~0.27 kW with about 7 distinct values a day; z5_Plug and "
                   "z2_Plug sit pinned for up to 2.6 days at a time. Why temperature never reads "
                   "frozen is on the Methods page.")
    else:
        st.caption("Why temperature never reads frozen (it is only judged while cooling runs) is on "
                   "the Methods page.")

# --------------------------------------------------------------------------- #
with tab_cross:
    left, right = st.columns(2, gap="large")
    with left:
        ui.md('<p class="cb-overline">Absolute humidity by zone</p>')
        hc = HL.humidity_consistency(hourly)
        if len(hc):
            st.altair_chart(ui.chart(
                alt.Chart(hc).mark_bar(color=ui.PRIMARY).encode(
                    y=alt.Y("zone:N", title=None),
                    x=alt.X("abs_hum:Q", title="absolute humidity (g/m³)", scale=alt.Scale(zero=False)),
                    tooltip=["zone", alt.Tooltip("abs_hum:Q", format=".2f"),
                             alt.Tooltip("dew:Q", format=".2f", title="dew point °C"),
                             alt.Tooltip("temp:Q", format=".1f"), alt.Tooltip("rh:Q", format=".1f")]),
                height=170), width="stretch")
            st.dataframe(
                hc.rename(columns={"abs_hum": "abs humidity g/m³", "dew": "dew point °C",
                                   "temp": "temp °C", "rh": "RH %", "abs_hum_dev": "vs median"})
                  [["zone", "abs humidity g/m³", "dew point °C", "temp °C", "RH %", "vs median"]].round(2),
                width="stretch", hide_index=True)
            with st.expander("Why this check works"):
                st.markdown(
                    "Temperature legitimately differs between zones; moisture content largely does "
                    "not, because they share an air path. A zone whose absolute humidity sits well "
                    "below the floor median is reading air that has just crossed a cooling coil."
                    + (" On this floor z1 sits ~1.4 g/m³ below the median and its dew point ~2.2 °C "
                       "below — the same verdict as the temperature argument (supply-air sensor), "
                       "reached without using temperature." if floor == 5 else ""))
    with right:
        ui.md('<p class="cb-overline">Temperature offset from peer median</p>')
        pv = HL.peer_deviation(hourly)
        if len(pv):
            tol = float(pv.tolerance.iloc[0])
            base = alt.Chart()
            st.altair_chart(ui.chart(
                alt.layer(
                    base.mark_area(opacity=0.35, color=ui.NO_DATA).encode(x="period:T", y="lo:Q", y2="hi:Q"),
                    base.mark_line(point=True, strokeWidth=2, color=ui.PRIMARY).encode(
                        x=alt.X("period:T", title=None, axis=alt.Axis(format="%b %y", tickCount=4)),
                        y=alt.Y("deviation:Q", title="°C vs peer median"),
                        tooltip=["zone", alt.Tooltip("period:T"), alt.Tooltip("deviation:Q", format=".2f"),
                                 alt.Tooltip("n:Q", title="hours")]),
                    data=pv[pv.reliable].assign(lo=-tol, hi=tol),
                ).properties(width=200, height=110)
                .facet(facet=alt.Facet("zone:N", title=None), columns=2), height=None),
                width="content")
            with st.expander("How to read it"):
                st.markdown(
                    f"Grey band: combined sensor tolerance ±{tol:.2f} °C, √(ε₁²+ε₂²) with the stated "
                    "±0.4 °C accuracy. Months with fewer than 40 valid hours are omitted. A persistent "
                    "offset is a spatial difference, not calibration drift: with 18 months, no reference "
                    "sensor and the 2019 sensor reinstall inside the record, drift cannot be separated "
                    "from one zone simply running warmer or colder than its peers."
                    + (" Here z5 runs ~1 °C above the peer median and z4 below it, and the offsets do "
                       "not grow." if floor == 5 else ""))

# --------------------------------------------------------------------------- #
with tab_trace:
    st.caption("Raw minute trace — confirm a flagged point by eye.")
    pts = [p for p in R.points(floor) if p.column]
    labels = {p.column: f"{p.zone} {p.channel} ({p.column})" for p in pts}
    first_bad = next((p for p in bad.point if p in labels), None)
    cc = st.columns([2, 1])
    col = cc[0].selectbox("Point", list(labels), format_func=labels.get,
                          index=list(labels).index(first_bad) if first_bad else 0)
    day = cc[1].date_input("Day", value=as_of.date(), min_value=D.RECORD_START.date(),
                           max_value=D.RECORD_END.date(), key="trace_day")
    m = ui.day_minutes(floor, str(day), (col,))
    if not len(m) or m[col].notna().sum() == 0:
        st.warning(f"{labels[col]} reported nothing on {day}.", icon=":material/warning:")
    else:
        t = m.reset_index().rename(columns={m.index.name or "index": "ts"})
        st.altair_chart(ui.chart(
            alt.Chart(t).mark_line(strokeWidth=1.3, color=ui.PRIMARY).encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y(f"{col}:Q", title=col, scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("ts:T"), alt.Tooltip(f"{col}:Q", format=".3f")]),
            height=260), width="stretch")
        row = S[(S.column == col) & (S.day == pd.Timestamp(day))]
        if len(row):
            r = row.iloc[0]
            k = st.columns(5)
            k[0].metric("Samples", f"{int(r.received)} / 1440", border=True)
            k[1].metric("Distinct values", int(r.distinct), border=True)
            k[2].metric("Longest flat run", f"{int(r.max_flat_nonzero)} min", border=True)
            k[3].metric("Out of range", int(r.out_of_range), border=True)
            k[4].metric("Spikes", int(r.spikes), border=True)
