"""Sensors — which points are down, stale or lying, worst first."""

from __future__ import annotations

import altair as alt
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

ui.page_header("Sensors", f"Floor {floor}", f"Status on {ui.day_txt(as_of)} | history {c['range']}")
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

tab_map, tab_hum, tab_trace = st.tabs(["Point × day map", "Humidity check", "Minute trace"])

#: Short form of health.verdict for the row labels; a healthy point gets no suffix.
VERDICT_SHORT = {"Constant — not a measurement": "constant", "Mostly offline": "mostly offline",
                 "Frequently frozen": "often frozen", "Intermittent": "intermittent",
                 "Occasional bad samples": "bad samples", "Healthy": ""}


def _row_names(S: pd.DataFrame) -> dict:
    """Row name per point: 'z2 temp', or 'z1 ac4' where a zone has several channels
    of one kind, so those points get a row each instead of being drawn on top of each other."""
    multi = S.groupby(["zone", "channel"]).column.nunique()
    multi = set(multi[multi > 1].index)
    names = {}
    for z, ch, col in S[["zone", "channel", "column"]].drop_duplicates().itertuples(index=False):
        names[col] = (f"{z} {col.split('(')[0].split('_', 1)[-1].lower()}" if (z, ch) in multi
                      else f"{z} {ch}")
    return names


def _point_summary(hm: pd.DataFrame) -> pd.DataFrame:
    """Per point over the period — the old per-point record. The planned sensor gap is
    left out of the denominator unless the period lies wholly inside it."""
    base = hm[~hm.in_blackout] if (~hm.in_blackout).any() else hm
    g = base.groupby(["column", "role"], as_index=False).agg(
        days=("day", "nunique"), received=("received", "sum"), expected=("expected", "sum"),
        offline_days=("received", lambda s: int((s == 0).sum())), flat_days=("flag_flatline", "sum"),
        out_of_range=("out_of_range", "sum"), spikes=("spikes", "sum"))
    g["completeness"] = g.received / g.expected * 100
    g["flat_day_share"] = g.flat_days / g.days * 100
    g["verdict"] = g.apply(HL.verdict, axis=1)
    return g


# --------------------------------------------------------------------------- #
with tab_map:
    ch = st.multiselect("Channels", sorted(S.channel.unique()),
                        default=[x for x in ["temp", "rh", "ac", "light", "plug"] if x in set(S.channel)])
    hm = S[(S.day >= start) & (S.day <= end)]
    if ch:
        hm = hm[hm.channel.isin(ch)]
    if len(hm):
        summ = _point_summary(hm)
        names = _row_names(S)
        summ["row"] = [names[c] + (f" · {VERDICT_SHORT.get(v, v.lower())}" if VERDICT_SHORT.get(v, v) else "")
                       for c, v in zip(summ.column, summ.verdict)]
        summ["rank"] = summ.verdict.map({v: i for i, v in enumerate(VERDICT_SHORT)})
        summ = summ.sort_values(["rank", "completeness", "flat_day_share"], ascending=[True, True, False])
        hm = hm.merge(summ[["column", "row", "completeness", "offline_days", "flat_day_share", "verdict"]]
                      .rename(columns={"completeness": "period_complete"}), on="column", how="left")
        hm["status"] = ui.day_status(hm)
        hm["mark"] = hm.status.map(ui.POINT_STATUS_MARK)
        n_days = hm.day.nunique()
        base = alt.Chart(hm).encode(
            x=alt.X("yearmonthdate(day):O", title=None,
                    axis=alt.Axis(labels=n_days <= 45, labelAngle=0, format="%-d", labelOverlap=True)),
            y=alt.Y("row:N", title=None, sort=summ.row.tolist(), axis=alt.Axis(minExtent=64, labelLimit=220)))
        layers = [base.mark_rect(strokeWidth=1.5 if n_days <= 90 else 0).encode(
            fill=alt.Fill("status:N", scale=ui.POINT_STATUS_COLOR,
                          legend=alt.Legend(orient="top", title=None, symbolType="square",
                                            symbolStrokeWidth=1.5, columnPadding=24, labelLimit=160)),
            stroke=alt.Stroke("status:N", scale=ui.POINT_STATUS_STROKE, legend=None),
            tooltip=[alt.Tooltip("column:N", title="point"), alt.Tooltip("day:T"), "status",
                     alt.Tooltip("completeness:Q", format=".0f", title="complete % (day)"),
                     alt.Tooltip("received:Q", title="samples (of 1440)"),
                     alt.Tooltip("flat_measured:Q", title="longest flat run (min)"),
                     alt.Tooltip("spikes:Q"), alt.Tooltip("distinct:Q", title="distinct values"),
                     alt.Tooltip("verdict:N", title="point verdict (period)"),
                     alt.Tooltip("period_complete:Q", format=".1f", title="complete % (period)"),
                     alt.Tooltip("offline_days:Q", title="days offline (period)"),
                     alt.Tooltip("flat_day_share:Q", format=".0f", title="days frozen % (period)")])]
        if n_days <= 45:
            layers.append(base.mark_text(fontSize=10).encode(
                text="mark:N", color=alt.Color("status:N", legend=None, scale=alt.Scale(
                    domain=ui.POINT_STATUS, range=[ui.INK, ui.C["status/warning-text"],
                                                   ui.C["text/on-accent"], ui.C["text/on-accent"], ui.INK_2]))))
        st.altair_chart(ui.chart(alt.layer(*layers), height=22 * hm.row.nunique() + 40), width="stretch")
        st.caption("Rows run worst first — by the point's verdict over the period (shown after the name), "
                   "then completeness; hover a cell for that day and for the point over the whole period (the planned "
                   "Sep 2018 – Mar 2019 sensor gap is left out of the period figures). ▲ BAD SAMPLES, "
                   "● FROZEN and '—' OFFLINE are down; ◐ PARTIAL is degraded. Temperature and humidity are "
                   "judged frozen only while cooling runs on a workday, since a room holds steady with the "
                   "AC off. Glyphs show for periods up to 45 days."
                   + (" The block on z1/z2 in Aug–Sep 2019 is a real 45- and 69-day outage." if floor == 5 else ""))
    else:
        st.info("No point-days in this period.", icon=":material/info:")

# --------------------------------------------------------------------------- #
with tab_hum:
    hc = HL.humidity_consistency(hourly)
    if len(hc) >= 2:
        st.altair_chart(ui.chart(
            alt.Chart(hc).mark_bar(color=ui.PRIMARY).encode(
                y=alt.Y("zone:N", title=None),
                x=alt.X("abs_hum:Q", title="absolute humidity (g/m³)"),
                tooltip=["zone", alt.Tooltip("abs_hum:Q", format=".2f"),
                         alt.Tooltip("dew:Q", format=".2f", title="dew point °C"),
                         alt.Tooltip("temp:Q", format=".1f"), alt.Tooltip("rh:Q", format=".1f")]),
            height=40 * len(hc) + 30), width="stretch")
        st.dataframe(
            hc.rename(columns={"abs_hum": "abs humidity g/m³", "dew": "dew point °C",
                               "temp": "temp °C", "rh": "RH %", "abs_hum_dev": "vs median g/m³"})
              [["zone", "abs humidity g/m³", "vs median g/m³", "dew point °C", "temp °C", "RH %"]].round(2),
            width="stretch", hide_index=True)
        st.caption(
            "Whole record, workday cooling hours. Temperature legitimately differs between zones; moisture "
            "content largely does not, because they share an air path. A zone well below the floor median "
            "is reading air that has just crossed a cooling coil — a sensor in the wrong place, not a "
            "colder room."
            + (" Here z1 sits ~1.4 g/m³ below the median and its dew point ~2.2 °C below: a supply-air "
               "sensor." if floor == 5 else ""))
    else:
        st.info("This check needs at least two zones with temperature and humidity sensors.",
                icon=":material/info:")

# --------------------------------------------------------------------------- #
with tab_trace:
    st.caption("Minute trace — confirm a flagged point by eye. The line is the cleaned feed: out-of-range "
               "values and spikes are removed and gaps of up to 5 minutes filled, so a spike counted in the "
               "tiles below will not show as a needle. The tiles count the raw feed.")
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
            k[0].metric("Samples (raw)", f"{int(r.received)} / 1440", border=True)
            k[1].metric("Distinct values", int(r.distinct), border=True)
            k[2].metric("Longest flat run", f"{int(r.max_flat_nonzero)} min", border=True)
            k[3].metric("Out of range", int(r.out_of_range), border=True)
            k[4].metric("Spikes", int(r.spikes), border=True)
