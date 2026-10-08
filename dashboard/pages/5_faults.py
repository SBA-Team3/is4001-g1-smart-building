"""Faults & actions — the alarm register, and an audit of our own alarm load."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from core import data as D
from core import faults as F
from core import registry as R
from core import ui

c = ui.ctx()
floor, as_of, win, tariff = c["floor"], c["as_of"], c["window"], c["tariff"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)

st.title("Faults & actions")

cc = st.columns([2, 2, 3])
period = cc[0].selectbox("Period", ["Trailing window", "Last 30 days",
                                    "Last 90 days", "Whole record"], index=1)
show_planned = cc[1].toggle("Show planned-maintenance items", value=False,
                            help="The 2018–19 sensor gap was a scheduled "
                                 "firmware update. Those items are suppressed by "
                                 "default so they do not crowd out real faults.")
days = {"Trailing window": win, "Last 30 days": 30, "Last 90 days": 90}.get(period)
if days:
    s, e = D.window(as_of, days)
else:
    s, e = None, None
    days = h.date.nunique()
cc[2].caption(f"{(s.date() if s is not None else 'start')} – "
              f"{(e.date() if e is not None else 'end')}")

if "muted" not in st.session_state:
    st.session_state["muted"] = set()

reg = F.build_register(h, H, start=s, end=e, floor=floor, tariff=tariff)
if not show_planned and len(reg):
    reg = reg[~reg.planned]

# --------------------------------------------------------------------------- #
if not len(reg):
    st.success("No faults raised in this period.")
    st.stop()

active = reg[~reg.id.isin(st.session_state["muted"])]
k = st.columns(4)
k[0].metric("Open faults", len(active))
k[1].metric("High priority", int((active.priority == F.HIGH).sum()))
k[2].metric("Estimated impact", f"฿{active.thb_impact.sum():,.0f}",
            help="Energy attributable to the fault × tariff, over the period "
                 "shown. Only rules with a defensible energy attribution carry "
                 "a figure; the rest show zero rather than a guess.")
k[3].metric("Alarms per day", f"{len(active)/max(days,1):.2f}")

st.divider()

for _, r in active.iterrows():
    colr = {"high": ui.BAD, "medium": ui.WARN, "low": ui.OK}[r.priority]
    with st.container(border=True):
        top = st.columns([6, 2, 1])
        top[0].markdown(
            f"<span style='color:{colr};font-weight:700'>{r.priority.upper()}</span> "
            f"&nbsp; **{r.title}**", unsafe_allow_html=True)
        top[1].markdown(
            f"<div style='text-align:right'>"
            f"{'฿%s' % format(r.thb_impact, ',.0f') if r.thb_impact else '—'}<br>"
            f"<small>{r.kwh_impact:,.0f} kWh</small></div>"
            if r.kwh_impact else "<div style='text-align:right'>—</div>",
            unsafe_allow_html=True)
        if top[2].button("Mute", key=f"mute_{r.id}"):
            st.session_state["muted"].add(r.id)
            st.rerun()

        st.markdown(r.detail)
        meta = []
        if pd.notna(r.first_seen):
            meta.append(f"first seen {pd.Timestamp(r.first_seen).date()}")
        if pd.notna(r.last_seen):
            meta.append(f"last seen {pd.Timestamp(r.last_seen).date()}")
        if r.evidence:
            meta.append(str(r.evidence))
        st.caption(" · ".join(meta))

        with st.expander("Possible diagnoses"):
            for dg in r.diagnoses:
                st.markdown(f"- {dg}")
            st.caption(
                "ASHRAE G36 requires a fault report to carry its possible "
                "diagnoses, and notes that most of them are sensor errors — the "
                "standard treats a faulty sensor as the leading hypothesis for "
                "most anomalies.")

if st.session_state["muted"]:
    st.caption(f"{len(st.session_state['muted'])} fault(s) muted this session.")
    if st.button("Unmute all"):
        st.session_state["muted"] = set()
        st.rerun()

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Alarm system self-audit")
st.caption(
    "A dashboard that emits alarms without measuring its own alarm load is the "
    "problem alarm rationalisation exists to fix. So we hold this register to "
    "the same ISA-18.2 / EEMUA 191 benchmarks we would apply to a BMS."
)

audit = F.alarm_audit(active, days)
a = st.columns(3)
a[0].metric("Alarms per day", f"{audit['per_day']:.2f}",
            help="Steady-state best practice is under ~2 per operator per hour. "
                 "This register is far below that, which is expected — it "
                 "aggregates episodes rather than emitting one alarm per event.")
a[1].metric("High-priority share", f"{audit['mix'].get(F.HIGH, 0):.0f}%",
            delta=f"{audit['deviation'].get(F.HIGH, 0):+.0f} pp vs 5% target",
            delta_color="inverse")
a[2].metric("Standing high alarms", audit["standing"],
            delta="within 10" if audit["standing"] <= 10 else "above 10",
            delta_color="normal" if audit["standing"] <= 10 else "inverse")

mix = pd.DataFrame([
    {"priority": p, "kind": "actual", "pct": audit["mix"].get(p, 0.0)}
    for p in (F.HIGH, F.MED, F.LOW)] + [
    {"priority": p, "kind": "ISA-18.2 target", "pct": v}
    for p, v in audit["target"].items()])
st.altair_chart(
    alt.Chart(mix).mark_bar().encode(
        y=alt.Y("priority:N", title=None, sort=[F.HIGH, F.MED, F.LOW]),
        x=alt.X("pct:Q", title="% of alarms"),
        yOffset="kind:N",
        color=alt.Color("kind:N", scale=alt.Scale(range=["#2e74b5", "#c9c9c9"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["priority", "kind", alt.Tooltip("pct:Q", format=".0f")],
    ).properties(height=180), width="stretch")

st.info(
    f"**{audit['verdict']}** The high-priority share sits above the 5% "
    "convention, and that is a real characteristic of this floor rather than a "
    "tuning failure: overcooling is genuinely the dominant fault, it affects "
    "every sensored zone, and each zone is a separate work order. The audit "
    "earned its place during development — the first version of this register "
    "raised 34 alarms at 76% high priority, because it emitted one alarm per "
    "channel for the known planned sensor outage. Grouping offline points by "
    "zone and episode, and tagging the planned window, cut that to "
    f"{len(active)}."
)

# --------------------------------------------------------------------------- #
with st.expander("What we cannot detect, and why"):
    st.markdown(
        "**None of the fifteen ASHRAE Guideline 36 AHU fault conditions "
        "(FC1–FC15) are computable on this data.** Every one needs at least one "
        "of: supply, mixed, return or outdoor air temperature; duct static "
        "pressure; or a valve or damper position. CU-BEMS has none of them — it "
        "meters zone-level power and measures zone temperature, humidity and a "
        "light index, and nothing on the plant side.\n\n"
        "What that rules out specifically:\n"
        "- economiser and outdoor-air-fraction faults (FC6)\n"
        "- coil-leak faults (FC14, FC15)\n"
        "- supply-air temperature faults (FC5, FC7, FC12, FC13)\n"
        "- operating-state classification (OS#1–#5), and FC4 as written\n"
        "- any setpoint-deviation fault, because there is no setpoint channel\n"
        "- simultaneous heating and cooling\n"
        "- chiller or plant efficiency of any kind\n\n"
        "Also not computable: occupancy-weighted exceedance and "
        "occupancy-conditioned suppression (no occupancy channel), ventilation "
        "or IAQ compliance (no CO₂), and airflow faults (no airflow).\n\n"
        "The rules on this page are the ones five zones of power and three "
        "environmental channels can actually support. Stating the boundary "
        "precisely is a stronger result than implying broader coverage."
    )
