"""Faults & work orders — the FDD register as a CMMS hand-off, and an audit of our own alarm load."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import faults as F
from sbanalytics import registry as R
from sbanalytics import workorders as W
import ui

c = ui.ctx()
floor, as_of, win, tariff = c["floor"], c["as_of"], c["window"], c["tariff"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)

st.title("Faults & work orders")
st.caption("Each fault carries its cost of inaction, the trade that owns it, the action and the "
           "check that proves it is fixed — the fields a CMMS work order needs. Lifecycle states "
           "are saved locally and survive a reload.")

cc = st.columns([2, 2, 2, 3])
# Keyed so Floor status's "All N open faults" link can open this page on its window.
st.session_state.setdefault(ui.FAULTS_PERIOD_KEY, "Last 30 days")
period = cc[0].selectbox("Period", ["Trailing window", "Last 30 days",
                                    "Last 90 days", "Whole record"], key=ui.FAULTS_PERIOD_KEY)
view = cc[1].selectbox("Show", ["Open", "All", "Resolved", "Muted"], index=0)
show_planned = cc[2].toggle("Planned-maintenance items", value=False,
                            help="The 2018–19 sensor gap was a scheduled "
                                 "firmware update. Those items are suppressed by "
                                 "default so they do not crowd out real faults.")
days = {"Trailing window": win, "Last 30 days": 30, "Last 90 days": 90}.get(period)
if days:
    s, e = D.window(as_of, days)
else:
    s, e = None, as_of
    days = h[h.ts <= as_of + pd.Timedelta(days=1)].date.nunique()
cc[3].caption(f"{(s.date() if s is not None else 'start of record')} – {e.date()}")

if "muted" not in st.session_state:
    st.session_state["muted"] = set()

reg = F.build_register(h, H, start=s, end=e, floor=floor, tariff=tariff, as_of=as_of)
reg = ui.with_states(reg, show_planned)   # session mutes still work, on top of saved states

# --------------------------------------------------------------------------- #
if not len(reg):
    st.success("No faults raised in this period.")
    st.stop()

active = reg[reg.is_open]
k = st.columns(5)
k[0].metric("Open faults", len(active),
            delta=f"{int((active.priority == F.HIGH).sum())} high" if len(active) else None,
            delta_color="inverse")
k[1].metric("At stake", f"฿{active.cost_per_day_thb.sum():,.0f}/day",
            help="Sum of the cost of inaction across open faults: energy attributable to each "
                 "fault ÷ calendar days since it was first seen × tariff. Rules with no defensible energy "
                 "attribution count zero rather than a guess.")
k[2].metric("Average days open", f"{active.days_open.mean():.0f}" if len(active) else "—",
            help=f"From first seen in the period to {as_of.date()}.")
k[3].metric("With energy attribution", f"{active.energy_attributed.mean() * 100:.0f}%"
            if len(active) else "—")
k[4].metric("Alarms per day", f"{len(active) / max(days, 1):.2f}")

shown = {"Open": reg[reg.is_open], "All": reg,
         "Resolved": reg[reg.state == "Resolved"], "Muted": reg[reg.state == "Muted"]}[view]

ex = W.export_frame(shown)
st.download_button("Export work orders (CSV)", ex.to_csv(index=False).encode("utf-8"),
                   file_name=f"work_orders_F{floor}_{as_of.date()}.csv", mime="text/csv",
                   icon=":material/download:",
                   help="One row per fault in the current view, in CMMS import order: id, trade, "
                        "priority, state, action, verification, cost of inaction.")

st.dataframe(
    shown[["id", "priority", "state", "trade", "title", "cost_per_day_thb", "days_open",
           "recurrence", "kwh_impact"]].rename(columns={
               "cost_per_day_thb": "฿/day", "days_open": "days open", "kwh_impact": "kWh"}),
    width="stretch", hide_index=True,
    column_config={"฿/day": st.column_config.NumberColumn(format="%.0f"),
                   "kWh": st.column_config.NumberColumn(format="%.0f"),
                   "title": st.column_config.TextColumn(width="large")})
st.caption("Sorted by priority, then cost of inaction. Recurrence = separate episodes in the "
           f"period (active days more than {R.conv('episode_gap_days'):.0f} days apart).")

st.divider()

for _, r in shown.iterrows():
    colr = {"high": ui.BAD, "medium": ui.WARN, "low": ui.OK}[r.priority]
    with st.container(border=True):
        top = st.columns([6, 2, 2])
        top[0].markdown(
            f"<span style='color:{colr};font-weight:700'>{r.priority.upper()}</span> "
            f"&nbsp; **{r.title}**  \n<small>`{r.id}` · {r.trade} · **{r.state}**</small>",
            unsafe_allow_html=True)
        top[1].markdown(
            f"<div style='text-align:right'>"
            + (f"<b>฿{r.cost_per_day_thb:,.0f}/day</b><br><small>{r.kwh_impact:,.0f} kWh in period"
               f"</small>" if r.kwh_impact else "—<br><small>no energy attribution</small>")
            + "</div>", unsafe_allow_html=True)
        top[2].markdown(
            f"<div style='text-align:right'>{r.days_open:.0f} days open<br>"
            f"<small>{r.recurrence:.0f} episode(s)</small></div>"
            if np.isfinite(r.days_open) else "", unsafe_allow_html=True)

        st.markdown(r.detail)
        meta = []
        if pd.notna(r.first_seen):
            meta.append(f"first seen {pd.Timestamp(r.first_seen).date()}")
        if pd.notna(r.last_seen):
            meta.append(f"last seen {pd.Timestamp(r.last_seen).date()}")
        if r.evidence:
            meta.append(str(r.evidence))
        st.caption(" · ".join(meta))

        a, b = st.columns(2)
        a.markdown(f"**Action** · {r.action}")
        b.markdown(f"**Verify** · {r.verification}")

        with st.expander("Possible diagnoses · update status"):
            for dg in r.diagnoses:
                st.markdown(f"- {dg}")
            st.caption(
                "ASHRAE G36 requires a fault report to carry its possible "
                "diagnoses, and notes that most of them are sensor errors — the "
                "standard treats a faulty sensor as the leading hypothesis for "
                "most anomalies.")
            u = st.columns([2, 4, 1, 1])
            new_state = u[0].selectbox("State", W.STATES, index=W.STATES.index(r.state),
                                       key=f"state_{r.id}")
            note = u[1].text_input("Note", value=r.state_note, key=f"note_{r.id}",
                                   placeholder="who, what, work-order number")
            if u[2].button("Save", key=f"save_{r.id}"):
                W.set_state(ui.FAULT_STATES, r.id, new_state, note)
                st.session_state["muted"].discard(r.id)
                st.rerun()
            if u[3].button("Mute", key=f"mute_{r.id}"):
                st.session_state["muted"].add(r.id)
                W.set_state(ui.FAULT_STATES, r.id, "Muted", note or "muted from the register")
                st.rerun()
        if pd.notna(r.get("state_updated")):
            st.caption(f"Status {r.state} since {pd.Timestamp(r.state_updated):%Y-%m-%d %H:%M}"
                       + (f" — {r.state_note}" if r.state_note else ""))

if st.session_state["muted"]:
    st.caption(f"{len(st.session_state['muted'])} fault(s) muted this session.")
    if st.button("Unmute all"):
        for fid in st.session_state["muted"]:
            W.set_state(ui.FAULT_STATES, fid, "New", "unmuted")
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
            delta=(f"within {F.ALARM_KPI['max_standing']}"
                   if audit["standing"] <= F.ALARM_KPI["max_standing"]
                   else f"above {F.ALARM_KPI['max_standing']}"),
            delta_color="normal" if audit["standing"] <= F.ALARM_KPI["max_standing"] else "inverse")

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
