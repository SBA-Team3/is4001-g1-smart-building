"""Alarms — the fault register as a dense, filterable work queue with a CMMS
hand-off. Same register and period as the Floor page, so the counts agree."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import faults as F
from sbanalytics import registry as R
from sbanalytics import workorders as W
import ui

c = ui.ctx()
floor, as_of, tariff = c["floor"], c["as_of"], c["tariff"]
if not ui.require_wired(floor):
    st.stop()
st.session_state.setdefault("muted", set())

hd = st.columns([5, 1], vertical_alignment="bottom")
with hd[0]:
    ui.page_header("Alarms", f"Floor {floor} alarms",
                   f"{c['period_label']} · what is broken, what it costs per day, who fixes it, "
                   "and how we know it's fixed")
export_slot = hd[1].empty()

reg = ui.register(floor, c, st.session_state.get("alarms_planned", False))
ui.floor_band(floor, as_of, reg)
f1, f2, f3, f4 = st.columns([2.2, 4, 2.6, 1.4], vertical_alignment="bottom")
f4.toggle("Planned", value=False, key="alarms_planned",
          help="The 2018–19 sensor gap was a scheduled firmware update. Those items "
               "are hidden by default so they do not crowd out real faults.")

sev = f1.pills("Severity", [F.HIGH, F.MED, F.LOW], selection_mode="multi", key="alarms_sev",
               format_func=ui.sev_glyph)
trades = sorted(reg.trade.unique()) if len(reg) else []
trade = f2.pills("Trade", trades, selection_mode="multi", key="alarms_trade",
                 format_func=lambda t: f"{t[:1].upper()}{t[1:]}")
view = f3.segmented_control("State", ["Open", "All", "Resolved", "Muted"], default="Open",
                            required=True, key="alarms_state")

if not len(reg):
    st.info("No alarms raised on this floor in this period.", icon=":material/check_circle:")
    st.stop()

active = reg[reg.is_open]
k = st.columns(3)
k[0].metric("Open faults", len(active),
            delta=f"{int((active.priority == F.HIGH).sum())} high" if len(active) else None,
            delta_color="inverse", delta_arrow="off", border=True)
k[1].metric("At stake", f"฿{active.cost_per_day_thb.sum():,.0f}/day" if tariff else "set rate in Assumptions",
            border=True,
            help="Sum of the cost of inaction across open faults: attributable energy ÷ calendar days "
                 "since first seen × tariff. Rules with no defensible attribution count zero.")
k[2].metric("Average days open", f"{active.days_open.mean():.0f}" if len(active) else "—", border=True,
            help=f"From first seen to {as_of.date()}.")

shown = {"Open": reg[reg.is_open], "All": reg, "Resolved": reg[reg.state == "Resolved"],
         "Muted": reg[reg.state == "Muted"]}[view]
if sev:
    shown = shown[shown.priority.isin(sev)]
if trade:
    shown = shown[shown.trade.isin(trade)]
shown = ui.sort_faults(shown).reset_index(drop=True)

ex = W.export_frame(shown)
export_slot.download_button(
    "Export CSV", ex.to_csv(index=False).encode("utf-8"), type="secondary",
    file_name=f"work_orders_F{floor}_{as_of.date()}.csv", mime="text/csv",
    icon=":material/download:", width="stretch",
    help="One row per alarm in the current view, in CMMS import order.")

now = pd.Timestamp.now()
in_state = [(now - pd.Timestamp(u)).days if pd.notna(u) else d
            for u, d in zip(shown.get("state_updated", pd.Series([pd.NaT] * len(shown))), shown.days_open)]
aging = ["⚠ aging" if (s_ == "New" and d > 3) or (o and d > 14) else ""
         for s_, d, o in zip(shown.state, shown.days_open, shown.is_open)]
tbl = pd.DataFrame({
    "Sev": [ui.sev_glyph(p) for p in shown.priority],
    "ID": shown.id,
    "Zone": [f"F{f} {z}" for f, z in zip(shown.floor, shown.zone)],
    "Title": shown.title,
    "฿/day": np.where((shown.cost_per_day_thb > 0) & bool(tariff), shown.cost_per_day_thb, np.nan),
    "Days open": shown.days_open,
    "In state": in_state,
    "State": shown.state,
    "Trade": shown.trade,
    "Aging": aging,
})


@st.dialog("Alarm", width="large")
def _detail(fid: str) -> None:
    r = reg[reg.id == fid].iloc[0]
    first = pd.Timestamp(r.first_seen)
    ui.md(f'<div class="cb-row">{ui.state_chip(r)}<span class="cb-cap cb-muted">{ui.esc(r.id)} · '
          f'{ui.esc(r.trade)} · first seen {first.day} {first:%b %Y} · {ui.esc(r.state)}</span></div>')
    st.subheader(r.title)
    st.markdown(r.detail)
    if r.evidence:
        st.caption(r.evidence)
    a, b = st.columns(2)
    a.markdown(f"**Action**  \n{r.action}")
    b.markdown(f"**Fixed when**  \n{r.verification}")
    cost = f"฿{r.cost_per_day_thb:,.0f}/day" if r.cost_per_day_thb and tariff else "—"
    st.caption(f"Cost of inaction {cost} · "
               + (f"{r.kwh_impact:,.0f} kWh in period" if r.kwh_impact else "no energy attribution")
               + f" · {r.recurrence:.0f} episode(s)")
    with st.expander(f"Possible causes ({len(r.diagnoses)})"):
        for dg in r.diagnoses:
            st.markdown(f"- {dg}")
        st.caption("ASHRAE G36 asks a fault report to carry its possible diagnoses; most are sensor "
                   "errors, so a faulty sensor is the leading hypothesis.")
    note = st.text_input("Note", value=r.state_note, placeholder="who, what, work-order number",
                         key=f"note_{fid}")
    bb = st.columns(4)
    for col, (label, state, kind) in zip(bb, [("Acknowledge", "Acknowledged", "secondary"),
                                              ("In progress", "In progress", "primary"),
                                              ("Resolve", "Resolved", "secondary"),
                                              ("Mute", "Muted", "secondary")]):
        if col.button(label, key=f"dlg_{state}_{fid}", type=kind, width="stretch",
                      disabled=r.state == state):
            if state == "Muted":
                ui.mute_fault(fid)
                W.set_state(ui.FAULT_STATES, fid, "Muted", note)
            else:
                ui.set_fault_state(fid, state, note)
            st.rerun()
    if pd.notna(r.get("state_updated")):
        st.caption(f"{r.state} since {pd.Timestamp(r.state_updated):%Y-%m-%d %H:%M}"
                   + (f" — {r.state_note}" if r.state_note else ""))


def _pick() -> None:
    sel = st.session_state["alarms_tbl"].selection.rows
    if sel:
        st.session_state["_alarm_dialog"] = tbl.ID.iloc[sel[0]]


if not len(tbl):
    st.info("No alarms in this view.", icon=":material/info:")
else:
    st.dataframe(
        ui.glyph_style(tbl, ["Sev", "Aging"], {"฿/day": "{:,.0f}", "Days open": "{:.0f}",
                                               "In state": "{:.0f}"}),
        hide_index=True, width="stretch", key="alarms_tbl", on_select=_pick,
        selection_mode="single-row", placeholder=ui.DASH, height=min(35 * len(tbl) + 38, 560),
        column_config={
            "Title": st.column_config.Column(width="large"),
            "ID": st.column_config.Column(width="small"),
            "฿/day": st.column_config.Column(help="Cost of inaction. '—' where no energy is attributable."),
            "In state": st.column_config.Column(help="Days in the current state."),
            "Aging": st.column_config.Column(
                help="New for more than 3 days, or open for more than 14."),
        })
    st.caption("Click a row for the action, how we know it's fixed, possible causes, a note and the "
               f"state buttons. Recurrence = separate episodes (active days more than "
               f"{R.conv('episode_gap_days'):.0f} days apart).")
_da = st.session_state.pop("_alarm_dialog", None)
if _da is not None and (reg.id == _da).any():
    _detail(_da)

if st.session_state["muted"]:
    st.caption(f"{len(st.session_state['muted'])} alarm(s) muted this session.")
    if st.button("Unmute all"):
        for fid in st.session_state["muted"]:
            W.set_state(ui.FAULT_STATES, fid, "New", "unmuted")
        st.session_state["muted"] = set()
        st.rerun()
