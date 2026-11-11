"""Alerts — the fault register as a dense work queue, labelled and filtered by
fault type (never severity), with a CMMS hand-off. Same register and period as
the Floor page, so the counts agree. With Bldg selected (the default on entry) it
is every wired floor's register, the same one the Overview counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import registry as R
from sbanalytics import workorders as W
import ui

c = ui.ctx()
floor, as_of, tariff = c["floor"], c["as_of"], c["tariff"]
bldg = st.session_state.get("building_view", False)
WIRED = R.wired_floors()
scope = "Bldg" if bldg else f"F{floor}"
if not bldg and not ui.require_wired(floor):
    st.stop()
st.session_state.setdefault("muted", set())

hd = st.columns([5, 1], vertical_alignment="bottom")
with hd[0]:
    ui.page_header("Alerts", "Building alerts" if bldg else f"Floor {floor} alerts", c["range"])
export_slot = hd[1].empty()
# One floor's file end does not stop the building queue, and it carries no note there.
if not bldg:
    ui.file_end_gate(floor, c["start"], c["end"])

# Session keys keep their historical "alarms_" names; only the UI says "alerts".
planned = st.session_state.get("alarms_planned", False)
reg = ui.building_register(c, planned) if bldg else ui.register(floor, c, planned)
EMPTY_BLDG = f"No open alerts on F{WIRED[0]}–F{WIRED[-1]} in this period."
f1, f2, f3, f4 = st.columns([4.2, 2.6, 2.6, 1.2], vertical_alignment="bottom")
STATE_VIEWS = ["Open", "Cleared", "Muted", "All"]
if st.session_state.get("alarms_state") not in STATE_VIEWS:      # e.g. an old "Resolved"
    st.session_state.pop("alarms_state", None)
view = f3.segmented_control("State", STATE_VIEWS, default="Open", required=True, key="alarms_state",
                            help="Alerts clear automatically once the condition has not been seen for "
                                 f"more than {R.conv('episode_gap_days'):.0f} days.")
f4.toggle("Planned", value=False, key="alarms_planned",
          help="The 2018–19 sensor gap was a scheduled firmware update. Those items "
               "are hidden by default so they do not crowd out real faults.")

if not len(reg):
    st.info(EMPTY_BLDG if bldg else "No alerts raised on this floor in this period.",
            icon=":material/check_circle:")
    st.stop()

shown = {"Open": reg[reg.is_open], "Cleared": reg[reg.cleared & (reg.state != "Muted")],
         "Muted": reg[reg.state == "Muted"], "All": reg}[view or "Open"]
type_counts = shown.rule.value_counts()
types = f1.pills("Fault type", list(type_counts.index), selection_mode="multi", key="alerts_type",
                 format_func=lambda r: f"{ui.type_text(r)} · {type_counts.get(r, 0)}")
trades = sorted(shown.trade.unique())
trade = f2.pills("Trade", trades, selection_mode="multi", key="alarms_trade",
                 format_func=lambda t: f"{t[:1].upper()}{t[1:]}")

active = reg[reg.is_open]
k = st.columns(3)
t_label, t_n = ui.top_type(active)
k[0].metric("Open alerts", len(active),
            delta=f"{t_n} {t_label.lower()}" if t_n else None,
            delta_color="off", delta_arrow="off", border=True,
            help="Open = not cleared and not muted. The sub-line is the most common type.")
k[1].metric("At stake", f"฿{active.cost_per_day_thb.sum():,.0f}/day" if tariff else "set rate in Assumptions",
            border=True,
            help="Sum of the cost of inaction across open alerts: attributable energy ÷ calendar days "
                 "since first seen × tariff. Rules with no defensible attribution count zero.")
k[2].metric("Average days open", f"{active.days_open.mean():.0f}" if len(active) else "—", border=True,
            help=f"Open alerts only, from first seen to {as_of.date()}.")

if types:
    shown = shown[shown.rule.isin(types)]
if trade:
    shown = shown[shown.trade.isin(trade)]
shown = ui.sort_faults(shown).reset_index(drop=True)

# CMMS export: the fault type, not a severity; no "fixed when".
ex = W.export_frame(shown)
ex = pd.DataFrame({
    "id": ex["id"], "type": [ui.type_label(r) for r in shown.rule], "floor": ex["floor"],
    "zone": ex["zone"], "description": ex["title"], "trade": ex["trade"],
    "state": shown.state_label.values, "cost_per_day_thb": ex["cost_per_day_thb"],
    "first_seen": ex["first_seen"], "last_seen": ex["last_seen"], "options": ex["action"],
    "note": ex["state_note"] if "state_note" in ex else ""})
export_slot.download_button(
    "Export CSV", ex.to_csv(index=False).encode("utf-8"), type="secondary",
    file_name=f"work_orders_{scope}_{as_of.date()}.csv", mime="text/csv",
    icon=":material/download:", width="stretch",
    help="One row per alert in the current view, in CMMS import order.")

tbl = pd.DataFrame({
    "Type": [ui.type_text(r) for r in shown.rule],
    "ID": shown.id,
    "Zone": [f"F{f} {z}" for f, z in zip(shown.floor, shown.zone)],
    "Description": shown.title,
    "Cost / day": np.where((shown.cost_per_day_thb > 0) & bool(tariff), shown.cost_per_day_thb, np.nan),
    "State": shown.state_label,
})


def _day(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.day} {t:%b %Y}" if pd.notna(t) else "—"


@st.dialog("Alert", width="large")
def _detail(fid: str) -> None:
    r = reg[reg.id == fid].iloc[0]
    ui.md(f'<div class="cb-row">{ui.state_chip(r)}</div>')
    st.subheader(r.title)
    st.markdown(r.detail)
    if r.evidence:
        st.caption(r.evidence)
    cost = f"฿{r.cost_per_day_thb:,.0f}/day" if r.cost_per_day_thb and tariff else "—"
    in_period = (f"฿{r.kwh_impact * tariff:,.0f} in period" if r.kwh_impact and tariff
                 else f"{r.kwh_impact:,.0f} kWh in period" if r.kwh_impact else "no energy attribution")
    st.caption(f"**Cost / day** {cost} · {in_period} · first seen {_day(r.first_seen)} · "
               f"last seen {_day(r.last_seen)} · {r.trade} · {r.id}")
    if r.cleared:
        st.info(f"Cleared automatically — not seen since {_day(r.last_seen)}; it reappears if the "
                "condition returns.", icon=":material/check:")
    with st.expander(f"Possible causes ({len(r.diagnoses)})"):
        for dg in r.diagnoses:
            st.markdown(f"- {dg}")
    ui.md('<p class="cb-overline" style="margin-top:8px">Options to consider</p>')
    st.markdown(r.action)
    note = st.text_input("Note", value=r.state_note, placeholder="who, what, work-order number",
                         key=f"note_{fid}")
    bb = st.columns(3)
    for col, (label, state, kind) in zip(bb, [("Acknowledge", "Acknowledged", "secondary"),
                                              ("Work order", "In progress", "primary"),
                                              ("Mute", "Muted", "secondary")]):
        if col.button(label, key=f"dlg_{state}_{fid}", type=kind, width="stretch",
                      disabled=r.state == state):
            if state == "Muted":
                ui.mute_fault(fid)
                W.set_state(ui.FAULT_STATES, fid, "Muted", note)
            else:
                ui.set_fault_state(fid, state, note or ("work order raised" if state == "In progress" else ""))
            st.rerun()
    if pd.notna(r.get("state_updated")) and r.state != "New":
        st.caption(f"{r.state} since {pd.Timestamp(r.state_updated):%Y-%m-%d %H:%M}"
                   + (f" — {r.state_note}" if r.state_note else ""))


# Keyed per view so a selection never carries over to a different set of rows.
TBL_KEY = f"alarms_tbl_{view}_{scope}_{as_of:%Y%m%d}_{c['period']}"


def _pick() -> None:
    sel = st.session_state[TBL_KEY].selection.rows
    if sel:
        st.session_state["_alarm_dialog"] = tbl.ID.iloc[sel[0]]


if not len(tbl):
    st.info(EMPTY_BLDG if bldg and view == "Open" and not (types or trade) else "No alerts in this view.",
            icon=":material/info:")
else:
    st.dataframe(
        tbl.style.format({"Cost / day": "฿{:,.0f}"}, na_rep=ui.DASH),
        hide_index=True, width="stretch", key=TBL_KEY, on_select=_pick,
        selection_mode="single-row", placeholder=ui.DASH, height="auto" if len(tbl) <= 15 else 560,
        # ID stays in tbl so a row click can open its dialog; it is shown there, not here.
        column_order=["Type", "Zone", "Description", "Cost / day", "State"],
        column_config={
            "Description": st.column_config.Column(width="large"),
            "Cost / day": st.column_config.Column(help="Cost of inaction. '—' where no energy is attributable."),
            "State": st.column_config.Column(
                help="New, Acknowledged, In progress or Muted are set by people. Cleared is set by the "
                     f"system once the condition has not been seen for more than "
                     f"{R.conv('episode_gap_days'):.0f} days."),
        })
    st.caption("Select a row for the details, possible causes, options and the state buttons.")
_da = st.session_state.pop("_alarm_dialog", None)
if _da is not None and (reg.id == _da).any():
    _detail(_da)

if st.session_state["muted"]:
    st.caption(f"{len(st.session_state['muted'])} alert(s) muted this session.")
    if st.button("Unmute all"):
        for fid in st.session_state["muted"]:
            W.set_state(ui.FAULT_STATES, fid, "New", "unmuted")
        st.session_state["muted"] = set()
        st.rerun()
