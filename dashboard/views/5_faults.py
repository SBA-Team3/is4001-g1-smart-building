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

hd = st.columns([4, 1], vertical_alignment="bottom")
with hd[0]:
    ui.page_header(f"Operations · Floor {floor}", "Faults & work orders",
                   "What is broken, what it costs per day, who fixes it, and how we know it's fixed. "
                   "Lifecycle states are saved locally and survive a reload.")
export_slot = hd[1].empty()

cc = st.columns([2, 2, 5, 2], vertical_alignment="bottom")
# Keyed so Floor status's "All N open faults" link can open this page on its window.
st.session_state.setdefault(ui.FAULTS_PERIOD_KEY, "Last 30 days")
period = cc[0].selectbox("Period", ["Trailing window", "Last 30 days",
                                    "Last 90 days", "Whole record"], key=ui.FAULTS_PERIOD_KEY)
view = cc[1].selectbox("Show", ["Open", "All", "Resolved", "Muted"], index=0)
show_planned = cc[3].toggle("Planned items", value=False,
                            help="The 2018–19 sensor gap was a scheduled "
                                 "firmware update. Those items are suppressed by "
                                 "default so they do not crowd out real faults.")
days = {"Trailing window": win, "Last 30 days": 30, "Last 90 days": 90}.get(period)
if days:
    s, e = D.window(as_of, days)
else:
    s, e = None, as_of
    days = h[h.ts <= as_of + pd.Timedelta(days=1)].date.nunique()

if "muted" not in st.session_state:
    st.session_state["muted"] = set()

reg = F.build_register(h, H, start=s, end=e, floor=floor, tariff=tariff, as_of=as_of)
reg = ui.with_states(reg, show_planned)   # session mutes still work, on top of saved states

# --------------------------------------------------------------------------- #
if not len(reg):
    st.success("No faults raised in this period.", icon=":material/check_circle:")
    st.stop()

shown = {"Open": reg[reg.is_open], "All": reg,
         "Resolved": reg[reg.state == "Resolved"], "Muted": reg[reg.state == "Muted"]}[view]
trades = shown.trade.value_counts()
trade = cc[2].pills("Trade", ["All"] + trades.index.tolist(), default="All", required=True,
                    format_func=lambda t: (f"All · {len(shown)}" if t == "All"
                                         else f"{t[:1].upper()}{t[1:]} · {trades[t]}"),
                    key="faults_trade")
if trade and trade != "All":
    shown = shown[shown.trade == trade]

ex = W.export_frame(shown)
export_slot.download_button(
    "Export work orders (CSV)", ex.to_csv(index=False).encode("utf-8"),
    file_name=f"work_orders_F{floor}_{as_of.date()}.csv", mime="text/csv",
    icon=":material/download:", width="stretch",
    help="One row per fault in the current view, in CMMS import order: id, trade, "
         "priority, state, action, verification, cost of inaction.")

active = reg[reg.is_open]
k = st.columns(5)
k[0].metric("Open faults", len(active),
            delta=f"{int((active.priority == F.HIGH).sum())} high" if len(active) else None,
            delta_color="inverse", border=True)
k[1].metric("At stake", f"฿{active.cost_per_day_thb.sum():,.0f}/day", border=True,
            help="Sum of the cost of inaction across open faults: energy attributable to each "
                 "fault ÷ calendar days since it was first seen × tariff. Rules with no defensible energy "
                 "attribution count zero rather than a guess.")
k[2].metric("Average days open", f"{active.days_open.mean():.0f}" if len(active) else "—", border=True,
            help=f"From first seen in the period to {as_of.date()}.")
k[3].metric("With energy attribution", f"{active.energy_attributed.mean() * 100:.0f}%"
            if len(active) else "—", border=True)
k[4].metric("Alarms per day", f"{len(active) / max(days, 1):.2f}", border=True)
st.caption(f"{(s.date() if s is not None else 'Start of record')} – {e.date()}. Sorted by priority, "
           "then cost of inaction (energy attributable ÷ calendar days since first seen × tariff). "
           "Faults without a defensible energy figure show '—', not a guess. Recurrence = separate "
           f"episodes in the period (active days more than {R.conv('episode_gap_days'):.0f} days apart).")


def _save_state(fid: str) -> None:
    W.set_state(ui.FAULT_STATES, fid, st.session_state[f"state_{fid}"],
                st.session_state.get(f"note_{fid}", ""))
    st.session_state["muted"].discard(fid)


def _meta(r) -> str:
    bits = []
    if r.kwh_impact:
        bits.append(f"{r.kwh_impact:,.0f} kWh in period")
    else:
        bits.append("No energy attribution")
    if np.isfinite(r.days_open):
        bits.append(f"{r.days_open:.0f} days open")
    bits.append(f"{r.recurrence:.0f} episode{'s' if r.recurrence != 1 else ''}")
    return " · ".join(bits)


if not len(shown):
    st.info("No faults in this view.", icon=":material/info:")

for n, (_, r) in enumerate(shown.iterrows()):
    first = (f" · first seen {pd.Timestamp(r.first_seen).day} {pd.Timestamp(r.first_seen):%b %Y}"
             if pd.notna(r.first_seen) else "")
    with ui.card(f"fault_{n}"):
        a, b, cst = st.columns([5, 4, 2], gap="large")
        a.markdown(
            f'<div class="cb-stack" style="gap:8px">{ui.chip(r.priority)}'
            f'<div class="cb-h3">{ui.esc(r.title)}</div>'
            f'<div class="cb-cap cb-muted">{ui.esc(r.id)} · {ui.esc(r.trade)}{first}</div>'
            f'<div class="cb-sub">{ui.esc(r.detail)}</div>'
            + (f'<div class="cb-cap cb-muted">{ui.esc(r.evidence)}</div>' if r.evidence else "")
            + "</div>", unsafe_allow_html=True)
        b.markdown(
            f'<div class="cb-stack" style="gap:12px"><div class="cb-stack"><p class="cb-overline">Action</p>'
            f'<p class="cb-body">{ui.esc(r.action)}</p></div><div class="cb-stack">'
            f'<p class="cb-overline">Fixed when</p><p class="cb-body">{ui.esc(r.verification)}</p>'
            f'</div></div>', unsafe_allow_html=True)
        with cst:
            cost = f"฿{r.cost_per_day_thb:,.0f}/day" if r.cost_per_day_thb else "—"
            ui.md(f'<div class="cb-stack" style="gap:6px"><p class="cb-overline">Cost of inaction</p>'
                  f'<div class="cb-metric">{cost}</div><div class="cb-cap">{_meta(r)}</div></div>')
            st.selectbox("State", W.STATES, index=W.STATES.index(r.state), key=f"state_{r.id}",
                         label_visibility="collapsed", on_change=_save_state, args=(r.id,))
        with st.expander(f"Why? {len(r.diagnoses)} possible cause{'s' if len(r.diagnoses) != 1 else ''}"
                         " · notes"):
            for dg in r.diagnoses:
                st.markdown(f"- {dg}")
            st.caption(
                "ASHRAE G36 requires a fault report to carry its possible "
                "diagnoses, and notes that most of them are sensor errors — the "
                "standard treats a faulty sensor as the leading hypothesis for "
                "most anomalies.")
            u = st.columns([5, 1], vertical_alignment="bottom")
            u[0].text_input("Note", value=r.state_note, key=f"note_{r.id}",
                            placeholder="who, what, work-order number")
            if u[1].button("Save note", key=f"save_{r.id}"):
                _save_state(r.id)
                st.rerun()
            meta = []
            if pd.notna(r.last_seen):
                meta.append(f"last seen {pd.Timestamp(r.last_seen).date()}")
            if pd.notna(r.get("state_updated")):
                meta.append(f"status {r.state} since {pd.Timestamp(r.state_updated):%Y-%m-%d %H:%M}"
                            + (f" — {r.state_note}" if r.state_note else ""))
            if meta:
                st.caption(" · ".join(meta))

if st.session_state["muted"]:
    st.caption(f"{len(st.session_state['muted'])} fault(s) muted this session.")
    if st.button("Unmute all"):
        for fid in st.session_state["muted"]:
            W.set_state(ui.FAULT_STATES, fid, "New", "unmuted")
        st.session_state["muted"] = set()
        st.rerun()

# --------------------------------------------------------------------------- #
audit = F.alarm_audit(active, days)
with ui.card("audit"):
    ui.section("Alarm system self-audit",
               "This register is held to the same ISA-18.2 / EEMUA 191 benchmarks we would apply to "
               "a BMS. A dashboard that emits alarms without measuring its own alarm load is the "
               "problem alarm rationalisation exists to fix.")
    left, right = st.columns([2, 3], gap="large", vertical_alignment="center")
    high_share = audit["mix"].get(F.HIGH, 0)
    over = high_share > audit["target"].get(F.HIGH, 5)
    std_ok = audit["standing"] <= F.ALARM_KPI["max_standing"]
    cells = [(f"{audit['per_day']:.2f}", "alarms per day", ui.INK,
              "Steady-state best practice is under ~2 per operator per hour. This register is far "
              "below that — it aggregates episodes rather than emitting one alarm per event."),
             (f"{high_share:.0f}%", f"high-priority (target {audit['target'].get(F.HIGH, 5):.0f}%)",
              ui.C["status/critical-text"] if over else ui.INK,
              f"{audit['deviation'].get(F.HIGH, 0):+.0f} pp vs the ISA-18.2 mix."),
             (f"{audit['standing']}", f"standing high (limit {F.ALARM_KPI['max_standing']})",
              ui.INK if std_ok else ui.C["status/critical-text"], "")]
    left.markdown('<div class="cb-between">' + "".join(
        f'<div title="{ui.esc(t)}"><div class="cb-metric" style="color:{col}">{v}</div>'
        f'<div class="cb-cap cb-muted">{ui.esc(l)}</div></div>' for v, l, col, t in cells)
        + "</div>", unsafe_allow_html=True)
    mix = pd.DataFrame([
        {"priority": p, "kind": "This register", "pct": audit["mix"].get(p, 0.0)}
        for p in (F.HIGH, F.MED, F.LOW)] + [
        {"priority": p, "kind": "ISA-18.2 target", "pct": v}
        for p, v in audit["target"].items()])
    mix["order"] = mix.priority.map({F.HIGH: 0, F.MED: 1, F.LOW: 2})
    with right:
        st.altair_chart(ui.chart(alt.Chart(mix).mark_bar(stroke=ui.SURFACE, strokeWidth=2,
                                                         height=12).encode(
            y=alt.Y("kind:N", title=None, sort=["ISA-18.2 target", "This register"]),
            x=alt.X("pct:Q", title=None, scale=alt.Scale(domain=[0, 100]), stack="zero",
                    axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'")),
            color=alt.Color("priority:N", legend=None, scale=alt.Scale(
                domain=[F.HIGH, F.MED, F.LOW],
                range=[ui.BAD, ui.WARN, ui.C["status/neutral"]])),
            order=alt.Order("order:Q"),
            tooltip=["kind", "priority", alt.Tooltip("pct:Q", format=".0f", title="% of alarms")],
        ), height=90), width="stretch")
        ui.md(" ".join(ui.chip(p) for p in (F.HIGH, F.MED, F.LOW)))
    st.caption(
        f"**{audit['verdict']}** The high-priority share sits above the 5% convention, and that is "
        "a real characteristic of this floor rather than a tuning failure: overcooling is genuinely "
        "the dominant fault, it affects every sensored zone, and each zone is a separate work order. "
        "The audit earned its place during development — the first version of this register raised "
        "34 alarms at 76% high priority, because it emitted one alarm per channel for the known "
        "planned sensor outage. Grouping offline points by zone and episode, and tagging the planned "
        f"window, cut that to {len(active)}.")

# --------------------------------------------------------------------------- #
with st.expander("What this register cannot detect, and why"):
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
