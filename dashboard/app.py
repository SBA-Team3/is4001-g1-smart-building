"""
CU-BEMS building — facilities operations dashboard.

    streamlit run dashboard/app.py

Build the shared cleaned dataset first (from the repo root):

    python -m sbanalytics.pipeline

Audience is the university FM team first, then finance and leadership. The
navigation is grouped by who asks the question: the overview for whoever opens
it first, operations for the technicians and supervisors, energy management for
the energy manager and finance, and the reference page for anyone who needs to
check a number. Within each group the order is the order the questions get
asked: can I trust the data, what is broken, how are we performing, and only
then what could we change.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))   # works without `pip install -e .`

from sbanalytics import data as D          # noqa: E402
from sbanalytics import registry as R      # noqa: E402

import ui                                  # noqa: E402

st.set_page_config(page_title="CU-BEMS — FM operations", page_icon=":material/apartment:",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
_DESIGN = os.path.join(_HERE, "design")
st.logo(os.path.join(_DESIGN, "logo.svg"), size="large",
        icon_image=os.path.join(_DESIGN, "logo_mark.svg"))

# --------------------------------------------------------------------------- #
# context sidebar — set once, read by every page through ui.ctx()
# --------------------------------------------------------------------------- #

if not D.cache_ready(5):
    st.sidebar.error("Cleaned data not built.")
    st.title("Cleaned data not built")
    st.markdown(
        "The dashboard reads the shared cleaned dataset in data/cleaned/ rather "
        "than the raw CSVs, because minute-level fault detection over 790,560 rows "
        "cannot run on a page load.\n\n```bash\npython -m sbanalytics.pipeline\n```"
    )
    st.stop()

sb = st.sidebar
sb.markdown('<p class="cb-overline">Context</p>', unsafe_allow_html=True)

floor_labels = {
    f: f"Floor {f}" + ("" if meta["wired"] else "  · not wired")
    for f, meta in sorted(R.FLOORS.items())
}
# Keyed so a Portfolio floor card can open Floor status on its own floor.
st.session_state.setdefault("floor_select", 5)
floor = sb.selectbox(
    "Floor", list(floor_labels), format_func=floor_labels.get, key="floor_select",
    help="The operational pages show wired floors only. The Portfolio overview "
         "reads every floor; the analysis/ notebooks cover all seven.")
st.session_state["floor"] = floor

as_of = sb.date_input(
    "Data as of", value=D.DEFAULT_AS_OF.date(),
    min_value=D.RECORD_START.date(), max_value=D.RECORD_END.date(), format="DD/MM/YYYY",
    help="The record runs 1 Jul 2018 – 31 Dec 2019, so 'today' is a date you "
         "choose. Every page shows that day's status plus trailing windows.")
st.session_state["as_of"] = pd.Timestamp(as_of)

st.session_state["window_days"] = sb.segmented_control(
    "Trailing window (days)", [1, 7, 14, 30, 90], default=7, required=True,
    key="window_select", width="stretch")

with sb.expander("Assumptions", icon=":material/tune:", expanded=False):
    st.session_state["tariff"] = st.number_input(
        "Tariff (฿/kWh)", 0.0, 20.0, float(R.conv("tariff_thb")), 0.10)
    st.session_state["area"] = st.number_input(
        "Floor area (m²)", 100.0, 5000.0, float(R.conv("floor_area_m2")), 10.0,
        help="11,700 m² total ÷ 7 floors. The paper gives no per-floor split.")
    st.session_state["ef"] = st.number_input(
        "Grid emission factor (kgCO₂e/kWh)", 0.0, 1.5, float(R.conv("co2_kg_per_kwh")), 0.01,
        format="%.4f", help=R.CONVENTIONS["co2_kg_per_kwh"][2])
    st.session_state["demand_charge"] = st.number_input(
        "Demand charge (฿/kW/month)", 0.0, 1000.0, float(R.conv("demand_charge_thb_per_kw")), 10.0,
        help=R.CONVENTIONS["demand_charge_thb_per_kw"][2])
    st.caption("Full list with sources on the Methods page.")

# Data status for the chosen floor and date, always visible.
sb.divider()
rows = []
if R.FLOORS[floor]["wired"]:
    _H = ui.health(floor)
    if (_H.day == pd.Timestamp(as_of).normalize()).any():
        phi = ui.HL.point_health_index(_H, as_of, floor)
        up = phi["points"] - phi["offline"]
        rows.append(ui.dot("good" if not phi["offline"] else "warning",
                           f"{up} / {phi['points']} points reporting"))
        rows.append(ui.dot("good" if phi["completeness"] >= R.conv("completeness_target") else "warning",
                           f"Points {phi['completeness']:.0f}% complete today"))
    else:
        rows.append(ui.dot("critical", "No data for this date"))
    for n in D.data_status(as_of, floor):
        rows.append(ui.dot("critical" if n["level"] == "error" else "warning", n["title"]))
else:
    rows.append(ui.dot("offline", "Floor not wired — Portfolio only"))
sb.markdown('<p class="cb-overline" style="margin-bottom:8px">Data status</p>'
            + "".join(f"<div style='margin-bottom:6px'>{r}</div>" for r in rows),
            unsafe_allow_html=True)

sb.divider()
sb.caption("CU-BEMS · 7-storey office, Bangkok  \n1-minute data, Jul 2018 – Dec 2019")

# --------------------------------------------------------------------------- #
# navigation
# --------------------------------------------------------------------------- #

# Grouped by persona -> question. Overview: "where do I send the team?" (FM
# lead, leadership). Operations: "what is broken and can I trust it?"
# (supervisors, technicians). Energy management: "are we on target and what
# would a change save?" (energy manager, finance). Reference: "where does this
# number come from?"
P = "views"
nav = st.navigation(position="top", pages={
    "Overview": [
        st.Page(f"{P}/0_portfolio.py", title="Portfolio overview", icon=":material/apartment:",
                default=True),
        st.Page(f"{P}/1_floor_status.py", title="Floor status", icon=":material/dashboard:"),
    ],
    "Operations": [
        st.Page(f"{P}/5_faults.py", title="Faults & work orders", icon=":material/build:"),
        st.Page(f"{P}/3_zones.py", title="Zones & comfort", icon=":material/grid_view:"),
        st.Page(f"{P}/2_sensor_health.py", title="Sensor & data health", icon=":material/sensors:"),
    ],
    "Energy management": [
        st.Page(f"{P}/4_energy.py", title="Energy performance", icon=":material/bolt:"),
        st.Page(f"{P}/6_opportunities.py", title="Opportunities & simulator", icon=":material/savings:"),
    ],
    "Reference": [
        st.Page(f"{P}/8_methods.py", title="Methods & assumptions", icon=":material/science:"),
    ],
})
nav.run()
