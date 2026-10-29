"""
Floor 5 — facilities operations dashboard.

    streamlit run dashboard/app.py

Build the shared cleaned dataset first (from the repo root):

    python -m sbanalytics.pipeline

Audience is a facilities manager, not an analyst. The page order follows the
order the questions actually get asked: can I trust the data, what is broken,
how is the floor performing, and only then what could we change.
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

st.set_page_config(page_title="Floor 5 — FM operations", page_icon="🏢",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# global sidebar — set once, read by every page through ui.ctx()
# --------------------------------------------------------------------------- #

st.sidebar.title("Floor 5 operations")

if not D.cache_ready(5):
    st.sidebar.error("Cleaned data not built.")
    st.title("Cleaned data not built")
    st.markdown(
        "The dashboard reads the shared cleaned dataset in data/cleaned/ rather "
        "than the raw CSVs, because minute-level fault detection over 790,560 rows "
        "cannot run on a page load.\n\n```bash\npython -m sbanalytics.pipeline\n```"
    )
    st.stop()

floor_labels = {
    f: f"Floor {f}" + ("" if meta["wired"] else "  · not wired")
    for f, meta in sorted(R.FLOORS.items())
}
floor = st.sidebar.selectbox(
    "Floor", list(floor_labels), format_func=floor_labels.get,
    index=list(floor_labels).index(5),
    help="Only Floor 5 is wired into these pages. Every floor is in the registry "
         "and the cleaned dataset; the analysis/ notebooks cover all seven.")
st.session_state["floor"] = floor

as_of = st.sidebar.date_input(
    "As of", value=D.DEFAULT_AS_OF.date(),
    min_value=D.RECORD_START.date(), max_value=D.RECORD_END.date(),
    help="The record runs 1 Jul 2018 – 31 Dec 2019, so 'today' is a date you "
         "choose. Every page shows that day's status plus trailing windows.")
st.session_state["as_of"] = pd.Timestamp(as_of)

st.session_state["window_days"] = st.sidebar.select_slider(
    "Trailing window", [1, 7, 14, 30, 90], value=7)

with st.sidebar.expander("Assumptions", expanded=False):
    st.session_state["tariff"] = st.number_input(
        "Tariff (฿/kWh)", 0.0, 20.0, float(R.conv("tariff_thb")), 0.10)
    st.session_state["area"] = st.number_input(
        "Floor area (m²)", 100.0, 5000.0, float(R.conv("floor_area_m2")), 10.0,
        help="11,700 m² total ÷ 7 floors. The paper gives no per-floor split.")
    st.caption("Full list with sources on the Methods page.")

# Data-quality warnings for the chosen date, always visible.
notes = D.data_status(as_of, floor) if R.FLOORS[floor]["wired"] else []
if notes:
    st.sidebar.divider()
    for n in notes:
        (st.sidebar.error if n["level"] == "error" else st.sidebar.warning)(
            f"**{n['title']}**")

st.sidebar.divider()
st.sidebar.caption(
    "CU-BEMS, Chulalongkorn University · 7-storey office, Bangkok · "
    "1-minute data, Jul 2018 – Dec 2019.")

# --------------------------------------------------------------------------- #
# navigation
# --------------------------------------------------------------------------- #

P = "pages"
nav = st.navigation({
    "Operations": [
        st.Page(f"{P}/1_floor_status.py", title="Floor status", icon=":material/dashboard:", default=True),
        st.Page(f"{P}/2_sensor_health.py", title="Sensor & data health", icon=":material/sensors:"),
        st.Page(f"{P}/3_zones.py", title="Zones", icon=":material/grid_view:"),
        st.Page(f"{P}/5_faults.py", title="Faults & actions", icon=":material/warning:"),
    ],
    "Performance": [
        st.Page(f"{P}/4_energy.py", title="Energy", icon=":material/bolt:"),
        st.Page(f"{P}/6_opportunities.py", title="Opportunities", icon=":material/savings:"),
    ],
    "Reference": [
        st.Page(f"{P}/7_building.py", title="Building overview", icon=":material/apartment:"),
        st.Page(f"{P}/8_methods.py", title="Methods & assumptions", icon=":material/science:"),
    ],
})
nav.run()
