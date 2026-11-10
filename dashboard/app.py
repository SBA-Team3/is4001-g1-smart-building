"""
CU-BEMS building — facilities operations dashboard.

    streamlit run dashboard/app.py

Build the shared cleaned dataset first (from the repo root):

    python -m sbanalytics.pipeline

The FM team's first questions are energy, cost and which sensors are down, so
the Overview answers those for the whole building on one 1440×900 screen. The
navigation is flat; one context bar (floor, as-of date, period, assumptions)
sits above every page and is read through ui.ctx().
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
                   layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()
_DESIGN = os.path.join(_HERE, "design")
st.logo(os.path.join(_DESIGN, "logo.svg"), size="large",
        icon_image=os.path.join(_DESIGN, "logo_mark.svg"))

if not D.cache_ready(5):
    st.title("Cleaned data not built")
    st.markdown(
        "The dashboard reads the shared cleaned dataset in data/cleaned/ rather "
        "than the raw CSVs, because minute-level fault detection over 790,560 rows "
        "cannot run on a page load.\n\n```bash\npython -m sbanalytics.pipeline\n```"
    )
    st.stop()

# A table row (Overview floors) can ask to open another page; its callback
# cannot switch pages itself, so it leaves a note here.
_goto = st.session_state.pop("_goto", None)
if _goto:
    st.switch_page(_goto)

# --------------------------------------------------------------------------- #
# navigation — flat, in the order the FM asks: energy & cost, alarms, sensors
# --------------------------------------------------------------------------- #
P = "views"
OVERVIEW = st.Page(f"{P}/0_portfolio.py", title="Overview", icon=":material/apartment:",
                   default=True)
nav = st.navigation(position="top", pages=[
    OVERVIEW,
    st.Page(f"{P}/1_floor_status.py", title="Floor", icon=":material/dashboard:"),
    st.Page(f"{P}/5_faults.py", title="Alarms", icon=":material/notifications:"),
    st.Page(f"{P}/2_sensor_health.py", title="Sensors", icon=":material/sensors:"),
    st.Page(f"{P}/3_zones.py", title="Comfort", icon=":material/thermostat:"),
    st.Page(f"{P}/4_energy.py", title="Energy", icon=":material/bolt:"),
    st.Page(f"{P}/6_opportunities.py", title="Savings", icon=":material/savings:"),
    st.Page(f"{P}/8_methods.py", title="Methods", icon=":material/science:"),
])
on_overview = nav.url_path == OVERVIEW.url_path

# --------------------------------------------------------------------------- #
# context bar — set once, read by every page through ui.ctx()
# --------------------------------------------------------------------------- #
WIRED = [f"F{f}" for f in R.wired_floors()]
FLOOR_OPTS = ["Bldg"] + WIRED
st.session_state.setdefault("floor", 5)
st.session_state.setdefault("floor_select", "Bldg" if on_overview else f"F{st.session_state['floor']}")
# Older sessions (and the Overview's row click) may hold a bare int.
if isinstance(st.session_state["floor_select"], int):
    st.session_state["floor_select"] = f"F{st.session_state['floor_select']}"
if st.session_state["floor_select"] not in FLOOR_OPTS:
    st.session_state["floor_select"] = f"F{st.session_state['floor']}"
# Floor pages need a wired floor: "Bldg" there falls back to the last one used.
if not on_overview and st.session_state["floor_select"] == "Bldg":
    st.session_state["floor_select"] = f"F{st.session_state['floor']}"
st.session_state.setdefault("period", ui.DEFAULT_PERIOD)
st.session_state.setdefault("as_of", D.DEFAULT_AS_OF.date())
if isinstance(st.session_state["as_of"], pd.Timestamp):
    st.session_state["as_of"] = st.session_state["as_of"].date()
for k, name in [("tariff", "tariff_thb"), ("area", "floor_area_m2"), ("ef", "co2_kg_per_kwh"),
                ("demand_charge", "demand_charge_thb_per_kw")]:
    st.session_state.setdefault(k, float(R.conv(name)))

with st.container(key="ctxbar"):
    cb = st.columns([3.3, 1.7, 4.2, 1.6], vertical_alignment="bottom", gap="medium")
    sel = cb[0].segmented_control(
        "Floor", FLOOR_OPTS, key="floor_select", required=True,
        help="Bldg = the whole building (Overview only). F1 is not an office floor and F2 is "
             "not wired; both are in the building totals.")
    if sel == "Bldg" and not on_overview:
        st.switch_page(OVERVIEW)
    if sel != "Bldg":
        st.session_state["floor"] = int(sel[1:])
    cb[1].date_input(
        "Data as of", key="as_of",
        min_value=D.RECORD_START.date(), max_value=D.RECORD_END.date(), format="DD/MM/YYYY",
        help="The record runs 1 Jul 2018 – 31 Dec 2019, so 'today' is a date you choose.")
    cb[2].segmented_control(
        "Period", ui.PERIODS, key="period", required=True,
        help="Every page reads this one period. MTD = month to date; Record = from 1 Jul 2018 "
             "to the as-of date.")
    with cb[3].popover("Assumptions", icon=":material/tune:", width="stretch"):
        st.number_input("Tariff (฿/kWh)", 0.0, 20.0, step=0.10, key="tariff",
                        help="0 hides every ฿ figure behind 'set rate in Assumptions'.")
        st.number_input("Floor area (m²)", 100.0, 5000.0, step=10.0, key="area",
                        help="11,700 m² total ÷ 7 floors. The paper gives no per-floor split.")
        st.number_input("Grid emission factor (kgCO₂e/kWh)", 0.0, 1.5, step=0.01, key="ef",
                        format="%.4f", help=R.CONVENTIONS["co2_kg_per_kwh"][2])
        st.number_input("Demand charge (฿/kW/month)", 0.0, 1000.0, step=10.0, key="demand_charge",
                        help=R.CONVENTIONS["demand_charge_thb_per_kw"][2])
        st.caption("Full list with sources on the Methods page.")

st.session_state["building_view"] = sel == "Bldg"
_s, _e, _ = ui.period_bounds(st.session_state["as_of"], st.session_state["period"])
st.session_state["window_days"] = int((_e - _s).days) + 1

nav.run()
