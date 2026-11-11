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
# navigation — flat, in the order the FM asks: energy & cost, alerts, sensors
# --------------------------------------------------------------------------- #
P = "views"
OVERVIEW = st.Page(f"{P}/0_portfolio.py", title="Overview", icon=":material/apartment:",
                   default=True)
nav = st.navigation(position="top", pages=[
    OVERVIEW,
    st.Page(f"{P}/1_floor_status.py", title="Floor", icon=":material/dashboard:"),
    st.Page(f"{P}/5_faults.py", title="Alerts", icon=":material/notifications:"),
    st.Page(f"{P}/2_sensor_health.py", title="Sensors", icon=":material/sensors:"),
    st.Page(f"{P}/3_zones.py", title="Comfort", icon=":material/thermostat:"),
    st.Page(f"{P}/4_energy.py", title="Energy", icon=":material/bolt:"),
    st.Page(f"{P}/7_carbon.py", title="Carbon", icon=":material/eco:"),
    st.Page(f"{P}/6_opportunities.py", title="Savings", icon=":material/savings:"),
    st.Page(f"{P}/8_methods.py", title="Methods", icon=":material/science:"),
])
on_overview = nav.url_path == OVERVIEW.url_path
#: Pages that offer "Bldg" (ui.BUILDING_PAGES): Overview and Carbon highlight a selected
#: floor; Alerts narrows to it.
on_building_page = nav.url_path in ui.BUILDING_PAGES
#: Arriving at Alerts from another page (top nav, Overview link) starts at Bldg: the
#: morning queue is building-wide. Picking a floor there narrows it until you leave.
entering_alerts = nav.url_path == "faults" and st.session_state.get("_prev_page") != "faults"
st.session_state["_prev_page"] = nav.url_path

# --------------------------------------------------------------------------- #
# context bar — set once, read by every page through ui.ctx()
# --------------------------------------------------------------------------- #
WIRED = [f"F{f}" for f in R.wired_floors()]
# Bldg on Overview, Alerts and Carbon; F1 (lobby, reporting only) on Overview, Carbon,
# Floor and Energy; the office floors everywhere (ui.floor_options).
FLOOR_OPTS = ui.floor_options(nav.url_path)
on_carbon = nav.url_path == "carbon"
st.session_state.setdefault("floor", 5)
st.session_state.setdefault("office_floor", 5)          # last office floor, for fallbacks
st.session_state.setdefault("floor_select", "Bldg" if on_building_page else f"F{st.session_state['floor']}")
# Older sessions (and the Overview's row click) may hold a bare int.
if isinstance(st.session_state["floor_select"], int):
    st.session_state["floor_select"] = f"F{st.session_state['floor_select']}"
if entering_alerts:
    st.session_state["floor_select"] = "Bldg"
# Anything not offered here ("Bldg" on a floor page, F1 on an office-only page) falls
# back to the last office floor (default F5).
if st.session_state["floor_select"] not in FLOOR_OPTS:
    st.session_state["floor_select"] = f"F{st.session_state['office_floor']}"
st.session_state.setdefault("period", ui.DEFAULT_PERIOD)
st.session_state.setdefault("as_of", D.DEFAULT_AS_OF.date())
if isinstance(st.session_state["as_of"], pd.Timestamp):
    st.session_state["as_of"] = st.session_state["as_of"].date()

# Assumptions persist in non-widget keys (ui.ASSUMPTIONS): a page that doesn't
# render an input would otherwise lose its widget key and fall back to the default.
for name, (store, default) in ui.ASSUMPTIONS.items():
    st.session_state.setdefault(store, default())
EF_DEFAULT = ui.ASSUMPTIONS["ef"][1]()


def _keep(widget: str, store: str) -> None:
    st.session_state[store] = float(st.session_state[widget])


def _input(label: str, widget: str, store: str, lo: float, hi: float, step: float, container=st, **kw):
    st.session_state[widget] = st.session_state[store]          # copy in before rendering
    return container.number_input(label, lo, hi, step=step, key=widget, on_change=_keep,
                                  args=(widget, store), **kw)


BAND_DEFAULT = (ui.ASSUMPTIONS["comfort_lo"][1](), ui.ASSUMPTIONS["comfort_hi"][1]())


def _keep_band(widget: str, store: str) -> None:
    """Keep low < high: moving one edge past the other pushes the other along."""
    _keep(widget, store)
    lo, hi = st.session_state["_comfort_lo"], st.session_state["_comfort_hi"]
    if lo >= hi:
        if widget == "comfort_lo":
            st.session_state["_comfort_hi"] = lo + 0.5
        else:
            st.session_state["_comfort_lo"] = hi - 0.5


def _reset_band() -> None:
    st.session_state["_comfort_lo"], st.session_state["_comfort_hi"] = BAND_DEFAULT


def _reset_ef() -> None:
    st.session_state["_ef"] = EF_DEFAULT
    st.session_state["ef"] = EF_DEFAULT


with st.container(key="ctxbar"):
    _custom = ui.band_is_custom(st.session_state["_comfort_lo"], st.session_state["_comfort_hi"])
    cb = st.columns([2.9, 1.45, 3.55, 3.9] if on_carbon
                    else [3.1, 1.6, 4.0, 2.3] if _custom else [3.3, 1.7, 4.2, 1.6],
                    vertical_alignment="top" if on_carbon else "bottom", gap="medium")
    sel = cb[0].segmented_control(
        "Floor", FLOOR_OPTS, key="floor_select", required=True,
        help=("Bldg = the whole building. " if on_building_page else "")
             + ("F1 is the lobby (reporting only: energy, cost, carbon); it is in the building totals."
                if "F1" in FLOOR_OPTS else
                "F1 is the lobby (reporting only), so it is not offered here; it is in the building totals."))
    if sel != "Bldg":
        st.session_state["floor"] = int(sel[1:])
        if int(sel[1:]) in R.wired_floors():
            st.session_state["office_floor"] = int(sel[1:])
    cb[1].date_input(
        "Data as of", key="as_of",
        min_value=D.RECORD_START.date(), max_value=D.RECORD_END.date(), format="DD/MM/YYYY",
        help="The record runs 1 Jul 2018 – 31 Dec 2019, so 'today' is a date you choose.")
    cb[2].segmented_control(
        "Period", ui.PERIODS, key="period", required=True,
        help="Every page reads this one period. MTD = month to date; Record = from 1 Jul 2018 "
             "to the as-of date.")
    if on_carbon:
        # Carbon reports on the grid factor, so it replaces Assumptions here.
        e1, e2 = cb[3].columns([1.7, 1], vertical_alignment="top", gap="small")
        _input("Grid emission factor", "ef", "_ef", 0.0, 1.5, 0.0001, container=e1, format="%.4f",
               help="kgCO₂e/kWh. " + R.CONVENTIONS["co2_kg_per_kwh"][2]
                    + ". Change it only to a newer TGO figure.")
        if abs(st.session_state["_ef"] - EF_DEFAULT) > 1e-9:
            e2.markdown('<p class="cb-cap cb-muted" style="margin:0 0 6px;line-height:20px">custom factor</p>',
                        unsafe_allow_html=True)
            e2.button("Reset to TGO", on_click=_reset_ef, key="ef_reset", width="stretch")
        e1.caption("kgCO₂e/kWh · last updated 2022 (TGO)")
    else:
        custom_band = ui.band_is_custom(st.session_state["_comfort_lo"], st.session_state["_comfort_hi"])
        with cb[3].popover("Assumptions · custom band" if custom_band else "Assumptions",
                           icon=":material/tune:", width="stretch"):
            _input("Tariff (฿/kWh)", "tariff", "_tariff", 0.0, 20.0, 0.10,
                   help="0 hides every ฿ figure behind 'set rate in Assumptions'.")
            _input("Building floor area (m²)", "building_area", "_building_area", 1000.0, 50000.0, 100.0,
                   format="%.0f",
                   help="Gross floor area; per-floor figures use ÷7, the paper gives no split.")
            st.markdown('<p class="cb-cap" style="margin:4px 0 0">Comfort band (°C)</p>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2, gap="small")
            for col, label, widget, store, lo_, hi_ in [(b1, "Low", "comfort_lo", "_comfort_lo", 15.0, 28.0),
                                                       (b2, "High", "comfort_hi", "_comfort_hi", 18.0, 32.0)]:
                st.session_state[widget] = st.session_state[store]
                col.number_input(label, lo_, hi_, step=0.5, format="%.1f", key=widget, on_change=_keep_band,
                                 args=(widget, store),
                                 help="Overcooling = cooled occupied hours below Low; overheating above "
                                      "High. Default 23–26 °C (Thai field studies). The 40% target is fixed.")
            if custom_band:
                st.button(f"Reset to {BAND_DEFAULT[0]:g}–{BAND_DEFAULT[1]:g} °C", on_click=_reset_band,
                          key="band_reset", type="tertiary", icon=":material/restart_alt:")
            st.caption("The grid emission factor is set on the Carbon page. Full list with sources on "
                       "the Methods page.")

st.session_state["building_view"] = sel == "Bldg"
_s, _e, _ = ui.period_bounds(st.session_state["as_of"], st.session_state["period"])
st.session_state["window_days"] = int((_e - _s).days) + 1

nav.run()
