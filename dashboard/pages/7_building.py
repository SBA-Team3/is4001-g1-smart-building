"""Building overview — a shell. Only Floor 5 is wired."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data as D
from core import energy as E
from core import registry as R
from core import ui

c = ui.ctx()
area = c["area"]

st.title("Building overview")
st.caption("CU-BEMS · Chulalongkorn University · seven-storey office, Bangkok · "
           "11,700 m², 33 zones, 1-minute data Jul 2018 – Dec 2019.")

st.warning(
    "**This page is a placeholder.** Only Floor 5 is wired. The layout and the "
    "data contract are in place so the other floors can be added later, but "
    "nothing below reads their data yet — the rows are deliberately blank "
    "rather than filled with plausible-looking numbers."
)

# --------------------------------------------------------------------------- #
rows = []
for f, meta in sorted(R.FLOORS.items()):
    if meta["wired"]:
        h = ui.hourly(f)
        tot = E.floor_totals(h)
        ei = E.eui(h, area, f)
        rows.append({
            "floor": f"Floor {f}", "status": "wired", "zones": meta["zones"],
            "AC kWh": tot["ac_kwh"], "Light kWh": tot["light_kwh"],
            "Plug kWh": tot["plug_kwh"], "Total kWh": tot["total"],
            "AC share %": tot["ac_share"], "partial EUI": ei["eui"],
            "note": meta["note"]})
    else:
        rows.append({
            "floor": f"Floor {f}", "status": "not wired", "zones": meta["zones"],
            "AC kWh": np.nan, "Light kWh": np.nan, "Plug kWh": np.nan,
            "Total kWh": np.nan, "AC share %": np.nan, "partial EUI": np.nan,
            "note": meta["note"]})

bt = pd.DataFrame(rows)
st.subheader("Floors")
st.dataframe(bt.round(0), width="stretch", hide_index=True,
             column_config={"note": st.column_config.TextColumn(width="large")})

st.caption("Blank cells are floors this dashboard does not read, not floors "
           "with no energy.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("What wiring another floor involves")

st.markdown(
    "The data layer is floor-agnostic: every loader takes a floor number and "
    "gets its channel list from the point registry. Adding a floor is a data "
    "task, not a refactor.\n\n"
    "Every floor is already in `src/sbanalytics/registry.py` (one row per "
    "physical point, with the role that says what it *means*) and in the "
    "shared cleaned dataset built by `python scripts/build_clean_data.py`.\n\n"
    "1. Check the floor's registry rows and that each page renders sensibly.\n"
    "2. Flip `wired` to `True` in `FLOORS`.\n\n"
    "**Floors 3, 4 and 7 share Floor 5's column layout**; F6 has its four-AC "
    "zone in z4. Floor 2 has a different schema (36 "
    "columns, 14 AC units in z2) and Floor 1 has no environmental sensors at all "
    "and is not an office floor."
)

st.subheader("Known data problems in the other floors")
st.error(
    "**`data/2019Floor6.csv` is a truncated download.** It holds 424,168 of the "
    "expected 525,600 rows and the file ends mid-row, partway through a "
    "timestamp. Any analysis of Floor 6 that does not check this will silently "
    "lose the last ten weeks of 2019.\n\n"
    "This is exactly the class of fault the Sensor health page is built to "
    "catch — the completeness check that flags it is already written and would "
    "flag it the moment Floor 6 is wired."
)

st.info(
    "**Cross-floor peer comparison is the single highest-value addition here.** "
    "With no setpoint and no occupancy channel, comparing a zone against "
    "structurally identical zones on other floors is the strongest fault-"
    "detection signal available — it is what turns 'z4 is cold' into 'z4 is "
    "cold *and its peers are not*'. Floors 3, 4, 5 and 7 are structurally "
    "identical, which would roughly quadruple the comparison base."
)
