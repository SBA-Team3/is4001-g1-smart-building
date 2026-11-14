"""About the data — what is measured and what is not, how alerts behave, and
the assumptions behind the numbers. Written for the facilities team; the
methodology lives in the notebooks and the report."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sbanalytics import registry as R
import ui

c = ui.ctx()

ui.page_header("About", "About the data", "What the data covers, how alerts behave, and the assumptions in use")

t_scope, t_alerts, t_assume = st.tabs(["What is measured", "Alerts", "Assumptions"])

# --------------------------------------------------------------------------- #
with t_scope:
    st.subheader("Measured")
    st.markdown(
        "- **Energy:** AC, lighting and plug kWh per zone, every minute, on all seven floors.\n"
        "- **Indoor conditions:** temperature, humidity and a light reading per sensored zone "
        "(not every zone has one; Floor 1 has none).\n"
        "- **Outdoor weather:** not in the dataset. Every weather-normalised figure uses ERA5 "
        "reanalysis for central Bangkok (via Open-Meteo) — city-level, not a rooftop station.")
    st.subheader("Not measured")
    st.markdown(
        "- **Central plant, lifts, risers and common areas.** All energy, cost and carbon figures are "
        "submetered zone loads only, so they understate the building's bill.\n"
        "- **Setpoints.** Only the resulting room temperature is seen, so comfort is measured against a "
        "declared band, not as 'unmet load hours', and setpoint faults cannot be detected.\n"
        "- **Plant and air-handler signals** (supply/return air, pressure, valves, dampers). None of the "
        "ASHRAE Guideline 36 air-handler faults and no chiller efficiency (kW/ton, COP).\n"
        "- **Occupancy.** Occupied hours are taken from the measured AC schedule, not from people counts.\n"
        "- **CO₂ and ventilation**, so no indoor air quality.\n"
        "- **True illuminance.** The light channel tops out at 49–138 where a lit office reads 300–2,000; "
        "it is a scaled index, used only as a relative signal.")

# --------------------------------------------------------------------------- #
with t_alerts:
    gap = R.conv("episode_gap_days")
    st.markdown(
        "**Who sets what.** People move an alert from New → Acknowledged → In progress (a work order), "
        f"or Mute it. The system **clears** it once the condition has not been seen for more than "
        f"**{gap:.0f} days** — a gap, so a weekend does not clear an overcooling alert. A cleared alert "
        "that comes back reappears as open, keeping its last human state.\n\n"
        "**Order.** Alerts are grouped by fault type and ranked by cost of inaction (฿/day = attributable "
        "energy ÷ days since first seen × tariff). There is no severity label: the rule thresholds do not "
        "line up with the 40% overcooling target and some depend on the period length.\n\n"
        "**What clears each type:**\n"
        "- **Overcooling** — cooled, occupied hours below the comfort band stop occurring.\n"
        "- **AC after hours** — no AC energy outside the cooling schedule or on non-workdays.\n"
        "- **Lights overnight** — overnight lighting falls back to its night baseline.\n"
        "- **Short cycling** — the AC stops switching on and off faster than the rule's limit.\n"
        "- **Humidity** — indoor humidity back under the limit.\n"
        "- **Dead meter** — the meter reports changing values again.\n"
        "- **Point offline** — the point reports samples again.\n"
        "- **Meter change** — a dated event; it clears after the gap.\n\n"
        "**Frozen readings.** A temperature sensor is judged frozen only while cooling runs: a room "
        "legitimately holds steady overnight with the AC off. Power meters are judged on a 4-hour flat run.")

# --------------------------------------------------------------------------- #
with t_assume:
    st.caption("Values in use now. Tariff, building area and comfort band are set in Assumptions (top "
               "bar); the emission factor on the Carbon page.")
    area_custom = abs(c["building_area"] - R.conv("floor_area_m2") * ui.N_FLOORS) > 0.5
    rows = [
        ("Electricity tariff", f"{c['tariff']:.2f} ฿/kWh", "Assumptions",
         "MEA large-commercial rate; flat, no time-of-use or demand charge"),
        ("Floor area", f"{c['area']:,.0f} m² per floor ({c['building_area']:,.0f} m² building)",
         "Assumptions", ("custom value" if area_custom else "11,700 m² building")
         + " split equally across 7 floors — the paper gives no per-floor split"),
        ("Comfort band", f"{c['lo']:g}–{c['hi']:g} °C", "Assumptions", R.CONVENTIONS["comfort_lo"][2]),
        ("Overcooling target", f"below {R.conv('overcooled_target_pct'):.0f}% of cooled occupied hours",
         "fixed", "Project proposal KPI"),
        ("Grid emission factor", f"{c['ef']:.4f} kgCO₂e/kWh", "Carbon page",
         "Scope 2, location-based; " + R.CONVENTIONS["co2_kg_per_kwh"][2]),
        ("EUI benchmarks", f"{R.conv('eui_good'):.0f} good · {R.conv('eui_typical'):.0f} typical kWh/m²/yr",
         "fixed", "Bangkok office study (n=41); whole-building, so our partial EUI reads low against it"),
        ("Floor 1 use", "lobby", "fixed", "Team assumption — reported for energy, cost and carbon only"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["assumption", "value", "set on", "basis"]),
                 width="stretch", hide_index=True,
                 column_config={"basis": st.column_config.TextColumn(width="large")})
    st.markdown(
        "**Carbon** is Scope 2, location-based: kWh × grid factor ÷ 1000. Not reported for lack of data: "
        "Scope 1 (refrigerant, generators), market-based Scope 2 (no contract or REC data) and a "
        "whole-building inventory (no main meter).")
