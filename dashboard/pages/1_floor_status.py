"""Floor status — the morning screen."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import comfort as C
from core import data as D
from core import energy as E
from core import faults as F
from core import registry as R
from core import ui

c = ui.ctx()
floor, as_of, win = c["floor"], c["as_of"], c["window"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)
start, end = D.window(as_of, win)

st.title(f"Floor {floor} — status")
st.caption(f"As of **{as_of.date()}** · trailing window **{win} day(s)** "
           f"({start.date()} – {end.date()})")

# --- 1. trust band ---------------------------------------------------------- #
st.subheader("Can I trust today's data?")
phi = ui.trust_band(floor, as_of)
ui.status_notes(as_of, floor)

st.divider()

# --- 2. headline ------------------------------------------------------------ #
st.subheader("How is the floor running?")

day_tot = E.floor_totals(h, as_of, as_of)
base = E.same_weekday_baseline(h, as_of)
dem = E.demand_stats(h, start, end)
zc = C.zone_comfort(h, start=start, end=end)
reg = F.build_register(h, H, start=start, end=end, tariff=c["tariff"])

k = st.columns(5)
k[0].metric("Energy today", f"{day_tot['total']:,.0f} kWh",
            delta=(f"{base['delta_pct']:+.0f}% vs same weekday"
                   if base.get("delta_pct") is not None and np.isfinite(base.get("delta_pct", np.nan))
                   else "no baseline"),
            delta_color="inverse",
            help="Compared with the mean of the same weekday over the previous "
                 "4 weeks. Deliberately not year-on-year: the floor ran 2–3 °C "
                 "colder after the 2019 maintenance window, so a year-on-year "
                 "comparison would be measuring a configuration change.")
k[1].metric("Peak demand", f"{dem.get('peak_kw', float('nan')):,.0f} kW" if dem else "—",
            help=f"In the trailing window. Load factor "
                 f"{dem.get('load_factor', float('nan')):.2f}." if dem else None)
k[2].metric("AC share", f"{day_tot['ac_share']:.0f}%" if np.isfinite(day_tot['ac_share']) else "—")
out_of_band = int((zc.cold_hours + zc.warm_hours > 0).sum()) if len(zc) else 0
k[3].metric("Zones outside band", f"{out_of_band} / {len(zc)}" if len(zc) else "—",
            help=f"Declared band {C.band()[0]:g}–{C.band()[1]:g} °C.")
k[4].metric("Open faults", len(reg),
            delta=f"{int((reg.priority == F.HIGH).sum())} high" if len(reg) else None,
            delta_color="inverse")

# --- 3. zone triage --------------------------------------------------------- #
st.divider()
st.subheader("Zone triage")
st.caption("Sorted by severity: overcooling degree-hours weighted above "
           "overheating, because in cooling mode a cold hour is a comfort fault "
           "*and* wasted energy.")

rows = []
day_e = E.end_use(h, start, end, floor=floor).set_index("zone")
nfaults = reg.groupby("zone").size().to_dict() if len(reg) else {}
cm = zc.set_index("zone") if len(zc) else pd.DataFrame()

for z in R.ZONES[floor]:
    has_ac = R.has_channel(floor, z, "ac")
    tp = R.point_for(floor, z, "temp")
    comfort_valid = tp is not None and tp.role == R.ROLE_ZONE_TEMP
    r = {
        "zone": ui.zone_label(floor, z),
        "temp °C": (cm.loc[z, "mean_c"] if z in cm.index else np.nan)
                   if comfort_valid else np.nan,
        "RH %": (cm.loc[z, "rh"] if z in cm.index else np.nan) if comfort_valid else np.nan,
        "AC kWh": day_e.loc[z, "ac_kwh"] if has_ac and z in day_e.index else np.nan,
        "light kWh": day_e.loc[z, "light_kwh"] if z in day_e.index else np.nan,
        "cold h %": (cm.loc[z, "cold_pct"] if z in cm.index else np.nan) if comfort_valid else np.nan,
        "exceedance °C·h": (cm.loc[z, "edh"] if z in cm.index else np.nan) if comfort_valid else np.nan,
        "faults": sum(v for kk, v in nfaults.items() if z in str(kk)),
        "note": ("stairwell — no cooling or sensors" if not has_ac else
                 "supply-air sensor — comfort not valid" if not comfort_valid else ""),
    }
    rows.append(r)

triage = pd.DataFrame(rows)
st.dataframe(
    triage.round(1), width="stretch", hide_index=True,
    column_config={
        "AC kWh": st.column_config.NumberColumn(format="%.0f"),
        "light kWh": st.column_config.NumberColumn(format="%.0f"),
        "cold h %": st.column_config.ProgressColumn(
            "cold h %", min_value=0, max_value=100, format="%.0f"),
    })
st.caption("Blank cells are points that do not exist, not zeros. z3 is a "
           "stairwell with no cooling and no sensors; z1's temperature sensor "
           "sits in a supply-air path, so it has no valid room-comfort figure.")

# --- 4. top faults ---------------------------------------------------------- #
st.divider()
st.subheader("What needs attention")

if not len(reg):
    st.success("No faults raised in this window.")
else:
    for _, r in reg.head(3).iterrows():
        colr = {"high": ui.BAD, "medium": ui.WARN, "low": ui.OK}[r.priority]
        with st.container(border=True):
            cc = st.columns([5, 1])
            cc[0].markdown(f"**{r.title}**  \n{r.detail}")
            cc[1].markdown(
                f"<div style='text-align:right'><span style='color:{colr};"
                f"font-weight:600'>{r.priority.upper()}</span><br>"
                f"{'฿%s/period' % format(r.thb_impact, ',.0f') if r.thb_impact else ''}"
                f"</div>", unsafe_allow_html=True)
    st.page_link("pages/5_faults.py", label=f"All {len(reg)} faults →",
                 icon=":material/warning:")

# --- 5. the week ------------------------------------------------------------ #
st.divider()
st.subheader("The last 30 days")
dl = E.daily(h, as_of - pd.Timedelta(days=29), as_of)
if len(dl):
    m = dl.melt(id_vars=["date", "is_weekend"], value_vars=E.END_USES,
                var_name="end_use", value_name="kWh")
    m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("kWh:Q", title="kWh/day"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["date:T", "end_use", alt.Tooltip("kWh:Q", format=",.0f")],
        ).properties(height=240), width="stretch")
