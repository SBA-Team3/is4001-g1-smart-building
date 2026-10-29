"""Floor status — the morning screen."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import comfort as C
from sbanalytics import data as D
from sbanalytics import energy as E
from sbanalytics import faults as F
from sbanalytics import health as HL
from sbanalytics import kpis as K
from sbanalytics import mv as M
from sbanalytics import registry as R
import ui

c = ui.ctx()
floor, as_of, win = c["floor"], c["as_of"], c["window"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)
start, end = D.window(as_of, win)


@st.cache_resource(show_spinner="Fitting the weather-normalised baseline…")
def _model(floor: int) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, "ac_kwh")

st.title(f"Floor {floor} — status")
st.caption(f"As of **{as_of.date()}** · trailing window **{win} day(s)** "
           f"({start.date()} – {end.date()})")

# --- 1. trust band ---------------------------------------------------------- #
st.subheader("Can I trust today's data?")
phi = ui.trust_band(floor, as_of)
ui.status_notes(as_of, floor)
ui.file_end_note(floor, as_of)

# --- 2. the proposal's two KPIs --------------------------------------------- #
st.divider()
st.subheader("Against target")
target = R.conv("overcooled_target_pct")
m0 = as_of.replace(day=1)
p0 = (m0 - pd.Timedelta(days=1)).replace(day=1)


def _oc(s_, e_):
    o = K.overcooled_share(K.cooled_occupied(D.slice_hourly(h, s_, e_)), by=())
    return float(o.share_below.iloc[0] * 100) if len(o) else np.nan


oc_now, oc_prev = _oc(m0, as_of), _oc(p0, m0 - pd.Timedelta(days=1))
ex = M.month_excess(_model(floor), as_of, c["tariff"])
t, l_ = ex["this"], ex["last"]
k = st.columns(3)
k[0].metric(f"Cooled hours below {R.conv('comfort_lo'):g} °C, month to date",
            f"{oc_now:.0f}%" if np.isfinite(oc_now) else "no sensor data",
            delta=f"{oc_now - target:+.0f} pp vs {target:.0f}% target" if np.isfinite(oc_now) else None,
            delta_color="inverse",
            help="Proposal KPI 1: share of cooled, occupied zone-hours below the band on comfort "
                 "zones, weighted by zone AC energy. "
                 + (f"Last month {oc_prev:.0f}%." if np.isfinite(oc_prev) else "No data last month."))
per_day = lambda x: x["excess"] / x["days"] if x["days"] else np.nan
k[1].metric("Excess cooling, month to date",
            f"{t['excess']:+,.0f} kWh" if t["days"] else "no data",
            delta=f"฿{t['thb']:+,.0f} vs expected" if t["days"] else None, delta_color="inverse",
            help=f"Proposal KPI 2: actual − weather/calendar-normalised expected AC energy, "
                 f"{t['start'].date()} – {t['end'].date()} ({t['days']} usable days). Actual "
                 f"{t['actual']:,.0f} kWh vs expected {t['expected']:,.0f} kWh. Target: lower every "
                 "month.")
guard = ui.comparison_guard(l_["start"], t["end"]) if t["days"] and l_["days"] else []
if guard or not (t["days"] and l_["days"]):
    k[2].metric("vs last month (per day)", "—",
                help="Disabled: the two months straddle a change-point, or one has no usable days.")
else:
    k[2].metric("vs last month (per day)", f"{per_day(t):+,.0f} kWh/day",
                delta=f"{per_day(t) - per_day(l_):+,.0f} vs {per_day(l_):+,.0f} last month",
                delta_color="inverse",
                help="Per usable day, so a part month compares fairly with a full one.")
if t["in_train"]:
    st.caption("This month is inside the baseline's training window, where excess is close to "
               "zero by construction — read it as a check that the model fits, not a result.")

# --- 3. since yesterday ----------------------------------------------------- #
y_start, y_end = D.window(as_of - pd.Timedelta(days=1), win)
reg_y = F.build_register(h, H, start=y_start, end=y_end, floor=floor, tariff=c["tariff"])
reg_t = F.build_register(h, H, start=start, end=end, floor=floor, tariff=c["tariff"])
diff = F.register_diff(reg_y, reg_t)
lost = HL.went_offline(H, as_of)
st.markdown("**Since yesterday**")
sy = st.columns(3)
sy[0].metric("New faults", len(diff["new"]))
sy[1].metric("Cleared faults", len(diff["cleared"]))
sy[2].metric("Points went offline", len(lost))
detail = ([f"new: `{x}`" for x in diff["new"]] + [f"cleared: `{x}`" for x in diff["cleared"]]
          + [f"offline: `{x}`" for x in lost.column])
if detail:
    st.caption(" · ".join(detail))

st.divider()

# --- 4. headline ------------------------------------------------------------ #
st.subheader("How is the floor running?")

day_tot = E.floor_totals(h, as_of, as_of)
base = E.same_weekday_baseline(h, as_of)
dem = E.demand_stats(h, start, end)
zc = C.zone_comfort(h, start=start, end=end)
reg = reg_t

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

# --- 5. zone triage --------------------------------------------------------- #
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

# --- 6. top faults ---------------------------------------------------------- #
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
                f"{'฿%s/day' % format(r.cost_per_day_thb, ',.0f') if r.cost_per_day_thb else ''}"
                f"</div>", unsafe_allow_html=True)
    st.page_link("pages/5_faults.py", label=f"All {len(reg)} faults →",
                 icon=":material/warning:")

# --- 7. the month ----------------------------------------------------------- #
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
