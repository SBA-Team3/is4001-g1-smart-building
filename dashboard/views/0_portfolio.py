"""Portfolio overview — the whole building on one screen: where do I send the team?"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import portfolio as P
from sbanalytics import registry as R
import ui

c = ui.ctx()
as_of, tariff, ef, area = c["as_of"], c["tariff"], c["ef"], c["area"]
target = R.conv("overcooled_target_pct")
target_txt = f"{target:.0f}%"
LO = R.conv("comfort_lo")

h = ui.building_hourly()
H = ui.building_health()


@st.cache_data(show_spinner="Ranking floors…")
def _league(area: float) -> pd.DataFrame:
    return P.floor_league(D.load_hourly(None), area, D.load_health(None))


@st.cache_resource(show_spinner="Sizing each floor's levers…")
def _levers() -> pd.DataFrame:
    return P.floor_levers(D.load_hourly(None))


@st.cache_data(show_spinner="Building the overcooling map…")
def _heat() -> pd.DataFrame:
    return P.overcooled_by_month(D.load_hourly(None))


def _day(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.day} {t:%b %Y}"


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
periods = {"Month to date": (as_of.replace(day=1), as_of),
           f"Trailing {c['window']} day(s)": D.window(as_of, c["window"]),
           "Last 12 months": D.window(as_of, 365),
           "Whole record": (D.RECORD_START, D.RECORD_END)}
hd = st.columns([3, 1], vertical_alignment="bottom")
with hd[0]:
    ui.page_header("Overview", "Portfolio overview",
                   f"All seven floors · 11,700 m² · 33 zones · Where do I send the team, and are we "
                   f"on the {target_txt} overcooling target?")
pname = hd[1].selectbox("Period", list(periods), index=0)
s, e = periods[pname]
s = max(pd.Timestamp(s), D.RECORD_START)
hd[1].caption(f"{_day(s)} – {_day(e)} · data as of {_day(as_of)}")

# --------------------------------------------------------------------------- #
# banners + KPI strip
# --------------------------------------------------------------------------- #
bk = P.building_kpis(h, s, e, tariff, ef)
for f in bk["truncated_floors"]:
    st.warning(f"**Floor {f}'s file ends {R.floor_file_end(f):%Y-%m-%d %H:%M}.** Building totals for "
               f"days after that exclude Floor {f} — they are not metered, not zero.",
               icon=":material/warning:")
COLD = f"below {LO:g} °C %"
lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
if s <= hi and pd.Timestamp(e) >= lo:
    st.info(f"**This period overlaps the building-wide sensor gap ({lo.date()} – {hi.date()}).** "
            "Energy is metered throughout; the overcooled share only counts hours with live sensors.",
            icon=":material/info:")

k = st.columns(5)
k[0].metric("Building energy", f"{bk['kwh']:,.0f} kWh", border=True,
            help="Submetered zone energy (AC + lighting + plug) on all seven floors, including "
                 "Floor 1. Excludes central plant, lifts and common areas.")
k[1].metric("Cost", f"฿{bk['thb']:,.0f}", border=True, help=f"At {tariff:.2f} ฿/kWh (sidebar).")
k[2].metric("Emissions", f"{bk['tco2e']:,.1f} tCO₂e", border=True,
            help=f"At {ef:.4f} kgCO₂e/kWh — {R.CONVENTIONS['co2_kg_per_kwh'][2]}.")
oc = bk["overcooled_pct"]
k[3].metric(f"Cooled hours below {LO:g} °C", f"{oc:.0f}%" if np.isfinite(oc) else "no sensor data",
            delta=f"{oc - target:+.0f} pp vs {target_txt} target" if np.isfinite(oc) else None,
            delta_color="inverse", border=True,
            help="Office floors F2–F7: share of cooled, occupied zone-hours below "
                 f"{LO:g} °C, weighted by zone AC energy (the notebooks' "
                 "definition). Floor 1 has no sensors and is not an office floor.")
k[4].metric("Data completeness", f"{bk['meter_pct']:.0f}% meters",
            delta=(f"{bk['sensor_pct']:.0f}% sensors" if np.isfinite(bk["sensor_pct"]) else "no sensors"),
            delta_color="off", border=True,
            help=f"Share of zone-hours with usable metering / usable indoor temperature. "
                 f"Target {R.conv('completeness_target'):.0f}%.")
st.caption("**Floor 1 is not an office floor** (AC never switches off). Its energy is in the "
           "totals; it is excluded from the overcooled share and every schedule ranking.")

# --------------------------------------------------------------------------- #
# floor cards
# --------------------------------------------------------------------------- #
hd = st.columns([4, 1], vertical_alignment="bottom")
with hd[0]:
    ui.section("Floors", f"Whole record. Office floors worst-first by cooled hours below "
                         f"{LO:g} °C; Floor 1 last.")
hd[1].markdown(f"<div style='text-align:right'>{ui.legend([(f'{target_txt} target', ui.INK)], tick=True)}"
               "</div>", unsafe_allow_html=True)

L = _league(area)
lv = _levers()


def _open_floor(f: int) -> None:
    st.session_state["floor_select"] = f


rows = list(L.iterrows())
for i, (_, r) in enumerate(rows):
    if i % 3 == 0:                       # one st.columns per row, so a row's cards top-align
        cols = st.columns(3)
    f = int(r.floor)
    office = f not in R.NON_OFFICE_FLOORS
    with cols[i % 3]:
        with ui.card(f"floor_{f}", edge=ui.FLOOR_HEX[f]):
            ui.md(ui.floor_card_body(f, r.to_dict(), P.what_to_look_at(f, r.to_dict(), lv, tariff),
                                     "office" if office else "non-office"))
            if R.FLOORS[f]["wired"]:
                if st.button("Open floor status", key=f"open_{f}", type="tertiary",
                             icon=":material/arrow_forward:", on_click=_open_floor, args=(f,)):
                    st.switch_page("views/1_floor_status.py")
            else:
                st.caption("Floor status not wired yet — see the analysis/ notebooks.")

fails = [f"F{f}" for f in L.loc[L.baseline_meets_g14.eq(False), "floor"]]
st.caption("'vs baseline' is Jul–Dec 2019 actual vs the weather- and calendar-normalised baseline "
           "(notebook 02, total kWh); positive means more than expected."
           + (f" The baseline for {', '.join(fails)} fails ASHRAE G14 out of sample — read its figure "
              "as indicative only." if fails else ""))

with st.expander("All floors as a table", icon=":material/table:"):
    show = pd.DataFrame({
        "floor": [f"F{f}" + ("  · non-office" if f in R.NON_OFFICE_FLOORS else "")
                  + (f"  · file ends {R.floor_file_end(f):%d %b %y}" if f in R.FILE_END else "")
                  for f in L.floor],
        "partial EUI": L.eui,
        "AC share %": L.ac_share_pct,
        COLD: L.overcooled_pct,
        "after-hours AC %": L.after_hours_ac_pct,
        "vs baseline %": L.deviation_pct,
        "meters %": L.meter_pct,
        "sensors %": L.sensor_pct,
        "status": L.status,
    })
    st.dataframe(
        show.style.format(
            {"partial EUI": "{:.0f}", "AC share %": "{:.0f}", COLD: "{:.0f}",
             "after-hours AC %": "{:.1f}", "vs baseline %": "{:+.1f}", "meters %": "{:.0f}",
             "sensors %": "{:.0f}"}, na_rep="—"),
        width="stretch", hide_index=True,
        column_config={
            "partial EUI": st.column_config.NumberColumn(
                "partial EUI", help="kWh/m²/yr on the assumed floor area (sidebar). Submetered zone "
                                    "loads only — not comparable to a whole-building benchmark."),
            COLD: st.column_config.ProgressColumn(color=ui.ACTUAL, label=
                f"{COLD} (target {target:.0f})", min_value=0, max_value=100, format="%.0f"),
            "vs baseline %": st.column_config.NumberColumn(
                "vs baseline %", help="Jul–Dec 2019 actual vs the weather- and calendar-normalised "
                                      "baseline trained on Jul 2018 – Jun 2019 (notebook 02, total kWh). "
                                      "Positive = used more than expected."),
            "sensors %": st.column_config.NumberColumn(
                "sensors %", help="Includes the six-month building-wide sensor gap."),
            "status": st.column_config.TextColumn(width="large"),
        })

# --------------------------------------------------------------------------- #
# floor x month heat map
# --------------------------------------------------------------------------- #
with ui.card("heatmap"):
    ui.section("Where the building is overcooled, month by month",
               f"Share of cooled, occupied hours below {LO:g} °C, office floors. "
               f"The {target_txt} target is the middle of the scale.")
    hm = _heat().assign(floor=lambda d: "F" + d.floor.astype(str))
    hm["label"] = np.where(hm.overcooled_pct.notna(),
                           hm.overcooled_pct.round(0).astype("Int64").astype(str), "")
    hm["cell"] = np.where(hm.overcooled_pct.notna(), "data", "no data")
    # Dark ink on the light middle of the scale, light ink on its darker ends.
    hm["ink"] = np.where(hm.overcooled_pct.between(25, 75), "dark", "light")
    base = alt.Chart(hm).encode(
        x=alt.X("yearmonth(month):O", title=None,
                axis=alt.Axis(labelAngle=0, format="%b %y", orient="top", domain=False, ticks=False)),
        y=alt.Y("floor:N", title=None, sort=[f"F{f}" for f in P.office_floors()],
                axis=alt.Axis(domain=False, ticks=False)))
    heat = alt.layer(
        base.transform_filter("datum.cell == 'no data'").mark_rect(
            color=ui.NO_DATA, cornerRadius=4, stroke=ui.SURFACE, strokeWidth=2),
        base.transform_filter("datum.cell == 'data'").mark_rect(
            cornerRadius=4, stroke=ui.SURFACE, strokeWidth=2).encode(
            color=alt.Color("overcooled_pct:Q", scale=ui.OVERCOOL_COLOR, legend=None),
            tooltip=["floor", alt.Tooltip("month:T", format="%b %Y"),
                     alt.Tooltip("overcooled_pct:Q", format=".0f", title=COLD),
                     alt.Tooltip("hours:Q", title="cooled zone-hours"),
                     alt.Tooltip("zones:Q", title="zones counted")]),
        base.mark_text(fontSize=12, fontWeight=500).encode(
            text="label:N",
            color=alt.Color("ink:N", scale=alt.Scale(domain=["dark", "light"],
                                                     range=[ui.C["bg/page"], ui.INK]), legend=None)),
        alt.Chart(pd.DataFrame({"m": [as_of.replace(day=1)]})).mark_rect(
            fill=None, stroke=ui.INK, strokeWidth=2, cornerRadius=4).encode(x="yearmonth(m):O"),
    )
    st.altair_chart(ui.chart(heat, height=230), width="stretch")
    stops = (f"{ui.C['data/overcool-low']} 0%, {ui.C['data/overcool-mid']} {target:.0f}%, "
             f"{ui.C['data/overcool-high']} 100%")
    ui.md(f'<div class="cb-row" style="gap:16px;align-items:flex-start;margin-top:-8px">'
          f'<div style="width:220px"><div style="height:8px;border-radius:4px;'
          f'background:linear-gradient(90deg,{stops})"></div>'
          f'<div class="cb-cap cb-muted" style="position:relative;height:16px;margin-top:4px">'
          f'<span style="position:absolute;left:0">0%</span>'
          f'<span style="position:absolute;left:{target:.0f}%;transform:translateX(-50%)">'
          f'{target_txt} target</span><span style="position:absolute;right:0">100%</span></div></div>'
          f'{ui.legend([("No data", ui.NO_DATA)])}</div>')
    st.caption(
        f"Grey: the building-wide sensor gap ({lo:%b %Y} – {hi:%b %Y}) and "
        + ", ".join(f"Floor {f} after its file ends ({R.floor_file_end(f):%d %b %Y})" for f in R.FILE_END)
        + ". Floor 1 has no indoor sensors. White outline: the as-of month. The step from 2018 to "
        "2019 is real and registered as a change-point: every zone ran 2–3 °C colder after the "
        "sensors came back on 2019-03-06, so the whole-record shares on the floor cards average two "
        "configurations.")

# --------------------------------------------------------------------------- #
# energy by floor
# --------------------------------------------------------------------------- #
with ui.card("monthly"):
    ui.section("Monthly energy by floor",
               "Submetered kWh per month. Floor 1 is the largest single consumer and runs around "
               "the clock.")
    mo = P.monthly_energy(h)
    mo["floor"] = "F" + mo.floor.astype(str)
    mo["order"] = mo.floor.str[1:].astype(int)
    st.altair_chart(ui.chart(alt.Chart(mo).mark_bar(stroke=ui.SURFACE, strokeWidth=1).encode(
        x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(labelAngle=0, format="%b %y")),
        y=alt.Y("kWh:Q", title="kWh / month",
                axis=alt.Axis(format="~s", labelExpr="datum.value == 0 ? '0' : datum.label")),
        color=alt.Color("floor:N", scale=ui.FLOOR_COLOR,
                        legend=alt.Legend(orient="top", title=None, symbolType="square",
                                          labelExpr="'Floor ' + slice(datum.label, 1)")),
        order=alt.Order("order:Q"),
        tooltip=["floor", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("kWh:Q", format=",.0f")]),
        height=280), width="stretch")
    st.caption("Floor 6 has no bar after October 2019 because its file ends, not because it used "
               "nothing.")
