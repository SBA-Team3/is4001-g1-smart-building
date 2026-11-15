"""Carbon — Scope 2, location-based emissions from purchased electricity on the
submetered zone loads. Reporting only: the grid factor defaults to the cited TGO figure and is set in the
context bar on this page."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import energy as E
from sbanalytics import mv as M
from sbanalytics import portfolio as P
from sbanalytics import registry as R
import ui

c = ui.ctx()
as_of, s, e = c["as_of"], c["start"], c["end"]
EF_DEFAULT = R.conv("co2_kg_per_kwh")               # kgCO2e/kWh, TGO 2022
EF = float(c["ef"])                                 # set in the context bar on this page
CUSTOM = abs(EF - EF_DEFAULT) > 1e-9
EF_SRC = R.CONVENTIONS["co2_kg_per_kwh"][2]
AREA = float(c["area"])                             # per floor (assumption)
sel_floor = None if st.session_state.get("building_view", True) else c["floor"]
WIRED = R.wired_floors()
BASIS = f"F{WIRED[0]}–F{WIRED[-1]}"        # floors with a weather-normalised baseline
n_days = int((e - s).days) + 1
t = lambda kwh: kwh * EF / 1000                     # kWh -> tCO2e


@st.cache_resource(show_spinner="Fitting the weather-normalised baselines…")
def _model(floor: int) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, "total_kwh")


def _usable(floor: int, s_=None, e_=None) -> pd.DataFrame:
    d = _model(floor)["daily"]
    d = d[d.actual.notna() & d.expected.notna()]
    if s_ is not None:
        d = d[(d.date >= s_) & (d.date <= e_)]
    return d


h = ui.building_hourly()
ui.page_header("Carbon", "Carbon", f"Scope 2 · location-based | {c['range']}")

# --------------------------------------------------------------------------- #
# (1) tiles
# --------------------------------------------------------------------------- #
bk = P.building_kpis(h, s, e, ef=EF)
dev = pd.concat([_usable(f, s, e) for f in WIRED], ignore_index=True)
dev_kwh = float((dev.actual - dev.expected).sum()) if len(dev) else np.nan
exp_kwh = float(dev.expected.sum()) if len(dev) else np.nan
ts0, ts1 = D.window(as_of, 365)
ts0 = max(ts0, D.RECORD_START)
t12 = P.building_kpis(h, ts0, ts1, ef=EF)
total_area = float(c["building_area"])

k = st.columns(4)
with k[0]:
    ui.tile("Emissions · " + c["period"], f"{bk['tco2e']:,.1f} tCO₂e" if bk["kwh"] else "no data",
            sub=f"{bk['kwh']:,.0f} kWh × {EF:.4f} kgCO₂e/kWh · all 7 floors",
            help="Building submetered kWh (AC + lighting + plug, every floor) × the grid factor.")
with k[1]:
    if np.isfinite(dev_kwh) and exp_kwh:
        ui.tile("vs expected", f"{t(dev_kwh):+,.1f} tCO₂e",
                f"{'▲' if dev_kwh > 0 else '▼'} {dev_kwh / exp_kwh * 100:+.1f}% vs weather-normalised baseline",
                sub=f"{BASIS}, {dev.date.nunique()} usable day(s)",
                help="Actual − expected kWh on the wired floors' usable days (the Overview energy tile's "
                     "basis), × the grid factor. F1 (lobby) has no baseline.")
    else:
        ui.tile("vs expected", ui.DASH, sub=f"{BASIS} · no usable baseline days in this period")
with k[2]:
    inten = bk["tco2e"] * 1000 / total_area * 365 / n_days if bk["kwh"] else np.nan
    ui.tile("Intensity", f"{inten:,.1f} kgCO₂e/m²·yr" if np.isfinite(inten) else ui.DASH,
            sub=f"annualised from {n_days} day(s) · {total_area:,.0f} m² assumed",
            help=f"Building floor area {total_area:,.0f} m² (Assumptions; the dataset gives no "
                 "per-floor split). Submetered zone loads only, so not comparable with a whole-building "
                 "benchmark.")
with k[3]:
    ui.tile("Trailing 12 months", f"{t12['tco2e']:,.1f} tCO₂e",
            sub=f"{ui.day_txt(ts0)} – {ui.day_txt(ts1)}"
                + (" · record starts 1 Jul 2018" if ts0 == D.RECORD_START else ""),
            help="The 365 days to the as-of date, whatever the period control says.")

# --------------------------------------------------------------------------- #
# (2) monthly emissions
# --------------------------------------------------------------------------- #
ui.section("Monthly emissions", "Whole record · tCO₂e")
mo = P.monthly_energy(h)
mo["tCO2e"] = t(mo.kWh)
if sel_floor:
    mo["part"] = np.where(mo.floor == sel_floor, f"F{sel_floor}", "Rest of building")
else:
    mo["part"] = "Building"
bars_df = mo.groupby(["month", "part"], as_index=False).tCO2e.sum()
bars_df["order"] = (bars_df.part == "Rest of building").astype(int)
dom = [f"F{sel_floor}", "Rest of building"] if sel_floor else ["Building"]
rng = [ui.PRIMARY, ui.CONTEXT] if sel_floor else [ui.PRIMARY]

# Expected building = actual building − (office floors' actual − expected on usable days).
ud = pd.concat([_usable(f) for f in WIRED], ignore_index=True)
ud["month"] = ud.date.dt.to_period("M").dt.to_timestamp()
gap = ud.assign(d=ud.actual - ud.expected).groupby("month").d.sum()
tot = mo.groupby("month").tCO2e.sum()
ex = pd.DataFrame({"month": gap.index, "expected": (tot.reindex(gap.index) - t(gap)).values}).dropna()

base = alt.Chart(bars_df).encode(x=alt.X("yearmonth(month):T", title=None,
                                         axis=alt.Axis(format="%b %y", labelAngle=0, tickCount=18)))
asof_m = pd.Timestamp(as_of).replace(day=1)
layers = [
    alt.Chart(pd.DataFrame({"a": [asof_m], "b": [asof_m + pd.offsets.MonthEnd(1)]})).mark_rect(
        color=ui.RAISED, opacity=1).encode(x="a:T", x2="b:T"),
    base.mark_bar(size=22).encode(
        y=alt.Y("sum(tCO2e):Q", title="tCO₂e / month"),
        color=alt.Color("part:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        order=alt.Order("order:Q"),
        tooltip=["part", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("tCO2e:Q", format=",.1f")]),
    ui.expected_line(alt.Chart(ex)).encode(
        x="yearmonth(month):T", y="expected:Q",
        tooltip=[alt.Tooltip("month:T", format="%b %Y"),
                 alt.Tooltip("expected:Q", format=",.1f", title="expected tCO₂e")]),
]
for f in R.FILE_END:
    fe = R.floor_file_end(f)
    fe_df = pd.DataFrame({"d": [fe], "label": [f"F{f} file ends {fe.day} {fe:%b %Y}"]})
    layers += [alt.Chart(fe_df).mark_rule(color=ui.INK_3, strokeDash=[1, 3], strokeWidth=1.5).encode(x="d:T"),
               alt.Chart(fe_df).mark_text(color=ui.INK_3, align="left", dx=4, dy=4, baseline="top",
                                          fontSize=11).encode(x="d:T", y=alt.value(0), text="label:N")]
st.altair_chart(ui.chart(alt.layer(*layers), height=280), width="stretch")
st.caption(f"Grey column: the as-of month. Expected = building actual minus the {BASIS} deviation from their "
           "weather/calendar baselines; F1 (lobby) enters at actual.")

# --------------------------------------------------------------------------- #
# (3) breakdown
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2], gap="large")
with left:
    ui.md('<p class="cb-overline">By floor</p>')
    rows = []
    for f in sorted(R.FLOORS):
        kwh = E.floor_totals(h[h.floor == f], s, e)["total"]
        fe = R.floor_file_end(f)
        note = (ui.REPORTING_ONLY.get(f) or ("not monitored" if not R.FLOORS[f]["wired"] else ""))
        if fe is not None and e >= fe.normalize():
            note = f"file ends {fe.day} {fe:%b %y}"
        rows.append({"Floor": f"F{f}", "tCO₂e": t(kwh) if np.isfinite(kwh) else np.nan,
                     "Share %": np.nan, "kgCO₂e/m²·yr": t(kwh) * 1000 / AREA * 365 / n_days
                     if np.isfinite(kwh) else np.nan, "Note": note})
    bf = pd.DataFrame(rows)
    bf["Share %"] = bf["tCO₂e"] / bf["tCO₂e"].sum() * 100
    hi_row = bf.index[bf.Floor == f"F{sel_floor}"][0] if sel_floor else None
    st.dataframe(ui.glyph_style(bf, [], {"tCO₂e": "{:,.2f}", "Share %": "{:.1f}", "kgCO₂e/m²·yr": "{:,.1f}"},
                                highlight_row=hi_row),
                 hide_index=True, width="stretch", placeholder=ui.DASH,
                 column_config={"Note": st.column_config.Column(width="medium"),
                                "kgCO₂e/m²·yr": st.column_config.Column(
                                    help=f"Annualised, on {AREA:,.0f} m² per floor (building area ÷ 7).")})
    st.caption(f"Total {bf['tCO₂e'].sum():,.2f} tCO₂e. '—' = not metered in the period, not zero.")
with right:
    ui.md('<p class="cb-overline">By end use</p>')
    d = D.slice_hourly(h, s, e)
    eu = pd.DataFrame({"end_use": ["ac", "light", "plug"],
                       "tCO₂e": [t(d[col].sum()) for col in E.END_USES]})
    eu["Share %"] = eu["tCO₂e"] / eu["tCO₂e"].sum() * 100 if eu["tCO₂e"].sum() else np.nan
    eu["name"] = eu.end_use.map(ui.END_USE_NAME)
    st.altair_chart(ui.chart(alt.Chart(eu).mark_bar(size=26).encode(
        y=alt.Y("name:N", title=None, sort=list(eu.name)),
        x=alt.X("tCO₂e:Q", title="tCO₂e"),
        color=alt.Color("end_use:N", scale=ui.END_USE_COLOR, legend=None),
        tooltip=["name", alt.Tooltip("tCO₂e:Q", format=",.2f"), alt.Tooltip("Share %:Q", format=".1f")]),
        height=170), width="stretch")
    st.dataframe(eu[["name", "tCO₂e", "Share %"]].rename(columns={"name": "End use"}).style.format(
        {"tCO₂e": "{:,.2f}", "Share %": "{:.1f}"}, na_rep=ui.DASH), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# (4) reporting-year summary (ignores the period control)
# --------------------------------------------------------------------------- #
ui.section("Reporting-year summary", "Calendar years")
yr = []
for label, a, b, note in [
        ("2018 (Jul–Dec, part year)", "2018-07-01", "2018-12-31", "record starts 1 Jul 2018"),
        ("2019", "2019-01-01", "2019-12-31",
         "F6 to " + " ".join(f"{R.floor_file_end(f).day} {R.floor_file_end(f):%b}" for f in R.FILE_END)
         + " only"),
        ("Jul 2018 – Dec 2019", "2018-07-01", "2019-12-31", "whole record; a part year plus a full year")]:
    y = P.building_kpis(h, pd.Timestamp(a), pd.Timestamp(b), ef=EF)
    yr.append({"Year": label, "tCO₂e": y["tco2e"], "kWh": y["kwh"], "Meter coverage %": y["meter_pct"],
               "Note": note})
st.dataframe(pd.DataFrame(yr).style.format({"tCO₂e": "{:,.1f}", "kWh": "{:,.0f}",
                                            "Meter coverage %": "{:.1f}"}, na_rep=ui.DASH),
             hide_index=True, width="stretch",
             column_config={"Meter coverage %": st.column_config.Column(
                 help="Share of zone-hours with usable metering (after each floor's file end, not counted)."),
                 "Note": st.column_config.Column(width="medium")})

fc = st.columns([12, 1], vertical_alignment="center")
_name, _src = (EF_SRC.split(", ", 1) + [""])[:2]
_fac = (f"custom factor {EF:.4f} kgCO₂e/kWh (TGO 2022 default {EF_DEFAULT:g})" if CUSTOM
        else f"{_name} {EF:g} kgCO₂e/kWh ({_src})")
fc[0].caption(f"Scope 2, location-based · {_fac} · "
              "submetered zone loads only; excludes central plant, lifts and common areas.")
fc[1].page_link("views/8_about_data.py", label="About the data", icon=":material/arrow_forward:", icon_position="right")
