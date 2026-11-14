"""Overview — the whole building on one 1440×900 screen: energy, cost, which
sensors are down, and which floor and alert to look at first."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import energy as E
from sbanalytics import kpis as K
from sbanalytics import mv as M
from sbanalytics import portfolio as P
from sbanalytics import registry as R
import ui

c = ui.ctx()
as_of, tariff, ef = c["as_of"], c["tariff"], c["ef"]
s, e = c["start"], c["end"]
LO, HI = c["lo"], c["hi"]                  # comfort band from Assumptions
target = R.conv("overcooled_target_pct")
sel_floor = None if st.session_state.get("building_view", True) else c["floor"]
WIRED = R.wired_floors()


@st.cache_resource(show_spinner="Fitting the weather-normalised baselines…")
def _model(floor: int) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, "total_kwh")


def _vs_expected(floor: int, s_, e_) -> tuple[float, float]:
    """(actual, expected) kWh over the usable days of the period, from the floor's baseline."""
    d = _model(floor)["daily"]
    d = d[(d.date >= s_) & (d.date <= e_) & d.actual.notna() & d.expected.notna()]
    return (float(d.actual.sum()), float(d.expected.sum())) if len(d) else (np.nan, np.nan)


def _cool_share(h_, s_, e_) -> float:
    o = K.overcooled_share(K.cooled_occupied(D.slice_hourly(h_, s_, e_)), by=(), lo=LO, hi=HI)
    return float(o.share_below.iloc[0] * 100) if len(o) else np.nan


h = ui.building_hourly()
ui.page_header("Overview", "Chulalongkorn University CU-BEMS", f"7 floors | 33 zones | {c['range']}")

# every wired floor's register over the one global period (same as Floor and Alerts)
reg = ui.building_register(c)
ps = ui.building_point_status(as_of)
past_end = [f for f in R.FILE_END if as_of >= R.floor_file_end(f).normalize()]
lo_b, hi_b = (pd.Timestamp(x) for x in R.BLACKOUT)

# --------------------------------------------------------------------------- #
# per-floor numbers for the period
# --------------------------------------------------------------------------- #
rows = []
act_w = exp_w = 0.0
for f in WIRED:
    hf = h[h.floor == f]
    tot = E.floor_totals(hf, s, e, f)["total"]
    a_, x_ = _vs_expected(f, s, e)
    if np.isfinite(a_):
        act_w, exp_w = act_w + a_, exp_w + x_
    rf = reg[reg.floor == f] if len(reg) else reg
    op = rf[rf.is_open] if len(rf) else rf
    psf = ps[ps.floor == f]
    down = int(psf.is_down.sum())
    sl = D.slice_hourly(hf, s, e)
    meter_pct = float(sl.kwh_ok.mean() * 100) if len(sl) else np.nan
    no_data = (f in past_end and s >= R.floor_file_end(f).normalize()) or not np.isfinite(tot)
    fe = R.floor_file_end(f)
    notes = (["no data"] if no_data else []) + (
        [f"file ends {fe.day} {fe:%b %y}"] if fe is not None else
        ["meter gaps"] if np.isfinite(meter_pct) and meter_pct < R.conv("completeness_target") else [])
    cool = _cool_share(hf, s, e)
    t_label, t_n = ui.top_type(op)
    stake = float(op.cost_per_day_thb.sum()) if len(op) else 0.0
    rows.append({"Floor": f"F{f}", "kWh": tot,
                 "฿": tot * tariff if tariff and np.isfinite(tot) else np.nan,
                 "vs exp %": (a_ - x_) / x_ * 100 if np.isfinite(a_) and x_ else np.nan,
                 f"<{LO:g} °C %": cool, "Alerts": len(op),
                 "Top type": f"{t_label} · {t_n}" if t_n else "", "Down": down,
                 "Note": " · ".join(notes),
                 "_nodata": int(no_data), "_stake": stake, "_f": f})
# Ranked by ฿ at stake per day across open alerts; floors with no data sink to the bottom.
ft = pd.DataFrame(rows).sort_values(["_nodata", "_stake"], ascending=[True, False])
# F1 (lobby, reporting only): in the totals, out of every ranking.
extra = []
for f_, note_, go in ((1, ui.REPORTING_ONLY[1], 1),):
    kwh_ = E.floor_totals(h[h.floor == f_], s, e)["total"]
    extra.append({"Floor": f"F{f_}", "kWh": kwh_,
                  "฿": kwh_ * tariff if tariff and np.isfinite(kwh_) else np.nan,
                  "vs exp %": np.nan, f"<{LO:g} °C %": np.nan, "Alerts": np.nan, "Top type": "",
                  "Down": np.nan, "Note": note_, "_f": go})   # F1's row opens its Floor page
ft = pd.concat([ft, pd.DataFrame(extra)], ignore_index=True)

# --------------------------------------------------------------------------- #
# three tiles
# --------------------------------------------------------------------------- #
bk = P.building_kpis(h, s, e, tariff, ef, lo=LO, hi=HI)
dev_pct = (act_w - exp_w) / exp_w * 100 if exp_w else np.nan
t1, t2, t3 = st.columns(3)
with t1:
    if np.isfinite(dev_pct):
        delta = (f"{'▲' if dev_pct > 0 else '▼'} {dev_pct:+.1f}% vs expected "
                 f"({act_w - exp_w:+,.0f} kWh, F{WIRED[0]}–F{WIRED[-1]})")
    else:
        # No usable baseline days: fall back to the same span last month, if comparable.
        ps0, pe0 = s - pd.DateOffset(months=1), e - pd.DateOffset(months=1)
        prev = P.building_kpis(h, ps0, pe0)["kwh"]
        blocked = any(ui.comparison_guard(ps0, e, f, show=False) for f in WIRED)
        delta = ("no baseline; comparison blocked by a change-point" if blocked or not prev
                 else f"{(bk['kwh'] - prev) / prev * 100:+.1f}% vs same span last month")
    ui.tile("Energy · " + c["period"], f"{bk['kwh']:,.0f} kWh" if bk["kwh"] else "no data", delta,
            sub=f"{bk['days']} day(s) · submetered zone loads, all 7 floors",
            help="AC + lighting + plug on every floor, including F1 and F2. Expected = each wired "
                 "floor's weather- and calendar-normalised baseline (Energy page).")
with t2:
    ex_thb = (act_w - exp_w) * tariff if np.isfinite(dev_pct) else np.nan
    ui.tile("Cost · " + c["period"], ui.thb(bk["kwh"], tariff),
            (f"{ui.signed_thb(ex_thb)} vs expected" if tariff and np.isfinite(ex_thb) else ""),
            help=f"Energy × {tariff:.2f} ฿/kWh.")
with t3:
    live = ps[ps.status != "NO FILE"]
    down = live[live.is_down]
    ui.tile("Sensors down · today", f"{len(down)} of {len(live)} points",
            help="Down = OFFLINE, FROZEN or BAD SAMPLES on the as-of day (Sensors page). PARTIAL "
                 "points are degraded, not down. Floors past their file end are not counted.")
    st.page_link("views/2_sensor_health.py", label="Sensors", icon=":material/arrow_forward:")

# --------------------------------------------------------------------------- #
# floors table + top alerts
# --------------------------------------------------------------------------- #
left, right = st.columns([7, 5], gap="medium")


def _open_floor(order: list[int]) -> None:
    rows_ = st.session_state["floors_tbl"].selection.rows
    if rows_ and order[rows_[0]]:
        st.session_state["floor_select"] = f"F{order[rows_[0]]}"
        st.session_state["floor"] = order[rows_[0]]
        st.session_state["_goto"] = "views/1_floor_status.py"


with left:
    ui.md('<p class="cb-overline">Floors</p>')
    show = ft.drop(columns=[x for x in ft.columns if x.startswith("_")])
    hi_row = ft.index[ft._f == sel_floor][0] if sel_floor in ft._f.values else None
    show.index = ft.index
    _order = list(ft._f)
    st.dataframe(
        ui.glyph_style(show, [], {
            "kWh": "{:,.0f}", "฿": "{:,.0f}", "vs exp %": "{:+.1f}",
            f"<{LO:g} °C %": "{:.0f}", "Alerts": "{:.0f}", "Down": "{:.0f}"},
            highlight_row=hi_row),
        hide_index=True, width="stretch", key="floors_tbl", placeholder=ui.DASH,
        on_select=lambda: _open_floor(_order), selection_mode="single-row",
        column_config={
            "Floor": st.column_config.Column(width=58),
            "kWh": st.column_config.Column(width=66),
            "Alerts": st.column_config.Column(width=54, help="Open alerts in the period."),
            "Top type": st.column_config.Column(width=128, help="Most common open alert type."),
            "Note": st.column_config.Column(width=168),
            "฿": st.column_config.Column(help="Energy × tariff. Blank when the rate is 0."),
            "vs exp %": st.column_config.Column(
                help="Actual vs the floor's weather/calendar-normalised baseline over the period's "
                     "usable days. Positive = used more than expected."),
            f"<{LO:g} °C %": st.column_config.Column(
                help=f"Share of cooled, occupied zone-hours below {LO:g} °C (target < {target:.0f}%). "
                     "Only hours with live sensors count."),
            "Down": st.column_config.Column(width=50, help="Points OFFLINE, FROZEN or BAD SAMPLES today."),
        })
    st.caption("**Floor 1 is the lobby** (runs around the clock); it is in the building totals and out of "
               "every ranking. "
               + (f"F6's file ends {R.floor_file_end(6):%d %b %Y}: after that it is not metered, "
                  "not zero. " if 6 in R.FILE_END else "")
               + (f"This period overlaps the building-wide sensor gap ({lo_b:%d %b %Y} – "
                  f"{hi_b:%d %b %Y}), so the cooled-hours share is blank or partial."
                  if s <= hi_b and e >= lo_b else ""))

with right:
    ui.md('<p class="cb-overline">Top alerts</p>')
    op = ui.sort_faults(reg[reg.is_open]) if len(reg) else reg
    if not len(op):
        ui.md(f'<div class="cb-list"><div class="item">{ui.chip("ok", "No open alerts")}'
              f'<div class="txt cb-cap">Nothing raised on floors F{WIRED[0]}–F{WIRED[-1]} '
              f'in this period.</div></div></div>')
    else:
        # One divided list, one compact row per alert: chip · title · cost and age, centred
        # on the row (the title wraps to at most two lines; full text on hover).
        items = []
        for r in op.head(5).itertuples():
            cost = (f"฿{r.cost_per_day_thb:,.0f}/day" if r.cost_per_day_thb and tariff else ui.DASH)
            days = f" · {r.days_open:.0f} d open" if np.isfinite(r.days_open) else ""
            title = f"F{r.floor} {r.title}".replace(" °C", "\u00a0°C")   # keep "23 °C" together
            items.append(f'<div class="item">{ui.state_chip(r)}'
                         f'<span class="cb-strong txt" title="{ui.esc(title)}">{ui.esc(title)}</span>'
                         f'<span class="cost">{cost}<span class="cb-cap cb-muted">{days}</span></span></div>')
        ui.md(f'<div class="cb-list compact">{"".join(items)}</div>')
    st.page_link("views/5_faults.py", label=f"All {len(op)} open alerts",
                 icon=":material/arrow_forward:")

# --------------------------------------------------------------------------- #
# below the fold
# --------------------------------------------------------------------------- #
ui.section("Building energy by month", "Whole record · kWh")
mo = P.monthly_energy(h)
if sel_floor:
    # A floor is selected: it in data/primary, the rest of the building in data/context.
    mo["part"] = np.where(mo.floor == sel_floor, f"F{sel_floor}", "Rest of building")
    mo = mo.groupby(["month", "part"], as_index=False).kWh.sum()
    mo["order"] = (mo.part == "Rest of building").astype(int)
    color = alt.Color("part:N", scale=alt.Scale(domain=[f"F{sel_floor}", "Rest of building"],
                                                range=[ui.PRIMARY, ui.CONTEXT]),
                      legend=alt.Legend(orient="top", title=None, symbolType="square"))
    tip = ["part"]
else:
    # Bldg: stacked by floor, one categorical hue per floor (Overview only; DESIGN.md).
    mo["part"] = "F" + mo.floor.astype(str)
    mo["order"] = mo.floor
    color = alt.Color("part:N", scale=ui.FLOOR_CAT,
                      legend=alt.Legend(orient="top", title=None, symbolType="square", columns=7))
    tip = [alt.Tooltip("part:N", title="floor")]
st.altair_chart(ui.chart(alt.Chart(mo).mark_bar(stroke=ui.SURFACE, strokeWidth=1).encode(
    x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(labelAngle=0, format="%b %y")),
    y=alt.Y("sum(kWh):Q", title="kWh / month",
            axis=alt.Axis(format="~s", labelExpr="datum.value == 0 ? '0' : datum.label")),
    color=color, order=alt.Order("order:Q"),
    tooltip=tip + [alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("kWh:Q", format=",.0f")]),
    height=260), width="stretch")
st.caption("Floor 6 has nothing after October 2019 because its file ends, not because it used nothing.")

