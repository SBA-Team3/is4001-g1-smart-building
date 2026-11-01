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
from sbanalytics import workorders as W
import ui

c = ui.ctx()
floor, as_of, win = c["floor"], c["as_of"], c["window"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)
start, end = D.window(as_of, win)
target = R.conv("overcooled_target_pct")
LO = R.conv("comfort_lo")


@st.cache_resource(show_spinner="Fitting the weather-normalised baseline…")
def _model(floor: int) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, "ac_kwh")


def _day(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.day} {t:%b %Y}"


ui.page_header(f"Overview · Floor {floor}", f"Floor {floor} — status",
               f"As of {_day(as_of)} · trailing {win} day(s) ({start.day} {start:%b} – "
               f"{end.day} {end:%b}). Can I trust today's data, and what changed since yesterday?")

# --- 1. trust band ---------------------------------------------------------- #
phi = ui.trust_band(floor, as_of)
ui.status_notes(as_of, floor)
ui.file_end_note(floor, as_of)

# --- 2. the proposal's two KPIs + since yesterday --------------------------- #
ui.section("Against target", "The proposal's two KPIs, this month to date.")
m0 = as_of.replace(day=1)
p0 = (m0 - pd.Timedelta(days=1)).replace(day=1)


def _oc(s_, e_):
    o = K.overcooled_share(K.cooled_occupied(D.slice_hourly(h, s_, e_)), by=())
    return float(o.share_below.iloc[0] * 100) if len(o) else np.nan


oc_now, oc_prev = _oc(m0, as_of), _oc(p0, m0 - pd.Timedelta(days=1))
ex = M.month_excess(_model(floor), as_of, c["tariff"])
t, l_ = ex["this"], ex["last"]
k = st.columns(4)
k[0].metric(f"Cooled hours below {LO:g} °C, month to date",
            f"{oc_now:.0f}%" if np.isfinite(oc_now) else "no sensor data",
            delta=f"{oc_now - target:+.0f} pp vs {target:.0f}% target" if np.isfinite(oc_now) else None,
            delta_color="inverse", border=True,
            help="Proposal KPI 1: share of cooled, occupied zone-hours below the band on comfort "
                 "zones, weighted by zone AC energy. "
                 + (f"Last month {oc_prev:.0f}%." if np.isfinite(oc_prev) else "No data last month."))
per_day = lambda x: x["excess"] / x["days"] if x["days"] else np.nan
k[1].metric("Excess cooling, month to date",
            f"{t['excess']:+,.0f} kWh" if t["days"] else "no data",
            delta=f"{ui.signed_thb(t['thb'])} vs expected" if t["days"] else None, delta_color="inverse",
            border=True,
            help=f"Proposal KPI 2: actual − weather/calendar-normalised expected AC energy, "
                 f"{t['start'].date()} – {t['end'].date()} ({t['days']} usable days). Actual "
                 f"{t['actual']:,.0f} kWh vs expected {t['expected']:,.0f} kWh. Target: lower every "
                 "month.")
guard = ui.comparison_guard(l_["start"], t["end"], floor) if t["days"] and l_["days"] else []
if guard or not (t["days"] and l_["days"]):
    k[2].metric("vs last month, per day", "—", border=True,
                help="Disabled: the two months straddle a change-point, or one has no usable days.")
else:
    k[2].metric("vs last month, per day", f"{per_day(t):+,.0f} kWh/day",
                delta=f"{per_day(t) - per_day(l_):+,.0f} vs {per_day(l_):+,.0f} last month",
                delta_color="inverse", border=True,
                help="Per usable day, so a part month compares fairly with a full one.")

y_start, y_end = D.window(as_of - pd.Timedelta(days=1), win)
reg_y = F.build_register(h, H, start=y_start, end=y_end, floor=floor, tariff=c["tariff"])
reg_t = F.build_register(h, H, start=start, end=end, floor=floor, tariff=c["tariff"],
                         as_of=as_of)
diff = F.register_diff(reg_y, reg_t)
lost = HL.went_offline(H, as_of)
with k[3]:
    ui.stat_card("Since yesterday", [(len(diff["new"]), "new faults"),
                                     (len(diff["cleared"]), "cleared"),
                                     (len(lost), "went offline")])
detail = ([f"new: `{x}`" for x in diff["new"]] + [f"cleared: `{x}`" for x in diff["cleared"]]
          + [f"offline: `{x}`" for x in lost.column])
if detail:
    st.caption(" · ".join(detail))
if t["in_train"]:
    st.caption("This month is inside the baseline's training window, where excess is close to "
               "zero by construction — read it as a check that the model fits, not a result.")

# --- 3. today --------------------------------------------------------------- #
ui.section("How is the floor running?", f"Today, and the trailing {win}-day window.")

day_tot = E.floor_totals(h, as_of, as_of)
base = E.same_weekday_baseline(h, as_of)
dem = E.demand_stats(h, start, end)
zc = C.zone_comfort(h, start=start, end=end)
# Open faults counted exactly as the Faults page counts them (no planned items,
# nothing already resolved or muted), so the link below lands on the same number.
reg = ui.with_states(reg_t)
reg = reg[reg.is_open] if len(reg) else reg

k = st.columns(5)
today_metered = np.isfinite(day_tot["total"])
k[0].metric("Energy today", f"{day_tot['total']:,.0f} kWh" if today_metered else "no data",
            delta=(f"{base['delta_pct']:+.0f}% vs same weekday"
                   if today_metered and base.get("delta_pct") is not None
                   and np.isfinite(base.get("delta_pct", np.nan))
                   else "no baseline" if today_metered else None),
            delta_color="inverse", border=True,
            help="Compared with the mean of the same weekday over the previous "
                 "4 weeks. Deliberately not year-on-year: the floor ran 2–3 °C "
                 "colder after the 2019 maintenance window, so a year-on-year "
                 "comparison would be measuring a configuration change.")
k[1].metric("Peak demand", f"{dem.get('peak_kw', float('nan')):,.0f} kW" if dem else "—", border=True,
            help=f"In the trailing window. Load factor "
                 f"{dem.get('load_factor', float('nan')):.2f}." if dem else None)
k[2].metric("AC share", f"{day_tot['ac_share']:.0f}%" if np.isfinite(day_tot['ac_share']) else "—",
            border=True)
out_of_band = int((zc.cold_hours + zc.warm_hours > 0).sum()) if len(zc) else 0
k[3].metric("Zones outside band", f"{out_of_band} / {len(zc)}" if len(zc) else "—", border=True,
            help=f"Declared band {C.band()[0]:g}–{C.band()[1]:g} °C.")
k[4].metric("Open faults", len(reg),
            delta=f"{int((reg.priority == F.HIGH).sum())} high" if len(reg) else None,
            delta_color="inverse", border=True)

# --- 4. zones --------------------------------------------------------------- #
ui.section("Zones", "Worst first. Read-only: each card recommends a change instead of offering "
                    "a control.")

day_e = E.end_use(h, start, end, floor=floor).set_index("zone")
cm = zc.set_index("zone") if len(zc) else pd.DataFrame()
PRIO = {F.HIGH: 0, F.MED: 1, F.LOW: 2}


def _zone_faults(z: str) -> pd.DataFrame:
    if not len(reg):
        return reg
    return reg[reg.zone.astype(str).str.contains(z)].sort_values(
        "priority", key=lambda s: s.map(PRIO))


def _raise_order(fid: str) -> None:
    W.set_state(ui.FAULT_STATES, fid, "Acknowledged", "work order raised from Floor status")


cards = []
for z in R.ZONES[floor]:
    has_ac = R.has_channel(floor, z, "ac")
    tp = R.point_for(floor, z, "temp")
    comfort = tp is not None and tp.role == R.ROLE_ZONE_TEMP
    supply = tp is not None and tp.role == R.ROLE_SUPPLY_TEMP
    row = cm.loc[z] if z in cm.index else None
    zf = _zone_faults(z)
    ac = day_e.loc[z, "ac_kwh"] if has_ac and z in day_e.index else np.nan
    light = day_e.loc[z, "light_kwh"] if z in day_e.index else np.nan
    title = f"Zone {z[1:]}" + (" · stairwell" if ui.is_stairwell(floor, z) else "")

    cold = row.cold_pct if comfort and row is not None else np.nan
    if comfort and row is not None:
        status = (ui.dot("critical", f"Cooling below {LO:g} °C") if cold > target
                  else ui.dot("warning", "Running warm") if row.warm_pct > target
                  else ui.dot("good", "In band"))
        ic, temp, big = "thermostat", f"{row.mean_c:.1f} °C", True
    elif comfort:
        status, ic, temp, big = ui.dot("offline", "No data in window"), "thermostat", "No data", False
    elif supply:
        status, ic, temp, big = ui.dot("good", "Reporting"), "ac_unit", "Supply air", False
    else:
        status, ic, temp, big = (ui.dot("offline", "No AC meter" if not has_ac else "No sensor"),
                                 "sensors", "No sensor", False)

    if len(zf):
        top = zf.iloc[0]
        chip = ui.chip(top.priority, top.priority.capitalize() if len(zf) == 1 else f"{len(zf)} faults")
        rec = top.action
    else:
        chip = ui.chip("ok", "No faults")
        rec = (ui.zone_note(floor, z).capitalize() + "." if ui.zone_note(floor, z)
               else "No open faults in this window — no change needed.")
    if supply and not len(zf):
        rec = ("Sensor sits in the supply-air path: a supply-air diagnostic, not a room "
               "temperature.")
    rh = (f"RH {row.rh:.0f}%" if comfort and row is not None and np.isfinite(row.rh)
          else f"{light:,.0f} kWh light" if np.isfinite(light) else "—")
    body = ui.zone_card_body(
        title, status, chip, ic, f"Mean temp, {win} day(s)", temp, big,
        f"{cold:.0f}%" if np.isfinite(cold) else "—", np.isfinite(cold) and cold > target,
        f"{ac:,.0f} kWh" if np.isfinite(ac) else "Not metered", rh, rec)
    # Worst first: comfort zones by cold share, then the rest.
    cards.append(((0, -cold) if np.isfinite(cold) else (1, 0), z, body, zf))

cards.sort(key=lambda x: x[0])
for i, (_, z, body, zf) in enumerate(cards):
    if i % 3 == 0:
        cols = st.columns(3)
    with cols[i % 3]:
        with ui.card(f"zone_{z}"):
            ui.md(body)
            if len(zf):
                top = zf.iloc[0]
                if top.state == "New":
                    st.button("Create work order", key=f"wo_{z}", icon=":material/arrow_forward:",
                              on_click=_raise_order, args=(top.id,),
                              help=f"Marks {top.id} Acknowledged on the Faults page.")
                else:
                    st.caption(f"Work order: {top.state.lower()} · `{top.id}`")

with st.expander("Zones as a table", icon=":material/table:"):
    nfaults = reg.groupby("zone").size().to_dict() if len(reg) else {}
    rows = []
    for z in R.ZONES[floor]:
        has_ac = R.has_channel(floor, z, "ac")
        tp = R.point_for(floor, z, "temp")
        comfort_valid = tp is not None and tp.role == R.ROLE_ZONE_TEMP
        rows.append({
            "zone": ui.zone_label(floor, z),
            "temp °C": (cm.loc[z, "mean_c"] if z in cm.index else np.nan) if comfort_valid else np.nan,
            "RH %": (cm.loc[z, "rh"] if z in cm.index else np.nan) if comfort_valid else np.nan,
            "AC kWh": day_e.loc[z, "ac_kwh"] if has_ac and z in day_e.index else np.nan,
            "light kWh": day_e.loc[z, "light_kwh"] if z in day_e.index else np.nan,
            "cold h %": (cm.loc[z, "cold_pct"] if z in cm.index else np.nan) if comfort_valid else np.nan,
            "exceedance °C·h": (cm.loc[z, "edh"] if z in cm.index else np.nan) if comfort_valid else np.nan,
            "faults": sum(v for kk, v in nfaults.items() if z in str(kk)),
            "note": ui.zone_note(floor, z),
        })
    st.dataframe(
        pd.DataFrame(rows).round(1), width="stretch", hide_index=True,
        column_config={
            "AC kWh": st.column_config.NumberColumn(format="%.0f"),
            "light kWh": st.column_config.NumberColumn(format="%.0f"),
            "cold h %": st.column_config.ProgressColumn(color=ui.ACTUAL, label=
                "cold h %", min_value=0, max_value=100, format="%.0f"),
        })
    st.caption("Blank cells are points that do not exist, not zeros — the note column says which. "
               "Sorted by severity: overcooling degree-hours weighted above overheating, because in "
               "cooling mode a cold hour is a comfort fault *and* wasted energy.")
st.caption("Blank or '—' readings are points that do not exist, not zeros. "
           + ("On this floor z3 is a stairwell with no cooling and no sensors; z1's temperature "
              "sensor sits in a supply-air path, so it has no valid room-comfort figure."
              if floor == 5 else ""))

# --- 5. what needs attention ------------------------------------------------ #
ui.section("What needs attention", "Top three by priority, then cost of inaction.")
if not len(reg):
    st.success("No open faults in this window.", icon=":material/check_circle:")
else:
    items = []
    for _, r in reg.head(3).iterrows():
        cost = f"฿{r.cost_per_day_thb:,.0f}/day" if r.cost_per_day_thb else "—"
        items.append(f'<div class="item">{ui.chip(r.priority)}<div class="txt">'
                     f'<div class="cb-strong">{ui.esc(r.title)}</div>'
                     f'<div class="cb-cap">{ui.esc(r.detail)}</div></div>'
                     f'<div class="cost">{cost}</div></div>')
    ui.md(f'<div class="cb-list">{"".join(items)}</div>')
    ui.faults_link(len(reg))

# --- 6. the month ----------------------------------------------------------- #
with ui.card("month"):
    ui.section("The last 30 days", "Daily energy by end use, kWh. Weekends and holidays should sit "
                                   "near zero — except plug load.")
    dl = E.daily(h, as_of - pd.Timedelta(days=29), as_of)
    if not len(dl) or dl[E.END_USES].isna().all().all():
        st.info("No metered energy in these 30 days — the meters were offline.", icon=":material/info:")
    else:
        m = dl.melt(id_vars=["date", "is_weekend"], value_vars=E.END_USES,
                    var_name="end_use", value_name="kWh")
        m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
        m["order"] = m.end_use.map({"ac": 0, "light": 1, "plug": 2})
        names = {"ac": "Air-conditioning", "light": "Lighting", "plug": "Plug load"}
        bars = alt.Chart(m).mark_bar(stroke=ui.SURFACE, strokeWidth=1).encode(
            x=alt.X("yearmonthdate(date):O", title=None,
                    axis=alt.Axis(labelAngle=0, format="%-d %b", labelOverlap=True)),
            y=alt.Y("kWh:Q", title="kWh / day"),
            color=alt.Color("end_use:N", scale=ui.END_USE_COLOR,
                            legend=alt.Legend(orient="top", title=None, symbolType="square",
                                              labelExpr=f"{names}[datum.label]".replace("'", '"'))),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), "end_use",
                     alt.Tooltip("kWh:Q", format=",.0f")])
        # Holidays where the AC still ran: the one thing on this chart worth pointing at.
        hol = (h[(h.ts >= as_of - pd.Timedelta(days=29)) & (h.ts < as_of + pd.Timedelta(days=1))
                 & h.is_holiday]
               .groupby(["date", "holiday_name"], as_index=False).ac_kwh.sum(min_count=1))
        hol = hol[hol.ac_kwh > 1]
        layers = [bars]
        if len(hol):
            hol["date"] = pd.to_datetime(hol.date)
            hol["label"] = [f"{d.day} {d:%b} · {n} — AC still ran {k_:,.0f} kWh"
                            for d, n, k_ in zip(hol.date, hol.holiday_name, hol.ac_kwh)]
            ann = alt.Chart(hol).encode(x="yearmonthdate(date):O")
            layers += [ann.mark_rule(color=ui.WARN, strokeWidth=1),
                       ann.mark_text(color=ui.WARN, align="left", dx=6, dy=-4, baseline="top",
                                     fontWeight=500, fontSize=12).encode(
                           text="label:N", y=alt.value(0))]
        st.altair_chart(ui.chart(alt.layer(*layers), height=240), width="stretch")
