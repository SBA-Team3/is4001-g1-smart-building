"""Floor — the morning screen for one floor: data health, today's energy and
cost, the zones worst first, and the alerts to act on."""

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
floor, as_of, tariff = c["floor"], c["as_of"], c["tariff"]
start, end = c["start"], c["end"]
if not ui.require_wired(floor, allow_reporting=True):
    st.stop()

h = ui.hourly(floor)
H = ui.health(floor)
target = R.conv("overcooled_target_pct")
LO, HI = c["lo"], c["hi"]                  # comfort band from Assumptions
m0 = as_of.replace(day=1)


@st.cache_resource(show_spinner="Fitting the weather-normalised baseline…")
def _model(floor: int, target: str = "total_kwh") -> dict:
    return M.daily_model(D.load_hourly(floor), floor, target)


def _last_30_days() -> None:
    """Daily kWh by end use over the 30 days to the as-of date."""
    ui.section("The last 30 days", "Daily kWh by end use")
    dl = E.daily(h, as_of - pd.Timedelta(days=29), as_of)
    if not len(dl) or dl[E.END_USES].isna().all().all():
        st.info("No metered energy in these 30 days — the meters were offline.", icon=":material/info:")
    else:
        m = dl.melt(id_vars=["date", "is_weekend"], value_vars=E.END_USES,
                    var_name="end_use", value_name="kWh")
        m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
        m["order"] = m.end_use.map({"ac": 0, "light": 1, "plug": 2})
        bars = alt.Chart(m).mark_bar(stroke=ui.SURFACE, strokeWidth=1).encode(
            x=alt.X("yearmonthdate(date):O", title=None,
                    axis=alt.Axis(labelAngle=0, format="%-d %b", labelOverlap=True)),
            y=alt.Y("kWh:Q", title="kWh / day"),
            color=alt.Color("end_use:N", scale=ui.END_USE_COLOR,
                            legend=alt.Legend(orient="top", title=None, symbolType="square",
                                              labelExpr=f"{ui.END_USE_NAME}[datum.label]".replace("'", '"'))),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), "end_use",
                     alt.Tooltip("kWh:Q", format=",.0f")])
        # Holidays where the AC still ran: annotated in muted text, not a data colour.
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
            layers += [ann.mark_rule(color=ui.INK_3, strokeWidth=1, strokeDash=[1, 3]),
                       ann.mark_text(color=ui.INK_2, align="left", dx=6, dy=-4, baseline="top",
                                     fontWeight=500, fontSize=12).encode(text="label:N", y=alt.value(0))]
        st.altair_chart(ui.chart(alt.layer(*layers), height=240), width="stretch")


def _lobby() -> None:
    """A reporting-only floor (F1, the lobby): energy, cost and data health only —
    no comfort, no alerts, no baseline."""
    ui.page_header("Floor", ui.floor_name(floor), c["range"])
    end_ = ui.file_end_gate(floor, start, end)
    ui.status_notes(as_of, floor)

    day_tot = E.floor_totals(h, as_of, as_of)
    base = E.same_weekday_baseline(h, as_of)
    dem = E.demand_stats(h, start, end_)
    mtd_kwh = E.floor_totals(h, m0, as_of)["total"]
    metered = np.isfinite(day_tot["total"])
    k = st.columns(3)
    k[0].metric("Energy today", f"{day_tot['total']:,.0f} kWh" if metered else "no data",
                delta=(f"{base['delta_pct']:+.0f}% vs same weekday"
                       if metered and base.get("delta_pct") is not None
                       and np.isfinite(base.get("delta_pct", np.nan)) else None),
                delta_color="inverse", border=True,
                help="Compared with the mean of the same weekday over the previous 4 weeks.")
    k[1].metric("Cost MTD", ui.thb(mtd_kwh, tariff) if np.isfinite(mtd_kwh) else "no data", border=True,
                help=f"Energy × {tariff:.2f} ฿/kWh, {ui.day_txt(m0)} – {ui.day_txt(as_of)}.")
    k[2].metric("Peak demand", f"{dem['peak_kw']:,.0f} kW" if dem else "—", border=True,
                delta=f"at {pd.Timestamp(dem['peak_at']):%a %d %b %H:%M}" if dem else None,
                delta_color="off", delta_arrow="off",
                help=(f"Highest hourly-mean demand. Load factor {dem['load_factor']:.2f}." if dem
                      else "No metered hours in the period."))

    eu = E.end_use(h, start, end_, floor=floor)
    tbl = pd.DataFrame({"Zone": eu.zone, "AC kWh": eu.ac_kwh, "Lighting kWh": eu.light_kwh,
                        "Plug kWh": eu.plug_kwh, "Total kWh": eu.total_kwh,
                        "Note": [("" if R.has_channel(floor, z, "ac") else "no AC meter") for z in eu.zone]})
    st.session_state.setdefault("_floor_view", "Map")
    st.session_state["floor_view"] = st.session_state["_floor_view"]
    hl, hr = st.columns([5, 2], vertical_alignment="center")
    hl.markdown('<p class="cb-overline">Zones</p>', unsafe_allow_html=True)
    view = hr.segmented_control("View", ["Map", "Table"], key="floor_view", required=True,
                                label_visibility="collapsed", width="stretch",
                                on_change=lambda: st.session_state.__setitem__(
                                    "_floor_view", st.session_state["floor_view"]))
    if view == "Map":
        tot = dict(zip(eu.zone, eu.total_kwh))
        _, mc, _ = st.columns([1, 2, 1])            # full-width block: centre the map
        with mc:
            ui.zone_map(floor, as_of=as_of, neutral=True, height=330,
                        label_extra={z: (f"{v:,.0f} kWh" if pd.notna(v) else "no data")
                                     for z, v in tot.items()},
                        zone_tips={z: ("" if R.has_channel(floor, z, "ac") else "no AC meter; ")
                                   + f"{tot.get(z, float('nan')):,.0f} kWh in the period"
                                   for z in tot})
            ui.md('<p class="cb-cap cb-muted" style="text-align:center">Schematic, not to scale — traced '
                  'from CU-BEMS Fig. 2 (Methods). Total kWh per zone over the period.</p>')
    else:
        st.dataframe(tbl.style.format({c_: "{:,.0f}" for c_ in tbl.columns if c_.endswith("kWh")},
                                      na_rep=ui.DASH),
                     hide_index=True, width="stretch", placeholder=ui.DASH)
        st.caption("'—' is a meter that does not exist, never zero.")
    _last_30_days()


if floor in ui.REPORTING_ONLY:
    _lobby()
    st.stop()

ui.page_header("Floor", f"Floor {floor}", f"{c['range']} | Comfort band {LO:g}–{HI:g} °C")
end = ui.file_end_gate(floor, start, end)

# No data-health line under the subtitle: point status lives on Sensors.
reg = ui.register(floor, c)
ui.status_notes(as_of, floor)

# --- four tiles ------------------------------------------------------------- #
day_tot = E.floor_totals(h, as_of, as_of)
base = E.same_weekday_baseline(h, as_of)
dem = E.demand_stats(h, start, end)
ex = M.month_excess(_model(floor), as_of, tariff)
t_, l_ = ex["this"], ex["last"]
mtd_kwh = E.floor_totals(h, m0, as_of)["total"]
oc = K.overcooled_share(K.cooled_occupied(D.slice_hourly(h, m0, as_of)), by=(), lo=LO, hi=HI)
oc_now = float(oc.share_below.iloc[0] * 100) if len(oc) else np.nan

k = st.columns(4)
today_metered = np.isfinite(day_tot["total"])
k[0].metric("Energy today", f"{day_tot['total']:,.0f} kWh" if today_metered else "no data",
            delta=(f"{base['delta_pct']:+.0f}% vs same weekday"
                   if today_metered and base.get("delta_pct") is not None
                   and np.isfinite(base.get("delta_pct", np.nan))
                   else "no baseline" if today_metered else None),
            delta_color="inverse", border=True,
            help="Compared with the mean of the same weekday over the previous 4 weeks. "
                 "Deliberately not year-on-year: the floor ran 2–3 °C colder after the 2019 "
                 "maintenance window.")
per_day = lambda x: x["excess"] / x["days"] if x["days"] else np.nan
guard = ui.comparison_guard(l_["start"], t_["end"], floor, show=False) if t_["days"] and l_["days"] else [1]
lm_txt = ("Per usable day vs last month: comparison disabled (change-point or no data)." if guard
          else f"Per usable day: {per_day(t_):+,.0f} kWh/day vs {per_day(l_):+,.0f} last month.")
k[1].metric("Cost MTD", ui.thb(mtd_kwh, tariff) if np.isfinite(mtd_kwh) else "no data",
            delta=(f"{ui.signed_thb(t_['thb'])} vs expected" if t_["days"] and tariff else None),
            delta_color="inverse", border=True,
            help=f"Energy × {tariff:.2f} ฿/kWh, {ui.day_txt(m0)} – {ui.day_txt(as_of)}. Expected = "
                 f"the weather/calendar-normalised baseline over {t_['days']} usable days "
                 f"({t_['actual']:,.0f} kWh actual vs {t_['expected']:,.0f} expected). {lm_txt}"
                 + (" This month is inside the baseline's training window, so excess is ~0 by "
                    "construction." if t_["in_train"] else ""))
k[2].metric("Peak demand", f"{dem.get('peak_kw', float('nan')):,.0f} kW" if dem else "—", border=True,
            delta=f"at {pd.Timestamp(dem['peak_at']):%a %d %b %H:%M}" if dem else None,
            delta_color="off", delta_arrow="off",
            help=(f"Highest hourly-mean demand, {c['period_label']}. Load factor "
                  f"{dem.get('load_factor', float('nan')):.2f} (mean ÷ peak)." if dem else
                  "No metered hours in the period."))
k[3].metric(f"Cooled <{LO:g} °C MTD", f"{oc_now:.0f}%" if np.isfinite(oc_now) else "no sensor data",
            delta=(f"{'above' if oc_now > target else 'within'} {target:.0f}% target"
                   if np.isfinite(oc_now) else None),
            delta_color="inverse" if np.isfinite(oc_now) and oc_now > target else "off",
            delta_arrow="off", border=True,
            help="Share of cooled, occupied zone-hours below the band on comfort zones, weighted "
                 "by zone AC energy (proposal KPI 1). " + ui.REINSTALL_NOTE)

# --- zones (left) + active alerts (right) ----------------------------------- #
zc = C.zone_comfort(h, start=start, end=end, lo=LO, hi=HI, floor=floor)
cm = zc.set_index("zone") if len(zc) else pd.DataFrame()
eu = E.end_use(h, start, end, floor=floor).set_index("zone")
op = ui.sort_faults(reg[reg.is_open]) if len(reg) else reg

uncooled = ui.uncooled_zone_temps(floor, start, end)   # sensored zones without an AC meter
rows = []
for z in R.ZONES[floor]:
    has_ac = R.has_channel(floor, z, "ac")
    tp = R.point_for(floor, z, "temp")
    comfort = tp is not None and tp.role == R.ROLE_ZONE_TEMP
    r = cm.loc[z] if comfort and z in cm.index else None
    cold = float(r.cold_pct) if r is not None else np.nan
    if r is not None:
        status = (f"● overcooled" if cold > target else "▲ warm" if r.warm_pct > target
                  else "✓ in band")
    elif comfort and not has_ac:
        status = "ⓘ not AC-metered"          # sensor but no cooling meter (F2 z3)
    elif comfort:
        status = "○ no data"
    elif tp is not None:
        status = "ⓘ supply air"
    else:
        status = "ⓘ no sensor"
    nf = int(op.zone.astype(str).str.contains(z).sum()) if len(op) else 0
    rows.append({"Zone": z, "Status": status,
                 "Mean °C": float(r.mean_c) if r is not None else uncooled.get(z, np.nan),
                 "Cold h %": cold,
                 "AC kWh": eu.loc[z, "ac_kwh"] if has_ac and z in eu.index else np.nan,
                 "Alerts": nf, "Sensor note": ui.zone_note(floor, z),
                 "_k": (0, -cold) if np.isfinite(cold) else (1, -nf)})
zt = pd.DataFrame(rows)
zt = zt.iloc[sorted(range(len(zt)), key=lambda i: zt._k[i])].reset_index(drop=True)
z_order = zt.Zone.tolist()


@st.dialog("Zone detail", width="large")
def _zone_dialog(z: str) -> None:
    row = cm.loc[z] if z in cm.index else None
    st.subheader(ui.zone_label(floor, z))
    if ui.zone_note(floor, z):
        st.caption(ui.zone_note(floor, z).capitalize() + ".")
    d = st.columns(4)
    d[0].metric("Mean temperature", f"{row.mean_c:.1f} °C" if row is not None else "—", border=True)
    d[1].metric(f"Cold hours (<{LO:g} °C)", f"{row.cold_pct:.0f}%" if row is not None else "—",
                border=True)
    d[2].metric("RH", f"{row.rh:.0f}%" if row is not None and np.isfinite(row.rh) else "—", border=True)
    ac = eu.loc[z, "ac_kwh"] if R.has_channel(floor, z, "ac") and z in eu.index else np.nan
    d[3].metric("AC energy", f"{ac:,.0f} kWh" if np.isfinite(ac) else "not metered", border=True)
    zf = op[op.zone.astype(str).str.contains(z)] if len(op) else op
    ui.md('<p class="cb-overline" style="margin-top:8px">Options to consider</p>')
    if len(zf):
        for r in zf.itertuples():
            ui.md(f'<div class="cb-row" style="margin:6px 0">{ui.state_chip(r)}'
                  f'<span class="cb-strong">{ui.esc(r.title)}</span></div>'
                  f'<p class="cb-body">{ui.esc(r.action)}</p>')
    else:
        st.markdown("No open alerts in this period — no change needed.")
    if st.button("Open in Comfort", icon=":material/arrow_forward:", type="secondary"):
        st.session_state["comfort_zone"] = z
        st.switch_page("views/3_zones.py")


def _pick_zone() -> None:
    sel = st.session_state["zones_tbl"].selection.rows
    if sel:
        st.session_state["_zone_dialog"] = z_order[sel[0]]


left, right = st.columns([6, 5], gap="medium")
with left:
    # Map / Table, persisted in its own non-widget key (survives other pages).
    st.session_state.setdefault("_floor_view", "Map")
    st.session_state["floor_view"] = st.session_state["_floor_view"]
    hl, hr = st.columns([3, 2], vertical_alignment="center")
    hl.markdown('<p class="cb-overline">Zones</p>', unsafe_allow_html=True)
    fview = hr.segmented_control("View", ["Map", "Table"], key="floor_view", required=True,
                                 label_visibility="collapsed", width="stretch",
                                 on_change=lambda: st.session_state.__setitem__(
                                     "_floor_view", st.session_state["floor_view"]))
if fview == "Map":
    with left:
        COLD = f"% < {LO:g} °C"
        vals = pd.DataFrame({"zone": zt.Zone, COLD: zt["Cold h %"], "mean °C": zt["Mean °C"],
                             "AC kWh": zt["AC kWh"]})
        n_al = dict(zip(zt.Zone, zt.Alerts))
        ui.zone_map(floor, vals, COLD, as_of=as_of, height=300, zone_temps=uncooled,
                    tooltip_cols=("mean °C", COLD, "AC kWh"),
                    label_extra={z: f"{n} alert{'s' if n != 1 else ''}" for z, n in n_al.items() if n})
        st.caption("Schematic, not to scale. Switch to Table for each zone's detail.")
else:
    with left:
        st.dataframe(
            ui.glyph_style(zt.drop(columns="_k"), ["Status"],
                           {"Mean °C": "{:.1f}", "Cold h %": "{:.0f}", "AC kWh": "{:,.0f}"}),
            hide_index=True, width="stretch", key="zones_tbl", on_select=_pick_zone,
            selection_mode="single-row", placeholder=ui.DASH,
            column_config={"Cold h %": st.column_config.Column(
                help=f"Cooled, occupied hours below {LO:g} °C. Above {target:.0f}% reads ● overcooled."),
                "Sensor note": st.column_config.Column(width="medium")})
        st.caption("Click a row for the zone's detail. '—' is a point that does "
                   "not exist or has no data, never zero.")
_dz = st.session_state.pop("_zone_dialog", None)
if _dz:
    _zone_dialog(_dz)

with right:
    ui.md(f'<p class="cb-overline">Active alerts · {len(op)} open</p>')
    if not len(op):
        ui.md(f'<div class="cb-list"><div class="item">{ui.chip("ok", "No open alerts")}'
              f'<div class="txt cb-cap">Nothing raised on this floor in the period.</div></div></div>')
    else:
        # One bordered list with divided rows (not a card per alert). Scoped CSS: rows are
        # 44 px, the markdown block loses its own spacing so the chip and zone centre on
        # the buttons, and the buttons are compact. Tokens only, no hex.
        ui.md("""<style>
.st-key-alert_list{padding:0 !important;gap:0 !important}
.st-key-alert_list [class*="st-key-al_row_"]{padding:6px 12px;gap:0}
.st-key-alert_list > * + *{border-top:1px solid var(--color-border-default)}
.st-key-alert_list [class*="st-key-al_row_"] [data-testid="stHorizontalBlock"]{gap:8px;align-items:center}
.st-key-alert_list [class*="st-key-al_row_"] [data-testid="stMarkdownContainer"]:has(> .cb-row){margin:0}
.st-key-alert_list [class*="st-key-al_row_"] .cb-row{min-height:32px;align-items:center}
.st-key-alert_list [class*="st-key-al_row_"] .cb-row .cb-chip{align-self:center}
.st-key-alert_list [class*="st-key-al_row_"] button{min-height:32px;height:32px;padding:0 12px}
.st-key-alert_list [class*="st-key-al_row_"] button p{font-size:13px}
</style>""")
        with st.container(border=True, key="alert_list"):
            for r in op.head(5).itertuples():
                with st.container(key=f"al_row_{r.id}"):
                    a, b1, b2, b3 = st.columns([5.0, 2.2, 2.5, 1.6], vertical_alignment="center", gap="small")
                    # The fault-type chip and the zone; description, cost and age live on Alerts.
                    a.markdown(f'<div class="cb-row">{ui.state_chip(r)}'
                               f'<span class="cb-strong">{ui.esc(r.zone)}</span></div>',
                               unsafe_allow_html=True)
                    b1.button("Work order", key=f"wo_{r.id}", type="primary", width="stretch",
                              disabled=r.state == "In progress",
                              on_click=ui.set_fault_state, args=(r.id, "In progress", "work order raised"),
                              help="Sets the alert In progress with the note 'work order raised'.")
                    b2.button("Acknowledge", key=f"ack_{r.id}", width="stretch", disabled=r.state != "New",
                              on_click=ui.set_fault_state, args=(r.id, "Acknowledged"))
                    b3.button("Mute", key=f"mute_{r.id}", width="stretch", on_click=ui.mute_fault,
                              args=(r.id,), help="Hide this alert for this session. Alerts clear by "
                                                 "themselves once the condition stops.")
    st.page_link("views/5_faults.py", label=f"All {len(op)} open alerts", icon=":material/arrow_forward:")

_last_30_days()
