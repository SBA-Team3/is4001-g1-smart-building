"""Energy — use and cost over the period against the weather-normalised
expectation (ISO 50006 / IPMVP), G14 model quality, energy signature and
load profiles."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import energy as E
from sbanalytics import mv as M
from sbanalytics import registry as R
import ui

c = ui.ctx()
floor, tariff, area = c["floor"], c["tariff"], c["area"]
s, e = c["start"], c["end"]
if not ui.require_wired(floor, allow_reporting=True):
    st.stop()
LOBBY = floor in ui.REPORTING_ONLY          # reporting only: energy and cost, no baseline

h = ui.hourly(floor)


@st.cache_resource(show_spinner="Fitting the weather-normalised baseline…")
def _model(floor: int, target: str) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, target)


ui.page_header("Energy", ui.floor_name(floor), c["range"])
e = ui.file_end_gate(floor, s, e)

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
tot = E.floor_totals(h, s, e, floor)
dem = E.demand_stats(h, s, e)
if LOBBY:
    metered = np.isfinite(tot["total"])
    k = st.columns(3)
    k[0].metric("Energy", f"{tot['total']:,.0f} kWh" if metered else "no data", border=True,
                help=f"Submetered zone loads (AC + lighting + plug), {c['period_label']}.")
    k[1].metric("Cost", ui.thb(tot["total"], tariff) if metered else "no data", border=True,
                help=f"Energy × {tariff:.2f} ฿/kWh.")
    k[2].metric("Peak demand", f"{dem['peak_kw']:,.0f} kW" if dem else "—", border=True,
                delta=f"at {pd.Timestamp(dem['peak_at']):%a %d %b %H:%M}" if dem else None,
                delta_color="off", delta_arrow="off",
                help=f"Hourly-mean peak. Load factor {dem['load_factor']:.2f}." if dem else None)
    st.caption("Floor 1 (lobby) is reported for energy and cost only; it is not compared with an "
               "office baseline.")
else:
    mt = _model(floor, "total_kwh")
    dd = mt["daily"]
    dd = dd[(dd.date >= s) & (dd.date <= e) & dd.actual.notna() & dd.expected.notna()]
    act, expd = (dd.actual.sum(), dd.expected.sum()) if len(dd) else (np.nan, np.nan)

    metered = np.isfinite(tot["total"])
    k = st.columns(4)
    k[0].metric("Energy", f"{tot['total']:,.0f} kWh" if metered else "no data", border=True,
                help=f"Submetered zone loads (AC + lighting + plug), {c['period_label']}.")
    k[1].metric("Cost", ui.thb(tot["total"], tariff) if metered else "no data", border=True,
                help=f"Energy × {tariff:.2f} ฿/kWh.")
    if len(dd) and expd:
        k[2].metric("฿ vs expected", ui.signed_thb((act - expd) * tariff) if tariff else
                    f"{act - expd:+,.0f} kWh",
                    delta=f"{(act - expd) / expd * 100:+.1f}% vs expected", delta_color="inverse",
                    border=True,
                    help=f"Actual {act:,.0f} kWh vs expected {expd:,.0f} kWh over {len(dd)} usable days: "
                         "the weather- and calendar-normalised baseline for total energy. Positive = "
                         "used more than expected.")
    else:
        k[2].metric("฿ vs expected", "—", border=True, help="No usable baseline days in this period.")
    k[3].metric("Peak demand", f"{dem['peak_kw']:,.0f} kW" if dem else "—", border=True,
                delta=f"at {pd.Timestamp(dem['peak_at']):%a %d %b %H:%M}" if dem else None,
                delta_color="off", delta_arrow="off",
                help=(f"Hourly-mean peak. Load factor {dem['load_factor']:.2f} (mean ÷ peak): low means a "
                      "peaky profile — here a tightly scheduled 9-hour cooling day." if dem else None))

    # --------------------------------------------------------------------------- #
    # G14 in one line
    # --------------------------------------------------------------------------- #
    g_tot = M.g14_summary(mt)
    test = g_tot[g_tot.split != "train"]
    tr = test.iloc[0] if len(test) else g_tot.iloc[0]
    ok = bool(tr.meets_g14)
    gl = st.columns([8, 2], vertical_alignment="center")
    gl[0].markdown(
        ui.chip("ok" if ok else "warning",
                f"Baseline {'meets' if ok else 'does not meet'} ASHRAE G14 · CV(RMSE) {tr.cv_rmse:.1f}% · "
                f"NMBE {tr.nmbe:+.1f}%")
        + f' <span class="cb-cap cb-muted">&nbsp;out of sample, total energy · trained '
          f'{mt["train"][0]} → {mt["train"][1]}</span>', unsafe_allow_html=True)
    with gl[1].popover("G14 detail", icon=":material/fact_check:", width="stretch"):
        rows = []
        for tgt, name in [("total_kwh", "Total"), ("ac_kwh", "Cooling (AC)")]:
            for r in M.g14_summary(_model(floor, tgt)).itertuples():
                rows.append({"Energy": name, "Split": r.split, "Days": r.n, "CV(RMSE) %": r.cv_rmse,
                             "limit %": r.cv_limit, "NMBE %": r.nmbe, "|NMBE| limit %": r.nmbe_limit,
                             "G14": "✓ meets" if r.meets_g14 else "▲ fails"})
        st.dataframe(ui.glyph_style(pd.DataFrame(rows), ["G14"], {"CV(RMSE) %": "{:.1f}", "NMBE %": "{:+.1f}",
                                                                  "limit %": "{:.0f}", "|NMBE| limit %": "{:.0f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Residual lag-1 autocorrelation {mt['rho']:.2f} (widens the band on multi-day totals). "
                   "A daily model held to the hourly G14 limits — lenient, so read as indicative.")

    # --------------------------------------------------------------------------- #
TAB_NAMES = (["Monthly", "Profiles"] if LOBBY else
             ["Actual vs expected", "Signature & outlier days", "Profiles"])
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
n_days = int((e - s).days) + 1

if LOBBY:
    with tabs["Monthly"]:
        mo = D.slice_hourly(h, s, e).assign(month=lambda d: d.ts.dt.to_period("M").dt.to_timestamp())
        mo = mo.groupby("month")[E.END_USES].sum(min_count=1).sum(axis=1, min_count=1).rename("kWh").reset_index()
        mo = mo.dropna(subset=["kWh"])
        if len(mo):
            mo["thb"] = mo.kWh * tariff
            st.altair_chart(ui.chart(alt.Chart(mo).mark_bar(color=ui.PRIMARY, size=28).encode(
                x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(labelAngle=0, format="%b %y")),
                y=alt.Y("kWh:Q", title="kWh / month"),
                tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("kWh:Q", format=",.0f"),
                         alt.Tooltip("thb:Q", format=",.0f", title="฿")]), height=260), width="stretch")
            st.caption("Monthly energy over the period; hover for ฿ at the Assumptions tariff.")
        else:
            st.info("No metered energy in this period.", icon=":material/info:")
else:
    with tabs["Actual vs expected"]:
        cc = st.columns([2, 3, 5], vertical_alignment="bottom")
        target = cc[0].selectbox("Energy", ["total_kwh", "ac_kwh"],
                                 format_func={"ac_kwh": "Cooling (AC)", "total_kwh": "Total"}.get)
        rolls = ["Daily", "Weekly", "Monthly"]
        freq = cc[1].segmented_control("Roll-up", rolls, required=True, key="energy_roll",
                                       default="Daily" if n_days <= 45 else "Weekly" if n_days <= 120
                                       else "Monthly")
        model = _model(floor, target)
        tr0, tr1 = model["train"]
        cc[2].caption(f"Baseline trained {tr0} → {tr1}, tested {model['test'][0]} → {model['test'][1]} "
                      "(fixed by the model, not the period). "
                      + ("The window sits between the 2019-03-06 and 2019-09-27 change-points."
                         if floor == 5 else "No declared blocking change-point inside it."))
        ru = M.rollup(model, s, e, {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[freq])
        if len(ru):
            ru["config"] = np.where(ru.baseline_segment, "baseline configuration", "different configuration")
            bars = alt.Chart(ru).mark_bar(opacity=0.9, size=max(3, min(40, 900 // max(len(ru), 1)))).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("actual:Q", title=f"kWh per {freq.lower()[:-2] if freq != 'Daily' else 'day'}"),
                color=alt.Color("config:N", scale=alt.Scale(
                    domain=["baseline configuration", "different configuration"],
                    range=[ui.PRIMARY, ui.CONTEXT]), legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("actual:Q", format=",.0f"),
                         alt.Tooltip("expected:Q", format=",.0f"),
                         alt.Tooltip("deviation_pct:Q", format="+.1f", title="vs expected %"),
                         alt.Tooltip("band:Q", format=",.0f", title="± band"), "days:Q"])
            band = alt.Chart(ru).mark_area(opacity=0.18, color=ui.INK_2).encode(x="date:T", y="lo:Q", y2="hi:Q")
            line = ui.expected_line(alt.Chart(ru)).encode(x="date:T", y="expected:Q")
            layers = [bars, band, line]
            rules = ui.cps_rules(floor, s, e)
            if rules is not None:
                layers.append(rules)
            st.altair_chart(ui.chart(alt.layer(*layers), height=300), width="stretch")
            st.caption(f"Bars actual; dashed line expected; shaded band ±{R.conv('mv_confidence_z'):g}σ (90%) "
                       "with the G14 autocorrelation correction. Grey bars lie across a registered "
                       "change-point from the training window and need an IPMVP non-routine adjustment "
                       "before they count. Dotted rules: change-points and the sensor gap.")
            with st.expander("Table"):
                st.dataframe(ui.round_numeric(ru[["date", "days", "actual", "expected", "lo", "hi",
                                                  "deviation", "deviation_pct", "significant", "config"]]),
                             width="stretch", hide_index=True)
        else:
            st.info("No usable days in this period.", icon=":material/info:")

    with tabs["Signature & outlier days"]:
        model = _model(floor, "total_kwh")
        sig = M.signature(model, s, e)
        pts = sig["points"]
        if len(pts):
            sc = alt.Chart(pts).mark_circle(size=40, opacity=0.7).encode(
                x=alt.X("out_temp:Q", title="daily mean outdoor temperature (°C)", scale=alt.Scale(zero=False)),
                y=alt.Y("actual:Q", title="kWh/day"),
                color=alt.Color("day_kind:N", scale=ui.DAY_KIND_COLOR, legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("date:T"), "day_kind", alt.Tooltip("out_temp:Q", format=".1f"),
                         alt.Tooltip("actual:Q", format=",.0f"), alt.Tooltip("expected:Q", format=",.0f")])
            ln = ui.expected_line(alt.Chart(sig["lines"])).encode(x="out_temp:Q", y="expected:Q", detail="regime:N")
            st.altair_chart(ui.chart(sc + ln, height=320), width="stretch")
            p_w = model["params"].get("workday", {})
            if "cp" in p_w and np.isfinite(p_w.get("cp", np.nan)):
                st.caption(f"Workday line: {p_w['b0']:,.0f} kWh/day below a balance point of {p_w['cp']:.1f} °C, "
                           f"then +{p_w['b1']:,.0f} kWh/day per °C above it. The lower line is non-workdays.")
            ol = sig["outliers"]
            ui.md(f'<p class="cb-overline">Workdays above the line ({len(ol)})</p>')
            if len(ol):
                st.dataframe(ui.round_numeric(ol.assign(**{"฿ excess": ol.resid * tariff})[
                    ["date", "out_temp", "actual", "expected", "resid", "฿ excess"]].rename(
                        columns={"out_temp": "outdoor °C", "resid": "excess kWh"})).head(20),
                    width="stretch", hide_index=True)
                st.caption(f"More than {R.conv('signature_outlier_z'):g} training SDs above the workday line.")
        else:
            st.info("No usable days in this period.", icon=":material/info:")

with tabs["Profiles"]:
    ei = E.eui(h, area, floor)
    eu = E.end_use(h, s, e, floor=floor)
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.md('<p class="cb-overline">Carpet plot</p>')
        val = st.segmented_control("Channel", ["ac_kwh", "light_kwh", "plug_kwh"], default="ac_kwh",
                                   required=True, key="energy_carpet",
                                   format_func=lambda v: ui.END_USE_NAME[v.replace("_kwh", "")])
        cp = E.carpet(h, val, s, e)
        if len(cp):
            st.altair_chart(ui.chart(alt.Chart(cp).mark_rect().encode(
                x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("hour:O", title="hour of day", sort="descending"),
                color=alt.Color(f"{val}:Q", scale=alt.Scale(range=[ui.SEQ[0], ui.SEQ[-1]]),
                                legend=alt.Legend(title="kWh")),
                tooltip=[alt.Tooltip("date:T"), "hour:O", alt.Tooltip(f"{val}:Q", format=",.1f")]),
                height=360), width="stretch")
            if floor == 5:
                st.caption("The two dark horizontal stripes across the cooling block are 12:00 and 17:00 — "
                           "the AC is off by schedule on ~94% of weekdays.")
    with right:
        ui.md('<p class="cb-overline">Workday hourly profile</p>')
        d = D.slice_hourly(h, s, e)
        prof = d[~d.is_weekend & ~d.is_holiday].groupby("hour", as_index=False)[E.END_USES].mean()
        m = prof.melt(id_vars="hour", value_vars=E.END_USES, var_name="end_use", value_name="kW")
        m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
        st.altair_chart(ui.chart(alt.Chart(m).mark_bar().encode(
            x=alt.X("hour:O", title="hour of day"), y=alt.Y("kW:Q", title="mean kW per zone-hour"),
            color=alt.Color("end_use:N", scale=ui.END_USE_COLOR, legend=alt.Legend(orient="top", title=None)),
            tooltip=["hour", "end_use", alt.Tooltip("kW:Q", format=".2f")]), height=200), width="stretch")
        if floor == 5:
            st.caption(f"Measured schedule: cooling runs {R.AC_SCHEDULE['start']}–{R.AC_SCHEDULE['lunch_off']} "
                       f"and {R.AC_SCHEDULE['lunch_on']}–{R.AC_SCHEDULE['end']}; hours 7 and 16 are only "
                       "partly covered, so averaging them understates the on-load.")
        ui.md('<p class="cb-overline" style="margin-top:12px">End use by zone</p>')
        mz = eu.melt(id_vars="zone", value_vars=E.END_USES, var_name="end_use",
                     value_name="kWh").dropna(subset=["kWh"])
        mz["end_use"] = mz.end_use.str.replace("_kwh", "", regex=False)
        st.altair_chart(ui.chart(alt.Chart(mz).mark_bar().encode(
            x=alt.X("zone:N", title=None), y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color("end_use:N", scale=ui.END_USE_COLOR, legend=None),
            tooltip=["zone", "end_use", alt.Tooltip("kWh:Q", format=",.0f")]), height=180), width="stretch")
        no_ac = [z for z in R.ZONES[floor] if not R.has_channel(floor, z, "ac")]
        if no_ac:
            st.caption(f"{', '.join(no_ac)} {'has' if len(no_ac) == 1 else 'have'} no AC bar because there "
                       "is no AC meter — a zero there would be a lie.")
    st.caption(f"Partial EUI {ei['eui']:.0f} kWh/m²/yr ({ei['kwh_year']:,.0f} kWh/yr ÷ {area:,.0f} m²) — "
               f"submetered zone loads only, on an assumed floor area, so not comparable with the Bangkok "
               f"benchmarks ({ei['good']:.0f} good practice, {ei['typical']:.0f} typical) without that caveat.")

    if not LOBBY:
        ui.md('<p class="cb-overline" style="margin-top:12px">Off-hours energy</p>')
        b1, b2 = st.columns(2, gap="large")
        with b1:
            bl = E.baseload(h, s, e, floor).dropna(subset=["ratio_pct"])
            if len(bl):
                bl["point"] = bl.zone + " " + bl.end_use
                st.altair_chart(ui.chart(alt.Chart(bl).mark_bar().encode(
                    y=alt.Y("point:N", title=None, sort="-x"),
                    x=alt.X("ratio_pct:Q", title="baseload: night as % of occupied"),
                    color=alt.Color("end_use:N", scale=ui.END_USE_COLOR, legend=None),
                    tooltip=["point", alt.Tooltip("ratio_pct:Q", format=".1f"),
                             alt.Tooltip("night_kwh:Q", format=",.0f")]), height=300), width="stretch")
        with b2:
            ah = E.after_hours(h, s, e, floor).dropna(subset=["after_hours_pct"])
            if len(ah):
                ah["point"] = ah.zone + " " + ah.end_use
                st.dataframe(
                    ah.sort_values("after_hours_kwh", ascending=False).rename(columns={
                        "after_hours_kwh": "after-hours kWh", "total_kwh": "total kWh",
                        "after_hours_pct": "after-hours %"})
                    [["point", "after-hours kWh", "total kWh", "after-hours %"]].round(1),
                    width="stretch", hide_index=True)
        if floor == 5:
            st.caption("On this floor z3_Plug draws a flat 0.27 kW round the clock and is also a frozen meter, "
                       "so verify it at the panel before chasing the load. Stairwell lighting starts at 06:00 "
                       "and runs to 19:00; z4's lighting runs at 0.48 kW through the weekend.")
