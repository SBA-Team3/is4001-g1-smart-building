"""Energy performance — actual vs expected (ISO 50006 / IPMVP), CUSUM, energy
signature, peak demand, then EUI, end use, baseload and after-hours."""

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
floor, as_of, win, tariff, area = (c["floor"], c["as_of"], c["window"],
                                   c["tariff"], c["area"])
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)
st.title("Energy performance")


@st.cache_resource(show_spinner="Fitting the weather-normalised baseline…")
def _model(floor: int, target: str) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, target)


period = st.radio("Period", ["Whole record", "Last 12 months", "Trailing window"],
                  horizontal=True, index=0)
if period == "Trailing window":
    s, e = D.window(as_of, max(win, 30))
    st.caption(f"{s.date()} – {e.date()}")
elif period == "Last 12 months":
    s, e = D.window(as_of, 365)
    s = max(s, D.RECORD_START)
    st.caption(f"{s.date()} – {e.date()}")
else:
    s = e = None
    st.caption("1 Jul 2018 – 31 Dec 2019")
ui.file_end_note(floor, e)

tot = E.floor_totals(h, s, e, floor)
ei = E.eui(h, area, floor)
dem = E.demand_stats(h, s, e)

k = st.columns(5)
k[0].metric("Energy", f"{tot['total']:,.0f} kWh")
k[1].metric("Cost", f"฿{tot['total'] * tariff:,.0f}")
k[2].metric("AC share", f"{tot['ac_share']:.0f}%")
k[3].metric("Peak demand", f"{dem.get('peak_kw', np.nan):,.0f} kW" if dem else "—",
            help=f"at {dem['peak_at']}" if dem else None)
k[4].metric("Load factor", f"{dem.get('load_factor', np.nan):.2f}" if dem else "—",
            help="Mean ÷ peak demand. Low means a peaky profile — here a "
                 "consequence of a tightly scheduled 9-hour cooling day.")

# --------------------------------------------------------------------------- #
# actual vs expected — the EnPI view
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Actual vs expected")
st.caption("Expected = what this floor should have used given the weather and the calendar: a "
           "daily 3-parameter cooling change-point model, fitted separately for workdays and "
           "non-workdays (ASHRAE Inverse Modeling Toolkit). This is the ISO 50006 energy "
           "baseline and the IPMVP Option C counterfactual.")

cc = st.columns([2, 2, 4])
target = cc[0].selectbox("Energy", ["ac_kwh", "total_kwh"],
                         format_func={"ac_kwh": "Cooling (AC)", "total_kwh": "Total"}.get)
freq = cc[1].radio("Roll-up", ["Monthly", "Weekly"], horizontal=True)
model = _model(floor, target)
tr0, tr1 = model["train"]
cc[2].caption(f"Trained on **{tr0} → {tr1}**, tested on **{model['test'][0]} → "
              f"{model['test'][1]}**. On Floor 5 the window sits between the 2019-03-06 and "
              "2019-09-27 change-points, so the baseline never learns across a configuration "
              "change.")

g = M.g14_summary(model)
gc = st.columns(len(g) + 1)
for i, r in g.iterrows():
    gc[i].metric(f"{r.split.title()} CV(RMSE)", f"{r.cv_rmse:.1f}%",
                 delta=f"limit {r.cv_limit:.0f}%", delta_color="off")
    gc[i].metric(f"{r.split.title()} NMBE", f"{r.nmbe:+.1f}%",
                 delta=("meets G14" if r.meets_g14 else "fails G14"),
                 delta_color="normal" if r.meets_g14 else "inverse",
                 help=f"ASHRAE Guideline 14 limit |NMBE| ≤ {r.nmbe_limit:.0f}% (daily model held "
                      "to the hourly limits — lenient, so read as indicative).")
gc[-1].metric("Residual autocorrelation", f"{model['rho']:.2f}",
              help="Lag-1 autocorrelation of training residuals. Widens the band on a "
                   "multi-day total, because correlated days carry less information.")

ru = M.rollup(model, s, e, "MS" if freq == "Monthly" else "W-MON")
cps_in = R.crosses_change_point(s if s is not None else D.RECORD_START,
                                e if e is not None else D.RECORD_END, floor)
blocking = [x for x in cps_in if x.blocks_comparison]
if len(ru):
    ru["config"] = np.where(ru.baseline_segment, "baseline configuration",
                            "different configuration")
    z = R.conv("mv_confidence_z")
    bars = alt.Chart(ru).mark_bar(opacity=0.85).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("actual:Q", title=f"kWh per {freq.lower()[:-2]}"),
        color=alt.Color("config:N", scale=alt.Scale(
            domain=["baseline configuration", "different configuration"],
            range=[ui.ACTUAL, ui.MUTED]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("actual:Q", format=",.0f"),
                 alt.Tooltip("expected:Q", format=",.0f"),
                 alt.Tooltip("deviation_pct:Q", format="+.1f", title="vs expected %"),
                 alt.Tooltip("band:Q", format=",.0f", title="± band"), "days:Q"])
    band = alt.Chart(ru).mark_area(opacity=0.25, color=ui.EXPECTED).encode(
        x="date:T", y="lo:Q", y2="hi:Q")
    line = alt.Chart(ru).mark_line(color=ui.EXPECTED, strokeWidth=2, point=True).encode(
        x="date:T", y="expected:Q")
    layers = [bars, band, line]
    rules = ui.cps_rules(floor, s, e)
    if rules is not None:
        layers.append(rules)
    st.altair_chart(ui.chart(alt.layer(*layers), height=300), width="stretch")
    st.caption(f"Bars actual, line expected, shaded band ±{z:g}σ (90%) with the G14 "
               "autocorrelation correction. Grey bars fall on the far side of a registered "
               "change-point from the training window: the expected value there is a back- or "
               "forward-cast across a configuration change and needs an IPMVP non-routine "
               "adjustment before it counts as a saving or a loss. Red dashed lines: "
               "change-points and the sensor gap.")
    with st.expander("Table"):
        st.dataframe(ru[["date", "days", "actual", "expected", "lo", "hi", "deviation",
                         "deviation_pct", "significant", "config"]].round(1),
                     width="stretch", hide_index=True)
else:
    st.info("No usable days in this period.")

# --- CUSUM ------------------------------------------------------------------ #
st.markdown("**Cumulative sum of (actual − expected)**")
cu = M.cusum(model, s, e)
if len(cu):
    if blocking:
        st.warning("The CUSUM restarts at " + "; ".join(f"**{x.date}** ({x.title})" for x in blocking)
                   + ". A cumulative sum across a configuration change measures the change, "
                     "not the floor's performance, so each stretch is drawn separately.")
    cl = [alt.Chart(cu).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("cusum:Q", title="cumulative kWh vs expected"),
        color=alt.Color("segment:N", legend=alt.Legend(orient="top", title="stretch")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("cusum:Q", format=",.0f"),
                 alt.Tooltip("resid:Q", format=",.0f", title="day's deviation"), "segment"]),
        alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=ui.EXPECTED).encode(y="y:Q")]
    rules = ui.cps_rules(floor, cu.date.min(), cu.date.max())
    if rules is not None:
        cl.append(rules)
    st.altair_chart(ui.chart(alt.layer(*cl), height=260), width="stretch")
    seg = cu[cu.date <= as_of]
    if len(seg):
        cur = seg[seg.segment == seg.segment.iloc[-1]]
        k2 = st.columns(3)
        k2[0].metric("Net vs expected, current stretch", f"{cur.cusum.iloc[-1]:+,.0f} kWh",
                     delta=f"฿{cur.cusum.iloc[-1] * tariff:+,.0f}", delta_color="inverse",
                     help=f"From {cur.date.min().date()} to {cur.date.max().date()}, within "
                          "one configuration.")
        k2[1].metric("Days in stretch", f"{len(cur)}",
                     help=("Includes training days, where the CUSUM is ~0 by construction."
                           if cur.in_train.any() else "All out of sample."))
        k2[2].metric("Configuration", "baseline" if cur.baseline_segment.iloc[0] else "different",
                     help="Whether this stretch is the one the baseline was trained in.")
    st.caption("Read the slope, not the level: a line climbing steadily means the floor is using "
               "more than the weather explains, and the point where the slope changes is when "
               "performance drifted. Within the training window the line hovers near zero by "
               "construction.")

# --- energy signature ------------------------------------------------------- #
st.markdown("**Energy signature**")
sig = M.signature(model, s, e)
pts = sig["points"]
if len(pts):
    sc = alt.Chart(pts).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("out_temp:Q", title="daily mean outdoor temperature (°C)",
                scale=alt.Scale(zero=False)),
        y=alt.Y("actual:Q", title="kWh/day"),
        color=alt.Color("day_kind:N", scale=ui.DAY_KIND_COLOR,
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T"), "day_kind", alt.Tooltip("out_temp:Q", format=".1f"),
                 alt.Tooltip("actual:Q", format=",.0f"), alt.Tooltip("expected:Q", format=",.0f")])
    ln = alt.Chart(sig["lines"]).mark_line(strokeWidth=2.5, color=ui.EXPECTED).encode(
        x="out_temp:Q", y="expected:Q", detail="regime:N")
    st.altair_chart(ui.chart(sc + ln, height=320), width="stretch")
    p_w = model["params"].get("workday", {})
    if "cp" in p_w and np.isfinite(p_w.get("cp", np.nan)):
        st.caption(f"Workday line: {p_w['b0']:,.0f} kWh/day below a balance point of "
                   f"{p_w['cp']:.1f} °C, then +{p_w['b1']:,.0f} kWh/day per °C above it. "
                   "The lower line is non-workdays.")
    ol = sig["outliers"]
    st.markdown(f"**Workdays well above the line** — more than "
                f"{R.conv('signature_outlier_z'):g} training SDs: {len(ol)} day(s)")
    if len(ol):
        st.dataframe(ol.assign(**{"฿ excess": ol.resid * tariff})[
            ["date", "out_temp", "actual", "expected", "resid", "฿ excess"]].rename(
                columns={"out_temp": "outdoor °C", "resid": "excess kWh"}).round(1).head(20),
            width="stretch", hide_index=True)
        st.caption("The drill-down list: check the carpet plot below and the fault register "
                   "for these dates (late finishes, overrides, events).")

# --- peak demand ------------------------------------------------------------ #
st.divider()
st.subheader("Peak demand")
charge = c["demand_charge"]
dp = M.daily_peaks(h, s, e)
if len(dp):
    st.altair_chart(ui.chart(alt.Chart(dp).mark_line(color=ui.ACTUAL).encode(
        x=alt.X("date:T", title=None), y=alt.Y("peak_kw:Q", title="daily peak (kW, hourly mean)"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("peak_kw:Q", format=".1f"),
                 alt.Tooltip("peak_at:T", format="%H:%M")]), height=220), width="stretch")
    pc = st.columns([3, 2])
    with pc[0]:
        st.markdown("**Top 10 demand hours**")
        st.dataframe(M.top_peaks(h, s, e).rename(columns={
            "out_temp": "outdoor °C", "out_rh": "outdoor RH %", "day_type": "day"}).round(1),
            width="stretch", hide_index=True)
    with pc[1]:
        dc = M.demand_cost(h, s, e, charge)
        st.metric("Demand charges in period", f"฿{dc.charge_thb.sum():,.0f}",
                  help=f"Monthly maximum demand × {charge:,.0f} ฿/kW/month (sidebar). "
                       + R.CONVENTIONS["demand_charge_thb_per_kw"][2] + ".")
        if charge == 0:
            st.caption("Demand charge is set to 0 — MEA large-customer tariffs carry one. "
                       "Enter the contract rate in the sidebar to price these peaks.")
        st.caption("Peaks are hourly means; MEA bills on the 15-minute maximum, which is "
                   "higher. The peak hour is the morning pull-down, not the hottest afternoon.")

# --------------------------------------------------------------------------- #
st.divider()
left, right = st.columns([2, 3])

with left:
    st.subheader("Energy use intensity")
    st.metric("Partial EUI", f"{ei['eui']:.0f} kWh/m²/yr",
              help=f"{ei['kwh_year']:,.0f} kWh/yr ÷ {area:,.0f} m².")
    bench = pd.DataFrame({
        "label": ["Floor 5 (partial)", "Bangkok good practice", "Bangkok typical"],
        "value": [ei["eui"], ei["good"], ei["typical"]],
        "kind": ["measured", "benchmark", "benchmark"]})
    st.altair_chart(
        alt.Chart(bench).mark_bar().encode(
            y=alt.Y("label:N", title=None, sort=None),
            x=alt.X("value:Q", title="kWh/m²/yr"),
            color=alt.Color("kind:N", scale=alt.Scale(
                domain=["measured", "benchmark"], range=["#2e74b5", "#c9c9c9"]),
                legend=None),
            tooltip=["label", alt.Tooltip("value:Q", format=".0f")],
        ).properties(height=140), width="stretch")
    st.warning(
        "**This is a partial EUI and is not comparable to the benchmark without "
        "that caveat.** It covers submetered zone loads only — AC, lighting and "
        "plug inside the five zones. It excludes central plant, lifts, risers and "
        "common areas, and the floor area is an assumption (11,700 m² ÷ 7; the "
        "dataset paper gives no per-floor split). The benchmarks are "
        "whole-building figures from a 41-building Bangkok office study."
    )

with right:
    st.subheader("End use by zone")
    eu = E.end_use(h, s, e, floor=floor)
    m = eu.melt(id_vars="zone", value_vars=E.END_USES,
                var_name="end_use", value_name="kWh").dropna(subset=["kWh"])
    m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("zone:N", title=None),
            y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["zone", "end_use", alt.Tooltip("kWh:Q", format=",.0f")],
        ).properties(height=260), width="stretch")
    st.caption("z3 has no AC bar because it has no AC meter — it is a stairwell. "
               "A zero there would be a lie.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("When the floor uses energy")

t1, t2, t3 = st.tabs(["Carpet plot", "Load duration", "Hourly profile"])

with t1:
    val = st.selectbox("Channel", ["ac_kwh", "light_kwh", "plug_kwh"],
                       format_func=lambda v: v.replace("_kwh", "").upper())
    cp = E.carpet(h, val, s, e)
    st.altair_chart(
        alt.Chart(cp).mark_rect().encode(
            x=alt.X("yearmonthdate(date):O", title=None,
                    axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("hour:O", title="hour of day", sort="descending"),
            color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme="viridis"),
                            legend=alt.Legend(title="kWh")),
            tooltip=[alt.Tooltip("date:T"), "hour:O",
                     alt.Tooltip(f"{val}:Q", format=",.1f")],
        ).properties(height=420), width="stretch")
    st.caption("The two dark horizontal stripes across the cooling block are "
               "12:00 and 17:00 — the AC is off by schedule, on ~94% of "
               "weekdays. The notebook's assumed 08:00–18:00 occupied window "
               "averaged those two uncooled hours in with eight cooled ones.")

with t2:
    ld = E.load_duration(h, s, e)
    st.altair_chart(
        alt.Chart(ld).mark_area(opacity=0.7, color="#2e74b5").encode(
            x=alt.X("rank_pct:Q", title="% of hours at or above"),
            y=alt.Y("kw:Q", title="floor demand (kW)"),
            tooltip=[alt.Tooltip("rank_pct:Q", format=".1f"),
                     alt.Tooltip("kw:Q", format=".1f")],
        ).properties(height=300), width="stretch")

with t3:
    d = D.slice_hourly(h, s, e)
    prof = d[~d.is_weekend & ~d.is_holiday].groupby("hour", as_index=False)[
        E.END_USES].mean()
    m = prof.melt(id_vars="hour", value_vars=E.END_USES,
                  var_name="end_use", value_name="kW")
    m["end_use"] = m.end_use.str.replace("_kwh", "", regex=False)
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("hour:O", title="hour of day"),
            y=alt.Y("kW:Q", title="mean kW per zone-hour"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["hour", "end_use", alt.Tooltip("kW:Q", format=".2f")],
        ).properties(height=280), width="stretch")
    st.caption("Measured schedule: cooling runs 07:48–11:50 and 13:03–16:31. "
               "Hours 7 and 16 are only partly covered (16% and 45% of minutes), "
               "so averaging them with hours 8–10 understates the on-load.")

# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Off-hours energy — the quick wins")

b1, b2 = st.columns(2)
with b1:
    st.markdown("**Baseload ratio** (overnight mean ÷ occupied mean)")
    bl = E.baseload(h, s, e, floor).dropna(subset=["ratio_pct"])
    bl["point"] = bl.zone + " " + bl.end_use
    st.altair_chart(
        alt.Chart(bl.sort_values("ratio_pct", ascending=False)).mark_bar().encode(
            y=alt.Y("point:N", title=None, sort="-x"),
            x=alt.X("ratio_pct:Q", title="night as % of occupied"),
            color=alt.Color("end_use:N", scale=alt.Scale(
                domain=["ac", "light", "plug"],
                range=["#2e74b5", "#d98032", "#8d8d8d"]), legend=None),
            tooltip=["point", alt.Tooltip("ratio_pct:Q", format=".1f"),
                     alt.Tooltip("night_kwh:Q", format=",.0f")],
        ).properties(height=320), width="stretch")

with b2:
    st.markdown("**Energy outside the measured schedule**")
    ah = E.after_hours(h, s, e, floor).dropna(subset=["after_hours_pct"])
    ah["point"] = ah.zone + " " + ah.end_use
    st.dataframe(
        ah.sort_values("after_hours_kwh", ascending=False).rename(columns={
            "after_hours_kwh": "after-hours kWh", "total_kwh": "total kWh",
            "after_hours_pct": "after-hours %"})
        [["point", "after-hours kWh", "total kWh", "after-hours %"]].round(1),
        width="stretch", hide_index=True)

st.info(
    "**Two findings the cooling analysis missed entirely, because it excluded "
    "the stairwell and looked only at AC.** `z3_Plug` draws a flat 0.27 kW "
    "round the clock — 98% of its occupied level overnight, and 41% of its "
    "annual energy falls outside working hours. It is also a frozen meter (see "
    "Sensor health), so the right first action is to verify it at the panel "
    "rather than chase the load. Stairwell lighting starts at 06:00, nearly two "
    "hours before cooling, and runs until 19:00, two hours after it stops. z4's "
    "lighting runs at 0.48 kW through the weekend — about five times any other "
    "zone."
)
