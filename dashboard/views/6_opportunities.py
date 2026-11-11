"""Opportunities & simulator — the proposal's Intervention Simulator and Energy &
Cost Impact Summary, on top of the three engine tabs (setpoint, thermal model,
schedule trim) that size each lever."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import analysis as A
from sbanalytics import comfort as C
from sbanalytics import data as D
from sbanalytics import mv as M
from sbanalytics import registry as R
from sbanalytics import scenario as SC
import ui

c = ui.ctx()
floor, tariff = c["floor"], c["tariff"]
if not ui.require_wired(floor):
    st.stop()

h = ui.hourly(floor)


@st.cache_resource(show_spinner="Calibrating thermal model…")
def _physics(floor: int, use_solar: bool):
    return A.fit_physics(D.load_hourly(floor), use_solar=use_solar, floor=floor)


@st.cache_resource(show_spinner="Fitting regression…")
def _reg(floor: int):
    return A.run_regression(D.load_hourly(floor), floor=floor)


@st.cache_resource(show_spinner="Fitting zone baselines for the M&V plan…")
def _zone_cv(floor: int) -> float:
    _, perf = A.fit_baselines(D.load_hourly(floor), floor=floor)
    return A.zone_baseline_cv(perf)


@st.cache_resource(show_spinner="Fitting the floor baseline…")
def _floor_model(floor: int) -> dict:
    return M.daily_model(D.load_hourly(floor), floor, "ac_kwh")


@st.cache_resource(show_spinner="Measuring weekend and holiday AC…")
def _lockout(floor: int) -> dict:
    return SC.lockout_measure(D.load_hourly(floor), floor)


ui.page_header("Savings", f"Floor {floor} savings",
               "Build a package of operational changes and see what it saves as a range, what it does "
               "to comfort, and how you would prove it. The tariff comes from Assumptions; the "
               "grid emission factor is set on the Carbon page (TGO 2022 default).")
ui.floor_band(floor, c["as_of"])

phys = _physics(floor, True)
reg = _reg(floor)
est = A.estimator_range(phys, reg)
cz = R.comfort_zones(floor)
cap = float(R.conv("max_lift_c"))
max_trim = SC.max_trim(h, floor)
target = R.conv("overcooled_target_pct")
LO = R.conv("comfort_lo")
t_loc, ef_loc = float(tariff), float(c["ef"])


def _baht(v: float) -> str:
    return f"฿{v:,.0f}" if t_loc else "set rate in Assumptions"


# --------------------------------------------------------------------------- #
# simulator: inputs (left) and results (right)
# --------------------------------------------------------------------------- #
left, right = st.columns([4, 8], gap="large")
with left:
    with st.container(border=True, key="card_sim_inputs"):
        ui.md('<p class="cb-overline">Package</p>')
        lift = st.slider("Zone-temperature lift ΔT (°C), all comfort zones", 0.0, cap, 2.0, 0.25,
                         help="There is no setpoint channel, so a setpoint raise is modelled as a "
                              "shift of each zone's observed temperature distribution.")
        trim_n = st.slider("Schedule trim (hours off the end of the day)", 0, max_trim, 0, 1,
                           help="Switches cooling off for the last n cooled hours. Ending the day "
                                "early needs no recovery, so the metered energy is the saving.")
        lockout = st.toggle("Weekend / holiday lockout", value=False,
                            help="Tie AC start/stop to the holiday calendar; weekend runs need sign-off.")
        with st.expander("Per-zone lift"):
            per_zone = st.toggle("Set the lift per zone", value=False)
            if per_zone:
                d_temp = {z: st.slider(f"{ui.zone_label(floor, z)} lift (°C)", 0.0, cap, lift, 0.25,
                                       key=f"lift_{z}") for z in cz}
            else:
                d_temp = {z: lift for z in cz}
                st.caption("Off: every comfort zone takes the lift above.")
        with st.expander("Implementation costs"):
            cost = {"lift": st.number_input("Lift (฿)", 0.0, 1e7, 0.0, 1000.0,
                                            help="0 for an operational change made at the BMS."),
                    "trim": st.number_input("Schedule trim (฿)", 0.0, 1e7, 0.0, 1000.0),
                    "lockout": st.number_input("Lockout (฿)", 0.0, 1e7, 0.0, 1000.0)}
        st.caption(f"Tariff {t_loc:.2f} ฿/kWh (Assumptions).")

res = SC.simulate(h, est, phys, floor, d_temp=d_temp, trim_n=trim_n, lockout=lockout,
                  cost=cost, tariff=t_loc, ef=ef_loc,
                  parts={"lockout": _lockout(floor)} if lockout else None)
tbl, comb, cf = res["table"], res["combined"], res["comfort"]

with right:
    k = st.columns(4)
    k[0].metric("Saving, kWh/yr", f"{comb['low'] / 1e3:,.1f}k–{comb['high'] / 1e3:,.1f}k",
                help=f"Mid {comb['mid']:,.0f} kWh/yr. Low and high come from the two estimators "
                     "(and metered gross/net); never one number.", border=True)
    k[1].metric("Saving, ฿/yr", f"฿{comb['low'] * t_loc / 1e3:,.0f}k–{comb['high'] * t_loc / 1e3:,.0f}k" if t_loc
                else "set rate in Assumptions",
                delta=f"mid ฿{comb['mid'] * t_loc:,.0f}" if t_loc else None, delta_color="off",
                delta_arrow="off", border=True)
    k[2].metric("Emissions avoided", f"{comb['mid'] * ef_loc / 1000:,.1f} tCO₂e/yr",
                help=f"Mid case × the grid factor {ef_loc:.4f} kgCO₂e/kWh, set on the Carbon page "
                     "(TGO 2022 default).",
                border=True)
    k[2].page_link("views/7_carbon.py", label="Grid factor set on Carbon (TGO 2022 default)",
                   icon=":material/arrow_forward:")
    k[3].metric(f"Cooled <{LO:g} °C", f"{cf['below_now']:.0f}% → {cf['below_after']:.0f}%",
                delta=f"{'meets' if cf['below_after'] <= target else 'above'} {target:.0f}% target",
                delta_color="off" if cf["below_after"] <= target else "inverse", delta_arrow="off",
                help="Whole-record cooled occupied hours on comfort zones, AC-kWh-weighted, after "
                     "shifting each zone's temperatures by its lift.", border=True)
    warm = cf["above_after"]
    (st.warning if warm > cf["above_now"] else st.info)(
        f"**Comfort trade-off.** Cooled hours below {LO:g} °C go from {cf['below_now']:.0f}% to "
        f"{cf['below_after']:.0f}% (target {target:.0f}%); hours above {R.conv('comfort_hi'):g} °C from "
        f"{cf['above_now']:.0f}% to {warm:.0f}%. The projection shifts the observed distribution; it does "
        "not narrow it — that is a control-loop retune, not a setpoint change.",
        icon=":material/warning:" if warm > cf["above_now"] else ":material/info:")

    ui.md('<p class="cb-overline" style="margin-top:8px">M&amp;V plan</p>')
    fm = _floor_model(floor)
    floor_cv = float(fm["metrics"].query("split == 'test'").cv_rmse.iloc[0])
    floor_ac = M.annual_kwh(fm)
    plan = SC.mv_plan(floor, res, _zone_cv(floor), floor_cv, lockout, trim_n, floor_ac)
    if not len(plan):
        st.caption("Select at least one measure to see its M&V plan.")
    for _, p in plan.iterrows():
        with st.container(border=True):
            a, b, cc_ = st.columns([3, 2, 2], vertical_alignment="center")
            a.markdown(f"**{p.measure}** · IPMVP Option {p['IPMVP option']}  \n"
                       f"<small>Boundary: {p['measurement boundary']}</small>", unsafe_allow_html=True)
            b.metric("Minimum pilot", f"{ui.fmt_weeks(p['minimum pilot (weeks)'])} weeks",
                     help=f"Expected saving {p['expected saving %']:.1f}% at the baseline's "
                          f"CV(RMSE) {p['CV(RMSE) %']:.1f}%, α={R.conv('pilot_alpha'):g}, "
                          f"power {R.conv('pilot_power'):g}. Four weeks can detect "
                          f"{p['detectable in 4 weeks %']:.1f}%.", border=True)
            cc_.metric("Expected saving", f"{p['expected saving %']:.1f}%",
                       help="Of the measurement boundary's annual energy.", border=True)
            st.caption(f"Baseline period **{p['baseline period']}** · model: {p['baseline model']} · "
                       f"CV(RMSE) {p['CV(RMSE) %']:.1f}%  \n**Verification:** {p.verification}  \n"
                       f"**Exclusions:** {p.exclusions}")

# --------------------------------------------------------------------------- #
# below: measures, sensitivity, second lever, engine
# --------------------------------------------------------------------------- #
ui.section("Measures", f"One row per measure plus the combined package. The lift and trim act on the "
                       f"comfort-sensored zones ({', '.join(cz)}); the lockout is floor-wide.")
st.dataframe(
    tbl, width="stretch", hide_index=True,
    column_config={
        **{col: st.column_config.NumberColumn(format="%,.0f") for col in tbl.columns
           if col.startswith(("kWh", "฿", "cost"))},
        "tCO₂e/yr mid": st.column_config.NumberColumn(format="%.1f"),
        "payback yr": st.column_config.NumberColumn(format="%.1f", help="Cost ÷ mid ฿/yr. 0 = no capital cost."),
        "comfort after": st.column_config.TextColumn(width="medium"),
    })

ui.section("Which assumption moves the answer most")
tor = SC.tornado(h, est, phys, floor, d_temp=d_temp, trim_n=trim_n, lockout=lockout,
                 tariff=t_loc, max_trim=max_trim)
if comb["mid"] > 0 and t_loc:
    order = tor.assumption.tolist()
    bars = alt.Chart(tor).mark_bar(color=ui.PRIMARY, opacity=0.85).encode(
        y=alt.Y("assumption:N", sort=order, title=None),
        x=alt.X("low_thb:Q", title="combined mid saving (฿/yr)"), x2="high_thb:Q",
        tooltip=["assumption", alt.Tooltip("low_thb:Q", format=",.0f"), alt.Tooltip("high_thb:Q", format=",.0f")])
    base_rule = alt.Chart(pd.DataFrame({"x": [comb["mid"] * t_loc]})).mark_rule(
        color=ui.INK, strokeWidth=2).encode(x="x:Q")
    st.altair_chart(ui.chart(bars + base_rule, height=40 * len(tor) + 40), width="stretch")
    st.caption(f"Each bar moves one assumption across its declared range (tariff ±{R.conv('sens_tariff_pct'):.0f}%, "
               f"lift ±{R.conv('sens_dT_c'):g} °C, trim ±{R.conv('sens_trim_h'):g} h, %/°C between the physics "
               "and regression estimates) and holds the rest. White line: the package's mid case.")
elif not t_loc:
    st.caption("Set a tariff in Assumptions to size the ฿ sensitivity.")
else:
    st.caption("The package saves nothing yet — set a lift, a trim or the lockout.")

ui.section("The second lever: control stability",
           "Raising the setpoint moves the distribution; tightening the control deadband narrows it. "
           "They are different work orders and only one of them is a setpoint change.")
lo_b, hi_b = C.band()
f1, f2 = st.columns(2)
shift = f1.slider("Setpoint rise (°C)", 0.0, 4.0, 1.5, 0.25, key="frontier_shift")
sd_scale = f2.slider("Deadband tightening", 0.4, 1.0, 1.0, 0.05, key="frontier_sd",
                     help="1.00 leaves the spread unchanged; 0.60 is a control loop retuned to cut swing by 40%.")
fr = C.frontier(h, lo=lo_b, hi=hi_b, sd_scale=sd_scale, floor=floor)
fl = fr.melt(id_vars=["zone", "shift"], value_vars=["pct_cold", "pct_warm"], var_name="band", value_name="pct")
fl["band"] = fl["band"].map({"pct_cold": f"< {lo_b:g} °C", "pct_warm": f"> {hi_b:g} °C"})
st.altair_chart(ui.chart(alt.layer(
    alt.Chart(fl).mark_line(strokeWidth=2.5).encode(
        x=alt.X("shift:Q", title="setpoint rise (°C)"), y=alt.Y("pct:Q", title="% of cooled hours"),
        color=alt.Color("band:N", title=None, scale=ui.COLD_WARM, legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("zone:N", legend=alt.Legend(orient="top")),
        tooltip=["zone", "band", alt.Tooltip("shift:Q", format=".2f"), alt.Tooltip("pct:Q", format=".1f")]),
    alt.Chart(pd.DataFrame({"s": [shift]})).mark_rule(color=ui.INK, strokeDash=[5, 3]).encode(x="s:Q"),
), height=280), width="stretch")
dz = D.occupied(h, zones=R.comfort_zones(floor), require_temp=True, ac_on=True)
after = []
for z, g in dz.groupby("zone"):
    n = C.shift_distribution(g.temp_c, shift, sd_scale)
    after.append({"zone": z, "cold %": (n < lo_b).mean() * 100, "warm %": (n > hi_b).mean() * 100,
                  "mean °C": n.mean(), "sd °C": n.std(), "sd now": g.temp_c.std()})
af = pd.DataFrame(after)
if len(af):
    st.dataframe(af.drop(columns="sd now").round(1), width="stretch", hide_index=True)
    st.caption(f"At +{shift:g} °C with deadband ×{sd_scale:.2f}, roughly {af['cold %'].mean():.0f}% of cooled "
               f"hours are still below {lo_b:g} °C. Today's distributions are wide (σ ≈ "
               f"{af['sd now'].min():.1f}–{af['sd now'].max():.1f} °C), so a zone that stays volatile after a "
               "setpoint change is a controls-tuning item, not a setpoint item.")

# --------------------------------------------------------------------------- #
with st.expander("How each lever is sized", icon=":material/engineering:"):
    tab_sp, tab_phys, tab_sched = st.tabs(["Setpoint", "Where the load goes", "Schedule trim"])

# --------------------------------------------------------------------------- #
with tab_sp:
    st.subheader("If the occupied setpoint went up")

    cc = st.columns([2, 2, 2])
    which = cc[0].selectbox(
        "Sensitivity estimate", est.estimator.tolist(),
        index=len(est) - 1,
        help="The spread between these is method disagreement, not sampling "
             "noise. No larger sample would shrink it.")
    row = est[est.estimator == which].iloc[0]
    if not phys.get("pooled") and len(est):
        st.warning("**One estimator only.** The physics model is not identified on this "
                   "floor (see the thermal-model tab), so there is no lower bound and no "
                   "midpoint. This is the regression, which overstates the saving by "
                   "absorbing occupancy.")
    elif phys.get("unidentified"):
        st.caption("Physics fit not identified for "
                   f"{', '.join(phys['unidentified'])}; those zones use the pooled fit.")
    d_temp = cc[1].slider("Raise setpoint by (°C)", 0.0, 4.0, 1.5, 0.25)
    realisation = cc[2].slider("Realisation factor", 0.3, 1.0, 1.0, 0.05,
                               help="Share of the modelled saving you expect to "
                                    "bank. M&V practice discounts for overrides, "
                                    "drift and hot calls.")

    sim = A.setpoint_saving(h, row.pct_per_c, d_temp, realisation=realisation,
                            floor=floor)
    lo_s = A.setpoint_saving(h, min(row.lo, row.hi), d_temp, realisation=realisation, floor=floor)
    hi_s = A.setpoint_saving(h, max(row.lo, row.hi), d_temp, realisation=realisation, floor=floor)
    tot = sim.saving_kwh.sum()

    k = st.columns(4)
    k[0].metric("Sensitivity", f"{row.pct_per_c:+.1f} %/°C",
                help=f"{row.role} — {row.note}", border=True)
    k[1].metric("Energy saved", f"{tot:,.0f} kWh/yr", border=True)
    k[2].metric("Cost saved", f"฿{tot * tariff:,.0f}/yr", border=True)
    k[3].metric("Range", f"฿{min(lo_s.saving_kwh.sum(), hi_s.saving_kwh.sum())*tariff:,.0f}"
                         f" – ฿{max(lo_s.saving_kwh.sum(), hi_s.saving_kwh.sum())*tariff:,.0f}", border=True)

    st.dataframe(
        sim.assign(**{"saving ฿/yr": sim.saving_kwh * tariff})
           .rename(columns={"kwh_year": "AC kWh/yr", "saving_kwh": "saving kWh/yr"})
           .round(0), width="stretch", hide_index=True)

    st.markdown("**Every estimator at once**")
    grid = []
    for _, r in est.iterrows():
        for d in np.arange(0, 4.01, 0.25):
            s = A.setpoint_saving(h, r.pct_per_c, d, realisation=realisation,
                                  floor=floor).saving_kwh.sum()
            grid.append({"estimator": r.estimator, "dT": d, "kWh": s})
    g = pd.DataFrame(grid)
    st.altair_chart(
        alt.layer(
            alt.Chart(g).mark_line(strokeWidth=2.5).encode(
                x=alt.X("dT:Q", title="setpoint rise (°C)"),
                y=alt.Y("kWh:Q", title="saving (kWh/yr)"),
                color=alt.Color("estimator:N", legend=alt.Legend(orient="top",
                                                                 columns=1, title=None)),
                tooltip=["estimator", alt.Tooltip("dT:Q", format=".2f"),
                         alt.Tooltip("kWh:Q", format=",.0f")]),
            alt.Chart(pd.DataFrame({"d": [d_temp]})).mark_rule(
                color=ui.INK, strokeDash=[5, 3]).encode(x="d:Q"),
        ).properties(height=320), width="stretch")

    st.dataframe(est[["estimator", "pct_per_c", "lo", "hi", "role", "note"]].round(2),
                 width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
with tab_phys:
    st.subheader("Where the cooling load actually goes")
    st.caption("The most useful single chart here, because it explains why the "
               "two estimates disagree.")

    if not phys.get("pooled"):
        st.error("Not enough AC-off hours to calibrate." if phys["n"] < 50 else
                 f"**The thermal model is not identified on Floor {floor}.** Across "
                 f"{phys['n']:,} lunch-hour zone-days the capacitance coefficient's "
                 "confidence interval reaches zero, so C — and with it UA — is "
                 "unbounded. The setpoint estimate above uses the regression alone, "
                 "which is the confounded upper bound: read it as indicative.")
    else:
        p = phys["pooled"]
        dec = A.load_decomposition(h, phys, floor=floor)

        k = st.columns(4)
        k[0].metric("Thermal capacitance C", f"{p['C_kwh_per_c']:.2f} kWh/°C", border=True)
        k[1].metric("Envelope conductance UA", f"{p['UA_kw_per_c']:.3f} kW/°C", border=True)
        k[2].metric("Implied sensitivity", f"{p['pct_per_c']:+.1f} %/°C", border=True)
        k[3].metric("Calibration sample", f"{phys['n']:,} zone-days", border=True)

        dl = dec.reset_index().melt(
            id_vars="zone", value_vars=["envelope_kw", "internal_kw", "residual_kw"],
            var_name="component", value_name="kW")
        dl["component"] = dl.component.map({
            "envelope_kw": "Envelope UA·ΔT — a setpoint change acts here",
            "internal_kw": "Internal — metered light + plug, untouchable",
            "residual_kw": "Residual — latent, occupants, fan, plant"})
        st.altair_chart(
            alt.Chart(dl).mark_bar().encode(
                y=alt.Y("zone:N", title=None),
                x=alt.X("kW:Q", title="mean cooling load (kW) on cooled hours"),
                color=alt.Color("component:N", scale=alt.Scale(
                    range=[ui.PRIMARY, ui.SERIES_2, ui.CONTEXT]),
                    legend=alt.Legend(orient="bottom", columns=1, title=None)),
                order=alt.Order("component:N"),
                tooltip=["zone", "component", alt.Tooltip("kW:Q", format=".2f")],
            ).properties(height=200), width="stretch")

        st.dataframe(dec[["measured_kw", "envelope_pct", "internal_pct",
                          "residual_pct"]].round(1), width="stretch")

        st.info(
            f"**Envelope conduction is only {dec.envelope_pct.mean():.0f}% of the "
            f"cooling load.** A setpoint change acts on that share, plus some "
            f"unknown part of the latent load inside the residual. It acts on "
            f"none of the metered internal gains "
            f"({dec.internal_pct.mean():.0f}%). That is the arithmetic reason "
            f"the physics estimate lands near {p['pct_per_c']:+.1f} %/°C while a "
            f"regression on indoor temperature reports "
            f"{reg['pooled']['pct_per_c']:+.1f} %/°C — the regression is claiming "
            f"a saving larger than the entire envelope term, which it can only "
            f"do by absorbing occupancy."
        )

        # How the calibration works (plain text: no expander inside an expander)
        ui.md('<p class="cb-overline" style="margin-top:12px">How the calibration works</p>')
        st.markdown(
            "At 12:00 the AC draws ~0 kWh on every weekday. For that hour "
            "the zone is a free-running thermal mass whose only *measured* "
            "heat input is lighting plus plug load:\n\n"
            r"$$C\,\frac{dT}{dt} = UA\,(T_{out}-T_{in}) + Q_{int} + a\,I_{sol}$$"
            "\n\nRegressing the hourly drift on the metered internal gain "
            "identifies $1/C$; the driving-force coefficient then gives "
            "$UA$. In steady cooled operation the envelope term is the part "
            "of the load a setpoint change removes, so "
            r"$d(\text{load})/d(T_{set}) = -UA$."
            "\n\nCrucially, indoor temperature never appears on the "
            "right-hand side as a *cause* of energy — so unlike a regression "
            "of load on temperature, this cannot be confounded by occupancy. "
            + ("It also agrees in sign across all three zones, where the "
               "regression gives z5 the wrong sign entirely." if floor == 5 else
               "Where the regression and the physics disagree in sign for a zone, "
               "occupancy is the likely reason.")
        )
        L = phys["frame"]
        # One panel per zone (small multiples) rather than four colours on one plot.
        pts = alt.Chart().mark_circle(opacity=0.45, size=35).encode(
            x=alt.X("drive:Q", title="driving force T_out − T_in (°C)"),
            y=alt.Y("drift:Q", title="free-running drift (°C/hour)"),
            color=alt.value(ui.PRIMARY),
            tooltip=["zone", "date", alt.Tooltip("drive:Q", format=".1f"),
                     alt.Tooltip("drift:Q", format=".2f")])
        fit = alt.Chart().transform_regression(
            "drive", "drift", groupby=["zone"]).mark_line(strokeWidth=2, color=ui.INK).encode(
            x="drive:Q", y="drift:Q")
        st.altair_chart(ui.chart(
            alt.layer(pts, fit, data=L).properties(width=260, height=200)
            .facet(facet=alt.Facet("zone:N", title=None), columns=4), height=None),
            width="content")

# --------------------------------------------------------------------------- #
with tab_sched:
    st.subheader("Trimming the cooling schedule")
    st.caption("The only lever here that is measured rather than inferred — the "
               "energy in an hour you switch off is metered. What the physics "
               "adds is the recovery charge.")

    hp = A.hour_profile(h, floor=floor)
    on_hours = SC.on_hours(h, floor)
    st.markdown("Cooling currently runs in: " + ", ".join(f"`{x:02d}:00`" for x in on_hours))

    cc = st.columns([3, 1])
    drop = cc[0].multiselect("Switch cooling off in these hours", on_hours,
                             default=[max(on_hours)] if on_hours else [])
    recover = cc[1].toggle("Charge recovery back", value=True)

    if drop and phys.get("pooled"):
        tr = A.schedule_trim(h, phys, tuple(drop), recover, floor=floor)
        gross, net = tr.gross_kwh_year.sum(), tr.net_kwh_year.sum()
        k = st.columns(4)
        k[0].metric("Gross energy released", f"{gross:,.0f} kWh/yr", border=True)
        k[1].metric("Net of recovery", f"{net:,.0f} kWh/yr",
                    delta=f"{net - gross:,.0f} recovery", border=True)
        k[2].metric("Cost saved (net)", f"฿{net * tariff:,.0f}/yr", border=True)
        k[3].metric("Temp at end of gap", f"{tr.end_temp_c.mean():.1f} °C",
                    help=f"From the calibrated drift rate. Above {R.conv('comfort_hi'):g} °C is a "
                         "complaint risk.", border=True)

        st.dataframe(
            tr[["gross_kwh_year", "recovery_kwh", "net_kwh_year", "drift_c_per_h",
                "end_temp_c", "resumes_same_day"]].round(2), width="stretch")

        if not tr.resumes_same_day.any():
            st.info(
                "Cooling does not resume after this gap, so nothing has to be "
                "pulled back out — the gross saving is the net saving. Trimming "
                "the end of the day is the cheapest lever on the floor.")
        else:
            st.info(
                f"Cooling resumes the same day, so {gross - net:,.0f} kWh/yr of "
                f"the gross is spent pulling the zone back down. The net saving "
                f"is {net / max(gross, 1) * 100:.0f}% of the gross — which is "
                "why extending the lunch gap is a much worse deal than it looks.")
    elif drop:
        st.error("Thermal model not calibrated.")
    else:
        st.info("Select at least one hour to trim.")

    st.dataframe(hp.rename(columns={
        "mean_kwh": "mean kWh", "total_kwh": "total kWh",
        "frac_off": "fraction AC off", "mean_c": "mean °C"}).round(2),
        width="stretch")
