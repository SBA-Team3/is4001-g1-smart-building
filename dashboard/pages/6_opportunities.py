"""Opportunities — what a change would actually save, and how sure we are."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import analysis as A
from core import data as D
from core import registry as R
from core import ui

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


st.title("Opportunities")
st.caption("The analytical work, reduced to the part an FM can act on. Full "
           "method, uncertainty and limitations on the Methods page.")

phys = _physics(floor, True)
reg = _reg(floor)
est = A.estimator_range(phys, reg)

tab_sp, tab_phys, tab_sched = st.tabs(
    ["Setpoint", "Where the load goes", "Schedule trim"])

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
                help=f"{row.role} — {row.note}")
    k[1].metric("Energy saved", f"{tot:,.0f} kWh/yr")
    k[2].metric("Cost saved", f"฿{tot * tariff:,.0f}/yr")
    k[3].metric("Range", f"฿{min(lo_s.saving_kwh.sum(), hi_s.saving_kwh.sum())*tariff:,.0f}"
                         f" – ฿{max(lo_s.saving_kwh.sum(), hi_s.saving_kwh.sum())*tariff:,.0f}")

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
                color="#b0453a", strokeDash=[5, 3]).encode(x="d:Q"),
        ).properties(height=320), width="stretch")

    st.dataframe(est[["estimator", "pct_per_c", "lo", "hi", "role", "note"]].round(2),
                 width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
with tab_phys:
    st.subheader("Where the cooling load actually goes")
    st.caption("The most useful single chart here, because it explains why the "
               "two estimates disagree.")

    if not phys.get("pooled"):
        st.error("Not enough AC-off hours to calibrate.")
    else:
        p = phys["pooled"]
        dec = A.load_decomposition(h, phys, floor=floor)

        k = st.columns(4)
        k[0].metric("Thermal capacitance C", f"{p['C_kwh_per_c']:.2f} kWh/°C")
        k[1].metric("Envelope conductance UA", f"{p['UA_kw_per_c']:.3f} kW/°C")
        k[2].metric("Implied sensitivity", f"{p['pct_per_c']:+.1f} %/°C")
        k[3].metric("Calibration sample", f"{phys['n']:,} zone-days")

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
                    range=["#2e74b5", "#d98032", "#9b9b9b"]),
                    legend=alt.Legend(orient="bottom", columns=1, title=None)),
                order=alt.Order("component:N"),
                tooltip=["zone", "component", alt.Tooltip("kW:Q", format=".2f")],
            ).properties(height=200), width="stretch")

        st.dataframe(dec[["measured_kw", "envelope_pct", "internal_pct",
                          "residual_pct"]].round(1), width="stretch")

        st.success(
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

        with st.expander("How the calibration works"):
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
                "It also agrees in sign across all three zones, where the "
                "regression gives z5 the wrong sign entirely."
            )
            L = phys["frame"]
            st.altair_chart(
                alt.Chart(L).mark_circle(opacity=0.45, size=35).encode(
                    x=alt.X("drive:Q", title="driving force T_out − T_in (°C)"),
                    y=alt.Y("drift:Q", title="free-running drift (°C/hour)"),
                    color=alt.Color("zone:N", scale=ui.ZONE_COLOR),
                    tooltip=["zone", "date", alt.Tooltip("drive:Q", format=".1f"),
                             alt.Tooltip("drift:Q", format=".2f")],
                ).properties(height=280)
                + alt.Chart(L).transform_regression(
                    "drive", "drift", groupby=["zone"]).mark_line(
                    strokeWidth=2).encode(x="drive:Q", y="drift:Q",
                                          color=alt.Color("zone:N", scale=ui.ZONE_COLOR)),
                width="stretch")

# --------------------------------------------------------------------------- #
with tab_sched:
    st.subheader("Trimming the cooling schedule")
    st.caption("The only lever here that is measured rather than inferred — the "
               "energy in an hour you switch off is metered. What the physics "
               "adds is the recovery charge.")

    hp = A.hour_profile(h, floor=floor)
    on_hours = [int(x) for x in hp.index if hp.loc[x, "frac_off"] < 0.5]
    st.markdown("Cooling currently runs in: " + ", ".join(f"`{x:02d}:00`" for x in on_hours))

    cc = st.columns([3, 1])
    drop = cc[0].multiselect("Switch cooling off in these hours", on_hours,
                             default=[max(on_hours)] if on_hours else [])
    recover = cc[1].toggle("Charge recovery back", value=True)

    if drop and phys.get("pooled"):
        tr = A.schedule_trim(h, phys, tuple(drop), recover, floor=floor)
        gross, net = tr.gross_kwh_year.sum(), tr.net_kwh_year.sum()
        k = st.columns(4)
        k[0].metric("Gross energy released", f"{gross:,.0f} kWh/yr")
        k[1].metric("Net of recovery", f"{net:,.0f} kWh/yr",
                    delta=f"{net - gross:,.0f} recovery")
        k[2].metric("Cost saved (net)", f"฿{net * tariff:,.0f}/yr")
        k[3].metric("Temp at end of gap", f"{tr.end_temp_c.mean():.1f} °C",
                    help="From the calibrated drift rate. Above 26 °C is a "
                         "complaint risk.")

        st.dataframe(
            tr[["gross_kwh_year", "recovery_kwh", "net_kwh_year", "drift_c_per_h",
                "end_temp_c", "resumes_same_day"]].round(2), width="stretch")

        if not tr.resumes_same_day.any():
            st.success(
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
