"""
Floor 5 over-cooling — interactive what-if model.

Run with:   streamlit run streamlit_app.py

Every figure is recomputed from cache/f5_hourly.csv.gz when a control moves;
nothing here is a hardcoded notebook result. See f5_core.py for the methods.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import f5_core as f5

st.set_page_config(page_title="Floor 5 — Over-cooling what-if",
                   layout="wide", initial_sidebar_state="expanded")

alt.data_transformers.disable_max_rows()
ZCOLOR = alt.Scale(domain=["z1", "z2", "z4", "z5"],
                   range=["#c0392b", "#2e74b5", "#2e8b57", "#d98032"])


# --------------------------------------------------------------------------- #
# cached computation
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading hourly table…")
def _load():
    return f5.load_hourly()


@st.cache_data(show_spinner="Calibrating thermal model…")
def _physics(zones, hours, use_solar):
    return f5.fit_physics(_load(), list(zones), list(hours), use_solar)


@st.cache_resource(show_spinner="Training baseline models…")
def _baselines(zones, hours):
    return f5.fit_baselines(_load(), list(zones), list(hours))


@st.cache_data(show_spinner="Re-running matching…")
def _cem(zones, hours, t_hi, t_lo, ac_on, spec, n_boot):
    return f5.run_cem(_load(), list(zones), list(hours), t_hi, t_lo, ac_on, spec, n_boot)


@st.cache_data(show_spinner="Fitting regression…")
def _reg(zones, hours):
    return f5.run_regression(_load(), list(zones), list(hours))


h = _load()
ANN, N_WEEKDAYS, N_DAYS = f5.annual_factor(h)


# --------------------------------------------------------------------------- #
# sidebar — the global levers
# --------------------------------------------------------------------------- #

st.sidebar.title("Floor 5 controls")
st.sidebar.caption("These apply to every tab.")

zones = st.sidebar.multiselect(
    "Zones", f5.ALL_ZONES, default=f5.CORE_ZONES,
    help="z1 is excluded by default: its sensor sits in a supply-air path "
         "(notebook §4). Add it back to see how much the headline depends on "
         "that call. z3 has no AC metering at all.")
if not zones:
    st.sidebar.error("Select at least one zone.")
    st.stop()
if "z1" in zones:
    st.sidebar.warning("z1 included — its temperatures are supply air, not room air.")

window = st.sidebar.radio(
    "Occupied window",
    ["Metered AC schedule (07–16, no 12)", "Notebook assumption (08–18)", "Custom"],
    help="The AC draws ~0 kWh at 12:00 and 17:00 on every weekday. The "
         "notebook's 08–18 window therefore averages eight cooled hours "
         "together with two uncooled ones.")
if window.startswith("Metered"):
    hours = f5.AC_ON_HOURS
elif window.startswith("Notebook"):
    hours = f5.NOTEBOOK_HOURS
else:
    c1, c2 = st.sidebar.columns(2)
    hs = c1.number_input("from", 0, 23, 7)
    he = c2.number_input("to (excl)", 1, 24, 17)
    drop_lunch = st.sidebar.checkbox("drop 12:00", value=True)
    hours = [x for x in range(int(hs), int(he)) if not (drop_lunch and x == 12)]
    if not hours:
        st.sidebar.error("Empty window.")
        st.stop()

st.sidebar.divider()
tariff = st.sidebar.number_input("Tariff (THB/kWh)", 0.0, 20.0, 4.20, 0.10,
                                 help="MEA large-commercial rates sit near 4 THB/kWh.")
scope = st.sidebar.selectbox(
    "Apply result to", ["This floor only", "Floors 3+4+5+7 (identical schema)"],
    help="Floors 3, 4, 5 and 7 share an identical zone and channel layout. "
         "Scaling to them is an extrapolation, not a measurement.")
scope_mult = 4.0 if scope.startswith("Floors") else 1.0

st.sidebar.divider()
st.sidebar.caption(f"Record: {N_DAYS} days, {N_WEEKDAYS} weekdays "
                   f"(annualisation ×{ANN:.3f})")

hours_t = tuple(hours)
zones_t = tuple(zones)
phys = _physics(zones_t, hours_t, True)
reg = _reg(zones_t, hours_t)
energy = f5.energy_summary(h, zones, hours)
comfort = f5.comfort_summary(h, zones, hours)


# --------------------------------------------------------------------------- #
# the sensitivity estimate — the one number everything downstream depends on
# --------------------------------------------------------------------------- #

def estimator_menu(key_prefix, sidebar=True):
    """Build the menu of available %/degC estimates and return the chosen one."""
    opts = {}
    if phys.get("pooled"):
        p = phys["pooled"]
        opts["Physics — envelope conduction (floor)"] = (
            p["pct_per_c"], p["pct_per_c_lo"], p["pct_per_c_hi"],
            "Calibrated on the 12:00 shutdown. Immune to the occupancy confound, "
            "but counts envelope conduction only — a genuine lower bound.")
    if reg.get("pooled"):
        r = reg["pooled"]
        opts["Regression on indoor temperature (ceiling)"] = (
            r["pct_per_c"], r["lo"], r["hi"],
            "Largest estimate, and the most confounded: warm hours are also busy "
            "hours, so occupancy loads onto the temperature coefficient.")
    opts["Midpoint of the two"] = None  # filled below
    if phys.get("pooled") and reg.get("pooled"):
        a, b = phys["pooled"]["pct_per_c"], reg["pooled"]["pct_per_c"]
        opts["Midpoint of the two"] = (
            (a + b) / 2, min(a, b), max(a, b),
            "Assumes latent (dehumidification) savings roughly match the "
            "envelope savings. Defensible, but an assumption.")
    else:
        opts.pop("Midpoint of the two")
    opts["Custom"] = (-5.0, -8.0, -2.0, "Set it yourself.")

    box = st.sidebar if sidebar else st
    choice = box.selectbox("Setpoint sensitivity", list(opts),
                           index=len(opts) - 2 if len(opts) > 2 else 0,
                           key=f"{key_prefix}_est")
    val, lo, hi, note = opts[choice]
    if choice == "Custom":
        val = box.slider("%/°C (negative = saving)", -15.0, 2.0, -5.0, 0.1,
                         key=f"{key_prefix}_custom")
        lo, hi = val * 1.3, val * 0.7
    return choice, val, lo, hi, note, opts


st.sidebar.divider()
est_name, est, est_lo, est_hi, est_note, EST_OPTS = estimator_menu("side")
st.sidebar.caption(est_note)


# --------------------------------------------------------------------------- #
st.title("Floor 5 — over-cooling what-if model")
st.caption(
    "The notebook answers one question with one number. This answers "
    "*what happens if* — for each lever an FM team can actually pull, with the "
    "uncertainty kept visible instead of collapsed."
)

tabs = st.tabs([
    "① Headline", "② Setpoint", "③ Comfort frontier", "④ Where the load goes",
    "⑤ Weather", "⑥ Schedule", "⑦ How fragile", "⑧ Pilot design",
])


# =========================================================================== #
# 1. headline
# =========================================================================== #
with tabs[0]:
    st.subheader("What the floor actually does")

    occ_kwh = energy.kwh_record.sum()
    occ_yr = energy.kwh_year.sum() * scope_mult
    cold_share = (comfort.ac_kwh * comfort.pct_kwh_below_cold / 100).sum() / comfort.ac_kwh.sum() * 100
    mean_c = (comfort.mean_c * comfort.hours).sum() / comfort.hours.sum()

    c = st.columns(4)
    c[0].metric("Occupied AC energy", f"{occ_yr/1000:,.0f} MWh/yr",
                help=f"{occ_kwh:,.0f} kWh over the {N_DAYS}-day record, annualised"
                     + (" and ×4 for scope" if scope_mult > 1 else ""))
    c[1].metric("Mean occupied temp", f"{mean_c:.1f} °C",
                delta=f"{mean_c - 24:.1f} vs 24 °C target", delta_color="inverse")
    c[2].metric("Cooling into space < 23 °C", f"{cold_share:.0f}%",
                help="Exposure, not waste — some of it is what holds the zone "
                     "at its current temperature.")
    c[3].metric("Annual bill (occupied AC)", f"฿{occ_yr*tariff/1e6:,.2f}m")

    st.markdown(
        f"**Sensitivity in use: `{est:+.1f}%` per °C** ({est_name}). "
        f"Plausible range `{min(est_lo,est_hi):+.1f}%` to `{max(est_lo,est_hi):+.1f}%`."
    )

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Weekday profile — the schedule the notebook assumed isn't the one metered**")
        hp = f5.hour_profile(h, zones).reset_index()
        hp["in_window"] = hp.hour.isin(hours)
        base = alt.Chart(hp).encode(x=alt.X("hour:O", title="hour of day"))
        bars = base.mark_bar().encode(
            y=alt.Y("mean_kwh:Q", title="mean AC kWh per zone-hour"),
            color=alt.condition("datum.in_window", alt.value("#2e74b5"), alt.value("#d5d5d5")),
            tooltip=["hour", alt.Tooltip("mean_kwh:Q", format=".2f"),
                     alt.Tooltip("mean_c:Q", format=".1f", title="mean °C"),
                     alt.Tooltip("frac_off:Q", format=".0%", title="frac AC off")])
        line = base.mark_line(color="#c0392b", point=True, strokeWidth=2).encode(
            y=alt.Y("mean_c:Q", title="mean indoor °C",
                    scale=alt.Scale(zero=False)))
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent")
                        .properties(height=300), width="stretch")
        st.caption("Blue = in your selected window. The red line is indoor "
                   "temperature: it jumps ~3 °C in the two hours cooling is off "
                   "(12:00, 17:00), which is the drift this app calibrates on.")

    with right:
        st.markdown("**Per zone**")
        show = comfort[["hours", "mean_c", "sd_c", "pct_below_cold",
                        "pct_above_warm", "pct_kwh_below_cold"]].copy()
        show.columns = ["hours", "mean °C", "sd °C", "% hrs <23", "% hrs >26", "% kWh <23"]
        show["kWh/yr"] = (energy.kwh_year * scope_mult).round(0)
        st.dataframe(show.round(2), width="stretch")

        st.markdown("**Estimates on the table**")
        er = []
        for k, v in EST_OPTS.items():
            if v and k != "Custom":
                er.append({"estimate": k.split(" — ")[0].split(" (")[0],
                           "%/°C": round(v[0], 1),
                           "low": round(min(v[1], v[2]), 1),
                           "high": round(max(v[1], v[2]), 1)})
        st.dataframe(pd.DataFrame(er), width="stretch", hide_index=True)

    st.info(
        "**The reframe.** The notebook's verdict was *inconclusive* because one "
        "estimator (matching on indoor temperature) failed its balance test. But "
        "the 12:00 shutdown gives a second, independent estimator that needs no "
        "matching at all — and it agrees in sign on **all three zones**, including "
        "the z5 that flipped positive under matching. The honest answer is not "
        "\"we don't know\"; it is **a bounded range, −2% to −8% per °C**, with the "
        "physics at the bottom and the confounded regression at the top."
    )


# =========================================================================== #
# 2. setpoint simulator
# =========================================================================== #
with tabs[1]:
    st.subheader("If the occupied setpoint went up, what would it save?")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        dT = st.slider("Raise occupied setpoint by (°C)", 0.0, 4.0, 1.5, 0.25,
                       key="sp_dt")
    with c2:
        per_zone_est = st.toggle("Per-zone sensitivity", value=False,
                                 help="Use each zone's own estimate instead of "
                                      "the pooled one. Reveals that z5 disagrees.")
    with c3:
        realise = st.slider("Realisation factor", 0.3, 1.0, 1.0, 0.05,
                            help="Share of the modelled saving you expect to "
                                 "actually bank. IPMVP practice discounts for "
                                 "control overrides, drift and hot calls.")

    rows = []
    for z in zones:
        kwh_yr = energy.kwh_year.get(z, 0.0) * scope_mult
        if per_zone_est:
            if est_name.startswith("Physics") and z in phys["per_zone"]:
                e = phys["per_zone"][z]["pct_per_c"]
                elo = phys["per_zone"][z]["pct_per_c_lo"]
                ehi = phys["per_zone"][z]["pct_per_c_hi"]
            else:
                m = [r for r in reg["per_zone"] if r["zone"] == z]
                e, elo, ehi = (m[0]["pct_per_c"], m[0]["lo"], m[0]["hi"]) if m else (est, est_lo, est_hi)
        else:
            e, elo, ehi = est, est_lo, est_hi
        rows.append({
            "zone": z, "kWh/yr": kwh_yr, "%/°C": e,
            "saving kWh/yr": -kwh_yr * (e / 100) * dT * realise,
            "low": -kwh_yr * (min(elo, ehi) / 100) * dT * realise,
            "high": -kwh_yr * (max(elo, ehi) / 100) * dT * realise,
            "mean °C now": comfort.mean_c.get(z, np.nan),
            "mean °C after": comfort.mean_c.get(z, np.nan) + dT,
        })
    sim = pd.DataFrame(rows)
    tot = sim["saving kWh/yr"].sum()
    tlo, thi = sim["low"].sum(), sim["high"].sum()

    m = st.columns(4)
    m[0].metric("Energy saved", f"{tot:,.0f} kWh/yr",
                delta=f"{tot/max(sim['kWh/yr'].sum(),1)*100:.1f}% of occupied AC")
    m[1].metric("Cost saved", f"฿{tot*tariff:,.0f}/yr")
    m[2].metric("Range", f"฿{min(tlo,thi)*tariff:,.0f} – ฿{max(tlo,thi)*tariff:,.0f}")
    m[3].metric("CO₂ avoided", f"{tot*0.4999/1000:,.1f} t/yr",
                help="Thailand grid factor 0.4999 kgCO₂/kWh (TGO, 2022).")

    st.dataframe(
        sim.assign(**{"saving kWh/yr": sim["saving kWh/yr"].round(0),
                      "saving THB/yr": (sim["saving kWh/yr"] * tariff).round(0)})
           .round(2)[["zone", "kWh/yr", "%/°C", "mean °C now", "mean °C after",
                      "saving kWh/yr", "saving THB/yr"]],
        width="stretch", hide_index=True)

    st.divider()
    st.markdown("**Saving against setpoint rise — every estimator at once**")
    grid = []
    for k, v in EST_OPTS.items():
        if not v or k == "Custom":
            continue
        for d in np.arange(0, 4.01, 0.25):
            grid.append({"estimator": k.split(" — ")[0].split(" (")[0],
                         "dT": d,
                         "kWh": -energy.kwh_year.sum() * scope_mult * (v[0] / 100) * d * realise})
    g = pd.DataFrame(grid)
    band = pd.DataFrame([
        {"dT": d,
         "lo": -energy.kwh_year.sum() * scope_mult * (min(est_lo, est_hi) / 100) * d * realise,
         "hi": -energy.kwh_year.sum() * scope_mult * (max(est_lo, est_hi) / 100) * d * realise}
        for d in np.arange(0, 4.01, 0.25)])

    ch = alt.layer(
        alt.Chart(band).mark_area(opacity=0.15, color="#2e74b5").encode(
            x=alt.X("dT:Q", title="setpoint rise (°C)"),
            y=alt.Y("lo:Q", title="saving (kWh/yr)"), y2="hi:Q"),
        alt.Chart(g).mark_line(strokeWidth=2.5).encode(
            x="dT:Q", y="kWh:Q",
            color=alt.Color("estimator:N", legend=alt.Legend(orient="top", columns=2)),
            tooltip=["estimator", alt.Tooltip("dT:Q", format=".2f"),
                     alt.Tooltip("kWh:Q", format=",.0f")]),
        alt.Chart(pd.DataFrame({"dT": [dT]})).mark_rule(
            color="#c0392b", strokeDash=[5, 3]).encode(x="dT:Q"),
    ).properties(height=340)
    st.altair_chart(ch, width="stretch")
    st.caption("The shaded band is the interval on your selected estimator. The "
               "spread *between the lines* is the real uncertainty — it is "
               "method disagreement, not sampling noise, and no bigger sample "
               "would shrink it.")


# =========================================================================== #
# 3. comfort frontier
# =========================================================================== #
with tabs[2]:
    st.subheader("Setpoint is one lever. Control stability is a second.")
    st.caption(
        "The notebook found that lifting each zone's mean to 24 °C still leaves "
        "39–48% of hours below 23 °C, because these distributions are wide, not "
        "just mis-centred. That means two separate work orders — and this is "
        "where you see which one buys the comfort."
    )

    c1, c2, c3, c4 = st.columns(4)
    shift = c1.slider("Setpoint rise (°C)", 0.0, 4.0, 1.5, 0.25, key="cf_shift")
    sd_scale = c2.slider("Deadband tightening", 0.4, 1.0, 1.0, 0.05,
                         help="1.00 = spread unchanged. 0.60 = control loop "
                              "retuned to cut swing by 40%.")
    cold = c3.slider("Cold threshold (°C)", 20.0, 25.0, 23.0, 0.5)
    warm = c4.slider("Warm threshold (°C)", 24.0, 29.0, 26.0, 0.5)

    d = f5.occupied(h, zones, hours, require_sensor=True, ac_on=True)
    parts, summ = [], []
    for z, g in d.groupby("zone"):
        after = f5.shift_distribution(g.temp_c, shift, sd_scale)
        parts.append(pd.DataFrame({"zone": z, "°C": g.temp_c.values, "state": "now"}))
        parts.append(pd.DataFrame({"zone": z, "°C": after, "state": "after"}))
        summ.append({
            "zone": z,
            "% <cold now": (g.temp_c < cold).mean() * 100,
            "% <cold after": (after < cold).mean() * 100,
            "% >warm now": (g.temp_c > warm).mean() * 100,
            "% >warm after": (after > warm).mean() * 100,
            "mean now": g.temp_c.mean(), "mean after": after.mean(),
            "sd now": g.temp_c.std(), "sd after": after.std(),
        })
    S = pd.DataFrame(summ)

    k = st.columns(4)
    k[0].metric("Hours below cold threshold",
                f"{S['% <cold after'].mean():.0f}%",
                delta=f"{S['% <cold after'].mean() - S['% <cold now'].mean():+.0f} pp",
                delta_color="inverse")
    k[1].metric("Hours above warm threshold",
                f"{S['% >warm after'].mean():.0f}%",
                delta=f"{S['% >warm after'].mean() - S['% >warm now'].mean():+.0f} pp",
                delta_color="inverse")
    k[2].metric("Mean temp", f"{S['mean after'].mean():.1f} °C")
    k[3].metric("Mean spread (sd)", f"{S['sd after'].mean():.2f} °C")

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Distribution, before and after**")
        # Faceting a layered chart needs one shared data source, so the two
        # threshold rules ride along as constant columns.
        P = pd.concat(parts).assign(_cold=cold, _warm=warm)
        area = alt.Chart().mark_area(opacity=0.45, interpolate="monotone").encode(
            x=alt.X("°C:Q", bin=alt.Bin(maxbins=45)),
            y=alt.Y("count()", stack=None, title="zone-hours"),
            color=alt.Color("state:N", scale=alt.Scale(
                domain=["now", "after"], range=["#9bb7d4", "#2e74b5"]),
                legend=alt.Legend(orient="top")))
        rules = alt.Chart().mark_rule(color="#c0392b", strokeDash=[4, 3])
        st.altair_chart(
            alt.layer(area,
                      rules.encode(x="mean(_cold):Q"),
                      rules.encode(x="mean(_warm):Q"),
                      data=P).properties(height=190).facet(
                          row=alt.Row("zone:N", title=None)),
            width="stretch")

    with cR:
        st.markdown("**The frontier — cold hours traded for warm hours**")
        fr = f5.comfort_frontier(h, zones, hours, cold, warm, sd_scale)
        fl = fr.melt(id_vars=["zone", "shift"], value_vars=["pct_cold", "pct_warm"],
                     var_name="band", value_name="pct")
        fl["band"] = fl["band"].map({"pct_cold": f"< {cold:g} °C",
                                     "pct_warm": f"> {warm:g} °C"})
        st.altair_chart(
            alt.layer(
                alt.Chart(fl).mark_line(strokeWidth=2.5, point=False).encode(
                    x=alt.X("shift:Q", title="setpoint rise (°C)"),
                    y=alt.Y("pct:Q", title="% of occupied hours"),
                    color=alt.Color("band:N", title=None, scale=alt.Scale(
                        range=["#2e74b5", "#c0392b"]), legend=alt.Legend(orient="top")),
                    strokeDash=alt.StrokeDash("zone:N", legend=alt.Legend(orient="top")),
                    tooltip=["zone", "band", alt.Tooltip("shift:Q", format=".2f"),
                             alt.Tooltip("pct:Q", format=".1f")]),
                alt.Chart(pd.DataFrame({"s": [shift]})).mark_rule(
                    color="#555", strokeDash=[5, 3]).encode(x="s:Q"),
            ).properties(height=340), width="stretch")
        st.caption("Move the *deadband* slider and watch both curves flatten at "
                   "once — tightening control cuts the cold tail without "
                   "creating a warm one, which a setpoint change cannot do.")

    st.dataframe(S.round(2), width="stretch", hide_index=True)
    st.warning(
        f"At +{shift:g} °C with the spread unchanged, roughly "
        f"**{S['% <cold after'].mean():.0f}%** of occupied hours are still below "
        f"{cold:g} °C. Setpoint alone does not close that gap — a zone that stays "
        "volatile after the change is a controls-tuning item, not a setpoint item."
    )


# =========================================================================== #
# 4. load decomposition
# =========================================================================== #
with tabs[3]:
    st.subheader("Where the cooling load actually goes")
    st.caption(
        "This is why the two estimators disagree, and it is the most useful "
        "single chart for the presentation."
    )

    use_solar = st.toggle(
        "Include solar in the drift calibration", value=True,
        help="Solar and the outdoor-indoor driving force correlate at ~0.62 at "
             "midday. Dropping solar moves UA by about a third — an honest "
             "sensitivity to show rather than hide.")
    ph = _physics(zones_t, hours_t, use_solar)

    if not ph.get("pooled"):
        st.error("Not enough AC-off hours in this selection to calibrate.")
    else:
        dec = f5.load_decomposition(h, ph, zones, hours)
        p = ph["pooled"]

        k = st.columns(4)
        k[0].metric("Thermal capacitance C", f"{p['C_kwh_per_c']:.2f} kWh/°C")
        k[1].metric("Envelope conductance UA", f"{p['UA_kw_per_c']:.3f} kW/°C")
        k[2].metric("Implied sensitivity", f"{p['pct_per_c']:+.1f} %/°C")
        k[3].metric("Calibration sample", f"{ph['n']:,} zone-days")

        cL, cR = st.columns([3, 2])
        with cL:
            dl = dec.reset_index().melt(
                id_vars="zone", value_vars=["envelope_kw", "internal_kw", "residual_kw"],
                var_name="component", value_name="kW")
            dl["component"] = dl.component.map({
                "envelope_kw": "Envelope — UA·ΔT (setpoint acts here)",
                "internal_kw": "Internal — metered light + plug (untouchable)",
                "residual_kw": "Residual — latent, occupants, fan, plant"})
            st.altair_chart(
                alt.Chart(dl).mark_bar().encode(
                    y=alt.Y("zone:N", title=None),
                    x=alt.X("kW:Q", title="mean cooling load (kW) on cooled hours"),
                    color=alt.Color("component:N", scale=alt.Scale(
                        range=["#2e74b5", "#d98032", "#9b9b9b"]),
                        legend=alt.Legend(orient="bottom", columns=1, title=None)),
                    order=alt.Order("component:N"),
                    tooltip=["zone", "component", alt.Tooltip("kW:Q", format=".2f")]
                ).properties(height=220), width="stretch")

        with cR:
            st.dataframe(
                dec[["measured_kw", "envelope_pct", "internal_pct", "residual_pct"]]
                   .rename(columns={"measured_kw": "load kW", "envelope_pct": "envelope %",
                                    "internal_pct": "internal %", "residual_pct": "residual %"})
                   .round(1), width="stretch")
            st.dataframe(
                pd.DataFrame([{"zone": z, "C kWh/°C": round(v["C_kwh_per_c"], 2),
                               "UA kW/°C": round(v["UA_kw_per_c"], 3),
                               "%/°C": round(v["pct_per_c"], 1),
                               "R²": round(v["r2"], 2), "n": v["n"]}
                              for z, v in ph["per_zone"].items()]),
                width="stretch", hide_index=True)

        env_share = dec.envelope_pct.mean()
        st.success(
            f"**Envelope conduction is only {env_share:.0f}% of the cooling load.** "
            f"A setpoint change acts directly on that share and on some unknown "
            f"part of the latent load inside the residual. It acts on *none* of "
            f"the metered internal gains ({dec.internal_pct.mean():.0f}%). That "
            f"is the arithmetic reason the physics estimate lands near "
            f"{p['pct_per_c']:+.1f}%/°C while a regression on indoor temperature "
            f"reports {reg['pooled']['pct_per_c']:+.1f}%/°C — the regression is "
            f"claiming savings larger than the entire envelope term, which it "
            f"can only do by absorbing occupancy."
        )

        with st.expander("How the calibration works, and the drift data"):
            st.markdown(
                "At 12:00 the AC draws ~0 kWh on every weekday. For that hour the "
                "zone is a free-running thermal mass whose only *measured* heat "
                "input is lighting + plug load:\n\n"
                "$$C\\,\\frac{dT}{dt} = UA\\,(T_{out}-T_{in}) + Q_{int} + a\\,I_{sol}$$\n\n"
                "Regressing the hourly drift on the metered internal gain "
                "identifies $1/C$; the driving-force coefficient then gives $UA$. "
                "In steady cooled operation the envelope term is the part of the "
                "load a setpoint change removes, so "
                "$d(\\text{load})/d(T_{set}) = -UA$.\n\n"
                "No indoor temperature appears on the right-hand side as a "
                "*cause* of energy, which is exactly the leakage the notebook "
                "warned about — so this estimate cannot be confounded by "
                "occupancy the way the matching design is."
            )
            st.code(str(ph["model"].summary()), language="text")
            L = ph["frame"]
            st.altair_chart(
                alt.Chart(L).mark_circle(opacity=0.45, size=35).encode(
                    x=alt.X("drive:Q", title="driving force T_out − T_in (°C)"),
                    y=alt.Y("drift:Q", title="free-running drift (°C/hour)"),
                    color=alt.Color("zone:N", scale=ZCOLOR),
                    tooltip=["zone", "date", alt.Tooltip("drive:Q", format=".1f"),
                             alt.Tooltip("drift:Q", format=".2f"),
                             alt.Tooltip("int_kwh:Q", format=".2f", title="internal kW")]
                ).properties(height=300)
                + alt.Chart(L).transform_regression("drive", "drift", groupby=["zone"])
                   .mark_line(strokeWidth=2).encode(
                       x="drive:Q", y="drift:Q", color=alt.Color("zone:N", scale=ZCOLOR)),
                width="stretch")


# =========================================================================== #
# 5. weather scenario
# =========================================================================== #
with tabs[4]:
    st.subheader("What would the energy have been? — the M&V baseline")
    st.caption(
        "The exogenous-feature model from notebook §6. It never sees indoor "
        "temperature, so it answers the counterfactual finance actually asks: "
        "what would we have spent if nothing had changed?"
    )

    fitted, perf = _baselines(zones_t, hours_t)
    if not fitted:
        st.error("Not enough data in this window to fit baselines.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        d_temp = c1.slider("Outdoor temp shift (°C)", -3.0, 5.0, 0.0, 0.25)
        solar_scale = c2.slider("Solar multiplier", 0.5, 1.5, 1.0, 0.05)
        d_rh = c3.slider("Humidity shift (pp)", -20.0, 20.0, 0.0, 1.0)
        mdl = c4.selectbox("Model", ["gbm", "ridge"],
                           format_func=lambda s: {"gbm": "GBM (selected)",
                                                  "ridge": "Ridge"}[s])

        sc = f5.weather_scenario(h, fitted, zones, hours, d_temp, solar_scale, d_rh, mdl)
        if len(sc):
            dk = sc.delta_kwh.sum() * ANN * scope_mult
            k = st.columns(4)
            k[0].metric("Predicted change in load", f"{dk:,.0f} kWh/yr",
                        delta=f"{sc.delta_kwh.sum()/sc.pred_base_kwh.sum()*100:+.1f}%")
            k[1].metric("Cost impact", f"฿{dk*tariff:,.0f}/yr")
            k[2].metric("Mean CV(RMSE), test",
                        f"{perf[(perf.model==mdl.upper().replace('GBM','GBM')) & perf.split.str.startswith('test')].cv_rmse.mean():.1f}%"
                        if len(perf) else "—")
            offset = -dk / max(energy.kwh_year.sum() * scope_mult, 1) * 100
            k[3].metric("Setpoint °C to offset this",
                        f"{offset/est:.2f} °C" if est else "—",
                        help="How much setpoint rise would cancel the weather "
                             "change, at your selected sensitivity.")

            st.dataframe(sc.round(1), width="stretch", hide_index=True)

            mo = f5.monthly_scenario(h, fitted, zones, hours, d_temp, solar_scale, d_rh, mdl)
            if len(mo):
                st.markdown("**Where the weather sensitivity lands across the year**")
                ml = mo.melt(id_vars="month", value_vars=["pred_base", "pred_scen"],
                             var_name="case", value_name="kWh")
                ml["case"] = ml.case.map({"pred_base": "baseline",
                                          "pred_scen": "scenario"})
                st.altair_chart(
                    alt.Chart(ml).mark_bar().encode(
                        x=alt.X("month:O", title="month"),
                        y=alt.Y("kWh:Q", title="predicted AC kWh"),
                        xOffset="case:N",
                        color=alt.Color("case:N", scale=alt.Scale(
                            range=["#9bb7d4", "#c0392b"]), legend=alt.Legend(orient="top")),
                        tooltip=["month", "case", alt.Tooltip("kWh:Q", format=",.0f")]
                    ).properties(height=260), width="stretch")

        st.divider()
        st.markdown("**Baseline quality — ASHRAE Guideline 14 (CV(RMSE) ≤ 30%, |NMBE| ≤ 10%)**")
        st.dataframe(perf.round(1), width="stretch", hide_index=True)
        bad = perf[(perf.model == "GBM") & (perf.meets_g14 == "no")
                   & perf.split.str.startswith("test")]
        if len(bad):
            st.warning(
                "Fails G14 on the test period for: "
                + ", ".join(bad.zone) +
                ". A zone whose baseline fails cannot be used to prove a saving "
                "— on z4 this is the unexplained Oct-2019 level shift."
            )

        st.markdown("**Actual vs baseline, held-out period**")
        rows = []
        for z, f in fitted.items():
            te = f["test"]
            pr = np.clip(f[mdl].predict(te[f5.FEATURES]), 0, None)
            t = pd.DataFrame({"ts": pd.to_datetime(te.ts), "actual": te.ac_kwh.values,
                              "baseline": pr}).set_index("ts").resample("D").sum()
            t["zone"] = z
            rows.append(t.reset_index())
        if rows:
            R = pd.concat(rows).melt(id_vars=["ts", "zone"],
                                     value_vars=["actual", "baseline"],
                                     var_name="series", value_name="kWh")
            st.altair_chart(
                alt.Chart(R).mark_line(strokeWidth=1.8).encode(
                    x=alt.X("ts:T", title=None),
                    y=alt.Y("kWh:Q", title="kWh/day"),
                    color=alt.Color("series:N", scale=alt.Scale(
                        range=["#2e74b5", "#c0392b"]), legend=alt.Legend(orient="top")),
                    strokeDash=alt.StrokeDash("series:N", legend=None),
                ).properties(height=170).facet(
                    column=alt.Column("zone:N", title=None)),
                width="stretch")


# =========================================================================== #
# 6. schedule trim
# =========================================================================== #
with tabs[5]:
    st.subheader("Trimming the cooling schedule")
    st.caption(
        "A different lever entirely, and the only one measured rather than "
        "inferred: the energy in an hour you switch off is metered. What the "
        "physics adds is the *recovery* charge — the heat you must pull back out "
        "when cooling resumes."
    )

    hp = f5.hour_profile(h, zones)
    on_hours = [x for x in sorted(hp.index) if hp.loc[x, "frac_off"] < 0.5]
    st.markdown("Cooling currently runs in: " + ", ".join(f"`{x:02d}:00`" for x in on_hours))

    drop = st.multiselect(
        "Switch cooling off in these hours", on_hours,
        default=[x for x in on_hours if x == max(on_hours)],
        help="Trimming the last hour of the day is nearly free — nothing has to "
             "be recovered. Extending the lunch gap is not, because cooling "
             "resumes at 13:00.")
    recover = st.toggle("Charge the recovery energy back", value=True)

    if drop and phys.get("pooled"):
        tr = f5.schedule_trim(h, phys, zones, tuple(drop), recover)
        gross = tr.gross_kwh_year.sum() * scope_mult
        net = tr.net_kwh_year.sum() * scope_mult

        k = st.columns(4)
        k[0].metric("Gross energy released", f"{gross:,.0f} kWh/yr")
        k[1].metric("Net of recovery", f"{net:,.0f} kWh/yr",
                    delta=f"{(net-gross):,.0f} recovery charge")
        k[2].metric("Cost saved (net)", f"฿{net*tariff:,.0f}/yr")
        k[3].metric("Temp at end of gap", f"{tr.end_temp_c.mean():.1f} °C",
                    help="Predicted from the calibrated drift rate. Above 26 °C "
                         "is a complaint risk.")

        st.dataframe(
            tr[["gross_kwh_year", "recovery_kwh", "net_kwh_year",
                "drift_c_per_h", "end_temp_c", "resumes_same_day"]]
              .rename(columns={"gross_kwh_year": "gross kWh/yr",
                               "recovery_kwh": "recovery kWh (record)",
                               "net_kwh_year": "net kWh/yr",
                               "drift_c_per_h": "drift °C/h",
                               "end_temp_c": "temp at end °C",
                               "resumes_same_day": "cooling resumes after"})
              .round(2), width="stretch")

        if tr.end_temp_c.mean() > 26:
            st.error(
                f"Predicted {tr.end_temp_c.mean():.1f} °C by the end of the gap — "
                "above the 26 °C complaint threshold. This trim is not free."
            )
        elif not tr.resumes_same_day.any():
            st.success(
                "Cooling does not resume after this gap, so there is no recovery "
                "charge: the gross saving is the net saving. This is the cheapest "
                "lever on the whole floor."
            )
        else:
            st.info(
                f"Cooling resumes the same day, so {gross-net:,.0f} kWh/yr of the "
                f"gross figure is spent pulling the zone back down. The net saving "
                f"is {net/max(gross,1)*100:.0f}% of the gross."
            )
    elif drop:
        st.error("Thermal model not calibrated for this selection.")
    else:
        st.info("Select at least one hour to trim.")

    st.divider()
    st.markdown("**Measured energy and temperature by hour**")
    st.dataframe(hp.rename(columns={"mean_kwh": "mean kWh", "total_kwh": "total kWh",
                                    "frac_off": "frac AC off", "mean_c": "mean °C"})
                   .round(2), width="stretch")
    st.caption(
        "Note hours 12 and 17: cooling is off ~94% of weekdays, yet the "
        "notebook's 08–18 occupied window counts them. That pulls the reported "
        "mean occupied temperature up by ~0.2 °C and dilutes the KPI denominator "
        "by two hours in ten."
    )


# =========================================================================== #
# 7. robustness
# =========================================================================== #
with tabs[6]:
    st.subheader("How fragile is the causal estimate?")
    st.caption(
        "The notebook declared a decision gate and reported *inconclusive*. "
        "Rather than take that on trust, move the design choices and watch the "
        "answer move. This re-runs the whole matching pipeline on every change."
    )

    c1, c2, c3 = st.columns(3)
    t_hi = c1.slider("Treated: temp ≥ (°C)", 23.0, 26.0, 24.0, 0.5)
    t_lo = c1.slider("Control: temp ≤ (°C)", 20.0, 23.0, 22.0, 0.5)
    ac_on = c2.slider("Count as cooling above (kWh)", 0.0, 3.0, 0.5, 0.1)
    spec = c2.selectbox("Matching specification", list(f5.SPECS), index=1)
    n_boot = c3.select_slider("Bootstrap reps", [100, 200, 500, 1000], value=200)
    run = c3.button("Re-run matching", type="primary", width="stretch")

    if t_lo >= t_hi:
        st.error("Control cut must be below the treated cut.")
    elif run or st.session_state.get("_ran_cem"):
        st.session_state["_ran_cem"] = True
        res = _cem(zones_t, hours_t, t_hi, t_lo, ac_on, spec, n_boot)
        if not res.get("att"):
            st.error("No matched strata survive this specification — the sample "
                     "collapsed. That is itself the finding: the design has very "
                     "little overlap to work with.")
        else:
            k = st.columns(4)
            k[0].metric("Estimate", f"{res['pct_per_c']:+.1f} %/°C")
            k[1].metric("95% CI",
                        f"[{res['ci'][0]:+.1f}, {res['ci'][1]:+.1f}]")
            k[2].metric("Matched treated N", f"{res['att']['n_treated']:,}")
            k[3].metric("Worst |SMD|", f"{res['worst_smd']:.3f}",
                        delta="balanced" if res["worst_smd"] < 0.1 else "imbalanced",
                        delta_color="normal" if res["worst_smd"] < 0.1 else "inverse")

            if res["verdict"] == "CONCLUSIVE":
                st.success(f"**Gate: {res['verdict']}** on this specification.")
            else:
                st.warning(f"**Gate: {res['verdict']}** on this specification.")

            gc = st.columns([2, 3])
            with gc[0]:
                st.dataframe(pd.DataFrame(
                    [{"criterion": k2, "value": str(v[1]),
                      "result": "PASS" if v[0] else "FAIL"}
                     for k2, v in res["gate"].items()]),
                    width="stretch", hide_index=True)
                st.markdown("**Covariate balance**")
                b = res["balance"].copy()
                b["ok"] = np.where(b.ok, "OK", "FAIL")
                st.dataframe(b.round(3), width="stretch")

            with gc[1]:
                st.markdown("**Bootstrap distribution** (day-clustered)")
                B = pd.DataFrame({"pct": res["boot"]})
                st.altair_chart(
                    alt.layer(
                        alt.Chart(B).mark_bar(opacity=0.8, color="#2e74b5").encode(
                            x=alt.X("pct:Q", bin=alt.Bin(maxbins=40),
                                    title="% of load per °C"),
                            y=alt.Y("count()", title="draws")),
                        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                            color="#c0392b", strokeWidth=2).encode(x="x:Q"),
                        alt.Chart(pd.DataFrame({"x": [res["pct_per_c"]]})).mark_rule(
                            color="#111", strokeDash=[5, 3]).encode(x="x:Q"),
                    ).properties(height=260), width="stretch")
                st.caption(f"{(res['boot'] >= 0).mean()*100:.1f}% of draws at or "
                           "above zero. The red rule is zero; the dashed rule is "
                           "the point estimate.")

            if res.get("counts") is not None:
                st.markdown("**Treated / control counts**")
                st.dataframe(res["counts"], width="stretch")
    else:
        st.info("Set the cuts, then press **Re-run matching**. "
                "Each run re-matches and bootstraps, so it takes a few seconds.")

    st.divider()
    st.markdown("**Per-zone regression estimates — where the pooled number breaks**")
    if reg["per_zone"]:
        RZ = pd.DataFrame(reg["per_zone"])
        st.altair_chart(
            alt.layer(
                alt.Chart(RZ).mark_rule(strokeWidth=3).encode(
                    y=alt.Y("zone:N", title=None),
                    x=alt.X("lo:Q", title="% of load per °C"), x2="hi:Q",
                    color=alt.Color("sign:N", scale=alt.Scale(
                        domain=["as expected", "WRONG SIGN"],
                        range=["#2e74b5", "#c0392b"]), legend=alt.Legend(orient="top"))),
                alt.Chart(RZ).mark_point(size=110, filled=True).encode(
                    y="zone:N", x="pct_per_c:Q",
                    color=alt.Color("sign:N", legend=None),
                    tooltip=["zone", "n", alt.Tooltip("pct_per_c:Q", format=".2f"),
                             "sign"]),
                alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                    color="#111", strokeDash=[4, 3]).encode(x="x:Q"),
            ).properties(height=160), width="stretch")
        st.dataframe(RZ.round(2), width="stretch", hide_index=True)
        st.caption(
            "z5's positive sign is the occupancy confound made visible: a busy "
            "day makes the zone warmer *and* raises its cooling load. Compare "
            "with tab ④, where the physics estimator gives all three zones the "
            "same sign — because indoor temperature never enters it."
        )


# =========================================================================== #
# 8. pilot design
# =========================================================================== #
with tabs[7]:
    st.subheader("Could a pilot even detect this?")
    st.caption(
        "The notebook's protocol specifies 1 °C steps held for two to four "
        "weeks. Whether that is long enough is a power question, and it depends "
        "on the baseline's own error — so it can be answered from the CV(RMSE) "
        "already measured."
    )

    fitted, perf = _baselines(zones_t, hours_t)
    default_cv = 30.0
    if len(perf):
        t = perf[(perf.model == "GBM") & perf.split.str.startswith("test")]
        if len(t):
            default_cv = float(round(t.cv_rmse.mean(), 1))

    c1, c2, c3, c4 = st.columns(4)
    cv = c1.slider("Baseline CV(RMSE) %", 10.0, 60.0, default_cv, 0.5,
                   help=f"Measured mean on the held-out period: {default_cv:.1f}%")
    step = c2.slider("Setpoint step (°C)", 0.5, 3.0, 1.0, 0.5)
    rho = c3.slider("Within-day correlation ρ", 0.0, 0.9, 0.5, 0.05,
                    help="Hours inside one day are not independent. ρ=0 is the "
                         "optimistic textbook case; 0.5 is realistic.")
    power = c4.slider("Power", 0.5, 0.95, 0.80, 0.05)

    hpd = len(hours)
    expected = abs(est * step)
    weeks_needed = f5.weeks_required(cv, expected, hpd, 5, 0.05, power, rho)

    k = st.columns(4)
    k[0].metric("Expected saving from the step", f"{expected:.1f}%",
                help=f"{est:+.1f}%/°C × {step:g} °C")
    k[1].metric("Weeks needed to detect it",
                f"{weeks_needed:.0f}" if np.isfinite(weeks_needed) else "> 5 years")
    mds4 = f5.detectable_saving(cv, 4, hpd, 5, 0.05, power, rho)
    k[2].metric("Detectable in 4 weeks", f"{mds4:.1f}%",
                delta="step is detectable" if mds4 <= expected else "step too small",
                delta_color="normal" if mds4 <= expected else "inverse")
    k[3].metric("Hours per day in window", f"{hpd}")

    W = pd.DataFrame({"weeks": np.arange(1, 53)})
    W["mds"] = [f5.detectable_saving(cv, w, hpd, 5, 0.05, power, rho) for w in W.weeks]
    st.altair_chart(
        alt.layer(
            alt.Chart(W).mark_line(strokeWidth=2.5, color="#2e74b5").encode(
                x=alt.X("weeks:Q", title="pilot length per step (weeks)"),
                y=alt.Y("mds:Q", title="minimum detectable saving (%)",
                        scale=alt.Scale(type="log")),
                tooltip=[alt.Tooltip("weeks:Q"), alt.Tooltip("mds:Q", format=".2f")]),
            alt.Chart(pd.DataFrame({"y": [expected]})).mark_rule(
                color="#c0392b", strokeWidth=2).encode(y="y:Q"),
            alt.Chart(pd.DataFrame({"y": [expected], "t": [
                f"expected effect {expected:.1f}%"]})).mark_text(
                align="left", dx=8, dy=-8, color="#c0392b").encode(y="y:Q", text="t:N"),
        ).properties(height=320), width="stretch")

    if np.isfinite(weeks_needed) and weeks_needed <= 4:
        st.success(
            f"A {step:g} °C step is detectable in about **{weeks_needed:.0f} "
            f"week(s)** against this baseline. The notebook's two-to-four-week "
            f"hold is adequate."
        )
    elif np.isfinite(weeks_needed):
        st.warning(
            f"A {step:g} °C step needs about **{weeks_needed:.0f} weeks** per "
            f"step at ρ={rho:g} — materially longer than the protocol's two to "
            f"four. Either hold each step longer, take a bigger first step, or "
            f"accept that the first step will not be individually significant "
            f"and judge the programme on the full sequence."
        )
    else:
        st.error(
            f"At {cv:.0f}% CV(RMSE) a {step:g} °C step is effectively undetectable "
            f"on this zone. Improving the baseline comes before running the pilot."
        )

    st.divider()
    st.markdown("**Per-zone readiness**")
    if len(perf):
        t = perf[(perf.model == "GBM") & perf.split.str.startswith("test")]
        rr = []
        for _, r in t.iterrows():
            w = f5.weeks_required(r.cv_rmse, expected, hpd, 5, 0.05, power, rho)
            sens = (phys["per_zone"].get(r.zone) or {}).get("pct_per_c", np.nan)
            rr.append({"zone": r.zone, "CV(RMSE) %": round(r.cv_rmse, 1),
                       "NMBE %": round(r.nmbe, 1), "meets G14": r.meets_g14,
                       "physics %/°C": round(sens, 1) if np.isfinite(sens) else None,
                       "weeks to detect": None if not np.isfinite(w) else int(w),
                       "verdict": ("pilot-ready" if r.meets_g14 == "yes"
                                   else "fix baseline first")})
        st.dataframe(pd.DataFrame(rr), width="stretch", hide_index=True)

    st.info(
        "**Reading this for the presentation.** The pilot recommendation is not "
        "just \"try it and see\" — it comes with a measurable design. Pick the "
        "zone whose baseline meets G14, size the step so the expected effect "
        "clears the detection floor, and hold it long enough that a null result "
        "actually means something."
    )
