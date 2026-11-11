"""Methods, assumptions, corrections and the explicit exclusion list."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import analysis as A
from sbanalytics import data as D
from sbanalytics import faults as F
from sbanalytics import registry as R
from sbanalytics import workorders as W
import ui

c = ui.ctx()
floor = c["floor"]
h = ui.hourly(floor) if R.FLOORS[floor]["wired"] else None

ui.page_header("Methods", "Methods & assumptions")

def _gaps(f: int) -> str:
    """This floor's per-zone exclusions, from the registry."""
    rows = []
    for z in R.ZONES[f]:
        if not R.has_channel(f, z, "ac"):
            rows.append(f"- All cooling metrics for **{z}** — no AC meter"
                        + (" (stairwell)" if ui.is_stairwell(f, z) else "") + ".")
        tp = R.point_for(f, z, "temp")
        if tp is not None and tp.role == R.ROLE_SUPPLY_TEMP:
            rows.append(f"- All room-comfort metrics for **{z}** — supply-air sensor path.")
    if not R.sensored_zones(f):
        rows.append("- Every comfort metric — this floor has no environmental sensors.")
    frozen = sorted({p.zone for p in R.points(f, "plug", roles={R.ROLE_SUSPECT_CONSTANT})})
    if frozen:
        rows.append(f"- Plug-load analysis for **{', '.join(frozen)}** — those meters are frozen.")
    dead = [p.column for p in R.points(f, roles={R.ROLE_DEAD}) if p.column]
    if dead:
        rows.append(f"- Anything from {', '.join(f'`{d}`' for d in dead)} — dead meter.")
    return ("\n".join(rows) or "- None declared in the registry.") + "\n\n"


t_conv, t_reg, t_cps, t_about = st.tabs(["Conventions", "Point registry", "Change-points", "About the data"])

# --------------------------------------------------------------------------- #
with t_conv:
    st.subheader("Declared conventions")
    st.caption("These are project decisions with sources, not standards. Where a "
               "standard exists we cite it; where it does not we say so.")
    conv = pd.DataFrame([
        {"parameter": k, "value": v[0], "unit": v[1],
         "basis": v[2] + (" — not used in the dashboard" if k == "demand_charge_thb_per_kw"
                          else " — default; adjustable in Assumptions (comfort band)"
                          if k in ("comfort_lo", "comfort_hi") else "")}
        for k, v in R.CONVENTIONS.items()])
    st.dataframe(conv, width="stretch", hide_index=True,
                 column_config={"basis": st.column_config.TextColumn(width="large")})

    st.subheader("Declared methods")
    st.caption("Rules rather than numbers: how each derived figure on the dashboard is computed.")
    st.dataframe(pd.DataFrame([{"method": k, "definition": v[0], "source": v[1]}
                               for k, v in R.METHODS.items()]),
                 width="stretch", hide_index=True,
                 column_config={"definition": st.column_config.TextColumn(width="large")})
    st.markdown(
        "**Fault lifecycle states** — " + " → ".join(W.STATES[:4]) + f", plus {W.STATES[4]}. "
        "Open = " + ", ".join(sorted(W.OPEN_STATES)) + ".\n\n"
        "**IPMVP option by measure** — zone-temperature lift and schedule trim: Option B "
        "(the comfort zones have their own AC meters); weekend/holiday lockout and the "
        "combined package: Option C (whole-floor AC). The lockout is verified by direct "
        "metering, because a weekend run is too small against a daily model's noise to "
        "detect statistically.")

    st.subheader("Carbon accounting")
    _ef = R.CONVENTIONS["co2_kg_per_kwh"]
    _fe = ", ".join(f"F{f} after {R.floor_file_end(f):%d %b %Y}" for f in R.FILE_END)
    st.markdown(
        f"- **Scope:** Scope 2, location-based — purchased grid electricity only.\n"
        f"- **Boundary:** the submetered zone loads (AC, lighting, plug) on all seven floors. Excluded: "
        f"central plant, lifts, risers and common areas; {_fe} (no file).\n"
        f"- **Formula:** tCO₂e = kWh × {_ef[0]:g} {_ef[1]} ÷ 1000. A grid-average factor, not a marginal one.\n"
        f"- **Factor:** {_ef[2]}. The default is the TGO 2022 figure; it is editable on the Carbon page "
        "so it can be updated to a newer TGO publication, and a custom value is flagged there and in "
        "the Carbon footnote.\n"
        "- **Not reported — data gaps:**\n"
        "  - *Scope 1* (refrigerant leakage, standby generators): no refrigerant or fuel data.\n"
        "  - *Market-based Scope 2*: no supply contract, green tariff or REC data.\n"
        "  - *Whole-building inventory*: needs main-meter data; the submeters cover zone loads only.")
    st.page_link("views/7_carbon.py", label="Carbon page", icon=":material/arrow_forward:")

    st.subheader("The measured schedule")
    st.markdown(
        f"Cooling runs **{R.AC_SCHEDULE['start']}–{R.AC_SCHEDULE['lunch_off']}** "
        f"and **{R.AC_SCHEDULE['lunch_on']}–{R.AC_SCHEDULE['end']}** on "
        "weekdays. This is read off the meter, not assumed from office norms.\n\n"
        f"- Fully covered hours: {R.AC_SCHEDULE['full_hours']}\n"
        f"- Partially covered: {R.AC_SCHEDULE['partial_hours']} — hour 7 is on "
        "for only ~16% of its minutes and hour 16 for ~45%, so averaging them "
        "with hours 8–10 understates the on-load\n"
        f"- Off by schedule: {R.AC_SCHEDULE['off_hours']} — the AC draws ~0 kWh "
        "at 12:00 and 17:00 on about 94% of weekdays\n\n"
        "This matters because the preceding notebook assumed an 08:00–18:00 "
        "occupied window, which averages two uncooled hours in with eight "
        "cooled ones and pulls the reported mean occupied temperature up."
    )

    st.subheader("Fault evaluation machinery")
    st.markdown(
        "Three mechanics are taken from **ASHRAE Guideline 36 Addendum u "
        "§5.16.14**, because an alarm system without them floods the operator:\n\n"
        "- **Persistence** — a condition must hold for `AlarmDelay` (30 min) "
        "before it is reported.\n"
        "- **State-based suppression** — faults are not evaluated while the "
        "equipment is off, nor for `ModeDelay` (30 min) after it starts. This is "
        "why temperature flatline detection runs only during cooling hours: the "
        "only two 60-minute flat runs in the record are overnight with the AC "
        "off, and reporting them would be manufacturing an alarm.\n"
        "- **Operator suppression** — every fault can be muted.\n\n"
        "The standard's own stated philosophy governs every threshold: defaults "
        "are *\"intentionally biased toward minimizing false alarms — if "
        "necessary, at the expense of missing real alarms\"*, because false "
        "alarms erode confidence."
    )

# --------------------------------------------------------------------------- #
with t_reg:
    st.subheader("Point registry")
    st.markdown(
        "The registry is the single source of truth for what each channel "
        "*means*. Encoding the metering traps as data rather than as special "
        "cases in analysis code is what stops them being silently baked into a "
        "headline number."
    )
    st.dataframe(R.registry_frame(floor), width="stretch", hide_index=True,
                 column_config={"note": st.column_config.TextColumn(width="large")})
    tags = R.haystack_frame()
    st.download_button(
        "Export tagged point list — all floors (CSV)", tags.to_csv(index=False).encode("utf-8"),
        file_name="cubems_points_haystack_brick.csv", mime="text/csv", icon=":material/download:",
        help="One row per existing point with Project Haystack 4 marker tags and a Brick class. "
             "z1's supply-air sensor is tagged discharge air; the lux channel gets a custom "
             "luxIndex tag, never illuminance; dead and frozen meters carry disabled / faulty.")
    st.caption(f"{len(tags)} points tagged across {tags.floorRef.nunique()} floors. Points that do "
               "not exist (z3's AC on Floor 5, sensors on Floor 1) are not in the export.")

    if floor != 5:
        multi = {z: [p.column for p in R.points(floor, "ac", z)] for z in R.ZONES[floor]}
        multi = {z: c for z, c in multi.items() if len(c) > 1}
        prim = {z: [p.column for p in R.points(floor, "ac", z, roles={R.ROLE_PRIMARY})]
                for z in multi}
        no_ac = [z for z in R.ZONES[floor] if not R.has_channel(floor, z, "ac")]
        st.warning(
            f"**The traps on Floor {floor}, as the registry declares them.**\n\n"
            + "".join(f"- **{z} has {len(c)} AC channels**; the load is on "
                      f"{', '.join(f'`{x}`' for x in prim[z]) or 'none'}. The rest are "
                      "residual or dead, so a `*_AC1` aggregate would miss it.\n"
                      for z, c in multi.items())
            + (f"- **{', '.join(no_ac)}: no AC meter.** Shown as 'not metered', never zero.\n"
               if no_ac else "")
            + "- **The lux channel is not lux** — a scaled index, relative signal only.\n\n"
            "Floor 5's version of this box, with the kWh evidence, is on the Floor 5 view.")
    if floor == 5:
        st.warning(
            "**Three traps this registry exists to prevent.**\n\n"
            "1. **z1's real cooling load is on AC4, not AC1.** Only z1 has four AC "
            "channels. `z1_AC4` carries 65,525 kWh; AC1+AC2+AC3 together carry "
            "225 kWh (0.34%). Every other zone's load is on AC1. Aggregating "
            "`*_AC1` across zones would silently drop 39% of the floor's cooling.\n"
            "2. **z3 has no AC meter** — it is a stairwell. A zero in a kW column "
            "that actually means 'no meter' is the classic FM dashboard lie.\n"
            "3. **The lux channel is not lux.** It maxes at 49–79 with integer "
            "resolution across all four zones, where a daylit office is 300–2,000. "
            "It is a scaled index or a mis-scaled register, usable as a relative "
            "signal only, and never labelled as illuminance."
        )


# --------------------------------------------------------------------------- #
with t_cps:
    st.subheader("Change-points")
    st.markdown("Dates across which the data is not comparable with itself. Any "
                "period-over-period comparison spanning one of these is disabled "
                "with a reason rather than silently shown.")
    st.dataframe(R.change_point_frame(floor), width="stretch", hide_index=True,
                 column_config={"detail": st.column_config.TextColumn(width="large")})

# --------------------------------------------------------------------------- #
with t_about:
    st.caption("Explanations that used to sit on the operational pages, and the project's practice, "
               "corrections, baseline, pilot and exclusion notes — one expander each.")

    with st.expander("Alarm system self-audit (ISA-18.2 / EEMUA 191)"):
        if h is not None:
            _reg = ui.register(floor, c)
            active = _reg[_reg.is_open] if len(_reg) else _reg
            audit = F.alarm_audit(active, c["window"])
            hs = audit["mix"].get(F.HIGH, 0)
            k = st.columns(3)
            k[0].metric("Alarms per day", f"{audit['per_day']:.2f}", border=True,
                        help="Open alarms ÷ days in the period. Steady-state best practice is under ~2 per "
                             "operator per hour; this register aggregates episodes rather than events.")
            k[1].metric("High priority", f"{hs:.0f}%", border=True,
                        help=f"ISA-18.2 target {audit['target'].get(F.HIGH, 5):.0f}%; "
                             f"{audit['deviation'].get(F.HIGH, 0):+.0f} pp vs the target mix.")
            k[2].metric("Standing high", f"{audit['standing']}", border=True,
                        help=f"Limit {F.ALARM_KPI['max_standing']}.")
            st.caption(f"Floor {floor}, {c['period_label']}. **{audit['verdict']}** A high-priority share "
                       "above the 5% convention is a real characteristic of these floors rather than a "
                       "tuning failure: overcooling is the dominant fault, it affects every sensored zone, "
                       "and each zone is a separate work order. The first version of this register raised "
                       "34 alarms at 76% high priority by emitting one alarm per channel for the planned "
                       "sensor outage; grouping offline points by zone and episode, and tagging the planned "
                       "window, fixed that.")

    with st.expander("What the alarm register cannot detect, and why"):
        st.markdown(
            "**None of the fifteen ASHRAE Guideline 36 AHU fault conditions (FC1–FC15) are computable on "
            "this data.** Every one needs at least one of: supply, mixed, return or outdoor air "
            "temperature; duct static pressure; or a valve or damper position. CU-BEMS meters zone-level "
            "power and measures zone temperature, humidity and a light index, and nothing on the plant side.\n\n"
            "That rules out economiser and outdoor-air-fraction faults (FC6), coil-leak faults (FC14, FC15), "
            "supply-air temperature faults (FC5, FC7, FC12, FC13), operating-state classification (OS#1–#5) "
            "and FC4 as written, any setpoint-deviation fault (no setpoint channel), simultaneous heating "
            "and cooling, and any chiller or plant efficiency. Also not computable: occupancy-weighted "
            "exceedance and occupancy-conditioned suppression, ventilation or IAQ compliance (no CO₂), and "
            "airflow faults.")

    with st.expander("Why comfort is not 'Unmet Load Hours'"):
        st.markdown(
            "UMLH is defined relative to a setpoint, and this building has no setpoint channel — we only "
            "ever see realised temperature, which is an *outcome* of the setpoint. So the Comfort page "
            "measures exceedance against a declared band (Thai field studies put thermal neutrality near "
            "26 °C; below 23 °C in a tropical air-conditioned office is overcooling). The ASHRAE 55 "
            "adaptive comfort model is not used either: it applies only to naturally conditioned spaces.")

    with st.expander("Why temperature reports zero frozen days"):
        if h is not None:
            from sbanalytics import health as HL
            t = ui.scored(floor)
            t = t[t.channel == "temp"]
            k = st.columns(2)
            k[0].metric("Temperature point-days flagged frozen", int(t.flag_flatline.sum()), border=True)
            k[1].metric("Longest flat run while cooling runs",
                        f"{int(t.max_flat_occupied.max())} min" if len(t) else "—",
                        help=f"Threshold is {HL.FLATLINE_MIN['temp']} min.", border=True)
        st.markdown(
            "Flatline thresholds differ by channel because the physics differs. A zone temperature cannot "
            "hold ±0.01 °C for an hour *while cooling is running*; a lighting circuit can sit at a constant "
            "load all afternoon. So sensors are judged on a 60-minute threshold evaluated **only during "
            "operating hours** — ASHRAE Guideline 36 §5.16.14.11 suspends fault evaluation when the "
            "equipment is off — and power channels on a 4-hour threshold. Long temperature flatlines "
            "overnight with the AC off are genuine thermal quiescence; reporting them would be "
            "manufacturing an alarm.")

    with st.expander("How a savings package is combined"):
        st.markdown(f"{R.METHODS['Combined savings'][0]}  \n*Source: {R.METHODS['Combined savings'][1]}*\n\n"
                    "Scope: the zone-temperature lift and the schedule trim act on the comfort-sensored zones "
                    "the estimators were calibrated on; the weekend/holiday lockout is floor-wide.")

    with st.expander("Industry practice this dashboard follows"):
        st.markdown(
            "The views follow the patterns commercial building-analytics platforms (Johnson "
            "Controls OpenBlue, Honeywell Forge, Siemens Building X / Desigo CC, Schneider "
            "EcoStruxure Building Advisor, SkySpark / KGS Clockworks) give a facilities team — "
            "within what zone-level power and three environmental channels can honestly support.\n\n"
            "| Pattern | Where | What we do |\n|---|---|---|\n"
            "| **ISO 50001 / 50006 EnPIs and CUSUM** | Energy performance | Actual vs a "
            "weather/calendar-normalised baseline, monthly and weekly; CUSUM of the difference, "
            "restarted at each registered change-point |\n"
            "| **IPMVP / ASHRAE Guideline 14 M&V** | Energy performance, simulator | Daily change-point "
            "baseline with CV(RMSE) and NMBE against G14 limits; 90% uncertainty band with the "
            "autocorrelation correction; IPMVP Option B/C per measure; minimum pilot length |\n"
            "| **ASHRAE Guideline 36 alarm delays and fault diagnoses** | Faults | AlarmDelay and "
            "ModeDelay persistence, state-based suppression, possible diagnoses on every fault |\n"
            "| **ISA-18.2 alarm rationalisation** | Faults | Self-audit of alarm rate, priority mix "
            "and standing alarms; muting as shelving with a note |\n"
            "| **FDD prioritisation by cost of inaction** | Faults | ฿/day = attributable energy ÷ "
            "calendar days since first seen × tariff; zero where there is no defensible attribution |\n"
            "| **CMMS work-order hand-off** | Faults | Lifecycle states, trade, action, "
            "verification check, CSV export in import order |\n"
            "| **Portfolio league and heat map** | Portfolio overview | One row per floor, "
            "normalised KPIs, floor × month overcooling map, generated next step per floor |\n")
        st.subheader("Deliberately absent")
        st.markdown(
            "- **Plant / chiller optimisation** (kW/ton, COP, chilled-water reset) — there is no "
            "plant, chiller, flow or chilled-water channel.\n"
            "- **AHU fault detection** (G36 FC1–FC15) — no supply, mixed, return or outdoor air "
            "temperature, no static pressure, no valve or damper position.\n"
            "- **Occupancy-based control and space utilisation** — no occupancy channel.\n"
            "- **Indoor air quality** (CO₂, ventilation compliance) — no CO₂ or airflow channel.\n"
            "- **Automated supervisory control** (writing setpoints or schedules back to the BMS) — "
            "this is a read-only historical dataset with no setpoint channel; every 'setpoint "
            "lift' here is modelled as a shift in observed zone temperature, and every action is "
            "a work order for a person.")

    with st.expander("Corrections to the preceding analysis"):
        st.markdown(
            "Building this dashboard surfaced two errors in the over-cooling "
            "notebook. Both come from the same place: an analysis that consumes a "
            "meter without auditing it will absorb a metering change as though it "
            "were physics."
        )

        if floor != 5:
            st.info("These corrections are to the Floor 5 notebook, so the figures "
                    "below are computed on Floor 5 whichever floor is selected.")
        st.markdown("#### 1. The z4 'level shift' is a z2↔z4 transfer, on 27 Sep 2019")
        if D.cache_ready(5):
            chk = A.combined_meter_check(ui.hourly(5), floor=5)
            if chk:
                k = st.columns(4)
                k[0].metric("z2 mean load", f"{chk['after']['z2']:.2f} kW",
                            delta=f"{chk['delta_a_pct']:+.1f}%", border=True)
                k[1].metric("z4 mean load", f"{chk['after']['z4']:.2f} kW",
                            delta=f"{chk['delta_b_pct']:+.1f}%", border=True)
                k[2].metric("z2 + z4 combined", f"{chk['after']['sum']:.2f} kW",
                            delta=f"{chk['delta_sum_pct']:+.1f}%", border=True)
                k[3].metric("z2 : z4 ratio",
                            f"{chk['ratio_before']:.2f} → {chk['ratio_after']:.2f}", border=True)
                st.info(
                    "**This is the signature of a re-apportionment, not a demand "
                    "change.** Each zone shifts sharply — z4 down 19%, z2 up 10% — "
                    "while their *sum* barely moves and the ratio between them "
                    "flips. If z4 had genuinely started using less cooling, the sum "
                    "would have fallen too.\n\n"
                    "It lands on the same afternoon (14:15–14:26 on 27 Sep 2019) "
                    "that z1's and z2's temperature sensors came back from 45- and "
                    "69-day outages. One site visit restored two sensors and "
                    "permanently re-apportioned cooling energy between two meters. "
                    "That is a CT re-assignment or a control re-zoning."
                )
                st.caption(f"Compared over {chk['n_before']:,} zone-hours before and "
                           f"{chk['n_after']:,} after.")

        st.markdown("#### 2. The baseline was trained across that boundary")
        st.warning(
            "The notebook trained its baseline on 2019-03-06 → 2019-10-01 and tested "
            "on 2019-10-01 onward, so the train/test boundary sat four days after "
            "the re-apportionment. Its headline result — *'z4 fails ASHRAE G14 with "
            "NMBE −23%, an unexplained level shift'* — was measuring a meter change "
            f"as though it were physics.\n\nTraining now stops at **{A.TRAIN[1]}**, "
            f"before the change, and testing starts at **{A.TEST[0]}**. The boundary "
            "is declared rather than discovered. z4's baseline still fails, and now "
            "fails *for a stated reason*: it cannot be used for M&V across a meter "
            "change at all. Either train after the change, or treat z2+z4 as one "
            "combined meter."
        )

        st.markdown("#### 3. Smaller corrections")
        st.markdown(
            "- The sensor gap ends **2019-03-06**, not 03-05, and does not start "
            "simultaneously: z1 and z2 fail mid-afternoon on **2018-09-12**, while "
            "z4 and z5 stop at exactly **00:00:00 on 2018-09-15** — a clean midnight "
            "boundary that looks like an export truncation rather than a hardware "
            "failure. Two different mechanisms are conflated in 'the blackout'.\n"
            "- **Every zone runs 2.1–2.9 °C cooler after the gap** and each sets a "
            "new record minimum, so any 2018-vs-2019 comparison compares two "
            "configurations.\n"
            "- The glitch-cap list was incomplete: `z2_Plug` spikes on the same "
            "logger-failure dates and `z4_Plug` reaches 16.33 kW against a 4.11 kW "
            "p99.99, neither capped. A 'peak plug demand' tile would have shown "
            "16 kW in z4 that never happened.\n"
            "- `sensor_alive` is too blunt: 11–14% of zone-hours are only partly "
            "covered, and an hour built from three valid minutes was treated "
            "identically to one built from sixty. The dashboard gates on "
            "`temp_minutes ≥ 30`."
        )

    with st.expander("Measurement & verification baseline"):
        st.markdown(
            "A gradient-boosted model on **exogenous features only** — outdoor "
            "temperature, humidity, solar radiation, their lags, and calendar terms. "
            "Indoor temperature is deliberately excluded: it is an *outcome* of "
            "cooling, so including it would let the model recover reverse causation. "
            "The exclusion is asserted in code, not just promised."
        )
        st.code(f"FEATURES  = {A.FEATURES}\nFORBIDDEN = {A.FORBIDDEN}\n"
                f"TRAIN = {A.TRAIN}\nTEST  = {A.TEST}", language="python")

        if h is not None:
            @st.cache_resource(show_spinner="Fitting baselines…")
            def _bl(f):
                return A.fit_baselines(D.load_hourly(f), floor=f)

            fitted, perf = _bl(floor)
            st.dataframe(perf.round(1), width="stretch", hide_index=True)
            st.caption("ASHRAE Guideline 14 for hourly models: CV(RMSE) ≤ 30%, "
                       "|NMBE| ≤ 10%.")

            gbm_test = perf[(perf.model == "GBM") & perf.split.str.startswith("test")]
            ok = gbm_test[gbm_test.meets_g14 == "yes"].zone.tolist()
            st.info(
                f"**Meets G14 on the test period: {', '.join(ok) if ok else 'none'}.** "
                + ("z4 does not, and the reason is now stated rather than unexplained — "
                   "its meter changed on 27 Sep 2019, so a model trained before that "
                   "date cannot predict it afterwards. The 2018 holdout fails for every "
                   "zone, which is the 2.1–2.9 °C level break showing up as model error."
                   if floor == 5 else
                   "The zone models use Floor 5's Mar–Sep 2019 window on every floor. "
                   "A zone that fails here has no declared change-point to explain it, "
                   "so check it against the 00a §7 meter-transfer scan before trusting "
                   "its zone-level savings.")
            )

    with st.expander("Could a pilot detect the saving?"):
        st.caption("Whether the protocol is long enough is a power question, and it "
                   "can be answered from the baseline's own error.")

        if h is not None:
            fitted, perf = _bl(floor)
            gt = perf[(perf.model == "GBM") & perf.split.str.startswith("test")]
            default_cv = float(round(gt.cv_rmse.mean(), 1)) if len(gt) else 30.0

            cc = st.columns(4)
            cv = cc[0].slider("Baseline CV(RMSE) %", 10.0, 60.0, default_cv, 0.5)
            step = cc[1].slider("Setpoint step (°C)", 0.5, 3.0, 1.0, 0.5)
            rho = cc[2].slider("Within-day correlation ρ", 0.0, 0.9, float(R.conv("pilot_rho")), 0.05,
                               help="Hours inside one day are not independent. ρ=0 is "
                                    "the optimistic textbook case; 0.5 is realistic.")
            power = cc[3].slider("Power", 0.5, 0.95, float(R.conv("pilot_power")), 0.05)

            phys = A.fit_physics(h, floor=floor)
            reg = A.run_regression(h, floor=floor)
            est = A.estimator_range(phys, reg)
            pct = float(est[est.role == "working figure"].pct_per_c.iloc[0]) \
                if (est.role == "working figure").any() else -5.0
            expected = abs(pct * step)
            hpd = len(R.AC_ON_HOURS)
            need = A.weeks_required(cv, expected, hpd, 5, R.conv("pilot_alpha"), power, rho)
            mds4 = A.detectable_saving(cv, 4, hpd, 5, R.conv("pilot_alpha"), power, rho)

            k = st.columns(4)
            k[0].metric("Expected saving from the step", f"{expected:.1f}%", border=True)
            k[1].metric("Weeks to detect it",
                        f"{need:.0f}" if np.isfinite(need) else "> 5 years", border=True)
            k[2].metric("Detectable in 4 weeks", f"{mds4:.1f}%",
                        delta="step detectable" if mds4 <= expected else "step too small",
                        delta_color="normal" if mds4 <= expected else "inverse", border=True)
            k[3].metric("Cooling hours per day", hpd, border=True)

            W = pd.DataFrame({"weeks": np.arange(1, 53)})
            W["mds"] = [A.detectable_saving(cv, w, hpd, 5, R.conv("pilot_alpha"), power, rho)
                        for w in W.weeks]
            st.altair_chart(
                alt.layer(
                    alt.Chart(W).mark_line(strokeWidth=2.5, color=ui.ACTUAL).encode(
                        x=alt.X("weeks:Q", title="pilot length per step (weeks)"),
                        y=alt.Y("mds:Q", title="minimum detectable saving (%)",
                                scale=alt.Scale(type="log")),
                        tooltip=["weeks:Q", alt.Tooltip("mds:Q", format=".2f")]),
                    alt.Chart(pd.DataFrame({"y": [expected]})).mark_rule(
                        color=ui.INK, strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q"),
                ).properties(height=300), width="stretch")

            st.warning(
                f"**A 1 °C step is not detectable in the two-to-four weeks a typical "
                f"protocol allows.** At {cv:.0f}% CV(RMSE) with realistic "
                f"within-day correlation, four weeks can only detect a saving of "
                f"{mds4:.0f}% or larger, while the step is expected to deliver "
                f"{expected:.1f}%. Either hold each step far longer "
                f"({need:.0f} weeks), take a bigger first step, or accept that no "
                "individual step will be significant and judge the programme on the "
                "full sequence."
            )

    with st.expander("What this dashboard cannot compute, and why"):
        st.markdown("Stating the boundary precisely is a stronger result than "
                    "implying broader coverage.")

        st.markdown(
            "#### Because there is no setpoint channel\n"
            "- **Unmet Load Hours** as conventionally defined (the ASHRAE 90.1 "
            "≤300 h/yr criterion). We measure exceedance against a declared band "
            "instead, and do not call it UMLH.\n"
            "- Setpoint-deviation faults, trim-and-respond assessment.\n"
            "- Simultaneous heating and cooling.\n\n"
            "#### Because there is no plant or AHU data\n"
            "- **All fifteen ASHRAE G36 AHU fault conditions, FC1–FC15.** Every one "
            "needs at least one of supply, mixed, return or outdoor air temperature, "
            "duct static pressure, or a valve/damper position.\n"
            "- Operating-state classification (OS#1–#5), and FC4 as written.\n"
            "- Economiser and outdoor-air-fraction faults, coil-leak faults.\n"
            "- Chiller kW/ton, COP, any plant efficiency metric.\n\n"
            "#### Because there is no occupancy channel\n"
            "- Occupant-weighted exceedance, occupancy-conditioned suppression, "
            "energy per occupant, space utilisation.\n"
            "- This is also the binding constraint on the causal estimate: indoor "
            "temperature partly proxies occupancy, which is why a regression of load "
            "on temperature gives z5 the wrong sign.\n\n"
            "#### Because of sensor scope\n"
            "- Operative temperature, mean radiant temperature, air velocity, "
            "draught rate — so no full PMV/PPD, and none of the ASHRAE 55 local "
            "discomfort criteria.\n"
            "- **The ASHRAE 55 adaptive comfort model** — excluded on *standards* "
            "grounds rather than data grounds: it applies only to naturally "
            "conditioned spaces, and this building is mechanically cooled.\n"
            "- Absolute calibration drift: there is no reference sensor, and 18 "
            "months containing a step change cannot separate drift from seasonal "
            "variation. We report relative peer offset and label it inconclusive.\n"
            "- CO₂ / IAQ compliance (ASHRAE 62.1, EN 16798 categories).\n\n"
            "#### Because of this floor's specific gaps\n"
            + _gaps(floor) +
            "#### Available only through an external join\n"
            "- CU-BEMS has **no outdoor weather channel**. Every weather-normalised "
            "figure here depends on ERA5 reanalysis for central Bangkok joined in "
            "from Open-Meteo. That is reanalysis for the city, not a rooftop "
            "station, so micro-climate and shading effects are not captured."
        )
