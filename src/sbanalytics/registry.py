"""
Point registry and change-point registry — the single source of truth.

Everything the dashboard knows about *what a channel means* lives here, as data.
The alternative (special-casing z1 and z3 inside every analysis function) is what
the prototype did, and it is how a metering trap ends up silently baked into a
headline number.

Three traps this registry exists to prevent:

1. **z1's real cooling load is on AC4, not AC1.** Only z1 has four AC channels.
   z1_AC4 carries 65,525 kWh; AC1+AC2+AC3 together carry 225 kWh (0.34%). Every
   other zone's load is on AC1. Aggregating `*_AC1` across zones silently drops
   39% of the floor's cooling.
2. **z3 has no AC meter at all** — it is a stairwell. A zero in a kW column that
   actually means "no meter" is the classic FM dashboard lie.
3. **The lux channel is not lux.** It maxes at 49-79 with integer resolution
   across all four zones; a daylit office is 300-2000 lux. It is a scaled index
   or a mis-scaled register, and must never be labelled or used as illuminance.

Roles drive behaviour everywhere downstream. Adding a floor means adding rows
here, not editing analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------- #
# roles
# --------------------------------------------------------------------------- #

#: Points that measure occupied-space temperature and may enter comfort ranking.
ROLE_ZONE_TEMP = "zone_air_temp"
#: Sensor sitting in a supply-air path — a diagnostic, never a comfort metric.
ROLE_SUPPLY_TEMP = "supply_air_temp"
#: The meter that carries a zone's real load for its channel.
ROLE_PRIMARY = "primary"
#: A metered channel that carries negligible load (standby / un-terminated CT).
ROLE_RESIDUAL = "residual"
#: Channel with no meter installed. Renders as "not metered", never as zero.
ROLE_ABSENT = "absent"
#: Meter that reports a near-constant value with no diurnal signal — not a
#: measurement. Render as a fixed baseload figure, do not trend.
ROLE_SUSPECT_CONSTANT = "suspect_constant"
#: Channel whose units are wrong or unknown. Usable as a relative signal only.
ROLE_SCALED_INDEX = "scaled_index"
#: Meter that reads exactly 0 for the whole record — disconnected or decommissioned.
#: Kept out of totals so it cannot dilute per-unit statistics.
ROLE_DEAD = "dead"

#: Roles that still carry usable energy for totals.
ENERGY_ROLES = {ROLE_PRIMARY, ROLE_RESIDUAL, ROLE_SUSPECT_CONSTANT}


@dataclass(frozen=True)
class Point:
    """One physical measurement point."""

    floor: int
    zone: str
    column: str | None          # None when the point does not exist
    channel: str                # ac | light | plug | temp | rh | lux
    unit: str
    role: str
    lo: float | None = None     # physical plausibility bounds
    hi: float | None = None
    resolution: float | None = None   # quantisation step, for flatline detection
    note: str = ""

    @property
    def key(self) -> str:
        return f"F{self.floor}.{self.zone}.{self.channel}"

    @property
    def exists(self) -> bool:
        return self.column is not None and self.role != ROLE_ABSENT


# --------------------------------------------------------------------------- #
# Floor 5 — the only wired floor
# --------------------------------------------------------------------------- #

def _floor5() -> list[Point]:
    pts: list[Point] = []

    # --- z1: four AC channels, and the real load is on AC4 -----------------
    pts.append(Point(5, "z1", "z1_AC4(kW)", "ac", "kW", ROLE_PRIMARY, 0, 80, 0.01,
                     "z1's real cooling load. NOT AC1 — see module docstring."))
    for n in (1, 2, 3):
        pts.append(Point(5, "z1", f"z1_AC{n}(kW)", "ac", "kW", ROLE_RESIDUAL, 0, 5, 0.01,
                         "Standby / un-terminated CT: 0.34% of zone AC combined."))

    # --- z2, z4, z5: single AC channel on AC1 -------------------------------
    for z in ("z2", "z4", "z5"):
        pts.append(Point(5, z, f"{z}_AC1(kW)", "ac", "kW", ROLE_PRIMARY, 0, 80, 0.01))

    # --- z3: stairwell, no cooling and no sensors ---------------------------
    pts.append(Point(5, "z3", None, "ac", "kW", ROLE_ABSENT,
                     note="Stairwell — not air-conditioned, no AC meter installed."))
    for ch in ("temp", "rh", "lux"):
        pts.append(Point(5, "z3", None, ch, "", ROLE_ABSENT,
                         note="Stairwell — no environmental sensor installed."))

    # --- lighting: every zone including the stairwell ------------------------
    for z in ("z1", "z2", "z3", "z4", "z5"):
        pts.append(Point(5, z, f"{z}_Light(kW)", "light", "kW", ROLE_PRIMARY, 0, 25, 0.01))

    # --- plug: three of the five meters are effectively dead -----------------
    pts.append(Point(5, "z3", "z3_Plug(kW)", "plug", "kW", ROLE_SUSPECT_CONSTANT, 0, 15, 0.01,
                     "Constant ~0.27 kW, 66 distinct values in 783k minutes, no "
                     "diurnal signal. Report as fixed baseload; do not trend."))
    for z in ("z2", "z5"):
        pts.append(Point(5, z, f"{z}_Plug(kW)", "plug", "kW", ROLE_SUSPECT_CONSTANT, 0, 15, 0.01,
                         "Pinned at 0.01-0.02 kW for up to 2.6 days at a time — "
                         "likely not resolving small loads."))
    for z in ("z1", "z4"):
        pts.append(Point(5, z, f"{z}_Plug(kW)", "plug", "kW", ROLE_PRIMARY, 0, 15, 0.01))

    # --- environmental sensors ----------------------------------------------
    for z in ("z2", "z4", "z5"):
        pts.append(Point(5, z, f"{z}_S1(degC)", "temp", "degC", ROLE_ZONE_TEMP,
                         15, 40, 0.01))
    # z1's sensor is in a supply-air path: excluded from comfort, kept as a
    # diagnostic. Its bounds are deliberately wider — it legitimately reads cold.
    pts.append(Point(5, "z1", "z1_S1(degC)", "temp", "degC", ROLE_SUPPLY_TEMP,
                     8, 40, 0.01,
                     "Ceiling sensor in a supply-air path. Drops ~5 degC when the "
                     "AHU runs vs ~2 degC for peers. All sub-15 degC readings are "
                     "post-2019-03-06, so this is a regime change, not a constant."))

    for z in ("z1", "z2", "z4", "z5"):
        pts.append(Point(5, z, f"{z}_S1(RH%)", "rh", "%", ROLE_PRIMARY, 20, 95, 0.01))
        pts.append(Point(5, z, f"{z}_S1(lux)", "lux", "index", ROLE_SCALED_INDEX,
                         0, 100, 1.0,
                         "NOT lux. Maxes at 49-79, integer-valued, where a daylit "
                         "office is 300-2000. Relative signal only."))
    return pts


# --------------------------------------------------------------------------- #
# Floors 1-4, 6, 7 — declared from a per-point profile of the raw CSVs
# --------------------------------------------------------------------------- #

#: Column layout per floor: zones, AC unit numbers per zone, and which zones
#: carry plug meters and an S1 environmental sensor. F3/F4/F5/F7 share a layout;
#: F6 has its four-AC zone in z4, not z1; F1 has no sensors; F2 has 14 ACs in z2.
_STD = {"zones": ["z1", "z2", "z3", "z4", "z5"],
        "ac": {"z1": [1, 2, 3, 4], "z2": [1], "z4": [1], "z5": [1]},
        "plug": ["z1", "z2", "z3", "z4", "z5"],
        "sensors": ["z1", "z2", "z4", "z5"]}
LAYOUT: dict[int, dict] = {
    1: {"zones": ["z1", "z2", "z3", "z4"], "ac": {"z2": [1, 2, 3, 4]},
        "plug": ["z1", "z2", "z3"], "sensors": []},
    2: {"zones": ["z1", "z2", "z3", "z4"],
        "ac": {"z1": [1], "z2": list(range(1, 15)), "z4": [1]},
        "plug": ["z1", "z2", "z3", "z4"], "sensors": ["z1", "z2", "z3", "z4"]},
    3: _STD, 4: _STD, 5: _STD, 7: _STD,
    6: {**_STD, "ac": {"z1": [1], "z2": [1], "z4": [1, 2, 3, 4], "z5": [1]}},
}

_RESIDUAL_NOTE = "Carries <1% of its zone's AC energy over the record; standby or un-terminated CT."

#: Role overrides found by profiling every point (share of zone AC energy,
#: distinct values, share of zero readings over Jul 2018 - Dec 2019).
ROLE_OVERRIDES: dict[int, dict[str, tuple[str, str]]] = {
    1: {"z2_AC3(kW)": (ROLE_RESIDUAL, "6 distinct values, 109 kWh in 18 months — never runs."),
        "z2_AC4(kW)": (ROLE_RESIDUAL, "7 distinct values, max 0.06 kW — never runs.")},
    2: {"z2_AC11(kW)": (ROLE_DEAD, "Reads exactly 0 in every sample of both years."),
        "z2_AC12(kW)": (ROLE_RESIDUAL, "Zero in 99.2% of minutes; 92 kWh in 18 months."),
        "z2_AC13(kW)": (ROLE_RESIDUAL, "Zero in 92.9% of minutes; 0.4% of zone AC."),
        "z2_AC14(kW)": (ROLE_RESIDUAL, "Zero in 99.1% of minutes; 94 kWh in 18 months."),
        "z4_Plug(kW)": (ROLE_SUSPECT_CONSTANT, "6 distinct values, max 0.06 kW — not resolving load.")},
    3: {f"z1_AC{n}(kW)": (ROLE_RESIDUAL, _RESIDUAL_NOTE) for n in (1, 2, 3)},
    4: {f"z1_AC{n}(kW)": (ROLE_RESIDUAL, _RESIDUAL_NOTE) for n in (1, 2, 3)},
    6: {f"z4_AC{n}(kW)": (ROLE_RESIDUAL, _RESIDUAL_NOTE) for n in (1, 2, 3)},
    7: {"z1_AC1(kW)": (ROLE_DEAD, "Reads exactly 0 in every sample of both years."),
        "z1_AC2(kW)": (ROLE_RESIDUAL, _RESIDUAL_NOTE),
        "z1_AC3(kW)": (ROLE_RESIDUAL, _RESIDUAL_NOTE)},
}

_LUX_NOTE = "NOT lux. Integer-valued, max 50-140 where a daylit office is 300-2000. Relative signal only."


def _generic_floor(floor: int) -> list[Point]:
    lay, over = LAYOUT[floor], ROLE_OVERRIDES.get(floor, {})
    pts: list[Point] = []

    def add(zone, column, channel, unit, role, lo=None, hi=None, res=None, note=""):
        role, note = over.get(column, (role, note)) if column else (role, note)
        pts.append(Point(floor, zone, column, channel, unit, role, lo, hi, res, note))

    for z in lay["zones"]:
        units = lay["ac"].get(z, [])
        if not units:
            add(z, None, "ac", "kW", ROLE_ABSENT, note="No AC meter in this zone.")
        for n in units:
            add(z, f"{z}_AC{n}(kW)", "ac", "kW", ROLE_PRIMARY, 0, 80, 0.01)
        add(z, f"{z}_Light(kW)", "light", "kW", ROLE_PRIMARY, 0, 150, 0.01)
        if z in lay["plug"]:
            add(z, f"{z}_Plug(kW)", "plug", "kW", ROLE_PRIMARY, 0, 80, 0.01)
        else:
            add(z, None, "plug", "kW", ROLE_ABSENT, note="No plug meter in this zone.")
        if z in lay["sensors"]:
            add(z, f"{z}_S1(degC)", "temp", "degC", ROLE_ZONE_TEMP, 12, 40, 0.01)
            add(z, f"{z}_S1(RH%)", "rh", "%", ROLE_PRIMARY, 20, 100, 0.01)
            add(z, f"{z}_S1(lux)", "lux", "index", ROLE_SCALED_INDEX, 0, 200, 1.0, _LUX_NOTE)
        else:
            for ch in ("temp", "rh", "lux"):
                add(z, None, ch, "", ROLE_ABSENT, note="No environmental sensor in this zone.")
    return pts


#: floor -> points. Floor 5 is hand-curated; the others are declared from LAYOUT.
POINTS: dict[int, list[Point]] = {
    f: (_floor5() if f == 5 else _generic_floor(f)) for f in LAYOUT}

#: Floors that exist in the dataset. `wired` = the Streamlit dashboard pages are
#: validated for it; every floor is in the registry and the cleaned dataset.
FLOORS: dict[int, dict] = {
    1: {"wired": False, "zones": 4, "note": "No sensors at all; lighting-dominated, load peaks at night — not an office profile."},
    2: {"wired": False, "zones": 4, "note": "Different schema — 36 columns, 14 AC units in z2; z2_AC11 is dead. "
                                         "No lunch shutdown, so the 12:00 thermal-model calibration has no AC-off hour."},
    3: {"wired": True,  "zones": 5, "note": "All meters offline 11 Oct - 21 Nov 2018."},
    4: {"wired": True,  "zones": 5, "note": "All meters offline 29 Oct - 21 Nov 2018; z1_AC1-3 offline Feb-May 2019."},
    5: {"wired": True,  "zones": 5, "note": "Wired."},
    6: {"wired": True,  "zones": 5, "note": "Source file is a truncated download: ends 2019-10-22 13:27, last row corrupt."},
    7: {"wired": True,  "zones": 5, "note": "z1_AC1 is dead; z1's load is on AC4."},
}

ZONES: dict[int, list[str]] = {f: list(lay["zones"]) for f, lay in LAYOUT.items()}

#: Single-minute logger artefacts capped by hand in the floor notebooks. Applied
#: in addition to the generic spike rule in `clean.py`.
GLITCH_CAPS: dict[int, dict[str, float]] = {
    4: {"z1_Plug(kW)": 15.0, "z1_Light(kW)": 20.0, "z5_Light(kW)": 15.0,
        "z2_Light(kW)": 15.0, "z4_Light(kW)": 20.0,
        "z1_AC1(kW)": 5.0, "z1_AC2(kW)": 8.0, "z1_AC3(kW)": 8.0},
    5: {"z1_Light(kW)": 20.0, "z1_Plug(kW)": 10.0, "z2_Light(kW)": 20.0},
}


# --------------------------------------------------------------------------- #
# change-points: dates across which the data is not comparable with itself
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChangePoint:
    date: str
    title: str
    detail: str
    severity: str               # info | warn | critical
    blocks_comparison: bool     # if True, year-on-year comparison is disabled
    zones: tuple[str, ...] = ()
    floors: tuple[int, ...] = (5,)


CHANGE_POINTS: list[ChangePoint] = [
    ChangePoint(
        "2018-09-12", "z1 and z2 sensors fail",
        "z2 stops at 11:09, z1 at 16:33 — mid-afternoon, consistent with a real fault.",
        "warn", False, ("z1", "z2")),
    ChangePoint(
        "2018-09-15", "z4 and z5 sensors stop at exactly 00:00:00",
        "A clean midnight boundary suggests an export/ingest truncation rather than "
        "a hardware failure. Two different mechanisms are conflated in 'the blackout'.",
        "warn", False, ("z4", "z5")),
    ChangePoint(
        "2019-03-06", "Sensors restored, and the floor gets colder",
        "All four sensors return within a 9-minute window (14:20-14:30). Every zone "
        "then runs 2.1-2.9 degC cooler than before and sets a new record minimum. "
        "Any 2018-vs-2019 comparison compares two different configurations.",
        "critical", True, ("z1", "z2", "z4", "z5")),
    ChangePoint(
        "2019-09-27", "Cooling energy re-apportioned between z2 and z4",
        "z4 loses 3.69 kW, z2 gains 1.73 kW; the combined z2+z4 share barely moves "
        "(0.533 -> 0.541) while the z2:z4 ratio flips from 0.90 to 1.23. It lands on "
        "the same afternoon (14:15-14:26) that z1 and z2 sensors were restored after "
        "45- and 69-day outages. This is a CT re-assignment or control re-zoning "
        "during a site visit, not a change in z4's cooling demand.",
        "critical", True, ("z2", "z4")),
    ChangePoint(
        "2019-03-06", "Indoor sensors reinstalled after the blackout",
        "Every floor's sensors return on 6 Mar 2019 (00a §2). On F5 the floor then reads "
        "2-3 degC colder; on this floor that has not been checked, so a 2018-vs-2019 "
        "comfort comparison may be measuring the reinstall rather than the building.",
        "warn", False, (), floors=(2, 3, 4, 6, 7)),
]

#: The building-wide sensor maintenance window (firmware update, per the dataset
#: paper). Excluded from completeness denominators so per-point figures are not
#: dominated by a known planned outage.
BLACKOUT = ("2018-09-12 11:09", "2019-03-06 14:30")

#: Known outages outside the blackout, for the sidebar and the health page:
#: (floors, zone, start, end, kind, note). F5's rows are to the minute from the
#: floor notebook; the others are day-level, from the point-day health table
#: (`portfolio.meter_outages` and a scan of zero-received AC days).
KNOWN_OUTAGES = [
    ((5,), "z2", "2019-07-20 07:36", "2019-09-27 14:15", "sensors", "69.3 days — no indoor temperature for Aug and most of Sep 2019"),
    ((5,), "z1", "2019-08-13 13:39", "2019-09-27 14:26", "sensors", "45.0 days"),
    ((5,), "*",  "2019-01-23 09:41", "2019-02-12 15:13", "ac", "20.2 days — all four primary AC channels to the minute, one logger"),
    ((3, 7), "*", "2019-01-24", "2019-02-11", "ac", "Every primary AC channel; the same Jan-Feb 2019 logger gap as F5"),
    ((2,), "*", "2019-01-24", "2019-02-11", "ac", "z1_AC1, z2_AC1 and z4_AC1; z2's other 13 units kept reporting"),
    ((3,), "*", "2018-10-11", "2018-11-21", "all meters", "All energy meters offline (full days 12 Oct - 20 Nov 2018)"),
    ((4,), "*", "2018-10-29", "2018-11-21", "all meters", "All energy meters offline (full days 30 Oct - 20 Nov 2018)"),
    ((5,), "z4", "2019-04-10 02:55", "2019-04-26 14:47", "sensors", "16.5 days"),
    ((5,), "z2", "2019-05-12 01:19", "2019-05-27 12:29", "sensors", "15.5 days"),
    ((5,), "z2", "2019-05-14 06:54", "2019-05-27 12:33", "light+plug", "13.2 days"),
    ((5,), "z4", "2019-12-13 17:12", "2019-12-19 11:35", "sensors", "5.8 days"),
]


# --------------------------------------------------------------------------- #
# declared conventions — every threshold the dashboard applies, in one place
# --------------------------------------------------------------------------- #

CONVENTIONS = {
    "comfort_lo": (23.0, "degC", "Thai field studies: below 23 degC in a tropical AC office is overcooling"),
    "comfort_hi": (26.0, "degC", "Thai field studies: thermal neutrality ~26 degC"),
    "rh_lo": (50.0, "%", "Project convention"),
    "rh_hi": (60.0, "%", "ASHRAE 55 humidity ratio limit 0.012 kg/kg ~ 60% RH at these conditions"),
    "humidity_ratio_max": (0.012, "kg/kg", "ASHRAE 55 Graphic Comfort Zone Method"),
    "floor_area_m2": (11700 / 7, "m2", "11,700 m2 total / 7 floors — stated assumption"),
    "tariff_thb": (4.20, "THB/kWh", "MEA large-commercial, user-adjustable"),
    "eui_good": (170.0, "kWh/m2/yr", "Bangkok office benchmark study, n=41"),
    "eui_typical": (225.0, "kWh/m2/yr", "Bangkok office benchmark study, n=41"),
    "sensor_acc_temp": (0.4, "degC", "CU-BEMS dataset paper"),
    "sensor_acc_rh": (2.0, "%", "CU-BEMS dataset paper"),
    "alarm_delay_min": (30, "min", "ASHRAE Guideline 36 Addendum u, AlarmDelay default"),
    "rolling_window_min": (5, "min", "ASHRAE G36 5.16.14.6 — 5-min rolling average of 1-min samples"),
    "mode_delay_min": (30, "min", "ASHRAE G36 ModeDelay — suspend evaluation after a mode change"),
    "completeness_target": (95.0, "%", "BMS/smart-meter literature, per point per day"),
    "temp_minutes_gate": (30, "min", "11-14% of zone-hours are partially covered"),
    "flatline_min": (60, "min", "Non-zero flatline; project convention"),
    "flatline_zero_hours": (24, "h", "Zero-run threshold, after Building Data Genome Project 2"),
    "spike_temp_c_per_min": (2.0, "degC/min", "Fires ~once a year on this floor"),
    "co2_kg_per_kwh": (0.4999, "kgCO2e/kWh", "Thailand grid emission factor, Thailand Greenhouse Gas "
                       "Management Organization (TGO), 2022 publication — verify the current figure before reporting"),

    # --- added for the v2 dashboard (portfolio, M&V, work orders, simulator) ---
    "overcooled_target_pct": (40.0, "%", "Project proposal KPI: % of cooled occupied hours below 23 degC, "
                              "68-83% on F2-F7 today, target below 40%"),
    "overcooled_min_hours": (50, "h", "Minimum cooled occupied zone-hours before a zone enters an overcooled "
                             "share (kpis.zone_comfort, notebook 01)"),
    "ac_on_kwh": (0.5, "kWh/h", "Zone AC counted as running above this hourly energy — the threshold "
                  "data.occupied and the fault rules already use"),
    "demand_charge_thb_per_kw": (0.0, "THB/kW/month", "Default 0. MEA large general-service and TOU tariffs "
                                 "carry a monthly demand charge — enter the contract rate. Peaks here are "
                                 "hourly means, which understate 15-min billing demand"),
    "episode_gap_days": (3, "days", "Active days more than this far apart start a new fault episode (the "
                         "same rule faults.dead_meters uses to group offline points)"),
    "verify_weeks_schedule": (2, "weeks", "Verification window for schedule/metering work orders — two "
                              "full weekends"),
    "verify_weeks_comfort": (4, "weeks", "Verification window for comfort work orders — one month of "
                             "occupied hours"),
    "night_light_ratio_pct": (10.0, "%", "Overnight lighting above this share of its occupied level raises "
                              "an after-hours lighting fault (faults.after_hours_operation)"),
    "short_cycle_per_day": (12, "transitions/day", "Healthy day is 2-4 AC on/off transitions; above 12 is "
                            "a short-cycling fault (computable analogue of G36 FC4)"),
    "humidity_alarm_share_pct": (20.0, "%", "Share of occupied hours above the ASHRAE 55 humidity ratio "
                                 "before a humidity fault is raised"),
    "mv_confidence_z": (1.645, "z", "90% two-sided confidence — the level IPMVP recommends for reporting "
                        "savings uncertainty"),
    "signature_outlier_z": (2.0, "sd", "Workday flagged on the energy signature when it sits more than 2 "
                            "training-residual SDs above the change-point line"),
    "pilot_alpha": (0.05, "", "Pilot power calculation: two-sided significance level"),
    "pilot_power": (0.80, "", "Pilot power calculation: power (Methods page default)"),
    "pilot_rho": (0.5, "", "Within-day correlation of hourly residuals (Methods page realistic default)"),
    "estimator_ci_max_ratio": (5.0, "x", "A %/degC estimate is used only if negative and its CI is no wider "
                               "than 5x its magnitude (notebook 06 reliability rule)"),
    "lockout_low_frac": (0.5, "", "Low end of the weekend/holiday lockout saving = half the metered "
                         "excess, because some runs are authorised (notebook 06)"),
    "sens_tariff_pct": (20.0, "%", "Tornado chart: tariff varied +/-20% around the sidebar value — project "
                        "assumption, roughly the spread between MEA TOU off-peak and on-peak blends"),
    "sens_dT_c": (0.5, "degC", "Tornado chart: zone-temperature lift varied +/-0.5 degC"),
    "sens_trim_h": (1, "h", "Tornado chart: schedule trim varied +/-1 hour"),
    "health_good": (85.0, "/100", "Point health index at or above this is green on the trust band"),
    "health_warn": (60.0, "/100", "Point health index below this is red on the trust band"),
    "max_lift_c": (3.0, "degC", "Simulator range for the zone-temperature lift; beyond 3 degC the "
                   "26 degC neutrality bound is exceeded on every floor"),
}

#: Methods that are rules rather than numbers. Shown beside CONVENTIONS on the
#: Methods page, so no formula the dashboard applies is undeclared.
METHODS: dict[str, tuple[str, str]] = {
    "Overcooled share": (
        "Cooled occupied zone-hours (workday, measured AC hours, temp_ok, AC > ac_on_kwh) below "
        "comfort_lo; per zone, then weighted by each zone's AC kWh in the window. Zones with fewer "
        "than overcooled_min_hours are left out.",
        "Notebook 01 (floor_kpis.csv) — the 68-83% figure in the proposal"),
    "tCO2e": ("kWh x co2_kg_per_kwh / 1000. Grid-average factor, not marginal.",
              "TGO grid emission factor"),
    "Cost of inaction (THB/day)": (
        "Energy attributable to the fault / number of days the condition was active x tariff. Zero, "
        "not estimated, where the rule has no defensible energy attribution. For overcooling the "
        "attributable energy is all cooling delivered while the zone was below the band, an upper bound "
        "on the avoidable share.",
        "FDD prioritisation practice (SkySpark / Clockworks 'cost of inaction')"),
    "Fault lifecycle": (
        "New -> Acknowledged -> In progress -> Resolved, plus Muted. Stored per fault id with "
        "timestamp and note in dashboard/cache/fault_states.csv (append-only; latest row wins).",
        "CMMS work-order states; ISA-18.2 shelving for Muted"),
    "Fault id": ("F<floor>.<zone>.<rule>[.<point or start date>] — stable across windows so a state "
                 "survives a page reload; offline episodes carry their start date.", "Project convention"),
    "Expected energy (M&V baseline)": (
        "Daily 3-parameter cooling change-point model on mean outdoor temperature, fitted separately "
        "for workdays and non-workdays (ASHRAE IMT). Floor 5 trains on 2019-03-07 to 2019-09-26 so the "
        "window sits between the 2019-03-06 and 2019-09-27 change-points; other floors use Jul 2018 - "
        "Jun 2019.", "ASHRAE Guideline 14, IPMVP Option C"),
    "Expected-energy band": (
        "+/- mv_confidence_z x sqrt(sum of daily residual variances x (1+rho)/(1-rho)), rho = lag-1 "
        "autocorrelation of training residuals.", "ASHRAE Guideline 14 Annex B autocorrelation correction"),
    "CUSUM": ("Cumulative sum of (actual - expected), restarted at every registered change-point that "
              "blocks comparison, so it never accumulates across a configuration change.",
              "ISO 50006 EnPI monitoring"),
    "Combined savings": (
        "Schedule trim and weekend lockout are metered and additive (disjoint hours). The setpoint "
        "%/degC is applied to the occupied AC baseline after the trim is removed: "
        "combined = S x (1 - T/B0) + T + L, for the low, mid and high case separately.",
        "Interactive-effects rule in IPMVP Core Concepts"),
    "IPMVP option": (
        "Option B (retrofit isolation) for measures acting on zones with their own AC meter "
        "(zone-temperature lift, schedule trim); Option C (whole facility) for floor-wide schedule "
        "measures (weekend/holiday lockout) and the combined package.", "IPMVP Core Concepts 2022"),
    "Minimum pilot length": (
        "analysis.weeks_required at the baseline's CV(RMSE): hourly zone model for Option B (cooled "
        "hours/day, rho = pilot_rho), daily floor model for Option C (one observation per workday).",
        "ASHRAE Guideline 14 / standard two-sample power calculation"),
}

#: Floors that are not office floors: kept in building totals, excluded from
#: schedule-based and overcooling rankings.
NON_OFFICE_FLOORS = {1}

#: Floors whose source file ends early. Nothing after this is data.
FILE_END: dict[int, str] = {6: "2019-10-22 13:27"}


def floor_file_end(floor: int) -> pd.Timestamp | None:
    return pd.Timestamp(FILE_END[floor]) if floor in FILE_END else None

#: The measured AC schedule. Derived from the meter, not assumed from office norms.
#: Hours 7 and 16 are partial (16.5% and 45% of minutes ON) — flagged so they are
#: not averaged naively with the full hours 8-10 and 13-15.
AC_SCHEDULE = {
    "start": "07:48", "lunch_off": "11:50", "lunch_on": "13:03", "end": "16:31",
    "full_hours": [8, 9, 10, 13, 14, 15],
    "partial_hours": [7, 11, 16],
    "off_hours": [12, 17],
}
AC_ON_HOURS = [7, 8, 9, 10, 11, 13, 14, 15, 16]

#: Measured AC schedule per floor: hours in which the floor's AC is on for at
#: least half of workdays (`features.measure_ac_hours`, re-run in
#: analysis/00a_data_quality §6). F2 has no lunch shutdown. F1's AC never
#: switches off (z2_AC1 median 44 kW around the clock), so it has no schedule
#: to read and gets the nominal 08-17 window. F5 keeps the dashboard's
#: hand-validated list, which also counts the partial hour 7 (16.5% on).
_MEASURED = [8, 9, 10, 11, 13, 14, 15, 16]
AC_ON_HOURS_BY_FLOOR: dict[int, list[int]] = {
    1: list(range(8, 17)),
    2: list(range(8, 17)),
    3: _MEASURED, 4: _MEASURED, 6: _MEASURED, 7: _MEASURED,
    5: AC_ON_HOURS,
}


def ac_on_hours(floor: int = 5) -> list[int]:
    return AC_ON_HOURS_BY_FLOOR.get(floor, AC_ON_HOURS)


# --------------------------------------------------------------------------- #
# lookups
# --------------------------------------------------------------------------- #

def points(floor: int = 5, channel: str | None = None, zone: str | None = None,
           roles: set[str] | None = None, existing_only: bool = True) -> list[Point]:
    """Filter the registry."""
    out = POINTS.get(floor, [])
    if channel:
        out = [p for p in out if p.channel == channel]
    if zone:
        out = [p for p in out if p.zone == zone]
    if roles:
        out = [p for p in out if p.role in roles]
    if existing_only:
        out = [p for p in out if p.exists]
    return out


def ac_columns(floor: int = 5, zone: str | None = None,
               primary_only: bool = False) -> list[str]:
    """Every AC column that carries energy.

    `primary_only=False` is the default because the residual z1_AC1/2/3 channels
    still carry 225 kWh — small, but dropping them silently is the habit this
    registry exists to break.
    """
    roles = {ROLE_PRIMARY} if primary_only else ENERGY_ROLES
    return [p.column for p in points(floor, "ac", zone, roles)]


def channel_columns(floor: int, channel: str, zone: str | None = None) -> list[str]:
    return [p.column for p in points(floor, channel, zone)]


def comfort_zones(floor: int = 5) -> list[str]:
    """Zones whose temperature legitimately represents occupied space."""
    return sorted({p.zone for p in points(floor, "temp", roles={ROLE_ZONE_TEMP})})


def sensored_zones(floor: int = 5) -> list[str]:
    return sorted({p.zone for p in points(floor, "temp")})


def has_channel(floor: int, zone: str, channel: str) -> bool:
    return bool(points(floor, channel, zone))


def point_for(floor: int, zone: str, channel: str) -> Point | None:
    ps = points(floor, channel, zone)
    return ps[0] if ps else None


def point_for_any(floor: int, zone: str, channel: str) -> Point | None:
    """Like point_for, but also returns ROLE_ABSENT placeholders (their notes say why)."""
    ps = points(floor, channel, zone, existing_only=False)
    return ps[0] if ps else None


def wired_floors() -> list[int]:
    return [f for f, meta in FLOORS.items() if meta["wired"]]


def registry_frame(floor: int = 5) -> pd.DataFrame:
    """The registry as a table, for display on the methods page."""
    rows = [{
        "zone": p.zone, "channel": p.channel, "column": p.column or "—",
        "role": p.role, "unit": p.unit,
        "range": f"{p.lo:g}–{p.hi:g}" if p.lo is not None else "—",
        "note": p.note,
    } for p in POINTS.get(floor, [])]
    return pd.DataFrame(rows).sort_values(["zone", "channel"]).reset_index(drop=True)


#: Project Haystack 4 marker tags and Brick class per (channel, role). The
#: lux channel gets no illuminance tag on purpose: it is a scaled index.
_TAGS = {
    ("ac", None): ("elec-meter", "point his sensor elec power hvac", "brick:Electric_Power_Sensor"),
    ("light", None): ("elec-meter", "point his sensor elec power lighting", "brick:Electric_Power_Sensor"),
    ("plug", None): ("elec-meter", "point his sensor elec power plugLoad", "brick:Electric_Power_Sensor"),
    ("temp", ROLE_ZONE_TEMP): ("zone-air-temp-sensor", "point his sensor zone air temp",
                               "brick:Zone_Air_Temperature_Sensor"),
    ("temp", ROLE_SUPPLY_TEMP): ("discharge-air-temp-sensor", "point his sensor discharge air temp",
                                 "brick:Supply_Air_Temperature_Sensor"),
    ("rh", None): ("zone-air-humidity-sensor", "point his sensor zone air humidity",
                   "brick:Zone_Air_Humidity_Sensor"),
    ("lux", None): ("light-level-index", "point his sensor zone luxIndex", "brick:Sensor"),
}


def haystack_frame(floors=None) -> pd.DataFrame:
    """The point registry with Haystack / Brick tags, for export to a BMS or a
    semantic model. Absent points are left out (nothing to tag); dead and
    suspect meters keep their tags plus a `disabled` / `faulty` marker so a
    consumer does not trend them as live."""
    rows = []
    for f in sorted(floors or POINTS):
        for p in points(f):
            proto, tags, brick = _TAGS.get((p.channel, p.role)) or _TAGS[(p.channel, None)]
            extra = {ROLE_DEAD: " disabled", ROLE_SUSPECT_CONSTANT: " faulty",
                     ROLE_RESIDUAL: " standby"}.get(p.role, "")
            rows.append({"id": f"cubems.F{f}.{p.zone}.{p.column}", "dis": f"F{f} {p.zone} {p.column}",
                         "siteRef": "cubems", "floorRef": f"cubems.F{f}", "spaceRef": f"cubems.F{f}.{p.zone}",
                         "proto": proto, "tags": tags + extra, "unit": p.unit,
                         "kind": "Number", "brick_class": brick, "role": p.role, "note": p.note})
    return pd.DataFrame(rows)


def change_point_frame(floor: int | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "floor": ", ".join(f"F{f}" for f in c.floors),
        "date": c.date, "event": c.title, "severity": c.severity,
        "zones": ", ".join(c.zones) if c.zones else "all",
        "blocks year-on-year": "yes" if c.blocks_comparison else "no",
        "detail": c.detail,
    } for c in CHANGE_POINTS if floor is None or floor in c.floors],
        columns=["floor", "date", "event", "severity", "zones", "blocks year-on-year", "detail"])


def crosses_change_point(start, end, floor: int | None = None) -> list[ChangePoint]:
    """Change-points strictly inside a period — used to disable comparisons."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [c for c in CHANGE_POINTS if s < pd.Timestamp(c.date) < e
            and (floor is None or floor in c.floors)]


def conv(name: str) -> float:
    """The value of a declared convention."""
    return CONVENTIONS[name][0]
