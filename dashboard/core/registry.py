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


#: floor -> points. Only floor 5 is wired; the rest are declared but empty.
POINTS: dict[int, list[Point]] = {5: _floor5()}

#: Floors that exist in the dataset, and whether the dashboard can read them.
FLOORS: dict[int, dict] = {
    1: {"wired": False, "zones": 4, "note": "No sensors at all; lighting-dominated, not an office floor."},
    2: {"wired": False, "zones": 5, "note": "Different schema — 37 columns, 16 AC channels."},
    3: {"wired": False, "zones": 5, "note": "Identical 30-column schema — wiring is a data task."},
    4: {"wired": False, "zones": 5, "note": "Identical 30-column schema — wiring is a data task."},
    5: {"wired": True,  "zones": 5, "note": "Wired."},
    6: {"wired": False, "zones": 5, "note": "Source file is a truncated download: 424,168 of 525,600 rows, ends mid-row."},
    7: {"wired": False, "zones": 5, "note": "Identical 30-column schema — wiring is a data task."},
}

ZONES: dict[int, list[str]] = {5: ["z1", "z2", "z3", "z4", "z5"]}


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
]

#: The building-wide sensor maintenance window (firmware update, per the dataset
#: paper). Excluded from completeness denominators so per-point figures are not
#: dominated by a known planned outage.
BLACKOUT = ("2018-09-12 11:09", "2019-03-06 14:30")

#: Known per-zone sensor outages outside the blackout, for the health page.
KNOWN_OUTAGES = [
    ("z2", "2019-07-20 07:36", "2019-09-27 14:15", "sensors", "69.3 days — no indoor temperature for Aug and most of Sep 2019"),
    ("z1", "2019-08-13 13:39", "2019-09-27 14:26", "sensors", "45.0 days"),
    ("*",  "2019-01-23 09:41", "2019-02-12 15:13", "ac", "20.2 days — all four AC channels to the minute, one logger"),
    ("z4", "2019-04-10 02:55", "2019-04-26 14:47", "sensors", "16.5 days"),
    ("z2", "2019-05-12 01:19", "2019-05-27 12:29", "sensors", "15.5 days"),
    ("z2", "2019-05-14 06:54", "2019-05-27 12:33", "light+plug", "13.2 days"),
    ("z4", "2019-12-13 17:12", "2019-12-19 11:35", "sensors", "5.8 days"),
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
    "co2_kg_per_kwh": (0.4999, "kgCO2/kWh", "Thailand grid factor, TGO 2022"),
}

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


def change_point_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "date": c.date, "event": c.title, "severity": c.severity,
        "zones": ", ".join(c.zones) if c.zones else "all",
        "blocks year-on-year": "yes" if c.blocks_comparison else "no",
        "detail": c.detail,
    } for c in CHANGE_POINTS])


def crosses_change_point(start, end) -> list[ChangePoint]:
    """Change-points strictly inside a period — used to disable comparisons."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [c for c in CHANGE_POINTS if s < pd.Timestamp(c.date) < e]


def conv(name: str) -> float:
    """The value of a declared convention."""
    return CONVENTIONS[name][0]
