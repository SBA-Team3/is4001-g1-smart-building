"""
Fault detection, with the anti-flood machinery that makes alarms usable.

Three mechanics are lifted directly from ASHRAE Guideline 36 Addendum u
(§5.16.14), because an alarm system without them floods the operator and gets
switched off:

  * **Persistence** — a fault condition must hold continuously for AlarmDelay
    (default 30 min) before it is reported.
  * **State-based suppression** — faults are not evaluated when the equipment is
    off, nor for ModeDelay (30 min) after it starts, which is the pull-down
    period when a zone is legitimately still warm or the load legitimately high.
  * **Operator suppression** — every fault can be muted.

G36's own stated philosophy applies to every threshold here: defaults are
"intentionally biased toward minimizing false alarms — if necessary, at the
expense of missing real alarms", because false alarms erode confidence.

What we cannot do, and say so plainly: **none of G36's fifteen AHU fault
conditions FC1-FC15 are computable on this data.** Every one of them needs at
least one of supply-, mixed-, return- or outdoor-air temperature, duct static
pressure, or a valve/damper position. CU-BEMS has none. The rules below are
built from what five zones of power and three environmental channels can
actually support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import registry as R

HIGH, MED, LOW = "high", "medium", "low"


#: What a work order needs beyond the diagnosis: which trade gets it, what they
#: do, and how anyone will know it worked. Verification text is filled from the
#: registry conventions, so the check and the alarm use the same thresholds.
PLAYBOOK: dict[str, dict] = {
    "overcooling": {
        "trade": "controls",
        "action": "Raise the zone's cooling setpoint in 0.5-1 °C steps (towards 24-25 °C) and check the "
                  "thermostat location and deadband; log the change so M&V can exclude the step day.",
        "verify": "Share of cooled occupied hours below {comfort_lo:g} °C in {zone} under "
                  "{overcooled_target_pct:.0f}% over the next {verify_weeks_comfort:.0f} weeks "
                  "(now {share:.0f}%)."},
    "after_hours_ac": {
        "trade": "controls",
        "action": "Check the BMS time schedule and holiday calendar for this zone, clear manual "
                  "overrides, and require sign-off for weekend runs.",
        "verify": "{zone} AC at or below {ac_on_kwh:g} kWh/h in every hour outside the schedule over the "
                  "next {verify_weeks_schedule:.0f} weekends and holidays."},
    "after_hours_light": {
        "trade": "electrical",
        "action": "Put the lighting contactor on the BMS time schedule or add occupancy sensing; confirm "
                  "whether egress lighting must stay on.",
        "verify": "Overnight lighting in {zone} below {night_light_ratio_pct:.0f}% of its occupied level "
                  "for {verify_weeks_schedule:.0f} weeks."},
    "dead_meter": {
        "trade": "metering-IT",
        "action": "Verify the meter/CT at the panel against a clamp reading; re-terminate or replace, then "
                  "confirm the logger is not caching the last value.",
        "verify": "No flatline flag and completeness at or above {completeness_target:.0f}% on every day "
                  "for {verify_weeks_schedule:.0f} weeks."},
    "point_offline": {
        "trade": "metering-IT",
        "action": "Check sensor power and Wi-Fi association, then the gateway; confirm the point is still "
                  "on the scan list.",
        "verify": "Point reporting at least {completeness_target:.0f}% of expected samples daily for "
                  "{verify_weeks_schedule:.0f} weeks."},
    "short_cycling": {
        "trade": "HVAC",
        "action": "Inspect the unit for short-cycling on pressure, then review control-loop gains and "
                  "deadband; check whether zones cycle together (plant-side event).",
        "verify": "No day with more than {short_cycle_per_day:.0f} on/off transitions over "
                  "{verify_weeks_schedule:.0f} weeks."},
    "humidity": {
        "trade": "HVAC",
        "action": "Check coil leaving-air condition and infiltration paths; review whether the coil can "
                  "run colder at part load.",
        "verify": "Humidity ratio above {humidity_ratio_max:g} kg/kg in under "
                  "{humidity_alarm_share_pct:.0f}% of occupied hours for {verify_weeks_comfort:.0f} weeks."},
    "step_change": {
        "trade": "metering-IT",
        "action": "Get the site-visit record for the date; confirm the CT/meter assignment; re-train the "
                  "baseline after the change or treat the affected meters as one combined meter.",
        "verify": "Site record obtained and the M&V baseline no longer straddles {date}."},
}


def _verify(rule: str, **kw) -> str:
    conv = {k: v[0] for k, v in R.CONVENTIONS.items()}
    return PLAYBOOK[rule]["verify"].format(**{**conv, **kw})


def episodes(dates, gap_days: float | None = None) -> int:
    """Separate runs of active days: a gap longer than `episode_gap_days` starts
    a new one. This is the recurrence count on a work order."""
    gap = R.conv("episode_gap_days") if gap_days is None else gap_days
    d = pd.Series(pd.to_datetime(pd.Series(list(dates)).unique())).sort_values()
    if not len(d):
        return 0
    return int((d.diff().dt.days.fillna(0) > gap).sum() + 1)


def _rule(zone, rule, priority, title, detail, diagnoses, first, last,
          kwh=0.0, evidence="", planned=False, dates=None, key="", verify=None):
    dates = list(pd.to_datetime(pd.Series(list(dates)).unique())) if dates is not None else []
    return {"zone": zone, "rule": rule, "priority": priority, "title": title,
            "detail": detail, "diagnoses": diagnoses, "first_seen": first,
            "last_seen": last, "kwh_impact": kwh, "evidence": evidence,
            "planned": planned, "days_active": len(dates) if dates else np.nan,
            "recurrence": episodes(dates) if dates else 1, "key": key,
            "verification": verify or _verify(rule, zone=zone, share=np.nan, date="")}


def _in_blackout(start, end) -> bool:
    """Whether a fault window lies entirely inside the planned maintenance gap.

    The 2018-09 to 2019-03 sensor outage was a scheduled firmware update, per the
    dataset paper. Raising one alarm per point for a known planned event is how
    an alarm system floods its operator: it produced 16 of 34 alarms here before
    this suppression, and the self-audit on this module is what caught it.
    """
    if start is None or end is None:
        return False
    lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    return pd.Timestamp(start) >= lo.normalize() and pd.Timestamp(end) <= hi


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #

def overcooling(h: pd.DataFrame, start=None, end=None, floor: int = 5,
                lo: float | None = None) -> list[dict]:
    """Cooling delivered into a zone already below the comfort band.

    Simultaneously a comfort fault and an energy fault — the one that matters
    most on this floor, and the reason the whole project exists.
    """
    lo = R.conv("comfort_lo") if lo is None else lo
    zones = R.comfort_zones(floor)
    d = D.occupied(D.slice_hourly(h, start, end, zones), zones=zones,
                   require_temp=True, ac_on=True)
    out = []
    for z, g in d.groupby("zone"):
        cold = g[g.temp_c < lo]
        if len(cold) < 3:
            continue
        share = len(cold) / len(g) * 100
        out.append(_rule(
            z, "overcooling", HIGH if share > 50 else MED,
            f"{z}: cooling into space already below {lo:g} °C",
            f"{share:.0f}% of cooled hours ({len(cold)} of {len(g)}) ran below "
            f"{lo:g} °C, drawing {cold.ac_kwh.sum():,.0f} kWh.",
            ["Setpoint set too low", "Control deadband too wide",
             "Thermostat sensor mislocated", "Zone oversized for its load"],
            cold.ts.min(), cold.ts.max(), cold.ac_kwh.sum(),
            f"mean {cold.temp_c.mean():.1f} °C while cooling", dates=cold.date,
            verify=_verify("overcooling", zone=z, share=share, comfort_lo=lo)))
    return out


def after_hours_operation(h: pd.DataFrame, start=None, end=None,
                          floor: int = 5) -> list[dict]:
    """Cooling or lighting running outside the measured weekday schedule."""
    d = D.slice_hourly(h, start, end)
    out = []

    off = d[(d.is_weekend | d.is_holiday | ~d.hour.isin(R.ac_on_hours(floor)))]
    ac_off = off[off.ac_kwh > R.conv("ac_on_kwh")]
    for z, g in ac_off.groupby("zone"):
        by_day = g.groupby("date").ac_kwh.sum().sort_values(ascending=False)
        worst = by_day.index[0]
        out.append(_rule(
            z, "after_hours_ac", HIGH if g.ac_kwh.sum() > 1000 else MED,
            f"{z}: cooling outside scheduled hours",
            f"{len(g)} zone-hours of cooling outside the metered weekday "
            f"schedule, {g.ac_kwh.sum():,.0f} kWh. Worst day {worst} "
            f"({by_day.iloc[0]:,.0f} kWh).",
            ["BMS schedule override left in place", "Holiday calendar not loaded",
             "Manual override at the panel", "Time schedule drift after a power cycle"],
            g.ts.min(), g.ts.max(), g.ac_kwh.sum(),
            f"worst day {worst}", dates=g.date))

    night = d[(d.hour >= 20) | (d.hour <= 5)]
    for z, g in night.groupby("zone"):
        if not R.has_channel(floor, z, "light"):
            continue
        day = d[(d.zone == z) & (~d.is_weekend) & d.hour.isin(R.ac_on_hours(floor))]
        day_mean, night_mean = day.light_kwh.mean(), g.light_kwh.mean()
        # Written as "not (x > y)" so a meter outage (all NaN) skips the zone:
        # NaN fails every comparison, and "NaN <= 0" alone would let it through.
        if not (day_mean > 0 and night_mean >= 0):
            continue
        ratio = night_mean / day_mean * 100
        if not ratio >= R.conv("night_light_ratio_pct"):
            continue
        lit = g.groupby("date").light_kwh.mean()
        lit = lit[lit / day_mean * 100 >= R.conv("night_light_ratio_pct")]
        out.append(_rule(
            z, "after_hours_light", MED if ratio > 25 else LOW,
            f"{z}: lighting running overnight",
            f"Overnight lighting averages {night_mean:.2f} kW, "
            f"{ratio:.0f}% of its occupied level — {g.light_kwh.sum():,.0f} kWh "
            f"over the period.",
            ["No occupancy sensing on this circuit", "Cleaning or security schedule",
             "Lighting contactor not on the BMS time schedule",
             "Stairwell/egress lighting required to stay on"],
            g.ts.min(), g.ts.max(), g.light_kwh.sum(),
            f"night/day ratio {ratio:.0f}%", dates=lit.index))
    return out


def dead_meters(H: pd.DataFrame, start=None, end=None, floor: int = 5) -> list[dict]:
    """Points reporting a constant, and points not reporting at all."""
    from . import health as HL
    S_all = HL.score_point_days(H)
    if end is not None:
        S_all = S_all[S_all.day <= pd.Timestamp(end)]
    S = S_all[S_all.day >= pd.Timestamp(start)] if start is not None else S_all
    out = []

    for (z, ch, col, role), g in S.groupby(["zone", "channel", "column", "role"]):
        share = g.flag_flatline.mean() * 100
        if share > 20:
            out.append(_rule(
                z, "dead_meter", MED,
                f"{z} {ch}: meter reporting a constant",
                f"Frozen on {share:.0f}% of days; longest run "
                f"{int(g.flat_measured.max()):,} min; "
                f"{g.distinct.median():.0f} distinct values on a median day.",
                ["Meter or CT failed", "Data logger caching the last value",
                 "Channel not actually connected to a load",
                 "Load genuinely constant (verify at the panel)"],
                g.day.min(), g.day.max(), 0.0, col,
                dates=g[g.flag_flatline].day, key=str(col)))

    # Offline points are grouped per zone and per outage episode. One alarm per
    # channel per zone turns a single gateway failure into a dozen alarms.
    # Episodes are found on the whole history up to `end`, then kept if they
    # reach into the window: an outage that began before the window keeps its
    # real start date, so its id and its days-open do not shift as the window moves.
    gap = R.conv("episode_gap_days")
    off = S_all[S_all.received == 0]
    for z, g in off.groupby("zone"):
        g = g.sort_values("day")
        episode = (g.day.diff().dt.days.fillna(1) > gap).cumsum()
        for _, ep in g.groupby(episode):
            days = ep.day.nunique()
            if days <= gap or (start is not None and ep.day.max() < pd.Timestamp(start)):
                continue
            planned = _in_blackout(ep.day.min(), ep.day.max())
            chans = sorted(ep.channel.unique())
            out.append(_rule(
                z, "point_offline", LOW if planned else (MED if days > 20 else LOW),
                f"{z}: {len(chans)} point(s) offline for {days} days"
                + (" (planned maintenance)" if planned else ""),
                f"{', '.join(chans)} reported nothing from {ep.day.min().date()} "
                f"to {ep.day.max().date()}."
                + (" This falls inside the scheduled building-wide sensor "
                   "firmware update and is expected." if planned else ""),
                ["Scheduled maintenance window"] if planned else
                ["Sensor lost Wi-Fi association", "Power to the sensor lost",
                 "Gateway or logger offline", "Point removed from the scan list"],
                ep.day.min(), ep.day.max(), 0.0, ", ".join(chans), planned,
                dates=ep.day, key=f"{ep.day.min():%Y%m%d}"))
    return out


def short_cycling(h: pd.DataFrame, start=None, end=None, floor: int = 5,
                  per_day: int | None = None) -> list[dict]:
    """Excessive AC on/off transitions.

    Our computable analogue of G36 FC4 ("too many changes in operating state").
    A healthy day here is 2-4 transitions: on, lunch off, lunch on, off.
    """
    per_day = R.conv("short_cycle_per_day") if per_day is None else per_day
    d = D.slice_hourly(h, start, end)
    d = d[~d.is_weekend & d.hour.isin(R.ac_on_hours(floor))].copy()
    if not len(d):
        return []
    d["on"] = (d.ac_kwh > R.conv("ac_on_kwh")).astype(int)
    out = []
    for z, g in d.groupby("zone"):
        if not R.has_channel(floor, z, "ac"):
            continue
        g = g.sort_values("ts")
        trans = g.groupby("date").on.apply(lambda s: s.diff().abs().sum())
        bad = trans[trans > per_day]
        if not len(bad):
            continue
        out.append(_rule(
            z, "short_cycling", MED,
            f"{z}: unstable cooling control",
            f"{len(bad)} day(s) with more than {per_day} on/off transitions "
            f"(worst {int(bad.max())} on {bad.idxmax()}). A healthy day is 2-4.",
            ["Control loop hunting / PI gains too aggressive",
             "Thermostat deadband too narrow", "Compressor short-cycling on pressure",
             "Plant-side or control-network event (check whether zones cycle together)"],
            pd.Timestamp(bad.index.min()), pd.Timestamp(bad.index.max()), 0.0,
            f"worst {int(bad.max())} transitions on {bad.idxmax()}", dates=bad.index))
    return out


def metering_step_change(h: pd.DataFrame, start=None, end=None,
                         floor: int = 5) -> list[dict]:
    """Registered change-points, surfaced as faults rather than buried.

    Only raised when the change-point falls inside the period being viewed — a
    step change from March is not an open fault in November, and leaving it in
    every window is exactly the "standing alarm" that alarm-management practice
    caps at ten.
    """
    out = []
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    for c in R.CHANGE_POINTS:
        if c.severity != "critical" or floor not in c.floors:
            continue
        cd = pd.Timestamp(c.date)
        if (lo is not None and cd < lo) or (hi is not None and cd > hi):
            continue
        out.append(_rule(
            ", ".join(c.zones) if c.zones else "floor", "step_change",
            HIGH, f"Step change: {c.title}", c.detail,
            ["Meter or CT re-assigned during a site visit",
             "Control re-zoning", "Sensor replaced or recalibrated",
             "Setpoint programme changed"],
            pd.Timestamp(c.date), pd.Timestamp(c.date), 0.0, c.date,
            dates=[c.date], key=c.date.replace("-", ""),
            verify=_verify("step_change", zone="", share=np.nan, date=c.date)))
    return out


def humidity_faults(h: pd.DataFrame, start=None, end=None,
                    floor: int = 5) -> list[dict]:
    from .comfort import humidity_exceedance
    he = humidity_exceedance(h, start=start, end=end, floor=floor)
    out = []
    wmax = R.conv("humidity_ratio_max")
    for _, r in he.iterrows():
        if not r.over_limit_pct >= R.conv("humidity_alarm_share_pct"):   # NaN-safe
            continue
        out.append(_rule(
            r.zone, "humidity", MED if r.over_limit_pct > 50 else LOW,
            f"{r.zone}: humidity above the ASHRAE 55 limit",
            f"{r.over_limit_pct:.0f}% of occupied hours exceed a humidity ratio "
            f"of {wmax:g} kg/kg (mean {r.mean_w:.4f}, mean RH {r.mean_rh:.0f}%).",
            ["Insufficient dehumidification at part load",
             "Outdoor air infiltration", "Coil not reaching dew point",
             "Reheat unavailable so the coil cannot run colder"],
            None, None, 0.0, f"mean dew point {r.mean_dew:.1f} °C"))
    return out


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #

def build_register(h: pd.DataFrame, H: pd.DataFrame, start=None, end=None,
                   floor: int = 5, tariff: float | None = None,
                   as_of=None, lo: float | None = None) -> pd.DataFrame:
    """Every rule's output as one work-order-ready table.

    Beyond the alarm itself each row carries what a CMMS hand-off needs: a
    stable id (so a lifecycle state survives a reload), the trade, the action,
    the verification check, days open, recurrence, and the cost of inaction —
    attributable energy / calendar days since first seen x tariff, zero rather than guessed where a
    rule has no defensible attribution.
    """
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    rules = (overcooling(h, start, end, floor, lo=lo)
             + after_hours_operation(h, start, end, floor)
             + dead_meters(H, start, end, floor)
             + short_cycling(h, start, end, floor)
             + humidity_faults(h, start, end, floor)
             + metering_step_change(h, start, end, floor))
    if not rules:
        return pd.DataFrame()

    f = pd.DataFrame(rules)
    f["floor"] = floor
    f["thb_impact"] = f.kwh_impact * tariff
    span = (pd.to_datetime(f.last_seen) - pd.to_datetime(f.first_seen)).dt.days + 1
    f["days_active"] = f.days_active.fillna(span).fillna(1).clip(lower=1)
    ref = pd.Timestamp(as_of if as_of is not None else end if end is not None else h.ts.max()).normalize()
    f["days_open"] = (ref - pd.to_datetime(f.first_seen).dt.normalize()).dt.days
    # Spread over calendar days since first seen, not active days: a one-off
    # holiday run of 72 kWh is not 72 kWh lost every day it stays open.
    exposure = (f.days_open + 1).fillna(f.days_active).clip(lower=1)
    f["cost_per_day_thb"] = np.where(f.kwh_impact > 0, f.kwh_impact / exposure * tariff, 0.0)
    f["trade"] = f.rule.map(lambda r: PLAYBOOK[r]["trade"])
    f["action"] = f.rule.map(lambda r: PLAYBOOK[r]["action"])
    f["energy_attributed"] = f.kwh_impact > 0
    order = {HIGH: 0, MED: 1, LOW: 2}
    f["_o"] = f.priority.map(order)
    f = f.sort_values(["_o", "cost_per_day_thb", "thb_impact"],
                      ascending=[True, False, False]).drop(columns="_o")
    zone_key = f.zone.astype(str).str.replace(", ", "+", regex=False)
    f.insert(0, "id", [f"F{floor}.{z}.{r}" + (f".{k}" if k else "")
                       for z, r, k in zip(zone_key, f.rule, f.key)])
    return f.drop(columns="key").reset_index(drop=True)


