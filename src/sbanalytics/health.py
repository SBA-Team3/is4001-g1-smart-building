"""
Sensor and data health.

This sits above everything else in the dashboard for a reason: the documented
failure mode of rule-based building FDD is that a rule has no way to know its
input has gone stale. A comfort or energy number computed on a dead point looks
exactly like one computed on a live point. So every other page is conditional on
this one, and says so.

Fault classes implemented (the standard non-fail-stop sensor taxonomy):
    data loss       missing samples over an interval
    stuck / frozen  zero variation while the point still reports
    out of bounds   physically implausible values
    spike           rate of change beyond what the physics allows
    comms loss      the point returns its LAST value rather than a new one
    inconsistency   individually plausible, but disagrees with physical neighbours

Deliberately NOT implemented: absolute calibration drift. With 18 months and no
reference sensor, true drift cannot be separated from seasonal change — and on
this floor the apparent trend is dominated by a step change and by two long
outages. We report relative peer offset and label it inconclusive rather than
dress a spatial difference up as drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import registry as R

# Weights for the Point Health Index. Completeness carries the score; the fault
# classes subtract. Biased toward reporting a point as healthy, consistent with
# ASHRAE G36's stated preference for missing an alarm over raising a false one.
PENALTY_FLATLINE = 45.0
PENALTY_RANGE = 25.0
PENALTY_SPIKE = 10.0

#: Flatline thresholds differ by channel because the physics differs. A zone
#: temperature in a Bangkok office cannot hold +/-0.01 degC for an hour while
#: cooling runs, so 60 minutes is diagnostic. A lighting or plug circuit can
#: legitimately sit at a constant load all afternoon, so only a multi-hour
#: constant indicates a dead meter — which is exactly what separates z3_Plug
#: (longest run 1,218 min) from z3_Light (71 min, a genuinely steady load).
FLATLINE_MIN = {"temp": 60, "rh": 60, "ac": 240, "light": 240, "plug": 240}

#: Channels evaluated only while cooling runs (G36 state-based suppression).
OCCUPIED_ONLY = {"temp", "rh"}


def _flatline_threshold(channel: str) -> int:
    return FLATLINE_MIN.get(channel, int(R.conv("flatline_min")))


def score_point_days(H: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-100 health score and fault flags to the point-day table.

    Zero-runs are treated asymmetrically from non-zero runs: an AC meter reading
    exactly 0 overnight is correct behaviour, not a stuck sensor, so a zero-run
    needs 24 hours before it counts. A *non-zero* constant is the real signature
    of a dead meter.
    """
    h = H.copy()
    thr = h.channel.map(_flatline_threshold)

    # Sensor channels are judged on their occupied-hours flatline only, so an
    # overnight quiescent period with the AC off is not reported as a fault.
    occ_only = h.channel.isin(OCCUPIED_ONLY)
    measured = np.where(occ_only, h.get("max_flat_occupied", h.max_flat_nonzero),
                        h.max_flat_nonzero)
    h["flat_measured"] = measured

    h["flag_flatline"] = measured >= thr
    # The lux channel is a scaled index of unknown units that legitimately pins
    # at low integers for hours. We report its flat runs but never alarm on them.
    h.loc[h.role == R.ROLE_SCALED_INDEX, "flag_flatline"] = False
    h["flag_zero_stuck"] = h.max_flat_zero >= R.conv("flatline_zero_hours") * 60
    h["flag_range"] = h.out_of_range > 0
    h["flag_spike"] = h.spikes > 0
    h["flag_offline"] = h.received == 0
    h["flag_partial"] = (h.received > 0) & (h.completeness < R.conv("completeness_target"))

    score = h.completeness.clip(0, 100).astype(float)
    score -= h.flag_flatline * PENALTY_FLATLINE
    score -= h.flag_range * PENALTY_RANGE
    score -= h.flag_spike * PENALTY_SPIKE
    h["health"] = score.clip(0, 100)

    # Lux legitimately sits at 0 all night, and the channel is a scaled index
    # anyway — do not let it drag a floor-level index around.
    h.loc[h.role == R.ROLE_SCALED_INDEX, "flag_zero_stuck"] = False
    return h


def completeness_table(H: pd.DataFrame, exclude_blackout: bool = True) -> pd.DataFrame:
    """Per-point completeness over the whole record.

    Excluding the blackout by default is the honest choice: with it included, z2
    reads 50.9% and every zone looks equally broken by one known planned outage.
    Excluding it shows z2 is still only ~75% complete, which is the number an FM
    would actually act on.
    """
    h = H[~H.in_blackout] if exclude_blackout else H
    g = h.groupby(["zone", "channel", "column", "role"], as_index=False).agg(
        days=("day", "nunique"),
        received=("received", "sum"),
        expected=("expected", "sum"),
        offline_days=("received", lambda s: int((s == 0).sum())),
        max_flat=("max_flat_nonzero", "max"),
        out_of_range=("out_of_range", "sum"),
        spikes=("spikes", "sum"),
    )
    g["completeness"] = g.received / g.expected * 100
    scored = score_point_days(h)
    g["flat_days"] = scored.groupby(["zone", "channel", "column", "role"]).flag_flatline.sum().values
    g["flat_day_share"] = g.flat_days / g.days * 100
    return g.sort_values("completeness")


def verdict(row) -> str:
    """One-line operator verdict per point."""
    if row.role == R.ROLE_SUSPECT_CONSTANT and row.flat_day_share > 40:
        return "Constant — not a measurement"
    if row.completeness < 60:
        return "Mostly offline"
    if row.flat_day_share > 20:
        return "Frequently frozen"
    if row.completeness < R.conv("completeness_target"):
        return "Intermittent"
    if row.out_of_range > 0 or row.spikes > 0:
        return "Occasional bad samples"
    return "Healthy"


def point_health_index(H: pd.DataFrame, day=None, floor: int = 5) -> dict:
    """Floor-level trust summary for the header band."""
    h = score_point_days(H)
    if day is not None:
        h = h[h.day == pd.Timestamp(day).normalize()]
    if not len(h):
        return {"index": np.nan, "points": 0, "offline": 0, "faulted": 0,
                "completeness": np.nan}
    faulted = h.flag_flatline | h.flag_range | h.flag_offline
    return {
        "index": float(h.health.mean()),
        "points": int(h.column.nunique()),
        "offline": int(h.flag_offline.sum()),
        "faulted": int(faulted.sum()),
        "completeness": float(h.completeness.mean()),
    }


def heatmap_frame(H: pd.DataFrame, channels=None, zones=None) -> pd.DataFrame:
    """Point x day health, for the operator's most-used data-quality visual."""
    h = score_point_days(H)
    if channels:
        h = h[h.channel.isin(list(channels))]
    if zones:
        h = h[h.zone.isin(list(zones))]
    h["point"] = h.zone + " " + h.channel
    return h[["point", "zone", "channel", "column", "day", "health", "completeness",
              "received", "max_flat_nonzero", "out_of_range", "spikes",
              "flag_offline", "flag_flatline", "in_blackout"]]


def last_good(H: pd.DataFrame, as_of, floor: int = 5) -> pd.DataFrame:
    """Most recent day each point reported anything, at or before as_of."""
    d = pd.Timestamp(as_of).normalize()
    h = H[(H.day <= d) & (H.received > 0)]
    g = h.groupby(["zone", "channel", "column"], as_index=False).day.max()
    g["days_stale"] = (d - g.day).dt.days
    return g.rename(columns={"day": "last_good_day"}).sort_values("days_stale", ascending=False)


def went_offline(H: pd.DataFrame, as_of) -> pd.DataFrame:
    """Points that reported something the day before as_of and nothing on as_of —
    the "since yesterday" line an operator reads first."""
    d = pd.Timestamp(as_of).normalize()
    today = H[H.day == d].set_index("column").received
    prev = H[H.day == d - pd.Timedelta(days=1)].set_index("column").received
    cols = today.index.intersection(prev.index)
    lost = [c for c in cols if today[c] == 0 and prev[c] > 0]
    return H[(H.day == d) & H.column.isin(lost)][["zone", "channel", "column"]]


# --------------------------------------------------------------------------- #
# cross-sensor consistency — the strongest check available without a reference
# --------------------------------------------------------------------------- #

def peer_deviation(h: pd.DataFrame, freq: str = "MS", zones=None) -> pd.DataFrame:
    """Each zone's temperature offset from the peer median, by period.

    Uses the RSS-combined tolerance form from ASHRAE G36 — |A - B| > sqrt(eA^2 +
    eB^2) — with the dataset's stated +/-0.4 degC sensor accuracy, which is the
    correct way to combine two instruments' error bands.
    """
    zones = zones or R.comfort_zones(D.floor_of(h))
    d = D.occupied(h, zones=zones, require_temp=True)
    if not len(d):
        return pd.DataFrame()

    d = d.assign(period=d.ts.dt.to_period(freq.rstrip("S")).dt.to_timestamp())
    piv = d.pivot_table(index="period", columns="zone", values="temp_c", aggfunc="mean")
    n = d.pivot_table(index="period", columns="zone", values="temp_c", aggfunc="size")

    med = piv.median(axis=1)
    dev = piv.sub(med, axis=0)
    eps = R.conv("sensor_acc_temp")
    tol = np.sqrt(2) * eps

    out = dev.reset_index().melt(id_vars="period", var_name="zone", value_name="deviation")
    out = out.merge(n.reset_index().melt(id_vars="period", var_name="zone", value_name="n"),
                    on=["period", "zone"], how="left")
    out["tolerance"] = tol
    out["exceeds"] = out.deviation.abs() > tol
    # Periods with almost no data produce meaningless deviations (z2 has months
    # with n=0 and n=8), so mark them rather than letting them drive a trend.
    out["reliable"] = out.n.fillna(0) >= 40
    return out


def humidity_consistency(h: pd.DataFrame, zones=None) -> pd.DataFrame:
    """Absolute humidity should be near-uniform across zones on one air path.

    Temperature legitimately differs between zones; moisture content largely does
    not. A zone whose absolute humidity diverges is either faulty or genuinely on
    a separate air path — which is an independent way to reach the z1 verdict
    without relying on the temperature argument at all.
    """
    zones = zones or R.sensored_zones(D.floor_of(h))
    d = D.occupied(h, zones=zones, require_temp=True).dropna(subset=["rh_pct"])
    if not len(d):
        return pd.DataFrame()
    d = d.assign(
        abs_hum=D.absolute_humidity(d.temp_c, d.rh_pct),
        dew=D.dew_point(d.temp_c, d.rh_pct),
        w=D.humidity_ratio(d.temp_c, d.rh_pct),
    )
    g = d.groupby("zone", as_index=False).agg(
        n=("abs_hum", "size"),
        abs_hum=("abs_hum", "mean"), dew=("dew", "mean"), w=("w", "mean"),
        temp=("temp_c", "mean"), rh=("rh_pct", "mean"))
    g["abs_hum_dev"] = g.abs_hum - g.abs_hum.median()
    g["dew_dev"] = g.dew - g.dew.median()
    return g.sort_values("abs_hum_dev")


def stuck_summary(H: pd.DataFrame) -> pd.DataFrame:
    """Which points are actually frozen, ranked. On this floor the answer is the
    plug meters, not the temperature sensors."""
    h = score_point_days(H)
    keys = (["floor"] if "floor" in h else []) + ["zone", "channel", "column", "role"]
    g = h.groupby(keys, as_index=False).agg(
        days=("day", "nunique"),
        flat_days=("flag_flatline", "sum"),
        longest_min=("flat_measured", "max"),
        median_distinct=("distinct", "median"),
    )
    g["flat_day_share"] = g.flat_days / g.days * 100
    return g.sort_values("flat_day_share", ascending=False)
