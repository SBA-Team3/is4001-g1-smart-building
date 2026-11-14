"""
Comfort metrics against a *declared band*.

A note on naming that matters for credibility: this is deliberately **not**
"Unmet Load Hours". UMLH is defined relative to a setpoint, and CU-BEMS has no
setpoint channel — we only ever observe realised temperature, which is an
*outcome* of the setpoint. Calling a band-exceedance metric UMLH would be a
category error, so we measure exceedance against a band we declare and cite.

We also deliberately do not use the ASHRAE 55 **adaptive** comfort model: it
applies only to naturally conditioned spaces and explicitly does not cover
mechanically cooled buildings. This building is air-conditioned.

Exceedance Degree-Hours (EDH) is preferred over plain exceedance hours, because
hour counts mask the *magnitude* of deviation: ten hours at 22.9 degC and ten
hours at 18 degC score identically on hours, and very differently on degree-hours.

The overcooling/overheating split is the operationally important one here. In
air-conditioning mode a zone that is too cold is simultaneously a comfort fault
*and* an energy fault, which is why it ranks above overheating on this floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import registry as R


def band(lo: float | None = None, hi: float | None = None) -> tuple[float, float]:
    return (R.conv("comfort_lo") if lo is None else lo), (R.conv("comfort_hi") if hi is None else hi)


def zone_comfort(h: pd.DataFrame, zones=None, start=None, end=None,
                 lo: float | None = None, hi: float | None = None,
                 ac_on: bool = True, floor: int = 5) -> pd.DataFrame:
    """Per-zone exceedance hours and degree-hours over a period.

    Only zones whose temperature point is tagged `zone_air_temp` are included —
    z1's sensor sits in a supply-air path, so a comfort statistic about it would
    describe the air being delivered, not the space anyone occupies.
    """
    lo = band()[0] if lo is None else lo
    hi = band()[1] if hi is None else hi
    zones = zones or R.comfort_zones(floor)

    d = D.slice_hourly(h, start, end, zones)
    d = D.occupied(d, zones=zones, require_temp=True, ac_on=ac_on)
    if not len(d):
        return pd.DataFrame()

    d = d.assign(
        cold_dh=(lo - d.temp_c).clip(lower=0),
        warm_dh=(d.temp_c - hi).clip(lower=0),
    )
    g = d.groupby("zone", as_index=False).agg(
        hours=("temp_c", "size"),
        mean_c=("temp_c", "mean"), median_c=("temp_c", "median"),
        sd_c=("temp_c", "std"), min_c=("temp_c", "min"), max_c=("temp_c", "max"),
        cold_hours=("temp_c", lambda s: int((s < lo).sum())),
        warm_hours=("temp_c", lambda s: int((s > hi).sum())),
        cold_dh=("cold_dh", "sum"), warm_dh=("warm_dh", "sum"),
        ac_kwh=("ac_kwh", "sum"),
        rh=("rh_pct", "mean"),
        temp_minutes=("temp_minutes", "mean"),
    )
    g["in_band_pct"] = (1 - (g.cold_hours + g.warm_hours) / g.hours) * 100
    g["cold_pct"] = g.cold_hours / g.hours * 100
    g["warm_pct"] = g.warm_hours / g.hours * 100
    g["edh"] = g.cold_dh + g.warm_dh
    g["edh_per_hour"] = g.edh / g.hours
    # Overcooling is weighted above overheating: in cooling mode a cold hour is
    # a comfort fault AND wasted energy, an overheated hour is only the former.
    g["severity"] = g.cold_dh * 1.5 + g.warm_dh
    g["cooling_below_band_kwh"] = [
        d[(d.zone == z) & (d.temp_c < lo)].ac_kwh.sum() for z in g.zone]
    g["cooling_below_band_pct"] = g.cooling_below_band_kwh / g.ac_kwh * 100
    g["data_confidence"] = np.where(g.temp_minutes >= 55, "high",
                                    np.where(g.temp_minutes >= 40, "medium", "low"))
    return g.sort_values("severity", ascending=False)


def humidity_exceedance(h: pd.DataFrame, zones=None, start=None, end=None,
                        floor: int = 5) -> pd.DataFrame:
    """Time above the ASHRAE 55 humidity-ratio limit (0.012 kg/kg).

    Using the humidity ratio rather than a flat 60% RH is the standards-correct
    computation, and temp + RH is enough to get there.
    """
    zones = zones or R.comfort_zones(floor)
    d = D.slice_hourly(h, start, end, zones)
    d = D.occupied(d, zones=zones, require_temp=True).dropna(subset=["rh_pct"])
    if not len(d):
        return pd.DataFrame()

    wmax = R.conv("humidity_ratio_max")
    d = d.assign(w=D.humidity_ratio(d.temp_c, d.rh_pct),
                 dew=D.dew_point(d.temp_c, d.rh_pct))
    return d.groupby("zone", as_index=False).agg(
        hours=("w", "size"), mean_w=("w", "mean"), mean_rh=("rh_pct", "mean"),
        mean_dew=("dew", "mean"),
        over_limit_pct=("w", lambda s: (s > wmax).mean() * 100),
        over_60rh_pct=("rh_pct", lambda s: (s > R.conv("rh_hi")).mean() * 100),
    )


def shift_distribution(temps, d_mean: float = 0.0, sd_scale: float = 1.0):
    """Shift the mean and optionally compress the spread of an observed sample.

    Two physically distinct levers: `d_mean` is a setpoint change (a work order),
    `sd_scale` is control-loop retuning (a commissioning job). Floor 5's
    distributions are wide rather than merely mis-centred, so the second lever
    matters as much as the first.
    """
    t = np.asarray(temps, dtype=float)
    mu = t.mean()
    return mu + d_mean + (t - mu) * sd_scale


