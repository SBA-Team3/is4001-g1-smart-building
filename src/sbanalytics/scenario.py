"""
Intervention simulator: a package of measures, sized as low / mid / high ranges,
with cost, carbon, payback, comfort impact and an M&V plan for each.

The engine is analysis.py. Nothing here re-estimates a sensitivity; it only
combines the existing estimators without double counting:

* zone-temperature lift — analysis.setpoint_saving at the physics (low),
  midpoint (mid) and regression (high) %/degC. There is no setpoint channel, so
  the lift is modelled as a shift of the observed zone-temperature distribution
  (comfort.shift_distribution), exactly as the setpoint tab does.
* schedule trim — analysis.schedule_trim on the last n cooled hours of the day.
  Metered energy; net of recovery (low) to gross (high).
* weekend/holiday lockout — anomaly.after_hours_ac excess on non-workdays,
  annualised. Metered; half (low) to all (high), because some runs are
  authorised (notebook 06).

Combined = S x (1 - T/B0) + T + L per case: the %/degC is applied to the
occupied baseline that is left after trimming. Lockout hours are non-workdays
and never overlap the occupied baseline, so they simply add.

Scope: the lift and the trim act on comfort-sensored zones only (on F5, z2, z4
and z5 — z1's sensor is a supply-air sensor and z3 a stairwell), because that is
the population the estimators were calibrated on. The lockout is floor-wide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis as A
from . import anomaly as AN
from . import baseline as B
from . import kpis as K
from . import registry as R

CASES = ("low", "mid", "high")


def sensitivities(est: pd.DataFrame) -> dict:
    """%/degC for the low, mid and high case from analysis.estimator_range.

    Low/high are the two estimators (physics lower bound, regression upper
    bound); mid is their midpoint. With one estimator, its CI stands in.
    """
    if (est.role == "working figure").any():
        r = est[est.role == "working figure"].iloc[0]
        lo, hi = sorted([r.lo, r.hi], key=abs)
        return {"low": lo, "mid": r.pct_per_c, "high": hi}
    r = est.iloc[0]
    lo, hi = sorted([r.lo, r.hi], key=abs)
    return {"low": lo, "mid": r.pct_per_c, "high": hi}


def on_hours(h: pd.DataFrame, floor: int) -> list[int]:
    hp = A.hour_profile(h, floor=floor)
    return [int(x) for x in hp.index if hp.loc[x, "frac_off"] < 0.5]


def max_trim(h: pd.DataFrame, floor: int) -> int:
    """Length of the last contiguous cooled block (the afternoon) — trimming
    further than that would cross the lunch shutdown and need recovery."""
    oh = on_hours(h, floor)
    n = 0
    for a, b in zip(reversed(oh), reversed([None] + oh[:-1])):
        n += 1
        if b is None or a - b > 1:
            break
    return n


def trim_hours(h: pd.DataFrame, floor: int, n: int) -> tuple[int, ...]:
    """The last n cooled hours of the day — the end-of-day trim, which needs no
    recovery because cooling does not resume."""
    oh = on_hours(h, floor)
    return tuple(oh[-n:]) if n > 0 else ()


# --------------------------------------------------------------------------- #
# measures
# --------------------------------------------------------------------------- #

def lift_measure(h: pd.DataFrame, est: pd.DataFrame, d_temp: dict, floor: int) -> dict:
    """Zone-temperature lift; `d_temp` maps zone -> degC."""
    sens = sensitivities(est)
    out = {"base_kwh": 0.0, **{c: 0.0 for c in CASES}}
    for z in R.comfort_zones(floor):
        dt = float(d_temp.get(z, 0.0))
        base = A.setpoint_saving(h, -1, 1, zones=[z], floor=floor).kwh_year.sum()
        out["base_kwh"] += base
        for c in CASES:
            out[c] += A.setpoint_saving(h, sens[c], dt, zones=[z], floor=floor).saving_kwh.sum()
    return out


def trim_measure(h: pd.DataFrame, phys: dict, floor: int, n_hours: int) -> dict:
    drop = trim_hours(h, floor, n_hours)
    if not drop or not phys.get("pooled"):
        return {"hours": drop, "low": 0.0, "mid": 0.0, "high": 0.0, "end_temp_c": np.nan,
                "resumes": False}
    t = A.schedule_trim(h, phys, drop, True, floor=floor)
    net, gross = t.net_kwh_year.sum(), t.gross_kwh_year.sum()
    return {"hours": drop, "low": min(net, gross), "mid": (net + gross) / 2, "high": max(net, gross),
            "end_temp_c": float(t.end_temp_c.mean()), "resumes": bool(t.resumes_same_day.any())}


def lockout_measure(h: pd.DataFrame, floor: int) -> dict:
    """Metered weekend/holiday/closure AC above the floor's own night baseload."""
    fh = B.floor_hourly(h)
    fd = B.floor_daily(fh)
    year = 365 / max(int(fd.ok.sum()), 1)
    ah = AN.after_hours_ac(fh)
    excess = 0.0
    if len(ah):
        excess = ah[ah.day_type.isin(["weekend", "holiday", "closure"])].excess_kwh.sum() * year
    lo = excess * R.conv("lockout_low_frac")
    return {"low": lo, "mid": (lo + excess) / 2, "high": excess}


# --------------------------------------------------------------------------- #
# comfort impact
# --------------------------------------------------------------------------- #

def comfort_after(h: pd.DataFrame, d_temp: dict, floor: int,
                  lo: float | None = None, hi: float | None = None) -> dict:
    """Projected overcooled / overheated share after the lift, same definition
    as the portfolio KPI (AC-kWh-weighted cooled occupied hours)."""
    d = K.cooled_occupied(h)
    now = K.overcooled_share(d, by=(), lo=lo, hi=hi)
    after = K.overcooled_share(d, by=(), shift={z: float(v) for z, v in d_temp.items()}, lo=lo, hi=hi)
    get = lambda f, c: float(f[c].iloc[0] * 100) if len(f) else np.nan
    return {"below_now": get(now, "share_below"), "above_now": get(now, "share_above"),
            "below_after": get(after, "share_below"), "above_after": get(after, "share_above")}


# --------------------------------------------------------------------------- #
# the package
# --------------------------------------------------------------------------- #

def _payback(cost: float, thb_per_year: float) -> float:
    if cost <= 0:
        return 0.0
    return cost / thb_per_year if thb_per_year > 0 else np.inf


def simulate(h: pd.DataFrame, est: pd.DataFrame, phys: dict, floor: int, *,
             d_temp: dict, trim_n: int = 0, lockout: bool = False,
             cost: dict | None = None, tariff: float | None = None,
             ef: float | None = None, parts: dict | None = None,
             lo: float | None = None, hi: float | None = None) -> dict:
    """Size every selected measure and the combined package.

    `parts` lets a caller pass pre-computed measure results (the tornado chart
    re-runs the package many times and the metered parts do not change).
    """
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    ef = R.conv("co2_kg_per_kwh") if ef is None else ef
    cost = cost or {}
    p = parts or {}
    lift = p.get("lift") or lift_measure(h, est, d_temp, floor)
    trim = p.get("trim") or trim_measure(h, phys, floor, trim_n)
    lock = p.get("lockout") or (lockout_measure(h, floor) if lockout
                                else {c: 0.0 for c in CASES})
    if not lockout:
        lock = {c: 0.0 for c in CASES}
    comfort = p.get("comfort") or comfort_after(h, d_temp, floor, lo=lo, hi=hi)

    b0 = lift["base_kwh"]
    combined = {c: lift[c] * (1 - (trim[c] / b0 if b0 else 0)) + trim[c] + lock[c] for c in CASES}

    lift_conf = "medium — two estimators bracket it" if (est.role == "working figure").any() \
        else "low — single estimator"
    trim_conf = ("high — metered, no recovery" if trim["hours"] and not trim["resumes"]
                 else "medium — recovery is modelled" if trim["hours"] else "—")
    rows = []
    for key, name, r, conf, comfort_txt in (
        ("lift", "Zone-temperature lift", lift, lift_conf,
         f"{comfort['below_after']:.0f}% below {R.conv('comfort_lo') if lo is None else lo:g} °C, "
         f"{comfort['above_after']:.0f}% above {R.conv('comfort_hi') if hi is None else hi:g} °C"),
        ("trim", "Schedule trim", trim, trim_conf,
         f"zones reach ≈{trim['end_temp_c']:.1f} °C at end of trimmed gap"
         if np.isfinite(trim.get("end_temp_c", np.nan)) else "—"),
        ("lockout", "Weekend/holiday lockout", lock,
         "high — metered; some runs may be authorised" if lockout else "—",
         "none in occupied hours" if lockout else "—"),
    ):
        rows.append(_row(name, r, cost.get(key, 0.0), tariff, ef, conf, comfort_txt))
    total_cost = sum(cost.get(k, 0.0) for k in ("lift", "trim", "lockout"))
    confs = [x["confidence"] for x in rows if x["kWh/yr mid"] > 0]
    worst = min(confs, key=lambda s: ["low", "medium", "high"].index(s.split(" ")[0])) if confs else "—"
    rows.append(_row("Combined (no double counting)", combined, total_cost, tariff, ef,
                     worst.split(" ")[0] if confs else "—",
                     f"{comfort['below_after']:.0f}% below / {comfort['above_after']:.0f}% above band"))
    return {"table": pd.DataFrame(rows), "lift": lift, "trim": trim, "lockout": lock,
            "combined": combined, "comfort": comfort, "base_kwh": b0}


def _row(name, r, cost, tariff, ef, conf, comfort_txt) -> dict:
    return {"measure": name,
            "kWh/yr low": r["low"], "kWh/yr mid": r["mid"], "kWh/yr high": r["high"],
            "฿/yr low": r["low"] * tariff, "฿/yr mid": r["mid"] * tariff, "฿/yr high": r["high"] * tariff,
            "tCO₂e/yr mid": r["mid"] * ef / 1000,
            "cost ฿": cost, "payback yr": _payback(cost, r["mid"] * tariff),
            "comfort after": comfort_txt, "confidence": conf}


# --------------------------------------------------------------------------- #
# M&V plan
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #

def tornado(h: pd.DataFrame, est: pd.DataFrame, phys: dict, floor: int, *,
            d_temp: dict, trim_n: int, lockout: bool, tariff: float,
            max_trim: int) -> pd.DataFrame:
    """Combined mid saving (THB/yr) as each assumption moves across its range.

    Ranges are declared conventions (sens_*). The estimator swing replaces the
    midpoint %/degC with the physics (low) and regression (high) estimate.
    """
    base_parts = {"lockout": lockout_measure(h, floor) if lockout else None}
    run = lambda **kw: simulate(h, est, phys, floor, d_temp=kw.get("d_temp", d_temp),
                                trim_n=kw.get("trim_n", trim_n), lockout=lockout,
                                tariff=kw.get("tariff", tariff),
                                parts={k: v for k, v in base_parts.items() if v})
    base = run()
    mid = base["combined"]["mid"] * tariff
    rows = []

    dt = R.conv("sens_dT_c")
    cap = R.conv("max_lift_c")
    shift = lambda s: {z: float(np.clip(v + s, 0, cap)) for z, v in d_temp.items()}
    lo_r, hi_r = run(d_temp=shift(-dt)), run(d_temp=shift(+dt))
    rows.append(("Zone-temperature lift ±%g °C" % dt,
                 lo_r["combined"]["mid"] * tariff, hi_r["combined"]["mid"] * tariff))

    tp = R.conv("sens_tariff_pct") / 100
    rows.append((f"Tariff ±{tp * 100:.0f}%", mid * (1 - tp), mid * (1 + tp)))

    lift = base["lift"]
    t, l_, b0 = base["trim"], base["lockout"], base["base_kwh"]
    adj = lambda s: (s * (1 - (t["mid"] / b0 if b0 else 0)) + t["mid"] + l_["mid"]) * tariff
    rows.append(("%/°C estimator (physics ↔ regression)", adj(lift["low"]), adj(lift["high"])))

    st = R.conv("sens_trim_h")
    lo_t = run(trim_n=int(max(trim_n - st, 0)))
    hi_t = run(trim_n=int(min(trim_n + st, max_trim)))
    rows.append((f"Schedule trim ±{st:g} h", lo_t["combined"]["mid"] * tariff,
                 hi_t["combined"]["mid"] * tariff))

    if lockout:
        rows.append(("Lockout: share of runs avoidable", adj_lock(base, "low", tariff),
                     adj_lock(base, "high", tariff)))

    out = pd.DataFrame(rows, columns=["assumption", "low_thb", "high_thb"])
    out["base_thb"] = mid
    out["swing"] = (out.high_thb - out.low_thb).abs()
    return out.sort_values("swing", ascending=False).reset_index(drop=True)


def adj_lock(res: dict, case: str, tariff: float) -> float:
    b0, l_, t, s = res["base_kwh"], res["lockout"], res["trim"], res["lift"]
    return (s["mid"] * (1 - (t["mid"] / b0 if b0 else 0)) + t["mid"] + l_[case]) * tariff
