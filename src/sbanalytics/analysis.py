"""
The surviving analytical work: setpoint sensitivity, the thermal calibration,
the M&V baseline, and the pilot power calculation.

Ported from the prototype's `f5_core.py`, with one substantive correction.

**The baseline training window has moved.** The prototype trained on
2019-03-06 -> 2019-10-01 and tested on 2019-10-01 onward, which straddles the
2019-09-27 re-apportionment of cooling energy between z2 and z4. Its headline
result — "z4 fails ASHRAE G14 with NMBE -23%, an unexplained level shift" — was
measuring a meter change as though it were physics. Training now stops at
2019-09-26 and the boundary is declared, not discovered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import data as D
from . import registry as R

FEATURES = ["out_temp", "out_rh", "out_solar", "out_temp_3h", "out_temp_24h",
            "hour_sin", "hour_cos", "dow", "month"]
FORBIDDEN = ["temp_c", "rh_pct", "lux_idx"]

#: Training stops before the 2019-09-27 metering change (see module docstring).
TRAIN = ("2019-03-07", "2019-09-26")
TEST = ("2019-09-30", "2019-12-31")
HOLDOUT = ("2018-07-01", "2018-09-11")

CONTROLS = "out_temp + out_temp_24h + out_temp_3h + out_solar + out_rh + C(hour) + C(month)"


def add_cyclic(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["hour_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.hour / 24)
    return d


# --------------------------------------------------------------------------- #
# thermal calibration on the lunch-hour shutdown
# --------------------------------------------------------------------------- #

def lunch_frame(h: pd.DataFrame, zones=None, off_hour: int = 12,
                ref_hour: int = 11, floor: int = 5) -> pd.DataFrame:
    """One row per zone-day: free-running drift while cooling is off.

    At 12:00 the AC draws ~0 kWh on every weekday, so for that hour the zone is a
    free-running thermal mass whose only *measured* heat input is lighting plus
    plug load. That identifies capacitance and conductance without ever putting
    indoor temperature on the right-hand side as a cause of energy.
    """
    zones = zones or R.comfort_zones(floor)
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(zones) & h.temp_ok].copy()
    d["int_kwh"] = d.light_kwh.fillna(0) + d.plug_kwh.fillna(0)
    w = d.pivot_table(index=["date", "zone"], columns="hour",
                      values=["temp_c", "ac_kwh", "out_temp", "out_solar", "int_kwh"])
    need = [("temp_c", ref_hour), ("temp_c", off_hour), ("out_temp", off_hour),
            ("out_solar", off_hour), ("int_kwh", off_hour), ("ac_kwh", off_hour)]
    if any(c not in w.columns for c in need):
        return pd.DataFrame()

    L = pd.DataFrame({
        "t_ref": w[("temp_c", ref_hour)], "t_off": w[("temp_c", off_hour)],
        "out_temp": w[("out_temp", off_hour)], "out_solar": w[("out_solar", off_hour)],
        "int_kwh": w[("int_kwh", off_hour)], "ac_off_kwh": w[("ac_kwh", off_hour)],
    }).dropna().reset_index()
    L = L[L.ac_off_kwh <= 0.5]
    L["drift"] = L.t_off - L.t_ref
    L["drive"] = L.out_temp - L.t_ref
    return L


def fit_physics(h: pd.DataFrame, zones=None, use_solar: bool = True,
                floor: int = 5) -> dict:
    """Calibrate C and UA, and read off d(load)/d(setpoint) = -UA.

    An envelope-conduction-only number: it excludes the latent (dehumidification)
    and fan work a warmer setpoint also saves, so it is a lower bound. Its value
    is that it cannot be confounded by occupancy, which is what broke the
    matching estimate.
    """
    L = lunch_frame(h, zones, floor=floor)
    out = {"n": len(L), "frame": L, "per_zone": {}, "pooled": None,
           "model": None, "use_solar": use_solar}
    if len(L) < 50:
        return out

    zt = " + C(zone)" if L.zone.nunique() > 1 else ""
    sol = " + out_solar" if use_solar else ""
    m = smf.ols(f"drift ~ int_kwh + drive{sol}{zt}", data=L).fit(
        cov_type="cluster", cov_kwds={"groups": L.date})
    out["model"] = m
    ci = m.conf_int()

    occ = D.occupied(h, zones=zones or R.comfort_zones(floor), ac_on=True)
    load = occ.ac_kwh.mean()

    def pack(inv_c, ua_c, inv_ci, ua_ci, mean_load):
        if inv_c <= 0 or not np.isfinite(mean_load) or mean_load <= 0:
            return None
        C = 1.0 / inv_c
        UA = C * ua_c
        lo_C, hi_C = 1.0 / max(inv_ci[1], 1e-9), 1.0 / max(inv_ci[0], 1e-9)
        cands = [c * u for c in (lo_C, hi_C) for u in ua_ci if c > 0 and u > 0]
        ua_lo, ua_hi = (min(cands), max(cands)) if cands else (np.nan, np.nan)
        return {"C_kwh_per_c": C, "UA_kw_per_c": UA, "mean_load_kwh": mean_load,
                "pct_per_c": -UA / mean_load * 100,
                "pct_per_c_lo": -ua_hi / mean_load * 100,
                "pct_per_c_hi": -ua_lo / mean_load * 100,
                "kwh_per_c": -UA}

    out["pooled"] = pack(m.params["int_kwh"], m.params["drive"],
                         tuple(ci.loc["int_kwh"]), tuple(ci.loc["drive"]), load)

    for z, g in L.groupby("zone"):
        if len(g) < 40:
            continue
        mz = smf.ols(f"drift ~ int_kwh + drive{sol}", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g.date})
        cz = mz.conf_int()
        lz = D.occupied(h, zones=[z], ac_on=True).ac_kwh.mean()
        p = pack(mz.params["int_kwh"], mz.params["drive"],
                 tuple(cz.loc["int_kwh"]), tuple(cz.loc["drive"]), lz)
        if p:
            p["n"], p["r2"] = len(g), mz.rsquared
            out["per_zone"][z] = p
    return out


def load_decomposition(h: pd.DataFrame, phys: dict, zones=None,
                       floor: int = 5) -> pd.DataFrame:
    """Split measured load into envelope / internal / residual.

    This is why the physics and regression estimates disagree: a setpoint change
    acts on the envelope term and some unknown share of latent load in the
    residual. It acts on none of the metered internal gains.
    """
    zones = zones or R.comfort_zones(floor)
    d = D.occupied(h, zones=zones, require_temp=True, ac_on=True)
    d = d.assign(int_kwh=d.light_kwh.fillna(0) + d.plug_kwh.fillna(0))
    rows = []
    for z, g in d.groupby("zone"):
        p = phys["per_zone"].get(z) or phys["pooled"]
        if not p:
            continue
        env = (p["UA_kw_per_c"] * (g.out_temp - g.temp_c)).clip(lower=0).mean()
        meas, intn = g.ac_kwh.mean(), g.int_kwh.mean()
        rows.append({"zone": z, "measured_kw": meas, "envelope_kw": env,
                     "internal_kw": intn, "residual_kw": meas - env - intn,
                     "envelope_pct": env / meas * 100,
                     "internal_pct": intn / meas * 100,
                     "residual_pct": (meas - env - intn) / meas * 100})
    return pd.DataFrame(rows).set_index("zone") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# regression elasticity (the upper bound)
# --------------------------------------------------------------------------- #

def run_regression(h: pd.DataFrame, zones=None, floor: int = 5) -> dict:
    """Regression-adjusted %/degC, pooled and per zone.

    Indoor temperature is the regressor, so this is the *upper* end of the range:
    on a busy day a zone is simultaneously warmer and drawing more cooling, and
    with no occupancy channel that confound stays in. z5 comes out with the wrong
    sign, which is the confound made visible.
    """
    zones = zones or R.comfort_zones(floor)
    full = D.occupied(h, zones=zones, require_temp=True, ac_on=True)
    out = {"pooled": None, "per_zone": []}
    if len(full) < 200:
        return out

    zt = " + C(zone)" if full.zone.nunique() > 1 else ""
    m = smf.ols(f"ac_kwh ~ temp_c + {CONTROLS}{zt}", data=full).fit(
        cov_type="cluster", cov_kwds={"groups": full.date})
    ml = full.ac_kwh.mean()
    ci = m.conf_int().loc["temp_c"]
    out["pooled"] = {"n": int(m.nobs), "r2": m.rsquared,
                     "pct_per_c": m.params["temp_c"] / ml * 100,
                     "lo": ci[0] / ml * 100, "hi": ci[1] / ml * 100}

    for z, g in full.groupby("zone"):
        if len(g) < 150:
            continue
        mm = smf.ols(f"ac_kwh ~ temp_c + {CONTROLS}", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g.date})
        cc = mm.conf_int().loc["temp_c"]
        mlz = g.ac_kwh.mean()
        out["per_zone"].append({
            "zone": z, "n": len(g), "pct_per_c": mm.params["temp_c"] / mlz * 100,
            "lo": cc[0] / mlz * 100, "hi": cc[1] / mlz * 100,
            "sign": "as expected" if mm.params["temp_c"] < 0 else "WRONG SIGN"})
    return out


def estimator_range(phys: dict, reg: dict) -> pd.DataFrame:
    """The estimators side by side. The spread between them is method
    disagreement, not sampling noise — no larger sample would shrink it."""
    rows = []
    if phys.get("pooled"):
        p = phys["pooled"]
        rows.append({"estimator": "Physics — envelope conduction",
                     "pct_per_c": p["pct_per_c"], "lo": p["pct_per_c_lo"],
                     "hi": p["pct_per_c_hi"], "role": "lower bound",
                     "note": "Calibrated on the 12:00 shutdown; immune to the "
                             "occupancy confound but counts envelope only."})
    if reg.get("pooled"):
        r = reg["pooled"]
        rows.append({"estimator": "Regression on indoor temperature",
                     "pct_per_c": r["pct_per_c"], "lo": r["lo"], "hi": r["hi"],
                     "role": "upper bound",
                     "note": "Largest and most confounded: warm hours are also "
                             "busy hours."})
    if len(rows) == 2:
        a, b = rows[0]["pct_per_c"], rows[1]["pct_per_c"]
        rows.append({"estimator": "Midpoint", "pct_per_c": (a + b) / 2,
                     "lo": min(a, b), "hi": max(a, b), "role": "working figure",
                     "note": "Assumes latent savings roughly match envelope "
                             "savings. Defensible, but an assumption."})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# M&V baseline
# --------------------------------------------------------------------------- #

def ashrae_g14(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (np.sqrt(((y - pred) ** 2).mean()) / y.mean() * 100,
            (y - pred).sum() / y.sum() * 100)


def fit_baselines(h: pd.DataFrame, zones=None, floor: int = 5):
    """Per-zone GBM and Ridge on exogenous features only."""
    assert not set(FEATURES) & set(FORBIDDEN), "LEAKAGE: indoor channel in features"
    zones = zones or R.comfort_zones(floor)
    data = add_cyclic(D.occupied(h, zones=zones)).dropna(subset=["ac_kwh"] + FEATURES)
    fitted, rows = {}, []

    for z, g in data.groupby("zone"):
        tr = g[(g.ts >= TRAIN[0]) & (g.ts <= TRAIN[1])]
        te = g[(g.ts >= TEST[0]) & (g.ts <= TEST[1])]
        ho = g[(g.ts >= HOLDOUT[0]) & (g.ts <= HOLDOUT[1])]
        if len(tr) < 200 or len(te) < 50:
            continue
        ridge = make_pipeline(
            ColumnTransformer([("spline", SplineTransformer(n_knots=6, degree=3),
                                ["out_temp", "out_temp_24h", "out_solar"])],
                              remainder="passthrough"),
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(tr[FEATURES], tr.ac_kwh)
        gbm = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=6,
            random_state=0).fit(tr[FEATURES], tr.ac_kwh)

        fitted[z] = {"ridge": ridge, "gbm": gbm, "train": tr, "test": te, "holdout": ho}
        for name, mdl in (("Ridge", ridge), ("GBM", gbm)):
            for split, dd in (("test (Oct–Dec 2019)", te),
                              ("holdout (Jul–Sep 2018)", ho)):
                if not len(dd):
                    continue
                pred = np.clip(mdl.predict(dd[FEATURES]), 0, None)
                cv, nmbe = ashrae_g14(dd.ac_kwh, pred)
                rows.append({"zone": z, "model": name, "split": split, "n": len(dd),
                             "cv_rmse": cv, "nmbe": nmbe,
                             "meets_g14": "yes" if (cv <= 30 and abs(nmbe) <= 10) else "no"})
    return fitted, pd.DataFrame(rows)


def combined_meter_check(h: pd.DataFrame, pair=("z2", "z4"),
                         split: str = "2019-09-27", floor: int = 5) -> dict:
    """Evidence that the 2019-09-27 step is a re-apportionment, not a demand change.

    If cooling energy moved *between* two meters, then each zone shifts sharply
    while their sum barely moves, and the ratio between them flips. If instead
    one zone genuinely started using less cooling, the sum would fall too. This
    distinguishes the two, and it is what disqualifies z4's baseline from M&V
    across the boundary.
    """
    a, b = pair
    d = D.occupied(h, zones=list(pair))
    cut = pd.Timestamp(split)
    before = d[(d.ts >= pd.Timestamp("2019-06-01")) & (d.ts < cut)]
    after = d[(d.ts >= cut) & (d.ts <= pd.Timestamp("2019-12-31"))]
    if not len(before) or not len(after):
        return {}

    def side(x):
        m = x.groupby("zone").ac_kwh.mean()
        tot = m.sum()
        return {a: m.get(a, np.nan), b: m.get(b, np.nan), "sum": tot,
                "ratio": m.get(a, np.nan) / m.get(b, np.nan) if m.get(b, 0) else np.nan}

    B, A_ = side(before), side(after)
    return {
        "pair": pair, "split": split, "before": B, "after": A_,
        "delta_a_pct": (A_[a] / B[a] - 1) * 100,
        "delta_b_pct": (A_[b] / B[b] - 1) * 100,
        "delta_sum_pct": (A_["sum"] / B["sum"] - 1) * 100,
        "ratio_before": B["ratio"], "ratio_after": A_["ratio"],
        "n_before": len(before), "n_after": len(after),
    }


# --------------------------------------------------------------------------- #
# levers
# --------------------------------------------------------------------------- #

def setpoint_saving(h: pd.DataFrame, pct_per_c: float, d_temp: float,
                    zones=None, realisation: float = 1.0, floor: int = 5,
                    annualise: bool = True) -> pd.DataFrame:
    zones = zones or R.comfort_zones(floor)
    occ = D.occupied(h, zones=zones)
    f = D.annual_factor(h) if annualise else 1.0
    rows = []
    for z, g in occ.groupby("zone"):
        kwh = g.ac_kwh.sum() * f
        rows.append({"zone": z, "kwh_year": kwh,
                     "saving_kwh": -kwh * (pct_per_c / 100) * d_temp * realisation})
    return pd.DataFrame(rows)


def schedule_trim(h: pd.DataFrame, phys: dict, drop_hours=(), recover: bool = True,
                  zones=None, floor: int = 5) -> pd.DataFrame:
    """Energy released by switching cooling off in `drop_hours`, net of recovery.

    The gross figure is metered. What the physics adds is the recovery charge —
    C * dT to pull the zone back down when cooling resumes — which is why
    trimming the last hour of the day is nearly free while extending the lunch
    gap is not.
    """
    zones = zones or R.comfort_zones(floor)
    f = D.annual_factor(h)
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(zones)]
    drop = sorted(drop_hours)
    rows = []
    for z, g in d.groupby("zone"):
        gross = g[g.hour.isin(drop)].ac_kwh.sum()
        p = phys["per_zone"].get(z) or phys["pooled"] or {}
        C, UA = p.get("C_kwh_per_c", np.nan), p.get("UA_kw_per_c", np.nan)

        prior = [x for x in R.ac_on_hours(floor) if drop and x < min(drop)]
        ref = g[g.temp_ok & g.hour.isin([max(prior)] if prior else R.ac_on_hours(floor))]
        start_t = ref.temp_c.mean()
        sub = g[g.hour.isin(drop)]

        rate = np.nan
        if np.isfinite(C) and C > 0 and len(sub):
            q = UA * (sub.out_temp.mean() - start_t) + \
                (sub.light_kwh.fillna(0) + sub.plug_kwh.fillna(0)).mean()
            rate = max(q, 0) / C

        resumes = any(x in R.ac_on_hours(floor) for x in range(max(drop) + 1, 18)) if drop else False
        n_days = sub.date.nunique() if drop else 0
        rec = C * rate * len(drop) * n_days if (recover and resumes and np.isfinite(rate) and n_days) else 0.0

        rows.append({"zone": z, "gross_kwh": gross, "recovery_kwh": rec,
                     "net_kwh": gross - rec, "gross_kwh_year": gross * f,
                     "net_kwh_year": (gross - rec) * f, "drift_c_per_h": rate,
                     "resumes_same_day": resumes,
                     "end_temp_c": start_t + rate * len(drop) if np.isfinite(rate) else np.nan})
    return pd.DataFrame(rows).set_index("zone")


def hour_profile(h: pd.DataFrame, zones=None, floor: int = 5) -> pd.DataFrame:
    zones = zones or R.comfort_zones(floor)
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(zones)]
    e = d.groupby("hour").ac_kwh.agg(mean_kwh="mean", total_kwh="sum", n="size")
    e["frac_off"] = d.assign(off=d.ac_kwh <= 0.5).groupby("hour").off.mean()
    t = d[d.temp_ok].groupby("hour").temp_c.mean().rename("mean_c")
    return e.join(t)


# --------------------------------------------------------------------------- #
# pilot design
# --------------------------------------------------------------------------- #

def detectable_saving(cv_rmse_pct, weeks, hours_per_day=9, days_per_week=5,
                      alpha=0.05, power=0.80, rho=0.0):
    """Minimum detectable saving for a baseline comparison of `weeks` length."""
    from scipy.stats import norm
    n = weeks * days_per_week * hours_per_day
    if n <= 1:
        return np.nan
    deff = 1 + (hours_per_day - 1) * rho
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * cv_rmse_pct / np.sqrt(max(n / deff, 1.0))


def weeks_required(cv_rmse_pct, target_pct, hours_per_day=9, days_per_week=5,
                   alpha=0.05, power=0.80, rho=0.0, max_weeks=260):
    for w in range(1, max_weeks + 1):
        if detectable_saving(cv_rmse_pct, w, hours_per_day, days_per_week,
                             alpha, power, rho) <= abs(target_pct):
            return w
    return np.nan
