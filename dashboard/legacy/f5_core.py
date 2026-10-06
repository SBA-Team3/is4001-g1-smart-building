"""
Floor 5 over-cooling — analysis core.

Everything the Streamlit app computes lives here, so the same functions can be
imported back into the notebook. No hardcoded results: every number the app
shows is derived from cache/f5_hourly.csv.gz at call time, which is what makes
the sliders meaningful rather than decorative.

Conventions carried over from `Floor 5 Over-cooling model.ipynb`:
  - z1 is excluded by default (supply-air sensor, see notebook s4)
  - z3 has no AC metering at all
  - a zone-hour counts as cooling if it drew > AC_ON_KWH
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

AC_ON_KWH = 0.5
COLD_THRESHOLD = 23.0
WARM_THRESHOLD = 26.0
COMFORT_TARGET = 24.0

ALL_ZONES = ["z1", "z2", "z4", "z5"]
CORE_ZONES = ["z2", "z4", "z5"]

# The AC schedule as metered, not as assumed. Hours 12 and 17 carry ~0 kWh on
# every weekday, so the notebook's 08:00-18:00 window averages eight cooled
# hours together with two uncooled ones. See notes in the app's Schedule tab.
AC_ON_HOURS = [7, 8, 9, 10, 11, 13, 14, 15, 16]
NOTEBOOK_HOURS = list(range(8, 18))

FEATURES = ["out_temp", "out_rh", "out_solar", "out_temp_3h", "out_temp_24h",
            "hour_sin", "hour_cos", "dow", "month"]
FORBIDDEN = ["temp_c", "rh_pct", "lux"]

BALANCE_COVS = ["out_temp", "out_solar", "out_rh", "hour", "month",
                "out_temp_24h", "out_temp_3h"]

SPECS = {
    "A. zone, hour, month, outdoor T, solar": ["zone", "hour", "month", "tbin", "sbin"],
    "B. A + 24h mean outdoor T":              ["zone", "hour", "month", "tbin", "sbin", "t24bin"],
    "C. B without month":                     ["zone", "hour", "tbin", "sbin", "t24bin"],
    "D. C + outdoor RH":                      ["zone", "hour", "tbin", "sbin", "t24bin", "rhbin"],
}

CONTROLS = "out_temp + out_temp_24h + out_temp_3h + out_solar + out_rh + C(hour) + C(month)"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_hourly():
    """The hourly zone table built by the notebook. Cached there, read-only here."""
    path = os.path.join(CACHE, "f5_hourly.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 'Floor 5 Over-cooling model.ipynb' through "
            "section 2 once to build the cache."
        )
    h = pd.read_csv(path, parse_dates=["ts"])
    h["int_kwh"] = h.light_kwh + h.plug_kwh          # metered internal heat gain
    h["hour_sin"] = np.sin(2 * np.pi * h.hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * h.hour / 24)
    return h


def occupied(h, zones=CORE_ZONES, hours=AC_ON_HOURS, require_sensor=True, ac_on=False):
    """Weekday, non-holiday, in-window zone-hours.

    require_sensor=False for anything that only needs the energy meter (totals,
    the baseline model); True for anything that reads indoor temperature.
    """
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.hour.isin(list(hours))
          & h.zone.isin(list(zones))]
    if require_sensor:
        d = d[d.sensor_alive].dropna(subset=["temp_c", "ac_kwh"])
    else:
        d = d.dropna(subset=["ac_kwh"])
    if ac_on:
        d = d[d.ac_kwh > AC_ON_KWH]
    return d.copy()


def annual_factor(h):
    """Weekdays in the record -> weekdays per year, so totals can be annualised."""
    span_days = h.date.nunique()
    wd = h[(~h.is_weekend) & (~h.is_holiday)].date.nunique()
    return 365.0 / span_days, wd, span_days


def energy_summary(h, zones=CORE_ZONES, hours=AC_ON_HOURS):
    """Metered occupied AC energy per zone, over the record and annualised."""
    d = occupied(h, zones, hours, require_sensor=False)
    f, _, _ = annual_factor(h)
    g = d.groupby("zone").ac_kwh.agg(["sum", "size"])
    g.columns = ["kwh_record", "zone_hours"]
    g["kwh_year"] = g.kwh_record * f
    g["kwh_per_hour"] = g.kwh_record / g.zone_hours
    return g


def comfort_summary(h, zones=CORE_ZONES, hours=AC_ON_HOURS, cold=COLD_THRESHOLD,
                    warm=WARM_THRESHOLD, ac_on=True):
    """Indoor-temperature picture per zone, on sensor-alive hours only."""
    d = occupied(h, zones, hours, require_sensor=True, ac_on=ac_on)
    rows = []
    for z, g in d.groupby("zone"):
        t = g.ac_kwh.sum()
        rows.append({
            "zone": z,
            "hours": len(g),
            "mean_c": g.temp_c.mean(),
            "median_c": g.temp_c.median(),
            "sd_c": g.temp_c.std(),
            "p5_c": g.temp_c.quantile(.05),
            "p95_c": g.temp_c.quantile(.95),
            "pct_below_cold": (g.temp_c < cold).mean() * 100,
            "pct_above_warm": (g.temp_c > warm).mean() * 100,
            "ac_kwh": t,
            "pct_kwh_below_cold": (g.loc[g.temp_c < cold, "ac_kwh"].sum() / t * 100) if t else np.nan,
        })
    return pd.DataFrame(rows).set_index("zone")


# --------------------------------------------------------------------------- #
# lever 1 -- thermal physics, calibrated on the lunch-hour shutdown
# --------------------------------------------------------------------------- #

def lunch_frame(h, zones=CORE_ZONES, off_hour=12, ref_hour=11):
    """One row per zone-day: the free-running drift while cooling is off.

    At `off_hour` the AC draws ~0 kWh on every weekday, so for that hour the
    zone is a free-running thermal mass with a *metered* internal gain
    (lighting + plug). That identifies both capacitance and conductance.
    """
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(list(zones)) & h.sensor_alive]
    w = d.pivot_table(index=["date", "zone"],
                      columns="hour",
                      values=["temp_c", "ac_kwh", "out_temp", "out_solar", "int_kwh"])
    need = [("temp_c", ref_hour), ("temp_c", off_hour), ("out_temp", off_hour),
            ("out_solar", off_hour), ("int_kwh", off_hour), ("ac_kwh", off_hour)]
    if any(c not in w.columns for c in need):
        return pd.DataFrame()
    L = pd.DataFrame({
        "t_ref":  w[("temp_c", ref_hour)],
        "t_off":  w[("temp_c", off_hour)],
        "out_temp": w[("out_temp", off_hour)],
        "out_solar": w[("out_solar", off_hour)],
        "int_kwh": w[("int_kwh", off_hour)],
        "ac_off_kwh": w[("ac_kwh", off_hour)],
    }).dropna().reset_index()
    L = L[L.ac_off_kwh <= AC_ON_KWH]              # keep only genuinely-off hours
    L["drift"] = L.t_off - L.t_ref                # degC gained in the hour
    L["drive"] = L.out_temp - L.t_ref             # outdoor-indoor driving force
    return L


def fit_physics(h, zones=CORE_ZONES, hours=AC_ON_HOURS, use_solar=True):
    """Calibrate a first-order thermal model and read off d(load)/d(setpoint).

    Free-running energy balance over the AC-off hour:

        C * dT/dt  =  UA * (T_out - T_in)  +  Q_internal  +  a * solar

    Regressing drift on the metered internal gain gives 1/C, so the conductance
    UA follows from the driving-force coefficient. In steady cooled operation
    the envelope term is the part of the cooling load that a setpoint change
    removes, so

        d(load)/d(T_setpoint)  =  -UA   [kW per degC]

    This is an *envelope-conduction-only* number. It deliberately excludes the
    latent (dehumidification) and fan work a warmer setpoint also saves, so it
    should be read as a lower bound on the true sensitivity -- and, unlike the
    matching estimate, it is immune to the occupancy confound because indoor
    temperature never enters as a regressor.

    `use_solar` matters: solar radiation and the driving force correlate at
    ~0.62 at midday, so dropping the solar term pulls UA/C down by about a
    third. Both are reported so the sensitivity is visible rather than hidden.
    """
    L = lunch_frame(h, zones)
    out = {"n": len(L), "frame": L, "per_zone": {}, "pooled": None, "model": None,
           "use_solar": use_solar}
    if len(L) < 50:
        return out

    zone_term = " + C(zone)" if L.zone.nunique() > 1 else ""
    solar_term = " + out_solar" if use_solar else ""
    m = smf.ols(f"drift ~ int_kwh + drive{solar_term}{zone_term}", data=L).fit(
        cov_type="cluster", cov_kwds={"groups": L.date})
    out["model"] = m

    load = occupied(h, zones, hours, require_sensor=False, ac_on=True).ac_kwh.mean()
    ci = m.conf_int()

    def pack(inv_c, ua_c, inv_c_ci, ua_c_ci, mean_load):
        if inv_c <= 0 or mean_load <= 0:
            return None
        C = 1.0 / inv_c
        UA = C * ua_c
        # Interval from the corner combinations of the two coefficient intervals.
        lo_C, hi_C = 1.0 / max(inv_c_ci[1], 1e-9), 1.0 / max(inv_c_ci[0], 1e-9)
        cands = [c * u for c in (lo_C, hi_C) for u in ua_c_ci if c > 0 and u > 0]
        ua_lo, ua_hi = (min(cands), max(cands)) if cands else (np.nan, np.nan)
        return {
            "C_kwh_per_c": C, "UA_kw_per_c": UA, "mean_load_kwh": mean_load,
            "pct_per_c": -UA / mean_load * 100,
            "pct_per_c_lo": -ua_hi / mean_load * 100,
            "pct_per_c_hi": -ua_lo / mean_load * 100,
            "kwh_per_c": -UA,
        }

    out["pooled"] = pack(m.params["int_kwh"], m.params["drive"],
                         tuple(ci.loc["int_kwh"]), tuple(ci.loc["drive"]), load)

    for z, g in L.groupby("zone"):
        if len(g) < 40:
            continue
        mz = smf.ols(f"drift ~ int_kwh + drive{solar_term}", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g.date})
        ciz = mz.conf_int()
        lz = occupied(h, [z], hours, require_sensor=False, ac_on=True).ac_kwh.mean()
        p = pack(mz.params["int_kwh"], mz.params["drive"],
                 tuple(ciz.loc["int_kwh"]), tuple(ciz.loc["drive"]), lz)
        if p:
            p["n"] = len(g)
            p["r2"] = mz.rsquared
            out["per_zone"][z] = p
    return out


def load_decomposition(h, phys, zones=CORE_ZONES, hours=AC_ON_HOURS):
    """Split the measured cooling load into envelope / internal / residual.

    This is the answer to why the physics and regression estimates disagree:

      envelope  = UA * (T_out - T_in)      -- what a setpoint change acts on
      internal  = metered lighting + plug  -- a setpoint change cannot touch it
      residual  = measured - the other two -- latent (dehumidification),
                  occupant heat, fan work and plant inefficiency

    A setpoint change removes the envelope term and some unknown share of the
    latent part of the residual. It removes none of the internal gains. So the
    envelope share is a floor on the sensitivity, and any estimate far above it
    is implicitly claiming large latent savings -- or is confounded.
    """
    d = occupied(h, zones, hours, require_sensor=True, ac_on=True)
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
# lever 2 -- causal estimates of setpoint sensitivity
# --------------------------------------------------------------------------- #

def coarsen(d):
    d = d.copy()
    d["tbin"] = d.out_temp.round().astype(int)
    d["t24bin"] = d.out_temp_24h.round().astype(int)
    d["sbin"] = pd.cut(d.out_solar, [-1, 1, 150, 400, 700, 2000], labels=False)
    d["rhbin"] = pd.cut(d.out_rh, [0, 60, 70, 80, 90, 101], labels=False)
    return d


def cem_match(d, keys):
    g = d.groupby(keys, observed=True).agg(nt=("treat", "sum"), n=("treat", "size")).reset_index()
    g["nc"] = g.n - g.nt
    return d.merge(g[(g.nt > 0) & (g.nc > 0)][keys], on=keys, how="inner")


def smd(df, col):
    t, c = df[df.treat == 1][col], df[df.treat == 0][col]
    s = np.sqrt((t.var() + c.var()) / 2)
    return (t.mean() - c.mean()) / s if s > 0 else 0.0


def att(df, keys):
    out = []
    for _, s in df.groupby(keys, observed=True):
        t, c = s[s.treat == 1], s[s.treat == 0]
        if len(t) == 0 or len(c) == 0:
            continue
        out.append({"w": len(t),
                    "dE": t.ac_kwh.mean() - c.ac_kwh.mean(),
                    "dT": t.temp_c.mean() - c.temp_c.mean(),
                    "base": c.ac_kwh.mean()})
    o = pd.DataFrame(out)
    if not len(o):
        return None
    W = o.w.sum()
    return {"strata": len(o), "n_treated": int(W),
            "dE": (o.dE * o.w).sum() / W,
            "dT": (o.dT * o.w).sum() / W,
            "base": (o.base * o.w).sum() / W}


def population(h, zones=CORE_ZONES, hours=AC_ON_HOURS, t_hi=24.0, t_lo=22.0,
               ac_on_kwh=AC_ON_KWH):
    d = occupied(h, zones, hours, require_sensor=True)
    d = d[d.ac_kwh > ac_on_kwh].copy()
    d["treat"] = (d.temp_c >= t_hi).astype(int)
    return d[(d.treat == 1) | (d.temp_c <= t_lo)].copy()


def run_cem(h, zones=CORE_ZONES, hours=AC_ON_HOURS, t_hi=24.0, t_lo=22.0,
            ac_on_kwh=AC_ON_KWH, spec="B. A + 24h mean outdoor T", n_boot=300, seed=0):
    """Coarsened exact matching + day-clustered bootstrap, re-runnable on any cut."""
    keys = [k for k in SPECS[spec] if not (k == "zone" and len(zones) < 2)]
    pop = population(h, zones, hours, t_hi, t_lo, ac_on_kwh)
    res = {"pop_n": len(pop), "pop_days": pop.date.nunique() if len(pop) else 0,
           "keys": keys, "counts": None, "att": None, "balance": None,
           "boot": None, "gate": None}
    if not len(pop):
        return res
    res["counts"] = pop.groupby(["zone", "treat"]).size().unstack(fill_value=0)

    d = cem_match(coarsen(pop), keys)
    if not len(d):
        return res
    a = att(d, keys)
    if a is None or a["dT"] == 0:
        return res

    bal = pd.DataFrame({
        "smd_before": [smd(pop, c) for c in BALANCE_COVS],
        "smd_after":  [smd(d, c) for c in BALANCE_COVS],
    }, index=BALANCE_COVS)
    bal["ok"] = bal.smd_after.abs() < 0.1

    rng = np.random.default_rng(seed)
    by_day = {k: v for k, v in d.groupby("date")}
    days = np.array(list(by_day))
    boot = []
    for _ in range(int(n_boot)):
        bs = pd.concat([by_day[x] for x in rng.choice(days, len(days), replace=True)],
                       ignore_index=True)
        r = att(bs, keys)
        if r and r["dT"] != 0 and r["base"] > 0:
            boot.append(r["dE"] / r["dT"] / r["base"] * 100)
    boot = np.array(boot)

    pct_per_c = a["dE"] / a["dT"] / a["base"] * 100
    worst = bal.smd_after.abs().max()
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if len(boot) else (np.nan, np.nan)

    g1 = a["n_treated"] >= 100
    g2 = worst < 0.1
    g3 = bool(len(boot)) and ci[1] < 0
    res.update({
        "matched": d, "att": a, "balance": bal, "boot": boot,
        "pct_per_c": pct_per_c, "kwh_per_c": a["dE"] / a["dT"],
        "worst_smd": worst, "ci": ci,
        "gate": {"pairs>=100": (g1, a["n_treated"]),
                 "all |SMD|<0.1": (g2, round(worst, 3)),
                 "95% CI excludes 0": (g3, f"[{ci[0]:.1f}%, {ci[1]:.1f}%]")},
        "verdict": "CONCLUSIVE" if (g1 and g2 and g3) else "INCONCLUSIVE",
    })
    return res


def run_regression(h, zones=CORE_ZONES, hours=AC_ON_HOURS, ac_on_kwh=AC_ON_KWH):
    """Regression-adjusted elasticity, pooled and per zone.

    Indoor temperature is the regressor here, so this is the *upper* end of the
    plausible range: on a busy day the zone is simultaneously warmer and drawing
    more cooling, and with no occupancy channel that confound stays in.
    """
    full = occupied(h, zones, hours, require_sensor=True, ac_on=True)
    full = full[full.ac_kwh > ac_on_kwh]
    out = {"pooled": None, "per_zone": []}
    if len(full) < 200:
        return out

    zone_term = " + C(zone)" if full.zone.nunique() > 1 else ""
    m = smf.ols(f"ac_kwh ~ temp_c + {CONTROLS}{zone_term}", data=full).fit(
        cov_type="cluster", cov_kwds={"groups": full.date})
    ml = full.ac_kwh.mean()
    ci = m.conf_int().loc["temp_c"]
    out["pooled"] = {"n": int(m.nobs), "r2": m.rsquared,
                     "kwh_per_c": m.params["temp_c"],
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
            "zone": z, "n": len(g),
            "pct_per_c": mm.params["temp_c"] / mlz * 100,
            "lo": cc[0] / mlz * 100, "hi": cc[1] / mlz * 100,
            "kwh_per_c": mm.params["temp_c"],
            "sign": "as expected" if mm.params["temp_c"] < 0 else "WRONG SIGN",
        })
    return out


# --------------------------------------------------------------------------- #
# lever 3 -- exogenous baseline (M&V engine)
# --------------------------------------------------------------------------- #

def ashrae_g14(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    cv = np.sqrt(((y - pred) ** 2).mean()) / y.mean() * 100
    nmbe = (y - pred).sum() / y.sum() * 100
    return cv, nmbe


def fit_baselines(h, zones=CORE_ZONES, hours=AC_ON_HOURS):
    """Per-zone GBM + Ridge on exogenous features only. Time-based splits."""
    assert not set(FEATURES) & set(FORBIDDEN), "LEAKAGE: indoor channel in features"
    data = occupied(h, zones, hours, require_sensor=False)
    fitted, rows = {}, []
    for z, g in data.groupby("zone"):
        tr = g[(g.ts >= "2019-03-06") & (g.ts < "2019-10-01")]
        te = g[g.ts >= "2019-10-01"]
        ho = g[g.ts < "2018-09-15"]
        if len(tr) < 200 or len(te) < 50:
            continue
        ridge = make_pipeline(
            ColumnTransformer([("spline", SplineTransformer(n_knots=6, degree=3),
                                ["out_temp", "out_temp_24h", "out_solar"])],
                              remainder="passthrough"),
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ).fit(tr[FEATURES], tr.ac_kwh)
        gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                            max_depth=6, random_state=0
                                            ).fit(tr[FEATURES], tr.ac_kwh)
        fitted[z] = {"ridge": ridge, "gbm": gbm, "train": tr, "test": te, "holdout": ho}
        for name, mdl in [("Ridge", ridge), ("GBM", gbm)]:
            for split, dd in [("test Oct-Dec 2019", te), ("holdout Jul-Sep 2018", ho)]:
                if not len(dd):
                    continue
                pred = np.clip(mdl.predict(dd[FEATURES]), 0, None)
                cv, nmbe = ashrae_g14(dd.ac_kwh, pred)
                rows.append({"zone": z, "model": name, "split": split, "n": len(dd),
                             "cv_rmse": cv, "nmbe": nmbe,
                             "meets_g14": "yes" if (cv <= 30 and abs(nmbe) <= 10) else "no"})
    return fitted, pd.DataFrame(rows)


def weather_scenario(h, fitted, zones=CORE_ZONES, hours=AC_ON_HOURS,
                     d_temp=0.0, solar_scale=1.0, d_rh=0.0, model="gbm"):
    """Re-predict the whole occupied record under a shifted climate."""
    data = occupied(h, zones, hours, require_sensor=False)
    base = data.copy()
    scen = data.copy()
    for c in ["out_temp", "out_temp_3h", "out_temp_24h"]:
        scen[c] = scen[c] + d_temp
    scen["out_solar"] = scen.out_solar * solar_scale
    scen["out_rh"] = (scen.out_rh + d_rh).clip(0, 100)

    rows = []
    for z in sorted(set(zones) & set(fitted)):
        mdl = fitted[z][model]
        b = base[base.zone == z]
        s = scen[scen.zone == z]
        pb = np.clip(mdl.predict(b[FEATURES]), 0, None)
        ps = np.clip(mdl.predict(s[FEATURES]), 0, None)
        rows.append({"zone": z, "actual_kwh": b.ac_kwh.sum(),
                     "pred_base_kwh": pb.sum(), "pred_scen_kwh": ps.sum(),
                     "n": len(b)})
    r = pd.DataFrame(rows)
    if len(r):
        r["delta_kwh"] = r.pred_scen_kwh - r.pred_base_kwh
        r["delta_pct"] = r.delta_kwh / r.pred_base_kwh * 100
    return r


def monthly_scenario(h, fitted, zones=CORE_ZONES, hours=AC_ON_HOURS,
                     d_temp=0.0, solar_scale=1.0, d_rh=0.0, model="gbm"):
    """Same scenario, broken out by month, to show where savings actually land."""
    data = occupied(h, zones, hours, require_sensor=False)
    frames = []
    for z in sorted(set(zones) & set(fitted)):
        mdl = fitted[z][model]
        b = data[data.zone == z].copy()
        s = b.copy()
        for c in ["out_temp", "out_temp_3h", "out_temp_24h"]:
            s[c] = s[c] + d_temp
        s["out_solar"] = s.out_solar * solar_scale
        s["out_rh"] = (s.out_rh + d_rh).clip(0, 100)
        b = b.assign(pred_base=np.clip(mdl.predict(b[FEATURES]), 0, None),
                     pred_scen=np.clip(mdl.predict(s[FEATURES]), 0, None))
        frames.append(b)
    if not frames:
        return pd.DataFrame()
    allz = pd.concat(frames)
    g = allz.groupby("month").agg(actual=("ac_kwh", "sum"),
                                  pred_base=("pred_base", "sum"),
                                  pred_scen=("pred_scen", "sum")).reset_index()
    g["delta_pct"] = (g.pred_scen - g.pred_base) / g.pred_base * 100
    return g


# --------------------------------------------------------------------------- #
# lever 4 -- comfort distribution under a setpoint shift and a tighter deadband
# --------------------------------------------------------------------------- #

def shift_distribution(temps, d_mean=0.0, sd_scale=1.0):
    """Shift the mean and optionally compress the spread of an observed sample.

    Two physically distinct levers: `d_mean` is a setpoint change (a work
    order), `sd_scale` is control-loop tuning (a commissioning job). The
    notebook showed that the first alone leaves 39-48% of hours below 23 degC,
    because the distributions are wide rather than merely mis-centred -- this
    makes the second lever explicit.
    """
    t = np.asarray(temps, dtype=float)
    mu = t.mean()
    return mu + d_mean + (t - mu) * sd_scale


def comfort_frontier(h, zones=CORE_ZONES, hours=AC_ON_HOURS, cold=COLD_THRESHOLD,
                     warm=WARM_THRESHOLD, sd_scale=1.0, shifts=np.arange(0, 3.51, 0.25)):
    d = occupied(h, zones, hours, require_sensor=True, ac_on=True)
    rows = []
    for s in shifts:
        for z, g in d.groupby("zone"):
            n = shift_distribution(g.temp_c, s, sd_scale)
            rows.append({"zone": z, "shift": s,
                         "pct_cold": (n < cold).mean() * 100,
                         "pct_warm": (n > warm).mean() * 100,
                         "mean_c": n.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# lever 5 -- schedule trim, priced with the physics model
# --------------------------------------------------------------------------- #

def hour_profile(h, zones=CORE_ZONES):
    """Measured mean AC kWh and indoor temperature by hour of the weekday."""
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(list(zones))]
    e = d.groupby("hour").ac_kwh.agg(mean_kwh="mean", total_kwh="sum", n="size")
    e["frac_off"] = d.assign(off=d.ac_kwh <= AC_ON_KWH).groupby("hour").off.mean()
    t = d[d.sensor_alive].groupby("hour").temp_c.mean().rename("mean_c")
    return e.join(t)


def schedule_trim(h, phys, zones=CORE_ZONES, drop_hours=(), recover=True):
    """Energy released by switching cooling off in `drop_hours`, net of recovery.

    The gross number is just metered energy in those hours. The net number
    charges back the heat that has to be pulled out again when cooling resumes,
    C * dT, using the calibrated capacitance and drift rate -- which is why
    trimming the *last* hour of the day is nearly free while extending the
    lunch break is not.
    """
    f, _, _ = annual_factor(h)
    d = h[(~h.is_weekend) & (~h.is_holiday) & h.zone.isin(list(zones))]
    drop = list(drop_hours)
    rows = []
    for z, g in d.groupby("zone"):
        gross = g[g.hour.isin(drop)].ac_kwh.sum()
        p = phys["per_zone"].get(z) or phys["pooled"] or {}
        C = p.get("C_kwh_per_c", np.nan)
        UA = p.get("UA_kw_per_c", np.nan)

        # Drift while off: UA*(Tout-Tin) + internal, expressed in degC/h.
        # Start from the temperature in the last cooled hour before the gap,
        # not the whole-window mean, so an end-of-day trim starts from the
        # right place.
        prior = [hh for hh in AC_ON_HOURS if drop and hh < min(drop)]
        ref = g[g.sensor_alive & g.hour.isin([max(prior)] if prior else AC_ON_HOURS)]
        occ_t = ref.temp_c.mean()
        sub = g[g.hour.isin(drop)]
        rate = np.nan
        if np.isfinite(C) and C > 0 and len(sub):
            q = UA * (sub.out_temp.mean() - occ_t) + sub.int_kwh.mean()
            rate = max(q, 0) / C

        # Recovery is only owed if cooling resumes the same day after the gap.
        resumes = any(hh in AC_ON_HOURS for hh in range(max(drop) + 1, 18)) if drop else False
        n_days = g[g.hour.isin(drop)].date.nunique() if drop else 0
        rec_kwh = 0.0
        if recover and resumes and np.isfinite(rate) and n_days:
            rec_kwh = C * rate * len(drop) * n_days       # C * degC recovered, per day

        rows.append({"zone": z, "gross_kwh": gross, "recovery_kwh": rec_kwh,
                     "net_kwh": gross - rec_kwh,
                     "gross_kwh_year": gross * f, "net_kwh_year": (gross - rec_kwh) * f,
                     "drift_c_per_h": rate, "resumes_same_day": resumes,
                     "end_temp_c": occ_t + rate * len(drop) if np.isfinite(rate) else np.nan})
    return pd.DataFrame(rows).set_index("zone")


# --------------------------------------------------------------------------- #
# lever 6 -- pilot design: is the effect even detectable?
# --------------------------------------------------------------------------- #

def detectable_saving(cv_rmse_pct, weeks, hours_per_day=9, days_per_week=5,
                      alpha=0.05, power=0.80, rho=0.0):
    """Minimum detectable saving for an ASHRAE-style baseline comparison.

    With n independent hourly comparisons and a baseline whose residual CV is
    cv_rmse, the standard error on a mean saving is cv_rmse/sqrt(n_eff). The
    usual two-sided test needs (z_a/2 + z_b) of those. `rho` inflates the
    variance for within-day correlation via a design effect.
    """
    from scipy.stats import norm
    n = weeks * days_per_week * hours_per_day
    if n <= 1:
        return np.nan
    deff = 1 + (hours_per_day - 1) * rho
    n_eff = max(n / deff, 1.0)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * cv_rmse_pct / np.sqrt(n_eff)


def weeks_required(cv_rmse_pct, target_pct, hours_per_day=9, days_per_week=5,
                   alpha=0.05, power=0.80, rho=0.0, max_weeks=260):
    for w in range(1, max_weeks + 1):
        if detectable_saving(cv_rmse_pct, w, hours_per_day, days_per_week,
                             alpha, power, rho) <= abs(target_pct):
            return w
    return np.nan
