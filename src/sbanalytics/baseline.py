"""
Floor-level baselines: "what should this floor have used, given the weather
and the calendar?"

A ladder from simplest to most flexible; the simplest model that meets ASHRAE
Guideline 14 is the one to report.

    daily   2P / 3P change-point regression on outdoor temperature (ASHRAE IMT)
    hourly  TOWT — time-of-week + piecewise temperature (LBNL / CalTRACK)
    hourly  gradient boosting on weather + calendar

The dashboard's spline-Ridge (analysis.fit_baselines) is kept for occupied-hour
zone models only: fitted on all hours, an additive model cannot represent the
AC switching off, predicts negative overnight load and, once clipped, carries
a -13 to -19% NMBE even in training.

Features are exogenous only (weather, calendar). Indoor temperature is an
outcome of cooling, not a cause, so it never enters — asserted below.

Guideline 14-2014 calibration limits: hourly CV(RMSE) <= 30%, |NMBE| <= 10%;
monthly <= 15% / 5%. G14 gives no daily limit; daily models are held to the
hourly limits here, which is lenient — read daily results as indicative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import analysis as A
from . import features as FT

G14 = {"hourly": (30.0, 10.0), "daily": (30.0, 10.0), "monthly": (15.0, 5.0)}

#: Floor-level baseline windows: train on a full year, test on the next half.
#: The dashboard's zone models train Mar-Sep 2019 only (analysis.TRAIN), to stay
#: clear of the F5 z2<->z4 meter transfer on 2019-09-27 — a zone-level change
#: that leaves the floor total almost flat (-5%). A floor model trained on those
#: seven warm months never sees Oct-Dec and over-predicts it by 10-35% on every
#: floor; a 12-month window covers every season once.
FLOOR_TRAIN = ("2018-07-01", "2019-06-30")
FLOOR_SPLITS = {"test (Jul-Dec 2019)": ("2019-07-01", "2019-12-31")}

ML_FEATURES = ["out_temp", "out_rh", "out_solar", "out_temp_3h", "out_temp_24h",
               "hour_sin", "hour_cos", "dow", "month", "is_workday"]
assert not set(ML_FEATURES) & set(A.FORBIDDEN), "LEAKAGE: indoor channel in features"

WEATHER_COLS = ["out_temp", "out_rh", "out_dew", "out_solar", "out_cloud", "out_temp_3h", "out_temp_24h"]


def g14(y, pred, resolution: str = "hourly") -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    cv, nmbe = A.ashrae_g14(y[ok], pred[ok])
    cv_max, nmbe_max = G14[resolution]
    return {"n": int(ok.sum()), "cv_rmse": cv, "nmbe": nmbe,
            "meets_g14": bool(cv <= cv_max and abs(nmbe) <= nmbe_max)}


# --------------------------------------------------------------------------- #
# floor-level frames
# --------------------------------------------------------------------------- #

def floor_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """Zone-hour -> floor-hour. An hour is `ok` only if every metered zone passed its gate."""
    g = h.groupby(["floor", "ts"])
    f = g[["ac_kwh", "light_kwh", "plug_kwh"]].sum(min_count=1)
    has_meter = h.ac_kwh.notna() | h.light_kwh.notna()
    f["ok"] = h.assign(bad=has_meter & ~h.kwh_ok).groupby(["floor", "ts"]).bad.sum() == 0
    f["ok"] &= f.ac_kwh.notna()
    first = g[["hour", "dow", "month", "date", "is_holiday", "day_type", "is_workday", *WEATHER_COLS]].first()
    f = f.join(first).reset_index()
    f["total_kwh"] = f[["ac_kwh", "light_kwh", "plug_kwh"]].sum(axis=1, min_count=3)
    f["occupied"] = FT.occupied_mask(f)
    return f


def floor_daily(fh: pd.DataFrame, cdd_base: float = 24.0) -> pd.DataFrame:
    g = fh.assign(day=fh.ts.dt.normalize()).groupby(["floor", "day"])
    d = g[["ac_kwh", "light_kwh", "plug_kwh", "total_kwh"]].sum(min_count=20)
    d["ok_hours"] = g.ok.sum()
    d["out_temp"] = g.out_temp.mean()
    d["out_solar"] = g.out_solar.mean()
    d["cdd"] = g.out_temp.apply(lambda s: ((s - cdd_base).clip(lower=0) / 24).sum())
    d["day_type"] = g.day_type.first()
    d["is_workday"] = g.is_workday.first()
    d["dow"] = g.dow.first()
    d = d.reset_index()
    d["ok"] = d.ok_hours >= 20
    return d


def split(df: pd.DataFrame, period, col: str = "ts") -> pd.DataFrame:
    lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    return df[(df[col] >= lo) & (df[col] < hi)]


# --------------------------------------------------------------------------- #
# daily change-point models (ASHRAE Inverse Modeling Toolkit)
# --------------------------------------------------------------------------- #

def fit_changepoint(x, y, kind: str = "3pc", grid: int = 41) -> dict:
    """2P: y = b0 + b1 x. 3PC: y = b0 + b1 * max(x - cp, 0), cp grid-searched.

    Fit separately per day type — a weekend is a different building.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return {"kind": kind, "n": len(x)}

    def ols(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, ((y - X @ beta) ** 2).sum()

    if kind == "2p":
        beta, sse = ols(np.c_[np.ones_like(x), x])
        cp = np.nan
    else:
        best = None
        for c in np.quantile(x, np.linspace(0.05, 0.85, grid)):
            beta, sse = ols(np.c_[np.ones_like(x), np.maximum(x - c, 0)])
            if beta[1] >= 0 and (best is None or sse < best[1]):
                best = (beta, sse, c)
        if best is None:
            return fit_changepoint(x, y, "2p")
        beta, sse, cp = best
    r2 = 1 - sse / ((y - y.mean()) ** 2).sum()
    return {"kind": kind, "n": len(x), "b0": beta[0], "b1": beta[1], "cp": cp, "r2": r2}


def predict_changepoint(m: dict, x) -> np.ndarray:
    x = np.asarray(x, float)
    if "b0" not in m:
        return np.full_like(x, np.nan)
    if m["kind"] == "2p" or not np.isfinite(m["cp"]):
        return m["b0"] + m["b1"] * x
    return m["b0"] + m["b1"] * np.maximum(x - m["cp"], 0)


# --------------------------------------------------------------------------- #
# TOWT — time-of-week and temperature
# --------------------------------------------------------------------------- #

def _temp_segments(t: np.ndarray, knots) -> np.ndarray:
    """Piecewise-linear temperature basis: one slope per segment between knots."""
    cols = [np.minimum(t, knots[0])]
    cols += [np.clip(t, lo, hi) - lo for lo, hi in zip(knots[:-1], knots[1:])]
    cols.append(np.maximum(t - knots[-1], 0))
    return np.column_stack(cols)


def _towt_design(d: pd.DataFrame, knots) -> np.ndarray:
    # Holidays and closures run like a Sunday; worked holidays like the weekday they are.
    dow = np.where(d.is_workday, d.dow, np.where(d.dow >= 5, d.dow, 6))
    how = (dow * 24 + d.hour).to_numpy()
    X_how = np.zeros((len(d), 168))
    X_how[np.arange(len(d)), how] = 1
    T = _temp_segments(d.out_temp.to_numpy(), knots)
    occ = d.occupied.to_numpy()[:, None].astype(float)
    return np.hstack([X_how, T * occ, T * (1 - occ)])


class TOWT:
    """Hour-of-week indicators + piecewise-linear temperature, split by occupancy."""

    def __init__(self, n_knots: int = 5, alpha: float = 1.0):
        self.n_knots, self.alpha = n_knots, alpha

    def fit(self, d: pd.DataFrame, y: pd.Series):
        self.knots = np.quantile(d.out_temp, np.linspace(0, 1, self.n_knots + 2)[1:-1])
        X = _towt_design(d, self.knots)
        A_ = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.beta = np.linalg.solve(A_, X.T @ y.to_numpy())
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return _towt_design(d, self.knots) @ self.beta


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #

def _gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0)


def fit_hourly_ladder(fh: pd.DataFrame, target: str = "ac_kwh",
                      train=FLOOR_TRAIN, splits=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit TOWT and GBM per floor on `train`, score on each split.

    Returns (metrics table, predictions frame with one column per model).
    """
    splits = splits or FLOOR_SPLITS
    data = FT.add_cyclic(fh[fh.ok].dropna(subset=[target, "out_temp", "out_solar"]))
    data["is_workday"] = data.is_workday.astype(int)
    rows, preds = [], []
    for f, g in data.groupby("floor"):
        tr = split(g, train)
        if len(tr) < 500:
            continue
        models = {"TOWT": TOWT().fit(tr, tr[target]),
                  "GBM": _gbm().fit(tr[ML_FEATURES], tr[target])}
        out = g[["floor", "ts", target]].copy()
        for name, m in models.items():
            p = m.predict(g) if name == "TOWT" else m.predict(g[ML_FEATURES])
            out[name] = np.clip(p, 0, None)
        preds.append(out)
        for split_name, period in {"train": train, **splits}.items():
            s = split(out, period)
            if len(s) < 50:
                continue
            for name in models:
                rows.append({"floor": f, "model": name, "split": split_name,
                             **g14(s[target], s[name], "hourly")})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def fit_daily_changepoint(fd: pd.DataFrame, target: str = "ac_kwh", x: str = "out_temp",
                          train=FLOOR_TRAIN, splits=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """3PC per floor and per workday/non-workday regime."""
    splits = splits or FLOOR_SPLITS
    data = fd[fd.ok].dropna(subset=[target, x]).copy()
    rows, preds, params = [], [], {}
    for f, g in data.groupby("floor"):
        tr = split(g, train, "day")
        g = g.copy()
        g["pred"] = np.nan
        for regime, mask_fn in (("workday", lambda d: d.is_workday), ("non-workday", lambda d: ~d.is_workday)):
            t = tr[mask_fn(tr)]
            m = fit_changepoint(t[x], t[target])
            params[(f, regime)] = m
            sel = mask_fn(g)
            g.loc[sel, "pred"] = predict_changepoint(m, g.loc[sel, x])
        preds.append(g)
        for split_name, period in {"train": train, **splits}.items():
            s = split(g, period, "day")
            if len(s) < 10:
                continue
            rows.append({"floor": f, "model": "3PC daily", "split": split_name,
                         **g14(s[target], s.pred, "daily")})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), params
