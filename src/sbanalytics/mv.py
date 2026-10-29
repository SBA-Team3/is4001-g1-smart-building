"""
Energy performance against an expected baseline — ISO 50001/50006 EnPI style,
with IPMVP / ASHRAE Guideline 14 statistics.

"Expected" is the daily 3-parameter cooling change-point model (baseline.py),
fitted separately for workdays and non-workdays, so a cold month or a holiday
week does not read as a saving.

The training window is per floor. Floor 5 trains on analysis.TRAIN
(2019-03-07 .. 2019-09-26), which sits *between* its two blocking
change-points: after the 2019-03-06 sensor restoration (the floor ran 2-3 degC
colder from then on) and before the 2019-09-27 z2<->z4 meter transfer. Other
floors use the notebooks' 12-month window. Anything on the far side of a
blocking change-point is still drawn, but the CUSUM restarts there, because a
cumulative sum across a configuration change measures the change, not the
performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis as A
from . import baseline as B
from . import energy as E
from . import registry as R

#: Training and test windows per floor. See module docstring for why F5 differs.
MV_TRAIN: dict[int, tuple[str, str]] = {5: A.TRAIN}
MV_TEST: dict[int, tuple[str, str]] = {5: A.TEST}


def train_window(floor: int) -> tuple[str, str]:
    return MV_TRAIN.get(floor, B.FLOOR_TRAIN)


def test_window(floor: int) -> tuple[str, str]:
    return MV_TEST.get(floor, next(iter(B.FLOOR_SPLITS.values())))


def blocking_change_points(floor: int) -> list:
    return [c for c in R.CHANGE_POINTS if c.blocks_comparison and floor in c.floors]


# --------------------------------------------------------------------------- #
# the daily model
# --------------------------------------------------------------------------- #

def daily_model(h: pd.DataFrame, floor: int, target: str = "ac_kwh") -> dict:
    """Fit the daily change-point baseline and return everything the page needs.

    Returns {"daily": frame, "metrics": G14 table, "params": {regime: fit},
    "rho": lag-1 autocorrelation of training residuals, "train", "test"}.
    """
    fd = B.floor_daily(B.floor_hourly(h))
    train, test = train_window(floor), test_window(floor)
    metrics, pred, params = B.fit_daily_changepoint(fd, target, train=train,
                                                    splits={"test": test})
    params = {regime: m for (f, regime), m in params.items() if f == floor}
    d = pred[pred.floor == floor].rename(columns={"day": "date", target: "actual",
                                                  "pred": "expected"})
    d["regime"] = np.where(d.is_workday, "workday", "non-workday")
    d["resid"] = d.actual - d.expected
    d["in_train"] = d.date.between(pd.Timestamp(train[0]), pd.Timestamp(train[1]))
    # Per-regime residual spread from training only: a workday is far noisier
    # than a weekend, and one pooled SD would mis-state both.
    sd = d[d.in_train].groupby("regime").resid.std()
    d["sd"] = d.regime.map(sd)
    tr = d[d.in_train].sort_values("date").resid
    rho = float(tr.autocorr(1)) if len(tr) > 2 else 0.0
    d["segment"] = segment_labels(d.date, floor)
    d["baseline_segment"] = d.segment == segment_labels(pd.Series([pd.Timestamp(train[0])]), floor).iloc[0]
    return {"daily": d.sort_values("date").reset_index(drop=True),
            "metrics": metrics[metrics.floor == floor].reset_index(drop=True),
            "params": params, "rho": max(min(rho, 0.95), 0.0),
            "train": train, "test": test, "target": target}


def segment_labels(dates: pd.Series, floor: int) -> pd.Series:
    """Label each date with the stretch between blocking change-points it falls in."""
    cps = sorted(pd.Timestamp(c.date) for c in blocking_change_points(floor))
    edges = [pd.Timestamp.min] + cps + [pd.Timestamp.max]
    names = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = "start" if lo == pd.Timestamp.min else f"{lo.date()}"
        b = "end" if hi == pd.Timestamp.max else f"{hi.date()}"
        names.append(f"{a} → {b}")
    idx = np.searchsorted(np.array(cps, dtype="datetime64[ns]"),
                          pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]"), side="right")
    return pd.Series([names[i] for i in idx], index=dates.index)


def annual_kwh(model: dict) -> float:
    """Annualised energy of the modelled channel, from usable days."""
    return float(model["daily"].actual.mean() * 365)


def g14_summary(model: dict) -> pd.DataFrame:
    """Train/test CV(RMSE) and NMBE against G14 limits, with pass/fail."""
    m = model["metrics"].copy()
    cv_max, nmbe_max = B.G14["daily"]
    m["cv_limit"], m["nmbe_limit"] = cv_max, nmbe_max
    return m[["split", "n", "cv_rmse", "nmbe", "cv_limit", "nmbe_limit", "meets_g14"]]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def rollup(model: dict, start=None, end=None, freq: str = "MS",
           z: float | None = None) -> pd.DataFrame:
    """Actual vs expected per period, with the uncertainty band on expected.

    Band = z * sqrt(sum of daily residual variances * (1+rho)/(1-rho)) — the G14
    autocorrelation correction, so a month of correlated days is not treated as
    thirty independent ones.
    """
    z = R.conv("mv_confidence_z") if z is None else z
    d = _window(model["daily"], start, end)
    if not len(d):
        return pd.DataFrame()
    infl = (1 + model["rho"]) / (1 - model["rho"])
    g = d.assign(var=d.sd ** 2).groupby(pd.Grouper(key="date", freq=freq))
    out = pd.DataFrame({"actual": g.actual.sum(min_count=1), "expected": g.expected.sum(min_count=1),
                        "days": g.size(), "var": g["var"].sum(),
                        "in_train_days": g.in_train.sum(),
                        "baseline_segment": g.baseline_segment.all()}).reset_index()
    out = out[out.days > 0]
    out["band"] = z * np.sqrt(out["var"] * infl)
    out["lo"], out["hi"] = out.expected - out.band, out.expected + out.band
    out["deviation"] = out.actual - out.expected
    out["deviation_pct"] = out.deviation / out.expected * 100
    out["significant"] = out.deviation.abs() > out.band
    return out.drop(columns="var")


def cusum(model: dict, start=None, end=None) -> pd.DataFrame:
    """Cumulative (actual - expected), restarted at every blocking change-point."""
    d = _window(model["daily"], start, end).sort_values("date")
    d = d.assign(cusum=d.groupby("segment").resid.cumsum())
    return d[["date", "actual", "expected", "resid", "cusum", "segment",
              "baseline_segment", "in_train"]]


def signature(model: dict, start=None, end=None, z: float | None = None) -> dict:
    """Daily energy vs mean outdoor temperature, the fitted lines, and the
    workdays sitting well above their line (the drill-down list)."""
    z = R.conv("signature_outlier_z") if z is None else z
    d = _window(model["daily"], start, end).copy()
    d["day_kind"] = np.select(
        [d.day_type.isin(["holiday", "closure"]), ~d.is_workday], ["holiday / closure", "weekend"],
        "workday")
    xs = np.linspace(d.out_temp.min(), d.out_temp.max(), 60) if len(d) else np.array([])
    lines = pd.concat([pd.DataFrame({"out_temp": xs, "expected": B.predict_changepoint(m, xs),
                                     "regime": regime})
                       for regime, m in model["params"].items()], ignore_index=True) \
        if len(xs) else pd.DataFrame()
    wd = d[(d.regime == "workday") & (d.resid > z * d.sd)].sort_values("resid", ascending=False)
    return {"points": d, "lines": lines, "outliers": wd}


def _window(d: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        d = d[d.date >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.date <= pd.Timestamp(end)]
    return d


def month_excess(model: dict, as_of, tariff: float | None = None) -> dict:
    """The proposal's second KPI: excess kWh and THB this month (to date) vs last
    month, against the weather/calendar-normalised expectation."""
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    d = pd.Timestamp(as_of).normalize()
    this0 = d.replace(day=1)
    last0 = (this0 - pd.Timedelta(days=1)).replace(day=1)
    out = {}
    for key, (s, e) in {"this": (this0, d), "last": (last0, this0 - pd.Timedelta(days=1))}.items():
        w = _window(model["daily"], s, e)
        out[key] = {"start": s, "end": e, "actual": w.actual.sum(), "expected": w.expected.sum(),
                    "excess": w.resid.sum(), "thb": w.resid.sum() * tariff, "days": len(w),
                    "in_train": bool(w.in_train.any())}
    return out


# --------------------------------------------------------------------------- #
# peak demand
# --------------------------------------------------------------------------- #

def daily_peaks(h: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Daily peak of hourly-mean floor demand (kW)."""
    s = E.floor_demand(h, start, end).dropna()
    if not len(s):
        return pd.DataFrame(columns=["date", "peak_kw", "peak_at"])
    g = s.groupby(s.index.normalize())
    return pd.DataFrame({"peak_kw": g.max(), "peak_at": g.idxmax()}).rename_axis("date").reset_index()


def top_peaks(h: pd.DataFrame, start=None, end=None, n: int = 10) -> pd.DataFrame:
    """The n highest demand hours, with outdoor temperature at the time."""
    s = E.floor_demand(h, start, end).dropna().nlargest(n)
    wx = h.groupby("ts")[["out_temp", "out_rh", "day_type"]].first()
    t = pd.DataFrame({"kW": s}).join(wx).rename_axis("hour").reset_index()
    return t


def demand_cost(h: pd.DataFrame, start=None, end=None, charge: float | None = None) -> pd.DataFrame:
    """Monthly maximum demand and the demand charge it would attract."""
    charge = R.conv("demand_charge_thb_per_kw") if charge is None else charge
    s = E.floor_demand(h, start, end).dropna()
    if not len(s):
        return pd.DataFrame(columns=["month", "max_kw", "charge_thb"])
    m = s.groupby(s.index.to_period("M")).max()
    return pd.DataFrame({"month": m.index.to_timestamp(), "max_kw": m.values,
                         "charge_thb": m.values * charge})
