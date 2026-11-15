"""
Day-ahead forecasting of hourly floor load, and an honest test of whether it
beats the obvious answer.

Use case: at midnight, forecast each hour of the coming day, so a facilities
manager can see a peak coming (or pre-cool). Every model is scored in a
rolling-origin backtest: refit weekly on everything before the origin, forecast
the next seven days one day at a time, never peeking past the origin.

Benchmark: seasonal-naive — "same hour, same weekday, last week". A model
that cannot beat it has no claim to value.

Caveat: weather inputs are ERA5 *actuals*, i.e. a perfect weather forecast.
Real day-ahead skill will be somewhat lower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import baseline as B
from . import features as FT

LAG_FEATURES = ["lag_24", "lag_168", "lag_24_mean_day"]
GBM_FEATURES = B.ML_FEATURES + LAG_FEATURES


def add_lags(fh: pd.DataFrame, target: str) -> pd.DataFrame:
    """Lags available at a midnight forecast origin: yesterday's same hour, last
    week's same hour, and yesterday's mean."""
    out = []
    for f, g in fh.sort_values("ts").groupby("floor"):
        s = g.set_index("ts")[target].asfreq("h")
        lag = pd.DataFrame({
            "lag_24": s.shift(24), "lag_168": s.shift(168),
            "lag_24_mean_day": s.shift(24).groupby(s.index.normalize()).transform("mean"),
        })
        out.append(g.merge(lag, left_on="ts", right_index=True, how="left"))
    d = pd.concat(out, ignore_index=True)
    d = FT.add_cyclic(d)
    d["is_workday"] = d.is_workday.astype(int)
    return d


def backtest(fh: pd.DataFrame, target: str = "total_kwh", start: str = "2019-07-01",
             end: str = "2019-12-31", refit_days: int = 7) -> pd.DataFrame:
    """Rolling-origin day-ahead backtest for every floor. Returns hourly predictions."""
    d = add_lags(fh, target)
    origins = pd.date_range(start, end, freq=f"{refit_days}D")
    rows = []
    for f, g in d.groupby("floor"):
        g = g.sort_values("ts")
        for o in origins:
            tr = g[(g.ts < o) & g.ok].dropna(subset=[target, "out_temp"] + LAG_FEATURES)
            te = g[(g.ts >= o) & (g.ts < o + pd.Timedelta(days=refit_days))]
            te = te[te.ok].dropna(subset=[target])
            if len(tr) < 2000 or te.empty:
                continue
            towt = B.TOWT().fit(tr, tr[target])
            gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                                random_state=0).fit(tr[GBM_FEATURES], tr[target])
            rows.append(pd.DataFrame({
                "floor": f, "ts": te.ts.to_numpy(), "actual": te[target].to_numpy(),
                "seasonal_naive": te.lag_168.to_numpy(),
                "persistence_24h": te.lag_24.to_numpy(),
                "TOWT (weather + calendar)": np.clip(towt.predict(te), 0, None),
                "GBM + lags": np.clip(gbm.predict(te[GBM_FEATURES]), 0, None),
                "day_type": te.day_type.to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


MODELS = ["seasonal_naive", "persistence_24h", "TOWT (weather + calendar)", "GBM + lags"]


def score(bt: pd.DataFrame, by: str = "floor") -> pd.DataFrame:
    """MAE, CV(RMSE) and skill vs seasonal-naive (1 - MAE_model / MAE_naive)."""
    rows = []
    for key, g in bt.groupby(by):
        g = g.dropna(subset=["actual", *MODELS])
        mae_naive = (g.actual - g.seasonal_naive).abs().mean()
        for m in MODELS:
            err = g.actual - g[m]
            rows.append({by: key, "model": m, "n": len(g), "mae_kwh": err.abs().mean(),
                         "cv_rmse": np.sqrt((err ** 2).mean()) / g.actual.mean() * 100,
                         "skill_vs_naive": 1 - err.abs().mean() / mae_naive})
    return pd.DataFrame(rows)


def peak_hit_rate(bt: pd.DataFrame, model: str) -> pd.Series:
    """Share of days on which the model puts the daily peak in the right hour (±1 h)."""
    g = bt.dropna(subset=["actual", model]).assign(day=lambda x: x.ts.dt.normalize())
    def hit(d):
        return abs(d.loc[d.actual.idxmax(), "ts"].hour - d.loc[d[model].idxmax(), "ts"].hour) <= 1
    return g.groupby(["floor", "day"]).apply(hit, include_groups=False).groupby("floor").mean()
