"""Unit tests for the analytics layer on synthetic data with known answers."""

import numpy as np
import pandas as pd

from sbanalytics import anomaly as AN
from sbanalytics import baseline as B


def test_changepoint_recovers_balance_point():
    rng = np.random.default_rng(1)
    t = rng.uniform(22, 34, 400)
    y = 500 + 40 * np.maximum(t - 27, 0) + rng.normal(0, 5, 400)
    m = B.fit_changepoint(t, y)
    assert abs(m["cp"] - 27) < 0.6 and abs(m["b1"] - 40) < 3


def _synthetic_hours(weeks=20, seed=2):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-01-07", periods=weeks * 168, freq="h")
    d = pd.DataFrame({"ts": ts, "hour": ts.hour, "dow": ts.dayofweek})
    d["out_temp"] = 28 + 4 * np.sin(2 * np.pi * (d.hour - 9) / 24) + rng.normal(0, 0.5, len(d))
    d["is_workday"] = d.dow < 5
    d["occupied"] = d.is_workday & d.hour.between(8, 16)
    d["y"] = np.where(d.occupied, 60 + 3 * (d.out_temp - 28), 2) + rng.normal(0, 1, len(d))
    return d


def test_towt_fits_schedule_and_temperature():
    d = _synthetic_hours()
    m = B.TOWT().fit(d, d.y)
    cv, nmbe = B.A.ashrae_g14(d.y, m.predict(d))
    assert cv < 10 and abs(nmbe) < 1


def test_g14_thresholds():
    y = np.full(100, 10.0)
    assert B.g14(y, y * 1.05)["meets_g14"]
    assert not B.g14(y, y * 1.2)["meets_g14"]          # |NMBE| 20% > 10%


def test_cusum_catches_sustained_shift_not_noise():
    rng = np.random.default_rng(3)
    x = pd.Series(np.r_[rng.normal(0, 1, 120), rng.normal(1.5, 1, 60)],
                  index=pd.date_range("2019-01-01", periods=180, freq="D"))
    c = AN.cusum(x)
    first_up = c[c.alarm == "up"].day.min()
    assert first_up >= pd.Timestamp("2019-05-01")      # not during the in-control period
    assert (c[c.day >= "2019-05-01"].alarm == "up").sum() >= 3


def test_flag_events_needs_persistence():
    ts = pd.date_range("2019-01-01", periods=10, freq="h")
    r = pd.DataFrame({"floor": 1, "ts": ts, "z": [0, 5, 0, 5, 5, 5, 0, -5, -5, 0],
                      "resid": 1.0, "day_type": "workday", "occupied": True})
    ev = AN.flag_events(r)
    assert list(ev.hours) == [3, 2] and list(ev.direction) == ["excess", "deficit"]
