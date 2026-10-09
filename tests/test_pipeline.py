"""Invariants of the cleaning pipeline and the cleaned dataset, for every floor."""

import numpy as np
import pandas as pd
import pytest

from sbanalytics import clean, io
from sbanalytics import config as C
from sbanalytics import data as D
from sbanalytics import registry as R


# == cleaning rules (unit) ==

def test_short_gaps_filled_long_gaps_kept():
    s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0,      # 5-gap -> filled
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 14.0,  # 6-gap -> kept
                   np.nan])                                               # edge -> kept
    out = clean.interpolate_short_gaps(s, max_len=5)
    assert out.iloc[1:6].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert out.iloc[7:13].isna().all()
    assert np.isnan(out.iloc[-1])


def test_spike_rule_removes_burst_keeps_load_step():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.uniform(0.5, 1.5, 2000))
    burst = base.copy()
    burst.iloc[500] = 400.0                     # one-minute reconnect burst
    step = base.copy()
    step.iloc[1000:1100] = 400.0                # sustained real load
    assert clean.spike_mask(burst).sum() == 1
    assert clean.spike_mask(step).sum() == 0


# == registry covers the raw files ==

@pytest.mark.parametrize("floor", C.FLOORS)
def test_every_raw_column_is_in_registry(floor):
    expected = {p.column for p in R.points(floor, existing_only=False) if p.column}
    for y in C.YEARS:
        cols = set(pd.read_csv(C.DATA_RAW / f"{y}Floor{floor}.csv", nrows=0).columns) - {"Date"}
        assert cols == expected


def test_dead_meters_excluded_from_totals():
    assert "z2_AC11(kW)" not in R.ac_columns(2)
    assert "z1_AC1(kW)" not in R.ac_columns(7)


# == cleaned dataset ==

def test_zone_hours_per_floor(building):
    n = building.groupby("floor").size()
    for f in C.FLOORS:
        assert n[f] == len(R.ZONES[f]) * 13176


def test_hourly_reconciles_with_cleaned_minutes(building):
    s = pd.read_csv(C.DATA_CLEANED / "build_summary.csv").set_index("floor")
    tot = building.groupby("floor")[["ac_kwh", "light_kwh", "plug_kwh"]].sum()
    for f in C.FLOORS:
        for ch in ("ac", "light", "plug"):
            assert tot.at[f, f"{ch}_kwh"] == pytest.approx(s.at[f, f"{ch}_kwh_clean"], rel=0.005)


def test_cleaning_changes_energy_by_under_half_a_percent():
    s = pd.read_csv(C.DATA_CLEANED / "build_summary.csv")
    for ch in ("ac", "light", "plug"):
        assert ((s[f"{ch}_kwh_clean"] / s[f"{ch}_kwh_raw"] - 1).abs() < 0.005).all()


def test_sensor_blackout_is_never_filled(building):
    mid = building[(building.ts >= "2018-10-01") & (building.ts < "2019-03-01")]
    assert mid.temp_c.isna().all()


def test_floor6_truncation_stays_missing(building):
    f6 = building[(building.floor == 6) & (building.ts >= "2019-10-23")]
    assert f6[["ac_kwh", "light_kwh", "plug_kwh"]].isna().all().all()


def test_weather_joined_on_local_time(building):
    assert building.out_temp.notna().all()
    w = D.load_weather()
    peak = w.groupby(w.ts.dt.hour).out_solar.mean().idxmax()
    assert peak in (11, 12)          # solar noon in Bangkok ~12:15 local


def test_holidays_and_day_types(building):
    d = building.drop_duplicates(["floor", "date"]).set_index(["floor", "date"])
    assert d.loc[(5, "2019-04-15"), "is_holiday"]           # Songkran
    assert set(d.day_type.unique()) <= {"workday", "weekend", "holiday", "closure", "holiday_worked"}


def test_strict_parse_drops_corrupt_floor6_row():
    raw = io.load_raw(6, years=(2019,), reindex=False)
    assert raw.index.max() == pd.Timestamp("2019-10-22 13:27")
