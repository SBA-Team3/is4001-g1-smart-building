"""
Ground-truth checks for Floor 5 — ported from dashboard/verify.py.

They assert against facts established by a minute-level reconnaissance of the
raw CSVs, so they catch a regression in the pipeline, the registry or the
health rules.
"""

import pandas as pd

from sbanalytics import analysis as A
from sbanalytics import config as C
from sbanalytics import faults as F
from sbanalytics import health as HL
from sbanalytics import registry as R


# == Reconciliation ==

def test_raw_ac_reconciles_to_eda_figure():
    s = pd.read_csv(C.DATA_CLEANED / "build_summary.csv").set_index("floor")
    assert abs(s.at[5, "ac_kwh_raw"] - 177914) <= 50


def test_cleaning_moves_ac_total_by_under_half_a_percent(h5):
    # Short-gap interpolation adds the few minutes a meter missed (+0.25% on F5).
    assert 177914 <= h5.ac_kwh.sum() <= 177914 * 1.005


def test_zone_hours(h5):
    assert len(h5) == 5 * 13176


# == Metering traps ==

def test_registry_uses_ac4_for_z1():
    assert "z1_AC4(kW)" in R.ac_columns(5, primary_only=True)


def test_naive_ac1_aggregate_misses_z1():
    naive = [c for c in R.ac_columns(5) if c.endswith("AC1(kW)")]
    assert "z1_AC4(kW)" not in naive


def test_z1_carries_about_37pct_of_cooling(h5):
    share = h5[h5.zone == "z1"].ac_kwh.sum() / h5.ac_kwh.sum()
    assert 0.30 < share < 0.42


def test_z3_ac_absent_not_zero(h5):
    assert h5[h5.zone == "z3"].ac_kwh.isna().all() and not R.has_channel(5, "z3", "ac")


def test_z1_temp_is_supply_air():
    assert "z1" not in R.comfort_zones(5) and "z1" in R.sensored_zones(5)


def test_lux_is_scaled_index_everywhere():
    for f in C.FLOORS:
        assert all(p.role == R.ROLE_SCALED_INDEX for p in R.points(f, "lux"))


# == Outage ground truth ==

def test_z1_z2_sensors_die_before_z4_z5(H5):
    first_dead = {}
    for z in ("z1", "z2", "z4", "z5"):
        t = H5[(H5.zone == z) & (H5.channel == "temp")]
        dead = t[(t.received == 0) & (t.day >= "2018-09-01") & (t.day < "2018-10-01")]
        first_dead[z] = dead.day.min()
    assert first_dead["z1"] < first_dead["z4"] and first_dead["z2"] < first_dead["z4"]


def test_sensors_return_2019_03_06(H5):
    back = H5[(H5.channel == "temp") & (H5.received > 0) & (H5.day >= "2019-03-01")].day.min()
    assert str(back.date()) == "2019-03-06"


def test_z2_summer_2019_outage(H5):
    z2dead = H5[(H5.zone == "z2") & (H5.channel == "temp") & (H5.received == 0)
                & (H5.day >= "2019-07-01") & (H5.day <= "2019-10-01")]
    assert 65 <= len(z2dead) <= 72
    aug = H5[(H5.zone == "z2") & (H5.channel == "temp")
             & (H5.day >= "2019-08-01") & (H5.day <= "2019-08-31")]
    assert aug.received.sum() == 0


# == Health rules ==

def test_no_stuck_temperature_sensors_when_occupied(S5):
    assert int(S5[S5.channel == "temp"].flag_flatline.sum()) == 0


def test_overnight_flat_runs_measured_not_alarmed(S5):
    t = S5[S5.channel == "temp"]
    assert t.max_flat_nonzero.max() >= 60 and t.max_flat_occupied.max() < 60


def test_frozen_points_are_plug_meters(H5):
    frozen = HL.stuck_summary(H5)
    frozen = frozen[frozen.flat_day_share > 20]
    assert set(frozen.channel) == {"plug"}
    assert frozen.iloc[0].zone == "z3" and frozen.iloc[0].median_distinct < 20


def test_uncapped_plug_spikes_caught(S5):
    sp = S5[S5.spikes > 0].groupby("column").spikes.sum()
    assert sp.get("z2_Plug(kW)", 0) >= 1 and sp.get("z4_Plug(kW)", 0) >= 1


def test_z2_completeness_excludes_blackout(H5):
    ct = HL.completeness_table(H5)
    z2t = ct[(ct.zone == "z2") & (ct.channel == "temp")].completeness.iloc[0]
    assert 70 < z2t < 80


# == Cross-sensor ==

def test_absolute_humidity_flags_z1(h5):
    hc = HL.humidity_consistency(h5)
    assert hc[hc.zone == "z1"].abs_hum_dev.iloc[0] < -0.8


# == Corrections ==

def test_z2_z4_transfer_on_2019_09_27(h5):
    chk = A.combined_meter_check(h5)
    assert chk["delta_b_pct"] < -10 and chk["delta_a_pct"] > 5
    assert abs(chk["delta_sum_pct"]) < 10
    assert chk["ratio_before"] < 1 < chk["ratio_after"]


def test_baseline_training_stops_before_metering_change():
    assert pd.Timestamp(A.TRAIN[1]) < pd.Timestamp("2019-09-27")


def test_no_indoor_channel_in_baseline_features():
    assert not set(A.FEATURES) & set(A.FORBIDDEN)


# == Alarm load ==

def test_alarm_load_normal_month(h5, H5):
    reg = F.build_register(h5, H5, start="2019-11-01", end="2019-11-30")
    audit = F.alarm_audit(reg, 30)
    assert audit["per_day"] < 10
    assert audit["standing"] <= 10
    assert not (reg.rule == "step_change").any()


def test_change_point_raised_in_its_month(h5, H5):
    sep = F.build_register(h5, H5, start="2019-09-01", end="2019-09-30")
    assert (sep.rule == "step_change").any()
