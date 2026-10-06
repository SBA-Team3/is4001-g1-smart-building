"""
Ground-truth checks for the dashboard.

    python3 dashboard/verify.py

These assert against facts established by a minute-level reconnaissance of the
raw CSVs, so they catch a regression in the cache pipeline, the registry or the
health rules. Run after any change to build_cache.py or core/.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import analysis as A      # noqa: E402
from core import data as D          # noqa: E402
from core import faults as F        # noqa: E402
from core import health as HL       # noqa: E402
from core import registry as R      # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def close(a, b, tol):
    return abs(a - b) <= tol


def main() -> int:
    h = D.load_hourly(5)
    H = D.load_health(5)
    S = HL.score_point_days(H)

    print("\n== Reconciliation ==")
    ac = h.ac_kwh.sum()
    check("total AC reconciles to the EDA figure (177,914 kWh)",
          close(ac, 177914, 50), f"{ac:,.0f}")
    check("zone-hours == 5 zones x 13,176 hours", len(h) == 65880, f"{len(h):,}")

    print("\n== Metering traps ==")
    # The AC1/AC4 inversion: a naive *_AC1 aggregate drops z1 entirely.
    raw_cols = R.ac_columns(5)
    naive = [c for c in raw_cols if c.endswith("AC1(kW)")]
    check("registry picks up z1_AC4 as a primary AC meter",
          "z1_AC4(kW)" in R.ac_columns(5, primary_only=True))
    check("a naive *_AC1 aggregate would miss z1's load",
          "z1_AC4(kW)" not in naive,
          f"naive set = {len(naive)} of {len(raw_cols)} AC columns")
    z1 = h[h.zone == "z1"].ac_kwh.sum()
    check("z1 carries ~37% of floor cooling (the part AC1-only would drop)",
          0.30 < z1 / ac < 0.42, f"{z1/ac*100:.1f}%")
    check("z3 AC is absent, not zero",
          h[h.zone == "z3"].ac_kwh.isna().all() and not R.has_channel(5, "z3", "ac"))
    check("z1 temp is tagged supply-air, so it is out of the comfort set",
          "z1" not in R.comfort_zones(5) and "z1" in R.sensored_zones(5))
    check("lux is tagged as a scaled index, never illuminance",
          all(p.role == R.ROLE_SCALED_INDEX for p in R.points(5, "lux")))

    print("\n== Outage ground truth ==")
    first_dead = {}
    for z in ("z1", "z2", "z4", "z5"):
        t = H[(H.zone == z) & (H.channel == "temp")].sort_values("day")
        dead = t[(t.received == 0) & (t.day >= "2018-09-01") & (t.day < "2018-10-01")]
        first_dead[z] = dead.day.min()
    check("z1/z2 sensors die before z4/z5 (12 Sep vs 15 Sep)",
          first_dead["z1"] < first_dead["z4"] and first_dead["z2"] < first_dead["z4"],
          f"z1 {first_dead['z1'].date()}, z4 {first_dead['z4'].date()}")
    back = H[(H.channel == "temp") & (H.received > 0) & (H.day >= "2019-03-01")].day.min()
    check("sensors return on 2019-03-06", str(back.date()) == "2019-03-06",
          str(back.date()))

    z2dead = H[(H.zone == "z2") & (H.channel == "temp") & (H.received == 0)
               & (H.day >= "2019-07-01") & (H.day <= "2019-10-01")]
    check("z2's 2019 summer sensor outage is present (~68 days)",
          65 <= len(z2dead) <= 72, f"{len(z2dead)} days")
    aug = H[(H.zone == "z2") & (H.channel == "temp")
            & (H.day >= "2019-08-01") & (H.day <= "2019-08-31")]
    check("z2 has no indoor temperature at all in Aug 2019",
          aug.received.sum() == 0, f"{int(aug.received.sum())} samples")

    print("\n== Health rules ==")
    t = S[S.channel == "temp"]
    check("no stuck temperature sensors under G36 occupied-only evaluation",
          int(t.flag_flatline.sum()) == 0, f"{int(t.flag_flatline.sum())} flagged")
    check("the two benign overnight flat runs are still measured but not alarmed",
          t.max_flat_nonzero.max() >= 60 and t.max_flat_occupied.max() < 60,
          f"all-day max {int(t.max_flat_nonzero.max())} min, "
          f"occupied max {int(t.max_flat_occupied.max())} min")

    ss = HL.stuck_summary(H)
    frozen = ss[ss.flat_day_share > 20]
    check("the frozen points are plug meters only",
          set(frozen.channel) == {"plug"}, f"{sorted(set(frozen.channel))}")
    check("z3_Plug is the worst offender and reads as a constant",
          frozen.iloc[0].zone == "z3" and frozen.iloc[0].median_distinct < 20,
          f"{frozen.iloc[0].median_distinct:.0f} distinct values/day")

    sp = S[S.spikes > 0].groupby("column").spikes.sum()
    check("the uncapped z2_Plug spike is caught", sp.get("z2_Plug(kW)", 0) >= 1)
    check("the uncapped z4_Plug spike is caught", sp.get("z4_Plug(kW)", 0) >= 1)

    ct = HL.completeness_table(H)
    z2t = ct[(ct.zone == "z2") & (ct.channel == "temp")].completeness.iloc[0]
    check("z2 sensor completeness excluding the blackout is ~75%, not ~51%",
          70 < z2t < 80, f"{z2t:.1f}%")

    print("\n== Cross-sensor checks ==")
    hc = HL.humidity_consistency(h)
    z1row = hc[hc.zone == "z1"]
    check("absolute humidity independently flags z1 as off the shared air path",
          len(z1row) and z1row.abs_hum_dev.iloc[0] < -0.8,
          f"{z1row.abs_hum_dev.iloc[0]:+.2f} g/m3 vs median")

    print("\n== Corrections ==")
    chk = A.combined_meter_check(h)
    check("z4 falls and z2 rises across 2019-09-27",
          chk["delta_b_pct"] < -10 and chk["delta_a_pct"] > 5,
          f"z4 {chk['delta_b_pct']:+.1f}%, z2 {chk['delta_a_pct']:+.1f}%")
    check("their combined load barely moves -> a transfer, not a demand change",
          abs(chk["delta_sum_pct"]) < 10, f"sum {chk['delta_sum_pct']:+.1f}%")
    check("the z2:z4 ratio flips across the boundary",
          chk["ratio_before"] < 1 < chk["ratio_after"],
          f"{chk['ratio_before']:.2f} -> {chk['ratio_after']:.2f}")
    check("baseline training stops before the metering change",
          pd.Timestamp(A.TRAIN[1]) < pd.Timestamp("2019-09-27"), A.TRAIN[1])
    check("no indoor channel can reach the baseline features",
          not set(A.FEATURES) & set(A.FORBIDDEN))

    print("\n== Alarm load ==")
    reg = F.build_register(h, H, start="2019-11-01", end="2019-11-30")
    audit = F.alarm_audit(reg, 30)
    check("a normal month does not flood the operator (<10 alarms/day)",
          audit["per_day"] < 10, f"{audit['per_day']:.2f}/day, {len(reg)} alarms")
    check("standing high-priority alarms stay within the ISA-18.2 cap of 10",
          audit["standing"] <= 10, f"{audit['standing']}")
    sep = F.build_register(h, H, start="2019-09-01", end="2019-09-30")
    check("the Sep-2019 change-point is raised in its own month",
          (sep.rule == "step_change").any())
    check("it is not raised in November",
          not (reg.rule == "step_change").any())

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed.")
    if FAIL:
        print("FAILED: " + "; ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
