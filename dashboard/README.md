# Floor 5 — facilities operations dashboard

A BMS-style operations dashboard for Floor 5 of the CU-BEMS building, built for a
facilities manager rather than an analyst.

## Run it

```bash
pip install -r ../requirements.txt
python3 build_cache.py        # once, ~1 minute
streamlit run app.py
```

`build_cache.py` reads the raw CSVs in `../data/` and writes four parquet
artefacts into `cache/` (gitignored). The dashboard never touches the raw files:
minute-level fault detection over 790,560 rows × 26 channels cannot run on a
page load.

```bash
python3 verify.py             # 29 ground-truth checks
```

## Why the pages are in this order

A facilities manager opening a dashboard asks three questions, in this order:

1. **Can I trust what I'm looking at?** → *Floor status* trust band, *Sensor &
   data health*
2. **What is wrong, and what do I do about it?** → *Zones*, *Faults & actions*
3. **How is the floor performing?** → *Energy*, *Opportunities*

Data health sits above everything else because the documented failure mode of
rule-based building FDD is that a rule cannot tell its input has gone stale. A
comfort figure computed on a dead sensor looks exactly like one computed on a
live sensor.

## Structure

```
app.py              navigation, global sidebar (floor, as-of date, tariff)
build_cache.py      raw CSV -> parquet artefacts
verify.py           ground-truth regression checks
core/
  registry.py       point registry + change-point registry — the source of truth
  data.py           floor-agnostic loaders, as-of windows, psychrometrics
  health.py         completeness, flatline, range, spike, peer consistency
  comfort.py        exceedance hours and degree-hours vs a declared band
  energy.py         EUI, end use, demand, baseload, after-hours
  faults.py         FDD rules + G36 persistence/suppression + alarm self-audit
  analysis.py       thermal calibration, setpoint sensitivity, M&V baseline
  ui.py             cached loaders, trust band, chart theme
pages/              one file per page
legacy/             the superseded 8-tab analyst prototype, for reference
```

## The registry is the important bit

`core/registry.py` holds one row per physical point, carrying what the channel
*means*. Three metering traps live there as data rather than as `if zone ==
"z1"` branches scattered through analysis code:

- **z1's real cooling load is on `AC4`, not `AC1`.** Only z1 has four AC
  channels. `z1_AC4` carries 65,525 kWh; AC1+AC2+AC3 together carry 225 kWh.
  Every other zone is on AC1, so a `*_AC1` aggregate silently drops 37 % of the
  floor's cooling. `verify.py` asserts against this.
- **z3 has no AC meter** — it is a stairwell. It renders as "not metered", never
  as zero.
- **The lux channel is not lux.** It maxes at 49–79 with integer resolution
  where a daylit office is 300–2,000, so it is tagged a scaled index and never
  labelled as illuminance.

A second registry holds *change-points* — dates across which the data is not
comparable with itself. Any period-over-period comparison spanning one is
disabled with a reason.

## Adding another floor

The data layer is floor-agnostic; every loader takes a floor number. Floors 3,
4, 6 and 7 share Floor 5's exact 30-column schema.

1. Add the floor's rows to `core/registry.py`
2. `python3 build_cache.py --floor N`
3. Set `wired: True` in `registry.FLOORS`

Nothing in the analysis code changes.

## Standards and conventions

Every threshold is declared with its source on the **Methods** page. The main
ones:

| Parameter | Value | Basis |
|---|---|---|
| Comfort band | 23–26 °C | Thai field studies (neutrality ≈ 26 °C) |
| Occupied schedule | 07:48–11:50, 13:03–16:31 | **Measured** from the meter |
| AlarmDelay | 30 min | ASHRAE G36 Addendum u §5.16.14 |
| Evaluation window | 5-min rolling mean | ASHRAE G36 §5.16.14.6 |
| EUI benchmark | 170 good / 225 typical | Bangkok office study, n=41 |
| Humidity limit | 0.012 kg/kg | ASHRAE 55 |

Deliberately **not** used: the term "Unmet Load Hours" (it is defined relative to
a setpoint, which this building does not expose) and the ASHRAE 55 adaptive
comfort model (it applies only to naturally conditioned spaces, and this
building is mechanically cooled).

## Corrections this dashboard makes to the earlier notebook

- The "unexplained z4 level shift around Oct 2019" is a **z2↔z4 transfer on
  2019-09-27**: z4 −19 %, z2 +10 %, combined sum −6 %, ratio flipping 0.89 →
  1.21, on the same afternoon two sensors were restored. A meter re-assignment,
  not a change in cooling demand.
- The baseline was trained straight across that boundary, so its headline G14
  failure was measuring a meter change. Training now stops 2019-09-26.
- The sensor gap does not start simultaneously (z1/z2 on 12 Sep, z4/z5 at exactly
  00:00 on 15 Sep) and ends 2019-03-06, not 03-05.
- Three of five plug meters are effectively frozen; two power spikes were
  uncapped.

See the Methods page for the full list with evidence.
