# CU-BEMS building — facilities operations dashboard

A BMS-style operations dashboard for the CU-BEMS building, built for the
university FM team (Office of Physical Resources Management) rather than an
analyst. The Portfolio overview reads all seven floors; the operational pages
are wired for Floors 3–7. Floor 5 carries the hand-validated findings (the
z2↔z4 transfer, the supply-air sensor, the stairwell); on the other floors those
passages are replaced by what the registry declares for that floor.

## Run it

From the repo root:

```bash
pip install -r requirements.txt && pip install -e .
python -m sbanalytics.pipeline         # once, ~2 minutes, all floors -> data/cleaned/
streamlit run dashboard/app.py
```

The dashboard reads the **shared cleaned dataset** in `data/cleaned/` — the same
files every notebook uses — built by `src/sbanalytics/pipeline.py`. It never
touches the raw files: minute-level fault detection over 790,560 rows × 26
channels cannot run on a page load.

## Why the pages are in this order

The navigation is grouped by **who asks the question**, the pattern commercial
platforms (OpenBlue, Forge, Building X, EcoStruxure Building Advisor,
SkySpark/Clockworks) use:

| Group | Page | Persona | Question |
|---|---|---|---|
| **Overview** | Portfolio overview *(default)* | FM lead, leadership | Where do I send the team? Are we on the 40% overcooling target? |
| | Floor status | Floor supervisor | Can I trust today's data, and what changed since yesterday? |
| **Operations** | Faults & work orders | Supervisor, technicians | What is broken, what does it cost per day, who fixes it, how do we know it's fixed? |
| | Zones & comfort | Supervisor | Which zones are too cold, and by how much? |
| | Sensor & data health | Controls / metering-IT | Which points are down, stale or lying? |
| **Energy management** | Energy performance | Energy manager, finance | Are we using more than the weather explains? When did it drift? |
| | Opportunities & simulator | Energy manager, finance, leadership | What would a package of changes save, in ฿ and tCO₂e, what does it do to comfort, and how would we prove it? |
| **Reference** | Methods & assumptions | Anyone checking a number | Where does this come from? |

Within each page the order is still: **can I trust it → what is wrong → how are
we performing → what could we change.** Data health sits above everything else
on the operational pages because the documented failure mode of rule-based
building FDD is that a rule cannot tell its input has gone stale.

### What each reworked page shows

- **Portfolio overview** — KPI strip (building kWh, ฿, tCO₂e, % of cooled
  occupied hours below 23 °C vs the 40% target, data completeness); floor league
  (partial EUI, AC share, overcooled share as a progress bar, after-hours AC,
  weather-normalised deviation, completeness, a generated status note); one
  generated "what to look at" sentence per floor; floor × month overcooling heat
  map; monthly energy by floor. Floor 1 is in the totals, last and greyed in the
  league, and out of every overcooling or schedule ranking. Floor 6 carries its
  file-end flag everywhere it appears.
- **Floor status** — trust band first, then the proposal's two KPIs (this
  month's overcooled share vs 40%; excess cooling kWh/฿ vs the normalised
  expectation, this month vs last, blocked across a change-point), then a
  "since yesterday" strip (new / cleared faults, points that went offline).
- **Faults & work orders** — each fault has a stable id, cost of inaction
  (฿/day), days open, recurrence, trade, action and a verification check;
  lifecycle New → Acknowledged → In progress → Resolved (+ Muted) saved to
  `dashboard/cache/fault_states.csv`; CSV export for a CMMS; the ISA-18.2
  self-audit and the "what we cannot detect" list are kept.
- **Energy performance** — actual vs expected (monthly/weekly, 90% band),
  G14 CV(RMSE)/NMBE with pass/fail, CUSUM restarted at each change-point,
  energy signature with the change-point lines and the drill-down list of
  workdays far above them, peak demand (daily trend, top-10 hours with outdoor
  temperature, demand charge); then the existing EUI, end use, carpet, load
  duration and off-hours sections.
- **Opportunities & simulator** — scenario builder (zone-temperature lift per
  zone or all, end-of-day trim, weekend/holiday lockout, cost per measure,
  tariff, emission factor) → one row per measure plus a combined row: kWh/yr
  low–mid–high, ฿/yr, tCO₂e/yr, payback, comfort after, confidence; an IPMVP
  M&V card per measure; a tornado chart. The setpoint / thermal-model /
  schedule-trim tabs below are the engine.

## Structure

```
app.py              navigation (persona groups), global sidebar (floor, as-of
                    date, tariff, emission factor, demand charge)
ui.py               cached loaders, trust band, chart theme (Streamlit-only)
pages/              one file per page; analytics come from ../src/sbanalytics/,
                    the same code the analysis notebooks use
cache/              local state (fault_states.csv) — gitignored
```

New analytics modules (pure pandas, no Streamlit):

| Module | What it does |
|---|---|
| `portfolio` | building KPIs, floor league, floor × month overcooling, meter-outage episodes, per-floor levers and the generated "what to look at" sentence |
| `mv` | daily change-point baseline per floor (F5 trains between its change-points), G14 summary, monthly/weekly roll-up with uncertainty band, segmented CUSUM, energy signature, peak demand |
| `scenario` | intervention simulator: lift / trim / lockout, combined without double counting, comfort after, IPMVP plan, tornado |
| `workorders` | fault lifecycle store (append-only CSV) and CMMS export |
| `faults` (extended) | stable ids, cost of inaction, days open, recurrence, trade, action, verification (`PLAYBOOK`) |
| `kpis` (extended) | `cooled_occupied` + `overcooled_share` — the one definition of the 68–83% KPI |

Smoke test: `python3 tests/test_dashboard_smoke.py` renders every page on
2019-06-14, 2018-11-15 (sensor gap), 2019-09-27 (change-point) and 2019-12-31,
and checks the Portfolio's F1 / F6 annotations.

## The registry is the important bit

`src/sbanalytics/registry.py` holds one row per physical point, carrying what the channel
*means*. Three metering traps live there as data rather than as `if zone ==
"z1"` branches scattered through analysis code:

- **z1's real cooling load is on `AC4`, not `AC1`.** Only z1 has four AC
  channels. `z1_AC4` carries 65,525 kWh; AC1+AC2+AC3 together carry 225 kWh.
  Every other zone is on AC1, so a `*_AC1` aggregate silently drops 37 % of the
  floor's cooling.
- **z3 has no AC meter** — it is a stairwell. It renders as "not metered", never
  as zero.
- **The lux channel is not lux.** It maxes at 49–79 with integer resolution
  where a daylit office is 300–2,000, so it is tagged a scaled index and never
  labelled as illuminance.

A second registry holds *change-points* — dates across which the data is not
comparable with itself. Any period-over-period comparison spanning one is
disabled with a reason.

## Adding another floor (operational pages)

The data layer is floor-agnostic; every loader takes a floor number. Floors 3,
4 and 7 share Floor 5's column layout; F6 has its four-AC zone in z4, F1 and F2
have their own layouts (all declared in `registry.LAYOUT`).

Every floor is in the registry and in `data/cleaned/`. Floors 3, 4, 6 and 7 are
wired; 1 and 2 are not. To wire one:

1. Check its registry rows (`LAYOUT`, `ROLE_OVERRIDES` in `src/sbanalytics/registry.py`)
2. Tag its outages and change-points with the floor (`KNOWN_OUTAGES`, `CHANGE_POINTS.floors`) —
   untagged history is Floor 5's and must not leak
3. Check each page renders sensibly for its zones, and that the thermal model is identified
   (Opportunities → thermal-model tab)
4. Set `wired: True` in `registry.FLOORS` and run the smoke test, which covers every wired floor

What is still open:

| Floor | Status |
|---|---|
| F1 | Not wired: no sensors, so Sensor health, Opportunities and Methods fail; not an office floor anyway. |
| F2 | Not wired: renders, but has no lunch shutdown, so the thermal model has no AC-off hour, and its 14-unit z2 needs its own triage view. |
| F4, F7 | Wired. The physics fit is not identified (1/C CI reaches zero), so the setpoint estimate is the regression alone and the page says so. |
| F6 | Wired. Unconfirmed z1/z5 meter-transfer candidate in Jun 2019 (00a §7), not yet in `CHANGE_POINTS`. |
| F2–F4, F6, F7 | The 2019-03-06 sensor reinstall is declared as a *warning*, not a blocking change-point: nobody has checked whether these floors also ran colder afterwards, as F5 did. |

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

Added for v2 (all in `R.CONVENTIONS`, with sources, on the Methods page):

| Parameter | Value | Basis |
|---|---|---|
| Overcooling target | < 40% of cooled occupied hours below 23 °C | Project proposal KPI |
| Grid emission factor | 0.4999 kgCO₂e/kWh | Thailand GHG Management Organization (TGO) — verify current figure |
| Demand charge | 0 ฿/kW/month (adjustable) | MEA large-customer tariffs carry one; enter the contract rate |
| Cost of inaction | attributable kWh ÷ active days × tariff | FDD prioritisation practice; zero where no defensible attribution |
| Fault episode gap | 3 days | Same rule that groups offline points |
| Verification windows | 2 weeks (schedule/metering), 4 weeks (comfort) | Project convention |
| M&V band | ±1.645 σ (90%), G14 autocorrelation correction | IPMVP / ASHRAE G14 |
| Energy-signature outlier | > 2 training SDs above the line | Project convention |
| Pilot power | α 0.05, power 0.8, ρ 0.5 | Methods-page defaults |
| Lockout low case | 50% of metered weekend/holiday excess | Notebook 06 |
| Tornado ranges | tariff ±20%, lift ±0.5 °C, trim ±1 h | Project assumption |

Rules (in `R.METHODS`): overcooled share = AC-kWh-weighted cooled occupied
hours (notebook 01); combined savings = S × (1 − T/B0) + T + L per case;
IPMVP Option B for zone-metered measures, C for floor-wide; CUSUM restarts at
each blocking change-point; fault lifecycle states.

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
