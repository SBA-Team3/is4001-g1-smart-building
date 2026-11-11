# CU-BEMS building — facilities operations dashboard

A BMS-style operations dashboard for the CU-BEMS building, built for the
university FM team (Office of Physical Resources Management). Its first
questions are energy use, cost, and which sensors are down, so the Overview
answers those for the whole building on one 1440×900 screen. The operational
pages are wired for Floors 3–7; F1 (non-office, no sensors) and F2 (not wired)
appear only in building totals.

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

## Layout

One **context bar** sits above every page: floor (`Bldg · F3 … F7` on Overview and
Carbon; `F3 … F7` on floor pages, which fall back to the last wired floor, default F5), the as-of date, the **period** (`Today · 7d · 30d · MTD · 12m · Record`,
default MTD) and an **Assumptions** popover (tariff, building floor area — 11,700 m²
by default; per-floor figures use ÷7; the comfort band, default 23–26 °C). Assumptions
persist across pages. A custom comfort band flows into every band-dependent figure
(overcooled shares, overcooling alerts, Comfort, levers,
the simulator's comfort after) through optional `lo`/`hi` arguments; the 40% target
stays fixed. When the band is not the default the button reads "Assumptions · custom band".
On the Carbon page the Assumptions popover is replaced by the grid emission
factor (default: the TGO 2022 figure in `R.CONVENTIONS["co2_kg_per_kwh"]`),
editable so it can be updated to a newer TGO figure; a custom value is flagged. Every page reads that one period, so counts agree across pages.
There is no sidebar.

The navigation is flat, in the order the FM asks:

| Page | Question |
|---|---|
| **Overview** *(default)* | How much energy and money this period, against expected? Which sensors are down? Which floor and which alert first? |
| **Floor** | For one floor: data health, energy today, cost MTD vs expected, peak, cooled <23 °C vs 40%; zones worst first; act on open alerts (Work order / Acknowledge / Mute) |
| **Alerts** | The fault register as a queue of Type · Zone · Description · Cost / day · State, filtered by fault type, trade and state (Open / Cleared / Muted / All); a pop-up with the facts, possible causes, options to consider, a note and Acknowledge / Work order / Mute; CSV export for a CMMS. No severity, and no manual Resolve: **alerts clear automatically** once the condition has not been seen for more than 3 days (the episode gap) |
| **Sensors** | Which points are not OK (OFFLINE / FROZEN / BAD SAMPLES / PARTIAL), since when; point × day status map; per-point record; cross-sensor checks; raw trace |
| **Comfort** | Which zones are too cold, by how much (degree-hours); floor × month overcooling map; zone detail |
| **Energy** | Energy, cost, ฿ vs expected, peak; G14 model quality in one line; actual vs expected, CUSUM, signature and outlier days, peak demand, profiles |
| **Carbon** | Scope 2, location-based emissions for the period (building-level; a selected floor is highlighted): tCO₂e, vs expected (F3–F7), intensity, trailing 12 months; monthly emissions; by floor and by end use; reporting-year summary |
| **Savings** | The intervention simulator: package inputs → kWh/yr and ฿/yr ranges, tCO₂e, comfort trade-off, IPMVP M&V plan; measures, tornado, control-stability lever, and how each lever is sized |
| **Methods** | Conventions, point registry, change-points, and "About the data" (why alerts have no severity, what cannot be detected, why not UMLH, flatline rationale, combination rule, practice, corrections, baseline, pilot, exclusions) |

Floor and Sensors carry one plain line of data health under the subtitle (points down
by name, point health, completeness today; on Floor also new / cleared / went offline
since yesterday). "Point down" is OFFLINE, FROZEN or BAD SAMPLES on the as-of day
(`ui.point_status`); PARTIAL is degraded, not down. Alerts are labelled by fault type,
never severity.

## Design

Dark only. Colours, type and components are in [DESIGN.md](DESIGN.md); the tokens
live in `design/tokens.json` and `.streamlit/config.toml` (repo root). Don't
hard-code a colour in a page.

## Structure

```
app.py              flat top navigation, context bar (floor, as-of date, period,
                    assumptions)
ui.py               cached loaders, design tokens, the global period, point status,
                    fault register, fault types, data-health caption, chips, tiles,
                    chart theme
design/             tokens.json, logo
views/              one file per page; analytics come from ../src/sbanalytics/,
                    the same code the analysis notebooks use
cache/              local state (fault_states.csv) — gitignored
```

Analytics modules (pure pandas, no Streamlit):

| Module | What it does |
|---|---|
| `portfolio` | building KPIs, floor league, floor × month overcooling, meter-outage episodes, per-floor levers and the generated "biggest lever" sentence |
| `mv` | daily change-point baseline per floor, G14 summary, roll-up with uncertainty band, segmented CUSUM, energy signature, peak demand |
| `scenario` | intervention simulator: lift / trim / lockout, combined without double counting, comfort after, IPMVP plan, tornado |
| `workorders` | alert-state store (append-only CSV) and CMMS export; the dashboard ignores saved "Resolved" states and derives Cleared itself |
| `faults` | stable ids, cost of inaction, days open, recurrence, trade, action, verification (`PLAYBOOK`) |
| `kpis` | `cooled_occupied` + `overcooled_share` — the one definition of the overcooling KPI |

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
   (Savings → "How each lever is sized" → thermal-model tab)
4. Set `wired: True` in `registry.FLOORS` and open every page for that floor, including the dates of its outages

What is still open:

| Floor | Status |
|---|---|
| F1 | Not wired: no sensors, so Sensors, Savings and Methods fail; not an office floor anyway. |
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
| Grid emission factor | 0.4999 kgCO₂e/kWh (default; editable on Carbon, custom value flagged) | Thailand GHG Management Organization (TGO) — verify current figure before external reporting |
| Demand charge | — | Not used in the dashboard (Energy shows the kW peaks only) |
| Cost of inaction | attributable kWh ÷ calendar days since first seen × tariff | FDD prioritisation practice; zero where no defensible attribution |
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
each blocking change-point; alert states (the dashboard overrides the registry's manual
lifecycle: people set Acknowledged / In progress / Muted, the system sets Cleared).

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
