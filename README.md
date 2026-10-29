# IS4001 G1 — Smart Building Analytics (CU-BEMS)

Final project for IS4001, using the CU-BEMS Smart Building Energy & Indoor Environmental Data dataset (7 floors, Jul 2018–Dec 2019, 1-minute sampling — AC power, lighting, plug loads, temperature, humidity, ambient light).

Kaggle source: https://www.kaggle.com/datasets/claytonmiller/cubems-smart-building-energy-and-iaq-data/data

**Goal:** understand energy consumption across the building and give facilities managers recommendations they can act on, each with an energy range, a confidence level and a way to verify it.

---

## Getting set up

### 1. Install VS Code (if you don't have it)

Download from [code.visualstudio.com](https://code.visualstudio.com) and add the **Python** and **Jupyter** extensions.

### 2. Clone the repo

```bash
git clone https://github.com/<org-or-user>/is4001-g1-smart-building.git
cd is4001-g1-smart-building
```

### 3. Set up your Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes the shared `sbanalytics` package importable
```

In VS Code, pick the `.venv` interpreter as the notebook kernel (top-right of a notebook).

### 4. Build the cleaned data (once, ~2 minutes)

```bash
python -m sbanalytics.pipeline
```

The hourly and daily cleaned files are committed, so the notebooks also run without step 4. Only the minute-level files (`data/cleaned/minute/`, used for drill-downs and by the dashboard) need it.

---

## Repo structure

```
is4001-g1-smart-building/
├── data/
│   ├── raw/                 CU-BEMS CSVs, untouched (2018FloorN.csv, 2019FloorN.csv)
│   ├── external/            weather (ERA5, UTC), Thai holidays, Chula academic calendar
│   ├── cleaned/             pipeline output — see data/cleaned/README.md
│   └── derived/             tables one analysis notebook hands to the next (KPIs, baselines)
├── src/sbanalytics/         the shared code: one set of rules for everyone
│                            (python -m sbanalytics.pipeline builds data/cleaned/)
├── analysis/                the analytics plan, A → F (run in order)
│   ├── 01_descriptive_kpis.ipynb      A  how does each floor normally behave?
│   ├── 02_baseline_models.ipynb       B  what should it have used? (ASHRAE G14 baselines)
│   ├── 03_comparison.ipynb            C  who is worse, and when?
│   ├── 04_anomaly_detection.ipynb     D  what needs attention? (fault register)
│   ├── 05_forecasting.ipynb           E  what is coming tomorrow?
│   └── 06_recommendations.ipynb       F  what should facilities management do?
└── dashboard/               Streamlit dashboard: portfolio view of all floors + Floor 5 operations, reads data/cleaned/
```

### `src/sbanalytics` — what lives where

| Module | Purpose |
|---|---|
| `config` | paths and cleaning thresholds (with the reason for each) |
| `registry` | every raw column on every floor: zone, channel, unit, **role** (primary / residual / dead / supply-air sensor / scaled index …), bounds; change-points, measured AC schedules |
| `io` | strict readers for raw CSVs, weather (UTC → local), holidays, calendar |
| `clean` | range → glitch caps → spike rule → ≤ 5-min gap interpolation, with a change log |
| `quality`, `health` | per-point daily health (computed before cleaning) and scoring |
| `resample`, `features` | minute → hourly/daily with coverage flags; calendar, day types, schedules |
| `pipeline` | builds `data/cleaned/` |
| `data` | loaders for the cleaned data (`D.load_hourly(floor=None)` = whole building) |
| `energy`, `comfort`, `faults`, `analysis` | the dashboard's analytics (end use, comfort, FDD rules, thermal model, setpoint savings, pilot power) |
| `kpis`, `baseline`, `anomaly`, `forecast` | the cross-floor analytics used in `analysis/` |
| `viz` | one colour-blind-safe chart style; floor N is always the same colour |

---

## How the data is cleaned (one set of rules, all floors)

| Problem | Rule |
|---|---|
| Impossible values | outside per-point bounds in the registry → NaN |
| Logger glitches found by floor owners (F4, F5) | kept as caps → NaN |
| Reconnect bursts (e.g. 4,762 kW on a light meter) | > 3×P99.9, ≤ 3 min, rolling median < cap/3 → NaN |
| Short gaps | interpolate interior gaps ≤ 5 min; **longer gaps are never filled** |
| Stuck sensors | flagged in the health table, not deleted |
| Hour / day usability | hour usable if ≥ 48 min present; day if ≥ 20 usable hours |
| Timestamps | Bangkok local time, no shift; weather converted UTC → local |
| Calendar | holidays from the shared CSV; closures and worked holidays detected from load |

Cleaning changes each floor's energy totals by less than 0.4%. Every changed minute is in `data/cleaned/cleaning_log.csv`.

---

## Analytics plan

| Layer | Question | Methods | Notebook |
|---|---|---|---|
| A · Descriptive baseline | How does each floor normally behave? | end-use split, baseload ratio, after-hours share, load factor, overcooling share; k-means on daily load shapes | `analysis/01` |
| B · Predictive baseline (M&V) | What should it have used, given weather and calendar? | 3-parameter change-point (daily), TOWT and gradient boosting (hourly); accepted only if ASHRAE Guideline 14 is met out of sample | `analysis/02` |
| C · Comparison | Who is worse, and when? | weather-normalised year-on-year; floor benchmark on normalised KPIs; zone overcooling league | `analysis/03` |
| D · Anomaly detection | What needs attention? | data-quality layer; rules (after-hours AC, always-on/dead units); robust z on cross-fitted baseline residuals + CUSUM drift; validated on known events | `analysis/04` |
| E · Forecasting | What is coming tomorrow? | day-ahead hourly forecasts, rolling-origin backtest against seasonal-naive | `analysis/05` |
| F · Recommendations | What should facilities management do? | kWh/THB ranges from two estimators, confidence, verification and pilot length | `analysis/06` |

### Headline findings (from the notebooks; re-run to refresh)

- **Data.** Indoor sensors are missing on every floor for ~6 months (Sep 2018 – Mar 2019); F6's file ends on 22 Oct 2019. Power meters are ~99% complete.
- **Overcooling everywhere.** 68–83% of occupied, cooled hours on F2–F7 are below 23 °C; some zones average 18.5–20 °C.
- **Baselines work.** The daily change-point model meets ASHRAE G14 out of sample on F2–F7; the hourly TOWT model on F2–F6.
- **Year on year, weather-normalised**, Jul–Dec 2019 is within ±3.5% of the 2018-trained baseline on F2–F7. F1 used ~12% less.
- **Anomalies.** The detector flags ~1% of hours on most floors and catches all 5 weekend AC runs found independently in the Floor 3 deep dive. F2 and F7 stand out with frequent full weekend AC programmes.
- **Forecasting** beats the seasonal-naive benchmark by 48–66% (MAE) on F2–F7 (17% on F1) — useful for peak warnings, secondary to B–D.
- **Biggest lever: raising cooling setpoints by 2 °C** (≈ 51–59 MWh/yr on the office floors), then **locking AC schedules to the calendar** (weekend/holiday runs). Full table: `analysis/06_recommendations.ipynb`.
- **Floor 1 is not an office floor** (evening-peaking, AC never off; a 7-hour clock offset cannot be ruled out). It is kept in totals but excluded from schedule-based measures.

---

## Floor owners

| Floor | Owner |
|---|---|
| Floor 1 | Leyna |
| Floor 2 | Shree |
| Floor 3 | Mik |
| Floor 4 | Kenneth |
| Floor 5 | Terence |
| Floor 6 | Williams |
| Floor 7 | Sana |

## Working together

- Change **cleaning or definitions** only in `src/sbanalytics/`, then rebuild with `python -m sbanalytics.pipeline`. Never patch the data inside a notebook.
- New analysis goes in `analysis/`, on a branch, via PR. Notebooks run in order: 03 and 06 read what 01 and 02 write to `data/derived/`.
- If you get stuck on setup, shout in the group chat before burning too much time on it.
