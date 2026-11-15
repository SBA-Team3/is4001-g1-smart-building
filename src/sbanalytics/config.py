"""
Paths and cleaning thresholds — one place, so every notebook and the dashboard
apply the same rules.

Each cleaning threshold carries the reason it was chosen. The analytical
conventions (comfort band, tariff, alarm rules) live in `registry.CONVENTIONS`.
"""

from __future__ import annotations

from pathlib import Path


def _find_repo() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data" / "raw").is_dir() and (p / "src" / "sbanalytics").is_dir():
            return p
    raise FileNotFoundError("Repo root not found (expected data/raw and src/sbanalytics).")


REPO = _find_repo()
DATA_RAW = REPO / "data" / "raw"
DATA_EXTERNAL = REPO / "data" / "external"
DATA_CLEANED = REPO / "data" / "cleaned"
MINUTE_DIR = DATA_CLEANED / "minute"          # large, gitignored
DATA_DERIVED = REPO / "data" / "derived"       # tables one analysis notebook hands to the next

WEATHER_CSV = DATA_EXTERNAL / "openmeteo_bangkok_hourly_2018-06-30_2019-12-31_UTC.csv"
HOLIDAYS_CSV = DATA_EXTERNAL / "thai_holidays_2018_2019.csv"

FLOORS = (1, 2, 3, 4, 5, 6, 7)
YEARS = (2018, 2019)

#: CSV timestamps are Bangkok wall-clock time (UTC+7, no DST). Office-hour peaks
#: at 08:00 and 13:00 with a lunch dip on F2/F3/F5/F6 confirm it. Timestamps are
#: kept tz-naive throughout; the weather join converts UTC to this zone first.
TZ = "Asia/Bangkok"
RECORD_START = "2018-07-01 00:00"
RECORD_END = "2019-12-31 23:59"

CLEAN = {
    # Gaps of <= 5 min are linearly interpolated (a meter changes little in 5 min);
    # anything longer stays NaN. Run-length based, so a 20-day gap is untouched —
    # pandas' interpolate(limit=60) would fill the first hour of every long gap.
    "short_gap_max_min": 5,
    # Power spike = value > factor x the column's P99.9 (floor 1 kW), lasting
    # <= max_run minutes, while the 31-min rolling median stays < cap/3. This is
    # the reconnect-burst signature (F1 notebook); a genuine load step persists.
    "spike_factor_p999": 3.0,
    "spike_min_cap_kw": 1.0,
    "spike_max_run_min": 3,
    "spike_context_win_min": 31,
    # An hour's energy is trusted for modelling when >= 48 of 60 minutes are
    # present. Totals still sum what was received; the flag gates models.
    "hour_min_minutes": 48,
    # A day is trusted when >= 20 of its 24 hours pass the hourly gate.
    "day_min_hours": 20,
}
