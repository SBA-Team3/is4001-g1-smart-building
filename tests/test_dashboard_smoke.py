"""
Smoke test: every dashboard page renders without an exception, for every
wired floor, on the as-of dates that exercise the dataset's landmines.

    python3 tests/test_dashboard_smoke.py          # no pytest needed
    python3 -m pytest tests/test_dashboard_smoke.py

Needs the cleaned data (`python -m sbanalytics.pipeline`). Dates:
  2019-06-14  an ordinary weekday, all sensors live
  2018-11-15  inside the building-wide sensor gap (and F3/F4 meter outages)
  2019-09-27  the F5 z2<->z4 meter-transfer change-point
  2019-12-31  end of record (F6 long gone)

Floor-specific checks: no other floor inherits F5's outages, change-points or
z3-is-a-stairwell narrative, and F6 past its file end says so.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from sbanalytics import data as D  # noqa: E402
from sbanalytics import registry as R  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
PAGES = ["views/0_portfolio.py", "views/1_floor_status.py", "views/5_faults.py",
         "views/3_zones.py", "views/2_sensor_health.py", "views/4_energy.py",
         "views/7_carbon.py", "views/6_opportunities.py", "views/8_methods.py"]
DATES = [dt.date(2019, 6, 14), dt.date(2018, 11, 15), dt.date(2019, 9, 27), dt.date(2019, 12, 31)]
FLOORS = R.wired_floors()
TIMEOUT = 300

# Keep the test out of the real fault-state store.
os.environ["FM_FAULT_STATES"] = os.path.join(tempfile.mkdtemp(), "fault_states.csv")


def _render(page: str, as_of: dt.date, floor: int = 5) -> AppTest:
    """Set floor and date through the context bar (by widget key), then open the page."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.segmented_control(key="floor_select").set_value(f"F{floor}")
    at.date_input(key="as_of").set_value(as_of)
    at.run()                                 # the date must land before the page switch
    got = at.session_state["as_of"]
    assert (got.date() if hasattr(got, "date") and callable(got.date) else got) == as_of
    assert at.session_state["floor"] == floor
    at.switch_page(page)
    at.run()
    assert at.title, f"{page} rendered no title"
    return at


def _text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "warning", "info", "error", "success", "title", "subheader"):
        parts += [str(getattr(e, "value", "")) for e in getattr(at, kind)]
    for df in at.dataframe:
        v = df.value
        v = getattr(v, "data", v)            # a Styler carries its frame in .data
        parts.append(v.to_string() if hasattr(v, "to_string") else str(v))
    return "\n".join(parts)


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


def test_every_page_renders():
    assert D.cache_ready(5), "cleaned data not built — run python -m sbanalytics.pipeline"
    failures = []
    for floor in FLOORS:
        for page in PAGES:
            for as_of in DATES:
                errs = _errors(_render(page, as_of, floor))
                if errs:
                    failures.append(f"F{floor} {page} @ {as_of}: {errs}")
    assert not failures, "\n".join(failures)


def test_no_floor_inherits_floor5_history():
    """F5's outages, change-points and stairwell are F5's alone."""
    for floor in [f for f in FLOORS if f != 5]:
        for as_of in (dt.date(2019, 8, 20), dt.date(2019, 9, 27)):   # F5 z2 outage; F5 transfer
            txt = _text(_render("views/1_floor_status.py", as_of, floor))
            assert "z2: sensors offline" not in txt, f"F{floor} @ {as_of}: shows F5's z2 outage"
            assert "re-apportioned" not in txt, f"F{floor} @ {as_of}: shows F5's z2/z4 change-point"
            assert "stairwell" not in txt, f"F{floor} @ {as_of}: calls a zone a stairwell"


def test_floor_specific_notes():
    txt = _text(_render("views/1_floor_status.py", dt.date(2018, 11, 15), 3))
    assert "all meters offline" in txt.lower(), "F3 Oct-Nov 2018 meter outage not flagged"
    txt = _text(_render("views/1_floor_status.py", dt.date(2019, 12, 31), 6))
    assert "Past the end of this floor's file" in txt, "F6 file end not flagged"


def _metric(at: AppTest, label: str) -> str:
    return next(str(m.value) for m in at.metric if m.label == label)


def test_outage_reads_as_no_data():
    """F3's meters are all offline on 2018-11-15: no zeros, no faults raised on NaN."""
    at = _render("views/1_floor_status.py", dt.date(2018, 11, 15), 3)
    txt = _text(at)
    assert not re.search(r"\bnan\b", txt), "a fault or caption was built from NaN"
    assert _metric(at, "Energy today") == "no data", "outage day shows a number, not 'no data'"
    assert _metric(at, "Peak demand") == "—", "outage window shows a peak demand"


def test_open_fault_counts_agree():
    """The Floor page's open-alarm count is what the Alarms page shows: both read the
    one global period (the context bar), so there is no per-page window to drift."""
    for floor, as_of in ((5, dt.date(2019, 11, 14)), (3, dt.date(2018, 11, 15))):
        txt = _text(_render("views/1_floor_status.py", as_of, floor))
        m = re.search(r"Active alarms · (\d+) open", txt)
        assert m, f"F{floor} @ {as_of}: Floor page shows no open-alarm count"
        at = _render("views/5_faults.py", as_of, floor)
        n_faults = _metric(at, "Open faults") if at.metric else "0"
        assert m.group(1) == n_faults, f"F{floor} @ {as_of}: floor {m.group(1)} vs alarms {n_faults}"


def test_portfolio_flags_f1_and_f6():
    for as_of in DATES:
        at = _render("views/0_portfolio.py", as_of)
        assert not _errors(at), _errors(at)
        txt = _text(at)
        assert "F1" in txt and "non-office" in txt, f"{as_of}: Floor 1 not annotated as non-office"
        assert "Floor 1 is not an office floor" in txt, f"{as_of}: F1 exclusion note missing"
        assert "F6" in txt and "file ends" in txt, f"{as_of}: Floor 6 file-end flag missing"


def test_carbon_page():
    """Carbon reports Scope 2, location-based, and flags F6 past its file end."""
    txt = _text(_render("views/7_carbon.py", dt.date(2019, 12, 31)))
    assert "Scope 2" in txt and "location-based" in txt, "Carbon page does not state its scope"
    assert "file ends 22 Oct 19" in txt, "Carbon by-floor table does not flag F6's file end"


def _carbon_numbers(at: AppTest) -> tuple[float, float]:
    """(Emissions tile, by-floor total) in tCO2e, as rendered."""
    txt = _text(at)
    tile = re.search(r'Emissions · \w+</div><div class="v">([\d,.]+) tCO₂e', txt)
    total = re.search(r"Total ([\d,.]+) tCO₂e", txt)
    assert tile and total, "Carbon emissions tile or by-floor total not found"
    return float(tile.group(1).replace(",", "")), float(total.group(1).replace(",", ""))


def test_carbon_emission_factor():
    """The grid factor is a context-bar input on Carbon (key `ef`, TGO default 0.4999); a custom
    value scales every figure and is flagged in the footnote."""
    at = _render("views/7_carbon.py", dt.date(2019, 11, 14))
    at.segmented_control(key="floor_select").set_value("Bldg").run()
    ef = at.number_input(key="ef")
    assert abs(ef.value - 0.4999) < 1e-9, f"default factor {ef.value}, expected 0.4999"
    tile0, tot0 = _carbon_numbers(at)
    assert abs(tile0 - tot0) < 0.051, f"tile {tile0} != by-floor total {tot0}"
    ef.set_value(0.5).run()
    assert not _errors(at), _errors(at)
    tile1, tot1 = _carbon_numbers(at)
    # Both totals are rounded to 0.01, so allow two half-steps of rounding.
    assert abs(tot1 - tot0 * 0.5 / 0.4999) < 0.011, f"total {tot0} -> {tot1}, not proportional"
    assert abs(tile1 - tot1) < 0.051, f"tile {tile1} != by-floor total {tot1}"
    at.number_input(key="ef").set_value(0.25).run()       # a change big enough to see
    tile2, tot2 = _carbon_numbers(at)
    assert abs(tot2 - tot0 * 0.25 / 0.4999) < 0.011, f"total {tot0} -> {tot2} at 0.25, not proportional"
    assert "custom factor" in _text(at), "custom factor not flagged in the footnote"


if __name__ == "__main__":
    t0 = time.time()
    n_fail = n_run = 0
    for floor in FLOORS:
        for page in PAGES:
            for as_of in DATES:
                t = time.time()
                at = _render(page, as_of, floor)
                errs = _errors(at)
                n_run += 1
                n_fail += bool(errs)
                print(f"{'FAIL' if errs else 'ok  '}  F{floor} {page:<28} {as_of}  "
                      f"{time.time() - t:5.1f}s" + (f"  {errs}" if errs else ""), flush=True)
    for name, fn in [("portfolio F1 / F6 annotations", test_portfolio_flags_f1_and_f6),
                     ("no floor inherits F5's history", test_no_floor_inherits_floor5_history),
                     ("F3 outage / F6 file-end notes", test_floor_specific_notes),
                     ("outage reads as no data", test_outage_reads_as_no_data),
                     ("open-fault counts agree", test_open_fault_counts_agree),
                     ("carbon scope and F6 note", test_carbon_page),
                     ("carbon emission factor input", test_carbon_emission_factor)]:
        n_run += 1
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            n_fail += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{n_run - n_fail} passed, {n_fail} failed in {time.time() - t0:.0f}s")
    sys.exit(1 if n_fail else 0)
