"""
Smoke test: every dashboard page renders without an exception on the as-of
dates that exercise the dataset's landmines.

    python3 tests/test_dashboard_smoke.py          # no pytest needed
    python3 -m pytest tests/test_dashboard_smoke.py

Needs the cleaned data (`python -m sbanalytics.pipeline`). Dates:
  2019-06-14  an ordinary weekday, all sensors live
  2018-11-15  inside the building-wide sensor gap (and F3/F4 meter outages)
  2019-09-27  the F5 z2<->z4 meter-transfer change-point
  2019-12-31  end of record (F6 long gone)
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from sbanalytics import data as D  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
PAGES = ["pages/0_portfolio.py", "pages/1_floor_status.py", "pages/5_faults.py",
         "pages/3_zones.py", "pages/2_sensor_health.py", "pages/4_energy.py",
         "pages/6_opportunities.py", "pages/8_methods.py"]
DATES = [dt.date(2019, 6, 14), dt.date(2018, 11, 15), dt.date(2019, 9, 27), dt.date(2019, 12, 31)]
TIMEOUT = 300

# Keep the test out of the real fault-state store.
os.environ["FM_FAULT_STATES"] = os.path.join(tempfile.mkdtemp(), "fault_states.csv")


def _render(page: str, as_of: dt.date) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.date_input[0].set_value(as_of)
    at.run()                                 # the date must land before the page switch
    assert at.session_state["as_of"].date() == as_of
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
    for page in PAGES:
        for as_of in DATES:
            errs = _errors(_render(page, as_of))
            if errs:
                failures.append(f"{page} @ {as_of}: {errs}")
    assert not failures, "\n".join(failures)


def test_portfolio_flags_f1_and_f6():
    for as_of in DATES:
        at = _render("pages/0_portfolio.py", as_of)
        assert not _errors(at), _errors(at)
        txt = _text(at)
        assert "F1" in txt and "non-office" in txt, f"{as_of}: Floor 1 not annotated as non-office"
        assert "Floor 1 is not an office floor" in txt, f"{as_of}: F1 exclusion note missing"
        assert "F6" in txt and "file ends" in txt, f"{as_of}: Floor 6 file-end flag missing"


if __name__ == "__main__":
    t0 = time.time()
    n_fail = 0
    for page in PAGES:
        for as_of in DATES:
            t = time.time()
            at = _render(page, as_of)
            errs = _errors(at)
            n_fail += bool(errs)
            print(f"{'FAIL' if errs else 'ok  '}  {page:<28} {as_of}  {time.time() - t:5.1f}s"
                  + (f"  {errs}" if errs else ""), flush=True)
    try:
        test_portfolio_flags_f1_and_f6()
        print("ok    portfolio F1 / F6 annotations on all four dates")
    except AssertionError as e:
        n_fail += 1
        print(f"FAIL  portfolio annotations: {e}")
    print(f"\n{len(PAGES) * len(DATES) + 1 - n_fail} passed, {n_fail} failed "
          f"in {time.time() - t0:.0f}s")
    sys.exit(1 if n_fail else 0)
