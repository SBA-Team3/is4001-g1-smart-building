"""
Rebuild data/cleaned/ from data/raw/ for every floor.

    python scripts/build_clean_data.py              # all floors
    python scripts/build_clean_data.py --floor 5    # one floor (overwrites outputs with that floor only)
    python scripts/build_clean_data.py --no-minute  # skip the large minute parquet

Takes a few minutes. Deterministic: same raw files -> same outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbanalytics import config as C  # noqa: E402
from sbanalytics import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, action="append")
    ap.add_argument("--no-minute", action="store_true")
    a = ap.parse_args()
    pipeline.build(tuple(a.floor) if a.floor else C.FLOORS, write_minute=not a.no_minute)
    print(f"done -> {C.DATA_CLEANED}")
