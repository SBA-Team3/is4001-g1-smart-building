"""
Build the dashboard's data. The dashboard now reads the shared cleaned dataset
in data/cleaned/, produced by the same pipeline every notebook uses:

    python dashboard/build_cache.py            # == python scripts/build_clean_data.py

Kept as an entry point so existing instructions still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbanalytics import config as C  # noqa: E402
from sbanalytics import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.build(C.FLOORS)
