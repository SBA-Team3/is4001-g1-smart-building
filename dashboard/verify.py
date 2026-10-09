"""
Ground-truth checks — now a pytest suite in tests/.

    python dashboard/verify.py        # == pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve().parents[1] / "tests"), "-q"]))
