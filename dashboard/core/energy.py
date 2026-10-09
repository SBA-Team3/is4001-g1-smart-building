"""Moved to src/sbanalytics/energy.py. Alias so `from core import energy` still works."""

import sys

from sbanalytics import energy as _m

sys.modules[__name__] = _m
