"""Moved to src/sbanalytics/faults.py. Alias so `from core import faults` still works."""

import sys

from sbanalytics import faults as _m

sys.modules[__name__] = _m
