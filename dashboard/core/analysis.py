"""Moved to src/sbanalytics/analysis.py. Alias so `from core import analysis` still works."""

import sys

from sbanalytics import analysis as _m

sys.modules[__name__] = _m
