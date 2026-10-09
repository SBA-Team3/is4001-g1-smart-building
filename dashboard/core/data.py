"""Moved to src/sbanalytics/data.py. Alias so `from core import data` still works."""

import sys

from sbanalytics import data as _m

sys.modules[__name__] = _m
