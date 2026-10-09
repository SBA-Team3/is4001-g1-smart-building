"""Moved to src/sbanalytics/health.py. Alias so `from core import health` still works."""

import sys

from sbanalytics import health as _m

sys.modules[__name__] = _m
