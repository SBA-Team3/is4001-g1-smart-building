"""Moved to src/sbanalytics/registry.py. Alias so `from core import registry` still works."""

import sys

from sbanalytics import registry as _m

sys.modules[__name__] = _m
