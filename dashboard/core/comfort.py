"""Moved to src/sbanalytics/comfort.py. Alias so `from core import comfort` still works."""

import sys

from sbanalytics import comfort as _m

sys.modules[__name__] = _m
