"""Dashboard core. The analysis modules now live in src/sbanalytics; each
`core/<name>.py` is an alias so `from core import registry` keeps working."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:      # works without `pip install -e .`
    sys.path.insert(0, str(_SRC))
