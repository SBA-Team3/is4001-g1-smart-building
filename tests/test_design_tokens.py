"""
Keep the dashboard on the design tokens (dashboard/DESIGN.md).

    python3 tests/test_design_tokens.py          # no pytest needed
    python3 -m pytest tests/test_design_tokens.py

1. .streamlit/config.toml uses the same colours as dashboard/design/tokens.json.
2. No page in dashboard/views/ hard-codes a hex colour; pages use ui.py's names.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = json.loads((ROOT / "dashboard" / "design" / "tokens.json").read_text(encoding="utf-8"))["color"]
THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]

#: config.toml key -> token it must equal.
PAIRS = {
    "primaryColor": "accent/default",
    "backgroundColor": "bg/page",
    "secondaryBackgroundColor": "bg/control",
    "textColor": "text/primary",
    "borderColor": "border/strong",          # input and card edges
    "dataframeBorderColor": "border/default",  # decorative dividers
    "dataframeHeaderBackgroundColor": "bg/raised",
    "dataframeHeaderTextColor": "text/secondary",
    "linkColor": "accent/hover",
    "redColor": "status/critical",
    "redTextColor": "status/critical-text",
    "redBackgroundColor": "status/critical-bg",
    "orangeColor": "status/warning",
    "orangeTextColor": "status/warning-text",
    "orangeBackgroundColor": "status/warning-bg",
    "yellowColor": "status/warning",
    "yellowTextColor": "status/warning-text",
    "yellowBackgroundColor": "status/warning-bg",
    "greenColor": "status/normal",
    "greenTextColor": "status/normal-text",
    "greenBackgroundColor": "status/normal-bg",
    "grayColor": "status/offline",
    "grayTextColor": "status/offline-text",
    "grayBackgroundColor": "status/offline-bg",
}

#: chartCategoricalColors, in order.
CATEGORICAL = ["data/primary", "data/series-2", "data/series-3", "data/context"]

#: Legacy names kept as aliases; they must stay equal to the new ones.
ALIASES = {"status/good": "status/normal", "status/neutral": "status/offline"}

#: Retired token families: floors and zones are identified by label, not hue.
RETIRED = ("data/floor-", "data/zone-", "data/overcool-", "data/enduse-", "data/cold", "data/warm")

HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def test_theme_matches_tokens():
    bad = [f"{k} = {THEME.get(k)} but {t} = {TOKENS[t]}" for k, t in PAIRS.items()
           if str(THEME.get(k, "")).upper() != TOKENS[t].upper()]
    assert THEME["sidebar"]["backgroundColor"].upper() == TOKENS["bg/surface"].upper()
    cats = [c.upper() for c in THEME["chartCategoricalColors"]]
    assert cats == [TOKENS[t].upper() for t in CATEGORICAL], "chartCategoricalColors != data tokens"
    assert not bad, "\n".join(bad)


def test_aliases_and_retired_tokens():
    for old, new in ALIASES.items():
        for suf in ("", "-text", "-bg"):
            assert TOKENS[old + suf].upper() == TOKENS[new + suf].upper(), f"{old + suf} != {new + suf}"
    left = [k for k in TOKENS if k.startswith(RETIRED)]
    assert not left, f"retired tokens still defined: {left}"


def test_no_hex_in_views():
    hits = []
    for p in sorted((ROOT / "dashboard" / "views").glob("*.py")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.lstrip().startswith("#") and HEX.search(line):
                hits.append(f"{p.name}:{n}: {line.strip()}")
    assert not hits, "hard-coded colours (use ui.C / ui.* instead):\n" + "\n".join(hits)


if __name__ == "__main__":
    fails = 0
    for name, fn in [("theme matches tokens", test_theme_matches_tokens),
                     ("aliases equal, retired tokens gone", test_aliases_and_retired_tokens),
                     ("no hex colours in views", test_no_hex_in_views)]:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
