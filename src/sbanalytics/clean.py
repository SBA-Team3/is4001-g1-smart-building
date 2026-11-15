"""
Minute-level cleaning: one rule per problem, applied identically to every floor.

Order matters and is fixed:

1. range     — values outside the registry's physical bounds -> NaN
2. caps      — the hand-found logger glitches from the F4/F5 notebooks -> NaN
3. spikes    — short reconnect bursts far above the column's normal range -> NaN
4. short gaps — interior NaN runs of <= 5 min are linearly interpolated

Nothing longer than 5 minutes is ever filled, so the sensor blackout and every
meter outage stay NaN and are visible downstream. Stuck sensors are *flagged*
by `quality.point_day_health`, not deleted: a frozen plug meter still reports a
plausible baseload, and deleting it would bias totals more than keeping it.

Every removal is written to a log, so a number in a notebook can be traced back
to exactly which minutes were changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import registry as R


def true_run_lengths(mask: pd.Series) -> pd.Series:
    """For each True element, the length of the consecutive True run it is in (0 elsewhere)."""
    m = mask.to_numpy(dtype=bool)
    if not len(m):
        return pd.Series(dtype=int, index=mask.index)
    gid = np.cumsum(np.r_[True, m[1:] != m[:-1]])
    lengths = np.bincount(gid)[gid]
    return pd.Series(np.where(m, lengths, 0), index=mask.index)


def interpolate_short_gaps(s: pd.Series, max_len: int = C.CLEAN["short_gap_max_min"]) -> pd.Series:
    """Linearly fill interior NaN runs of at most `max_len` samples; leave longer runs alone."""
    na = s.isna()
    if not na.any():
        return s
    short = na & (true_run_lengths(na) <= max_len)
    filled = s.interpolate(limit_area="inside")
    return s.where(~short, filled)


def spike_cap(s: pd.Series) -> float:
    q = s.quantile(0.999)
    if not np.isfinite(q):
        return np.inf
    return max(C.CLEAN["spike_factor_p999"] * float(q), C.CLEAN["spike_min_cap_kw"])


def spike_mask(s: pd.Series, cap: float | None = None) -> pd.Series:
    """Short excursions above `cap` whose surroundings stay well below it.

    A real load step persists and lifts the rolling median with it; a reconnect
    burst lasts a minute or two and the median ignores it.
    """
    cap = spike_cap(s) if cap is None else cap
    exceed = s > cap
    if not exceed.any():
        return exceed
    short = true_run_lengths(exceed) <= C.CLEAN["spike_max_run_min"]
    ctx = s.rolling(C.CLEAN["spike_context_win_min"], center=True, min_periods=1).median() < cap / 3
    return exceed & short & ctx


def _log(rows, floor, column, rule, mask, detail=""):
    n = int(mask.sum())
    if n:
        idx = mask.index[mask.to_numpy()]
        rows.append({"floor": floor, "column": column, "rule": rule, "minutes": n,
                     "first": idx.min(), "last": idx.max(), "detail": detail})


def clean_floor(df: pd.DataFrame, floor: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the four rules to a raw minute frame. Returns (clean frame, change log)."""
    out = df.copy()
    log: list[dict] = []
    caps = R.GLITCH_CAPS.get(floor, {})

    for p in R.points(floor, existing_only=False):
        if not p.column or p.column not in out.columns:
            continue
        s = out[p.column]

        if p.lo is not None:
            bad = s.notna() & ((s < p.lo) | (s > p.hi))
            _log(log, floor, p.column, "range", bad, f"outside [{p.lo:g}, {p.hi:g}] {p.unit}")
            s = s.mask(bad)

        if p.column in caps:
            bad = s > caps[p.column]
            _log(log, floor, p.column, "glitch_cap", bad, f"> {caps[p.column]:g} kW (floor notebook)")
            s = s.mask(bad)

        if p.channel in ("ac", "light", "plug"):
            cap = spike_cap(s)
            bad = spike_mask(s, cap)
            _log(log, floor, p.column, "spike", bad, f"> {cap:.2f} kW, <= {C.CLEAN['spike_max_run_min']} min")
            s = s.mask(bad)

        before = s.isna()
        s = interpolate_short_gaps(s)
        _log(log, floor, p.column, "interpolated", before & s.notna(),
             f"interior gaps <= {C.CLEAN['short_gap_max_min']} min")
        out[p.column] = s.astype("float32")

    return out, pd.DataFrame(log, columns=["floor", "column", "rule", "minutes", "first", "last", "detail"])
