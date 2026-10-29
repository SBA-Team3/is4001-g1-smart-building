"""
Fault lifecycle and CMMS hand-off.

States follow a work order, not an alarm: New -> Acknowledged -> In progress ->
Resolved, plus Muted (ISA-18.2 "shelved": suppressed on purpose, with a note).
The store is an append-only CSV — one row per change, latest row per fault id
wins — so the history of who moved what, and why, is never overwritten.

Pure pandas and file I/O; the dashboard passes the path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STATES = ["New", "Acknowledged", "In progress", "Resolved", "Muted"]
OPEN_STATES = {"New", "Acknowledged", "In progress"}
COLUMNS = ["id", "state", "updated", "note"]


def load_log(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(p, dtype={"id": str, "state": str, "note": str})
    log["updated"] = pd.to_datetime(log.updated, errors="coerce")
    log["note"] = log.note.fillna("")
    return log[COLUMNS]


def current_states(path) -> pd.DataFrame:
    """Latest state per fault id."""
    log = load_log(path)
    if not len(log):
        return log
    return log.sort_values("updated").groupby("id", as_index=False).last()


def set_state(path, fault_id: str, state: str, note: str = "", when=None) -> None:
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}; expected one of {STATES}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"id": fault_id, "state": state,
                         "updated": pd.Timestamp(when) if when is not None else pd.Timestamp.now().floor("s"),
                         "note": note}])
    row.to_csv(p, mode="a", header=not p.exists(), index=False)


def apply_states(reg: pd.DataFrame, path) -> pd.DataFrame:
    """Join the latest state onto a register. Unseen faults are New."""
    if not len(reg):
        return reg
    st = current_states(path)
    out = reg.merge(st.rename(columns={"updated": "state_updated", "note": "state_note"}),
                    on="id", how="left")
    out["state"] = out.state.fillna("New")
    out["state_note"] = out.state_note.fillna("")
    out["is_open"] = out.state.isin(OPEN_STATES)
    return out


#: Columns a CMMS import needs, in hand-off order.
EXPORT_COLUMNS = ["id", "floor", "zone", "trade", "priority", "state", "title", "detail", "action",
                  "verification", "first_seen", "last_seen", "days_open", "recurrence",
                  "kwh_impact", "cost_per_day_thb", "evidence", "state_note"]


def export_frame(reg: pd.DataFrame) -> pd.DataFrame:
    """The register as a CMMS work-order import (one row per fault)."""
    cols = [c for c in EXPORT_COLUMNS if c in reg.columns]
    out = reg[cols].copy()
    for c in ("first_seen", "last_seen"):
        if c in out:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    return out.round({"kwh_impact": 1, "cost_per_day_thb": 0})
