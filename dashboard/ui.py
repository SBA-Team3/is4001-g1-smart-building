"""Shared UI helpers: cached loaders, the trust band, formatting, chart theme."""

from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import health as HL
from sbanalytics import registry as R
from sbanalytics import viz

NOT_METERED = "—  not metered"

ZONE_COLOR = alt.Scale(domain=["z1", "z2", "z3", "z4", "z5"],
                       range=["#b0453a", "#2e74b5", "#8d8d8d", "#2e8b57", "#d98032"])
HEALTH_COLOR = alt.Scale(domain=[0, R.conv("health_warn"), R.conv("health_good"), 100],
                         range=["#b0453a", "#d98032", "#d9c24a", "#2e8b57"])
OK, WARN, BAD = "#2e8b57", "#d98032", "#b0453a"
END_USE_COLOR = alt.Scale(domain=["ac", "light", "plug"], range=["#2e74b5", "#d98032", "#8d8d8d"])
DAY_KIND_COLOR = alt.Scale(domain=["workday", "weekend", "holiday / closure"],
                           range=["#2e74b5", "#d98032", "#8d8d8d"])
#: Overcooled share: green at or under the 40% target, through amber to red.
OVERCOOL_COLOR = alt.Scale(domain=[0, R.conv("overcooled_target_pct"), 100],
                           range=["#2e8b57", "#d9c24a", "#b0453a"])
ACTUAL, EXPECTED, MUTED = "#2e74b5", "#444444", "#c9c9c9"
#: Floor N is always the same colour, matching the notebooks (sbanalytics.viz).
FLOOR_COLOR = alt.Scale(domain=[f"F{f}" for f in sorted(R.FLOORS)],
                        range=[viz.FLOOR_COLORS[f] for f in sorted(R.FLOORS)])

#: Fault lifecycle store. dashboard/cache/ is gitignored, so states stay local.
#: FM_FAULT_STATES overrides it (the smoke test points it at a temp file).
FAULT_STATES = os.environ.get(
    "FM_FAULT_STATES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "fault_states.csv"))


# --------------------------------------------------------------------------- #
# cached loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading hourly data…")
def hourly(floor: int) -> pd.DataFrame:
    return D.load_hourly(floor)


@st.cache_data(show_spinner="Loading point health…")
def health(floor: int) -> pd.DataFrame:
    return D.load_health(floor)


@st.cache_data(show_spinner="Loading the whole building…")
def building_hourly() -> pd.DataFrame:
    return D.load_hourly(None)


@st.cache_data(show_spinner="Loading building point health…")
def building_health() -> pd.DataFrame:
    return D.load_health(None)


@st.cache_data(show_spinner="Scoring points…")
def scored(floor: int) -> pd.DataFrame:
    return HL.score_point_days(D.load_health(floor))


@st.cache_data(show_spinner="Loading day…")
def day_minutes(floor: int, day: str, columns: tuple) -> pd.DataFrame:
    return D.load_day(floor, day, list(columns))


# --------------------------------------------------------------------------- #
# shared context
# --------------------------------------------------------------------------- #

def ctx() -> dict:
    """Sidebar selections, set once in app.py and read by every page."""
    return {
        "floor": st.session_state.get("floor", 5),
        "as_of": pd.Timestamp(st.session_state.get("as_of", D.DEFAULT_AS_OF)),
        "tariff": st.session_state.get("tariff", R.conv("tariff_thb")),
        "area": st.session_state.get("area", R.conv("floor_area_m2")),
        "window": st.session_state.get("window_days", 7),
        "ef": st.session_state.get("ef", R.conv("co2_kg_per_kwh")),
        "demand_charge": st.session_state.get("demand_charge", R.conv("demand_charge_thb_per_kw")),
    }


def require_wired(floor: int) -> bool:
    """Render the placeholder panel for an unwired floor. True means go ahead."""
    meta = R.FLOORS.get(floor, {})
    if meta.get("wired"):
        return True
    st.title(f"Floor {floor}")
    st.warning(f"**Not wired yet.** {meta.get('note','')}")
    st.markdown(
        "The dashboard is floor-agnostic — every loader takes a floor number and "
        "reads its channel list from the point registry. This floor is already in "
        "the registry and the cleaned dataset (see the analysis/ notebooks); wiring "
        "it here means checking the pages render sensibly, not changing analysis code."
    )
    st.caption(f"Wired floors: {', '.join(f'F{f}' for f in R.wired_floors())}. "
               "The Portfolio overview reads every floor.")
    return False


# --------------------------------------------------------------------------- #
# the trust band — sits at the top of every operational page
# --------------------------------------------------------------------------- #

def trust_band(floor: int, as_of, compact: bool = False) -> dict:
    """Data-health header. Everything below it on a page is conditional on it."""
    H = health(floor)
    phi = HL.point_health_index(H, as_of, floor)
    day_rows = H[H.day == pd.Timestamp(as_of).normalize()]

    if not len(day_rows):
        st.error(f"No data at all for {pd.Timestamp(as_of).date()}.")
        return phi

    c = st.columns(4)
    idx = phi["index"]
    c[0].metric("Point health index", f"{idx:.0f} / 100",
                help="Mean per-point health for the selected day: completeness, "
                     "less penalties for frozen, out-of-range and spiking points.")
    c[1].metric("Points reporting", f"{phi['points'] - phi['offline']} / {phi['points']}",
                delta=f"{phi['offline']} offline" if phi["offline"] else "all online",
                delta_color="inverse" if phi["offline"] else "normal")
    c[2].metric("Completeness today", f"{phi['completeness']:.0f}%",
                help=f"Target {R.conv('completeness_target'):.0f}% of expected "
                     "samples per point per day.")
    stale = HL.last_good(H, as_of, floor)
    worst = int(stale.days_stale.max()) if len(stale) else 0
    c[3].metric("Longest stale point", f"{worst} d",
                delta="current" if worst == 0 else "stale", delta_color="inverse" if worst else "normal")

    if not compact:
        band = (f"Everything below is conditional on this band. "
                f"{phi['faulted']} point-day(s) are flagged for the selected date.")
        (st.success if idx >= R.conv("health_good")
         else st.warning if idx >= R.conv("health_warn") else st.error)(band)
    return phi


def status_notes(as_of, floor: int = 5) -> None:
    """Blackout / outage / change-point warnings for the chosen date."""
    for note in D.data_status(as_of, floor):
        fn = st.error if note["level"] == "error" else st.warning
        fn(f"**{note['title']}** — {note['detail']}")


def comparison_guard(start, end, floor: int) -> list:
    """Block a period-over-period comparison that straddles a change-point."""
    cps = [c for c in R.crosses_change_point(start, end, floor) if c.blocks_comparison]
    for c in cps:
        st.error(f"**Comparison disabled — {c.title} ({c.date}).** {c.detail}")
    return cps


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def kwh(v, digits=0):
    return NOT_METERED if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:,.{digits}f}"


def money(v, tariff):
    return f"฿{v * tariff:,.0f}"


def is_stairwell(floor: int, zone: str) -> bool:
    """Only where the registry says so (F5 z3) — elsewhere a zone without an AC
    meter is just that, and calling it a stairwell would be a guess."""
    p = R.point_for_any(floor, zone, "ac")
    return bool(p and p.note.lower().startswith("stairwell"))


def zone_note(floor: int, zone: str) -> str:
    """The caveat an operator needs next to a zone, from the registry."""
    if is_stairwell(floor, zone):
        return "stairwell — no cooling or sensors"
    has_ac, tp = R.has_channel(floor, zone, "ac"), R.point_for(floor, zone, "temp")
    if not has_ac and tp is None:
        return "no AC meter, no sensor"
    if not has_ac:
        return "no AC meter"
    if tp is None:
        return "no sensor — comfort not measured"
    if tp.role == R.ROLE_SUPPLY_TEMP:
        return "supply-air sensor — comfort not valid"
    return ""


def zone_label(floor: int, zone: str) -> str:
    """Zone name with the caveat an operator needs to see next to it."""
    if is_stairwell(floor, zone):
        return f"{zone} · stairwell, no cooling"
    if not R.has_channel(floor, zone, "ac"):
        return f"{zone} · no AC meter"
    p = R.point_for(floor, zone, "temp")
    if p and p.role == R.ROLE_SUPPLY_TEMP:
        return f"{zone} · supply-air sensor"
    return zone


def chart(c, height=280):
    return c.properties(height=height).configure_view(strokeWidth=0)


def altair_heatmap(df, x, y, color, tooltip, title="", scheme=None):
    return alt.Chart(df).mark_rect().encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None),
        color=alt.Color(color, scale=scheme or HEALTH_COLOR,
                        legend=alt.Legend(title="health", orient="right")),
        tooltip=tooltip,
    ).properties(title=title)


def cps_rules(floor: int, start=None, end=None) -> alt.Chart | None:
    """Vertical rules for blocking change-points (and the sensor gap) inside a
    period — the annotation every time-series chart on the energy pages carries."""
    rows = [{"date": pd.Timestamp(c.date), "label": c.title} for c in R.CHANGE_POINTS
            if c.blocks_comparison and floor in c.floors]
    lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    rows += [{"date": lo.normalize(), "label": "sensor gap starts"},
             {"date": hi.normalize(), "label": "sensors restored"}]
    df = pd.DataFrame(rows).drop_duplicates("date")
    if start is not None:
        df = df[df.date >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.date <= pd.Timestamp(end)]
    if not len(df):
        return None
    return alt.Chart(df).mark_rule(color=BAD, strokeDash=[4, 3]).encode(
        x="date:T", tooltip=[alt.Tooltip("date:T"), "label:N"])


def file_end_note(floor: int, end=None) -> None:
    """Flag a floor whose source file ends early, wherever it appears."""
    fe = R.floor_file_end(floor)
    if fe is not None and (end is None or pd.Timestamp(end) >= fe.normalize()):
        st.warning(f"**Floor {floor}'s file ends {fe:%Y-%m-%d %H:%M}.** Everything after is "
                   "not metered — never compare it with an earlier period.")


def fmt_weeks(w) -> str:
    return "> 5 years" if w is None or not np.isfinite(w) else f"{w:.0f}"
