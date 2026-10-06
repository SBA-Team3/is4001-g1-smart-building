"""Shared UI helpers: cached loaders, the trust band, formatting, chart theme."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from . import data as D
from . import health as HL
from . import registry as R

NOT_METERED = "—  not metered"

ZONE_COLOR = alt.Scale(domain=["z1", "z2", "z3", "z4", "z5"],
                       range=["#b0453a", "#2e74b5", "#8d8d8d", "#2e8b57", "#d98032"])
HEALTH_COLOR = alt.Scale(domain=[0, 60, 85, 100],
                         range=["#b0453a", "#d98032", "#d9c24a", "#2e8b57"])
OK, WARN, BAD = "#2e8b57", "#d98032", "#b0453a"


# --------------------------------------------------------------------------- #
# cached loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading hourly data…")
def hourly(floor: int) -> pd.DataFrame:
    return D.load_hourly(floor)


@st.cache_data(show_spinner="Loading point health…")
def health(floor: int) -> pd.DataFrame:
    return D.load_health(floor)


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
        "reads its channel list from the point registry. Wiring this floor means "
        "adding its rows to `core/registry.py` and running `build_cache.py "
        f"--floor {floor}`, not changing analysis code."
    )
    st.caption("Only Floor 5 is wired for this project.")
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
        (st.success if idx >= 85 else st.warning if idx >= 60 else st.error)(band)
    return phi


def status_notes(as_of, floor: int = 5) -> None:
    """Blackout / outage / change-point warnings for the chosen date."""
    for note in D.data_status(as_of, floor):
        fn = st.error if note["level"] == "error" else st.warning
        fn(f"**{note['title']}** — {note['detail']}")


def comparison_guard(start, end) -> list:
    """Block a period-over-period comparison that straddles a change-point."""
    cps = [c for c in R.crosses_change_point(start, end) if c.blocks_comparison]
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


def zone_label(floor: int, zone: str) -> str:
    """Zone name with the caveat an operator needs to see next to it."""
    if not R.has_channel(floor, zone, "ac"):
        return f"{zone} · stairwell, no cooling"
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
