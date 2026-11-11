"""Shared UI helpers: cached loaders, design tokens, the global period, point
status, fault types, the data-health caption, chips, formatting and chart theme.

Every colour, size and radius comes from design/tokens.json (see DESIGN.md).
Pages use the names here (ui.C, ui.PRIMARY, ui.chip(), ...) and never a hex
code of their own.
"""

from __future__ import annotations

import html as _html
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sbanalytics import data as D
from sbanalytics import faults as F
from sbanalytics import health as HL
from sbanalytics import registry as R
from sbanalytics import workorders as W

NOT_METERED = "—  not metered"
DASH = "—"

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "design", "tokens.json"), encoding="utf-8") as _f:
    TOKENS = json.load(_f)

#: Colour tokens by name: C["text/primary"].
C = TOKENS["color"]

# status — only for severity and health, always with an icon and a word
OK, WARN, BAD, OFFLINE = (C["status/normal"], C["status/warning"], C["status/critical"],
                          C["status/offline"])
INK, INK_2, INK_3 = C["text/primary"], C["text/secondary"], C["text/muted"]
SURFACE, RAISED, GRID = C["bg/surface"], C["bg/raised"], C["border/default"]
#: "No data" fill in charts: bg/control reads as empty without looking like a value.
NO_DATA = C["bg/control"]

# data — never status, never the accent
PRIMARY, SERIES_2, SERIES_3, CONTEXT = (C["data/primary"], C["data/series-2"],
                                        C["data/series-3"], C["data/context"])
SEQ = [C[f"data/seq-{i}"] for i in range(1, 6)]
#: Overview monthly stack only: one hue per floor, chosen clear of the status hues
#: and the accent (DESIGN.md). Everywhere else floors are identified by label.
FLOOR_CAT = alt.Scale(domain=[f"F{f}" for f in range(1, 8)],
                      range=[C[f"data/floor-cat-{i}"] for i in range(1, 8)])
#: Ink that clears 4.5:1 on each sequential step (light on the two dark steps).
SEQ_INK = [INK, INK, C["text/on-accent"], C["text/on-accent"], C["text/on-accent"]]

ACTUAL, EXPECTED, MUTED = PRIMARY, INK_2, CONTEXT
END_USE_COLOR = alt.Scale(domain=["ac", "light", "plug"], range=[PRIMARY, SERIES_2, SERIES_3])
END_USE_NAME = {"ac": "Air-conditioning", "light": "Lighting", "plug": "Plug load"}
DAY_KIND_COLOR = alt.Scale(domain=["workday", "weekend", "holiday / closure"],
                           range=[PRIMARY, SERIES_2, CONTEXT])
#: Below the comfort band (cold, blue) / above it (warm, sand).
COLD, WARM = PRIMARY, SERIES_2
COLD_WARM = alt.Scale(range=[COLD, WARM])
#: Overcooled share as a blue magnitude: five steps of 20 pp. Pass/fail against
#: the 40% target is a chip beside it, never the colour.
OVERCOOL_BINS = [20, 40, 60, 80]
OVERCOOL_COLOR = alt.Scale(type="threshold", domain=OVERCOOL_BINS, range=SEQ)
#: Point-day status in the health map. Offline is the empty control fill.
POINT_STATUS = ["OK", "PARTIAL", "BAD SAMPLES", "FROZEN", "OFFLINE"]
POINT_STATUS_COLOR = alt.Scale(domain=POINT_STATUS,
                               range=[C["status/normal-fill"], C["status/warning-bg"], WARN, BAD, NO_DATA])
#: Cell outlines: PARTIAL is the dark amber step outlined in amber (degraded); BAD
#: SAMPLES is solid amber (down); OFFLINE is the empty control fill with a grey ring.
POINT_STATUS_STROKE = alt.Scale(domain=POINT_STATUS,
                                range=[C["bg/surface"], WARN, WARN, BAD, OFFLINE])
POINT_STATUS_MARK = {"OK": "", "PARTIAL": "◐", "BAD SAMPLES": "▲", "FROZEN": "●", "OFFLINE": "—"}
#: OK cells use the quiet normal background: normal is never a large bright fill.


def seq_step(pct: float) -> int:
    """Index 0..4 of a 0–100 value on the sequential ramp."""
    return int(np.searchsorted(OVERCOOL_BINS, pct, side="right")) if np.isfinite(pct) else -1


def highlight(field: str, value) -> alt.condition:
    """Selected floor/zone in data/primary, every other one in data/context."""
    return alt.condition(alt.datum[field] == value, alt.value(PRIMARY), alt.value(CONTEXT))


#: Fault lifecycle store. dashboard/cache/ is gitignored, so states stay local.
#: FM_FAULT_STATES overrides the path (e.g. to keep a demo's states separate).
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
# the global period
# --------------------------------------------------------------------------- #

PERIODS = ["Today", "7d", "30d", "MTD", "12m", "Record"]
DEFAULT_PERIOD = "MTD"


def period_bounds(as_of, period: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """(start, end, label) of the global period, ending on the as-of day."""
    period = period or st.session_state.get("period", DEFAULT_PERIOD)
    end = pd.Timestamp(as_of).normalize()
    if period == "Today":
        start = end
    elif period == "7d":
        start = D.window(end, 7)[0]
    elif period == "30d":
        start = D.window(end, 30)[0]
    elif period == "12m":
        start = D.window(end, 365)[0]
    elif period == "Record":
        start = D.RECORD_START
    else:
        start = end.replace(day=1)
    start = max(start, D.RECORD_START)
    if start == end:
        label = f"{end.day} {end:%b %Y}"
    elif start.year == end.year:
        label = f"{start.day} {start:%b} – {end.day} {end:%b %Y}"
    else:
        label = f"{start.day} {start:%b %Y} – {end.day} {end:%b %Y}"
    names = {"Today": "today", "7d": "last 7 days", "30d": "last 30 days", "MTD": "month to date",
             "12m": "last 12 months", "Record": "whole record to date"}
    return start, end, f"{names.get(period, period)} · {label}"


#: User assumptions live in these non-widget keys, so they survive pages that
#: don't render their inputs (Streamlit drops an unrendered widget's key).
#: app.py copies each into its widget key before rendering and back on change.
N_FLOORS = 7
ASSUMPTIONS = {"tariff": ("_tariff", lambda: float(R.conv("tariff_thb"))),
               "building_area": ("_building_area", lambda: float(R.conv("floor_area_m2")) * N_FLOORS),
               "ef": ("_ef", lambda: float(R.conv("co2_kg_per_kwh"))),
               "comfort_lo": ("_comfort_lo", lambda: float(R.conv("comfort_lo"))),
               "comfort_hi": ("_comfort_hi", lambda: float(R.conv("comfort_hi")))}


def band_is_custom(lo: float, hi: float) -> bool:
    return (lo, hi) != (float(R.conv("comfort_lo")), float(R.conv("comfort_hi")))


def assumption(name: str) -> float:
    key, default = ASSUMPTIONS[name]
    return float(st.session_state.get(key, default()))


def date_range(start, end) -> str:
    """'8 – 14 Nov 2019', '28 Oct – 3 Nov 2019', '15 Nov 2018 – 14 Nov 2019' or '14 Nov 2019'."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s.normalize() == e.normalize():
        return f"{e.day} {e:%b %Y}"
    if (s.year, s.month) == (e.year, e.month):
        return f"{s.day} – {e.day} {e:%b %Y}"
    if s.year == e.year:
        return f"{s.day} {s:%b} – {e.day} {e:%b %Y}"
    return f"{s.day} {s:%b %Y} – {e.day} {e:%b %Y}"


def ctx() -> dict:
    """Context-bar selections, set once in app.py and read by every page."""
    as_of = pd.Timestamp(st.session_state.get("as_of", D.DEFAULT_AS_OF))
    period = st.session_state.get("period", DEFAULT_PERIOD)
    start, end, label = period_bounds(as_of, period)
    building_area = assumption("building_area")
    return {
        "floor": st.session_state.get("floor", 5),
        "as_of": as_of,
        "tariff": assumption("tariff"),
        "building_area": building_area,
        "area": building_area / N_FLOORS,              # per floor: the paper gives no split
        "window": int((end - start).days) + 1,
        "ef": assumption("ef"),                        # set on Carbon; TGO default
        # Comfort band from Assumptions (default 23–26 °C). Passed explicitly to every
        # band-dependent sbanalytics call; R.CONVENTIONS is never mutated.
        "lo": assumption("comfort_lo"), "hi": assumption("comfort_hi"),
        "custom_band": band_is_custom(assumption("comfort_lo"), assumption("comfort_hi")),
        "period": period, "start": start, "end": end, "period_label": label,
        "range": date_range(start, end),
    }


#: Floors the dashboard reports for energy, cost and carbon only (a frontend notion;
#: the registry still marks them unwired). Floor 1 is the lobby — a team
#: assumption; the dataset does not say what it is.
REPORTING_ONLY = {1: "lobby"}
#: Pages that accept a reporting-only floor (by URL path); the rest need an office floor.
REPORTING_PAGES = {"", "carbon", "floor_status", "energy"}
BUILDING_PAGES = {"", "carbon"}


def floor_options(url_path: str) -> list[str]:
    """Floor choices in the context bar for a page: Bldg on building pages, F1
    (lobby) where reporting-only floors are allowed, then the wired floors."""
    opts = ["Bldg"] if url_path in BUILDING_PAGES else []
    if url_path in REPORTING_PAGES:
        opts += [f"F{f}" for f in sorted(REPORTING_ONLY)]
    return opts + [f"F{f}" for f in R.wired_floors()]


def floor_name(floor: int) -> str:
    return f"Floor {floor}" + (f" ({REPORTING_ONLY[floor]})" if floor in REPORTING_ONLY else "")


def require_wired(floor: int, allow_reporting: bool = False) -> bool:
    """Render the placeholder panel for an unwired floor. True means go ahead."""
    meta = R.FLOORS.get(floor, {})
    if meta.get("wired") or (allow_reporting and floor in REPORTING_ONLY):
        return True
    st.title(f"Floor {floor}")
    st.warning(f"**Not wired yet.** {meta.get('note','')}", icon=":material/warning:")
    st.caption(f"Wired floors: {', '.join(f'F{f}' for f in R.wired_floors())}. "
               "The Overview reads every floor.")
    return False


# --------------------------------------------------------------------------- #
# point status — one definition of "down", reused on every page
# --------------------------------------------------------------------------- #

DOWN = ("OFFLINE", "FROZEN", "BAD SAMPLES")
#: status -> (glyph, kind). Kind picks the colour; the glyph and word carry it too.
POINT_GLYPH = {"OFFLINE": ("○", "offline"), "FROZEN": ("●", "critical"),
               "BAD SAMPLES": ("▲", "warning"), "PARTIAL": ("◐", "warning"),
               "OK": ("✓", "normal"), "NO FILE": ("○", "offline")}


def point_glyph(status: str) -> str:
    return f"{POINT_GLYPH.get(status, ('', ''))[0]} {status}".strip()


@st.cache_data(show_spinner=False)
def point_status(floor: int, as_of) -> pd.DataFrame:
    """One row per point on the as-of day: floor, zone, channel, point, status,
    last good day, days stale. Status is the Sensors page's rule: OFFLINE (no
    samples), FROZEN (flatline), PARTIAL (below the completeness target),
    BAD SAMPLES (out of range or spikes), else OK. OFFLINE, FROZEN and BAD
    SAMPLES are "down"; PARTIAL is degraded. Past a floor's file end every point
    is NO FILE — not counted as down, because there is no file to be down in."""
    S = scored(floor)
    d = pd.Timestamp(as_of).normalize()
    day = S[S.day == d]
    cols = ["floor", "zone", "channel", "point", "status", "last_good_day", "days_stale", "is_down"]
    fe = R.floor_file_end(floor)
    if fe is not None and d > fe.normalize():
        pts = S[["zone", "channel", "column"]].drop_duplicates()
        out = pts.assign(status="NO FILE")
    elif len(day):
        status = np.select(
            [day.received == 0, day.flag_flatline.values,
             day.completeness < R.conv("completeness_target"),
             (day.out_of_range > 0) | (day.spikes > 0)],
            ["OFFLINE", "FROZEN", "PARTIAL", "BAD SAMPLES"], default="OK")
        out = day[["zone", "channel", "column"]].assign(status=status)
    else:
        pts = S[["zone", "channel", "column"]].drop_duplicates()
        out = pts.assign(status="OFFLINE")
    lg = HL.last_good(health(floor), as_of, floor)[["column", "last_good_day", "days_stale"]]
    out = out.merge(lg, on="column", how="left").rename(columns={"column": "point"})
    out["floor"] = floor
    out["is_down"] = out.status.isin(DOWN)
    order = {"FROZEN": 0, "BAD SAMPLES": 1, "OFFLINE": 2, "PARTIAL": 3, "NO FILE": 4, "OK": 5}
    out = out.sort_values(["status", "zone", "channel"], key=lambda s: s.map(order) if s.name == "status" else s)
    return out[cols].reset_index(drop=True)


def building_point_status(as_of) -> pd.DataFrame:
    return pd.concat([point_status(f, as_of) for f in R.wired_floors()], ignore_index=True)


def short_point(row) -> str:
    """'F5 z3 plug'."""
    return f"F{row.floor} {row.zone} {row.channel}"


# --------------------------------------------------------------------------- #
# fault register — one window, one set of states, so counts agree everywhere
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Building the fault register…")
def _raw_register(floor: int, start, end, as_of, tariff: float, lo: float) -> pd.DataFrame:
    return F.build_register(D.load_hourly(floor), D.load_health(floor), start=start, end=end,
                            floor=floor, tariff=tariff, as_of=as_of, lo=lo)


#: People only acknowledge and handle the work; the system clears an alert.
HUMAN_STATES = ["New", "Acknowledged", "In progress", "Muted"]


def cleared_mask(last_seen: pd.Series, as_of) -> pd.Series:
    """Cleared = not seen for more than the episode gap (3 days) before the as-of
    day. The gap, not 'not seen today', so a weekend does not clear an alert."""
    ls = pd.to_datetime(last_seen).dt.normalize()
    days = (pd.Timestamp(as_of).normalize() - ls).dt.days
    return (days > R.conv("episode_gap_days")).fillna(False)


def with_states(reg: pd.DataFrame, show_planned: bool = False, as_of=None) -> pd.DataFrame:
    """A fault register as the FM team sees it: planned-maintenance items hidden,
    saved human states and session mutes applied, and auto-clear derived.

    Adds `cleared`, `state_label` and `is_open` (= not cleared and not Muted). A
    saved "Resolved" (no longer a manual state) reads as New; an alert that comes
    back keeps its last human state."""
    if not len(reg):
        return reg
    if not show_planned:
        reg = reg[~reg.planned]
    reg = W.apply_states(reg, FAULT_STATES)
    if not len(reg):
        return reg
    reg.loc[reg.state == "Resolved", "state"] = "New"
    sm = reg.id.isin(st.session_state.get("muted", set()))
    reg.loc[sm, "state"] = "Muted"
    as_of = pd.Timestamp(as_of if as_of is not None else st.session_state.get("as_of", D.DEFAULT_AS_OF))
    reg["cleared"] = cleared_mask(reg.last_seen, as_of).to_numpy()
    reg["is_open"] = ~reg.cleared & (reg.state != "Muted")
    seen = pd.to_datetime(reg.last_seen)
    reg["state_label"] = [
        "Muted" if st_ == "Muted" else f"Cleared · last seen {d.day} {d:%b}" if c_ else st_
        for st_, c_, d in zip(reg.state, reg.cleared, seen)]
    return reg


#: Offline / metering rules a file end would trigger by itself.
FILE_END_RULES = ("point_offline", "dead_meter", "metering_step_change")


def register(floor: int, c: dict | None = None, show_planned: bool = False) -> pd.DataFrame:
    """The floor's register over the global period, with states applied."""
    c = c or ctx()
    raw = _raw_register(floor, c["start"], c["end"], c["as_of"], float(c["tariff"]), float(c["lo"]))
    fe = R.floor_file_end(floor)
    if fe is not None and len(raw):
        # Points "going offline" on or after the file end are the file ending, not
        # an outage. Outages that started before it stay.
        artefact = (raw.rule.isin(FILE_END_RULES)
                    & (pd.to_datetime(raw.first_seen).dt.normalize() >= fe.normalize()))
        raw = raw[~artefact]
    return with_states(raw.copy(), show_planned, c["as_of"])


def building_register(c: dict | None = None) -> pd.DataFrame:
    c = c or ctx()
    regs = [register(f, c) for f in R.wired_floors()]
    regs = [r for r in regs if len(r)]
    return pd.concat(regs, ignore_index=True) if regs else pd.DataFrame()


#: Alerts are labelled, filtered and summarised by fault type (the register's
#: `rule`), never by severity: the rule thresholds are not aligned with the 40%
#: target and some depend on the period length. rule -> (label, Material icon).
FAULT_TYPES = {
    "overcooling": ("Overcooling", "ac_unit"),
    "after_hours_ac": ("AC after hours", "schedule"),
    "after_hours_light": ("Lights overnight", "lightbulb"),
    "short_cycling": ("Short cycling", "autorenew"),
    "humidity": ("Humidity", "water_drop"),
    "dead_meter": ("Dead meter", "power_off"),
    "point_offline": ("Point offline", "sensors_off"),
    "step_change": ("Meter change", "swap_horiz"),
}


def type_label(rule: str) -> str:
    return FAULT_TYPES.get(rule, (str(rule).replace("_", " ").title(), ""))[0]


#: Plain-text stand-ins for the chip icons, for places that cannot render HTML
#: (st.dataframe cells, st.pills labels). Text symbols, not colour emoji (︎
#: forces text presentation), and none of the status glyphs (● ▲ ○ ◐ ⓘ).
TYPE_SYMBOL = {
    "overcooling": "❄︎",        # ❄ snowflake
    "after_hours_ac": "◷",           # ◷ clock
    "after_hours_light": "☾",        # ☾ moon
    "short_cycling": "↻",            # ↻
    "humidity": "≋",                 # ≋
    "dead_meter": "⊘",               # ⊘
    "point_offline": "✕",            # ✕
    "step_change": "⇄",              # ⇄
}


def type_text(rule: str) -> str:
    """'❄ Overcooling' — the chip's label with a text symbol, for tables and pills."""
    sym = TYPE_SYMBOL.get(rule, "")
    return f"{sym} {type_label(rule)}" if sym else type_label(rule)


def type_chip(rule: str, state: str = "New") -> str:
    """Neutral fault-type chip: icon + label, no status colour. Filled grey while
    New; an outline once Acknowledged / In progress."""
    label, ic = FAULT_TYPES.get(rule, (type_label(rule), "label"))
    cls = "type" + ("" if state == "New" else " hollow")
    return f'<span class="cb-chip {cls}">{icon(ic)}{esc(label)}</span>'


def top_type(reg: pd.DataFrame) -> tuple[str, int]:
    """(label, count) of the most common fault type in a register."""
    if not len(reg):
        return "", 0
    vc = reg.rule.value_counts()
    return type_label(vc.index[0]), int(vc.iloc[0])


def sort_faults(reg: pd.DataFrame) -> pd.DataFrame:
    """Open first, then by ฿/day (missing last) and days open — never by severity."""
    if not len(reg):
        return reg
    closed = (~reg.is_open).astype(int) if "is_open" in reg else 0
    return (reg.assign(_closed=closed)
            .sort_values(["_closed", "cost_per_day_thb", "days_open"], ascending=[True, False, False],
                         na_position="last").drop(columns="_closed"))


def set_fault_state(fid: str, state: str, note: str = "") -> None:
    W.set_state(FAULT_STATES, fid, state, note)
    st.session_state.setdefault("muted", set()).discard(fid)


def mute_fault(fid: str) -> None:
    st.session_state.setdefault("muted", set()).add(fid)


# --------------------------------------------------------------------------- #
# page chrome and components (HTML + CSS from the tokens)
# --------------------------------------------------------------------------- #

def _css() -> str:
    t = TOKENS["font"]["styles"]
    roots = "".join(f"--color-{k.replace('/', '-')}:{v};" for k, v in C.items())
    roots += "".join(f"--space-{k}:{v}px;" for k, v in TOKENS["space"].items())
    roots += "".join(f"--radius-{k}:{v}px;" for k, v in TOKENS["radius"].items())

    def ts(name):
        s = t[name]
        # !important: Streamlit's own markdown <p> rules would otherwise win.
        return (f"font-size:{s['size']}px !important;line-height:{s['lineHeight']}px !important;"
                f"font-weight:{s['weight']};letter-spacing:{s['letterSpacing']}px;")

    return f"""<style>
:root{{{roots}}}
.cb-ms{{font-family:'Material Symbols Rounded';font-weight:400;font-style:normal;line-height:1;
  display:inline-block;vertical-align:middle;-webkit-font-feature-settings:'liga';user-select:none}}
.cb-overline{{{ts('Label/Overline')}text-transform:uppercase;color:var(--color-text-muted);margin:0}}
.cb-sub{{{ts('Body/Regular')}color:var(--color-text-secondary);margin:0}}
.cb-cap{{{ts('Caption/Regular')}color:var(--color-text-secondary);margin:0}}
.cb-muted{{color:var(--color-text-muted)}}
.cb-h3{{{ts('Heading/H3')}color:var(--color-text-primary);margin:0}}
.cb-metric{{{ts('Display/Metric')}color:var(--color-text-primary)}}
.cb-h2{{{ts('Heading/H2')}color:var(--color-text-primary)}}
.cb-strong{{{ts('Body/Strong')}color:var(--color-text-primary)}}
.cb-body{{{ts('Body/Regular')}color:var(--color-text-primary)}}
.cb-row{{display:flex;align-items:center;gap:var(--space-2)}}
.cb-between{{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--space-3)}}
.cb-stack{{display:flex;flex-direction:column;gap:var(--space-1)}}
/* severity chip: icon + word, always both. Filled = new; hollow = acknowledged / in progress */
.cb-chip{{display:inline-flex;align-self:flex-start;align-items:center;gap:var(--space-1);padding:2px var(--space-2);
  border:1.5px solid;border-radius:var(--radius-full);{ts('Caption/Strong')}white-space:nowrap}}
.cb-chip .cb-ms{{font-size:14px}}
.cb-chip.critical{{background:var(--color-status-critical);border-color:var(--color-status-critical);color:var(--color-text-on-accent)}}
.cb-chip.warning{{background:var(--color-status-warning);border-color:var(--color-status-warning);color:var(--color-text-on-accent)}}
.cb-chip.normal{{background:var(--color-status-normal-bg);border-color:var(--color-status-normal);color:var(--color-status-normal-text)}}
.cb-chip.info{{background:var(--color-bg-control);border-color:var(--color-bg-control);color:var(--color-text-secondary)}}
.cb-chip.offline{{background:transparent;border-color:var(--color-status-offline);border-style:dashed;color:var(--color-status-offline-text)}}
.cb-chip.hollow{{background:transparent}}
.cb-chip.type{{background:var(--color-bg-control);border-color:var(--color-border-strong);color:var(--color-text-secondary)}}
.cb-chip.type .cb-ms{{color:var(--color-text-muted)}}
.cb-chip.type.hollow{{background:transparent}}
.cb-chip.hollow.critical{{color:var(--color-status-critical-text)}}
.cb-chip.hollow.warning{{color:var(--color-status-warning-text)}}
.cb-chip.hollow.info{{border-color:var(--color-border-strong)}}
/* status dot; offline is a hollow ring */
.cb-dot{{display:inline-flex;align-items:center;gap:6px;{ts('Caption/Regular')}color:var(--color-text-secondary);white-space:nowrap}}
.cb-dot i{{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--color-status-normal)}}
.cb-dot.warning i{{background:var(--color-status-warning)}}
.cb-dot.critical i{{background:var(--color-status-critical)}}
.cb-dot.offline i{{background:transparent;border:1.5px solid var(--color-status-offline)}}
.cb-dot.hollow i{{background:transparent;border:1.5px solid currentColor}}
.cb-health{{{ts('Caption/Regular')}color:var(--color-text-secondary);margin:-8px 0 4px}}
/* tiles */
.cb-tile{{min-height:124px;border:1px solid var(--color-border-strong);border-radius:var(--radius-lg);background:var(--color-bg-surface);
  padding:14px var(--space-5);height:100%}}
.cb-tile .lbl{{{ts('Label/Overline')}text-transform:uppercase;color:var(--color-text-muted)}}
.cb-tile .v{{{ts('Display/Metric')}color:var(--color-text-primary);margin-top:2px}}
.cb-tile .d{{{ts('Caption/Strong')}color:var(--color-text-secondary);margin-top:2px}}
.cb-tile .s{{{ts('Caption/Regular')}color:var(--color-text-muted);margin-top:4px}}
.cb-tile ul{{margin:6px 0 0;padding:0;list-style:none}}
.cb-tile li{{{ts('Caption/Regular')}color:var(--color-text-secondary);margin:0 0 2px}}
.cb-share{{position:relative;height:12px;margin:6px 0 4px}}
.cb-share .track{{position:absolute;left:0;right:0;top:3px;height:6px;border-radius:3px;background:var(--color-bg-control)}}
.cb-share .fill{{position:absolute;left:0;top:3px;height:6px;border-radius:3px;background:var(--color-data-primary)}}
.cb-share .target{{position:absolute;top:0;width:2px;height:12px;background:var(--color-text-primary)}}
.cb-legend{{display:inline-flex;align-items:center;gap:6px;{ts('Caption/Regular')}color:var(--color-text-secondary);margin-right:var(--space-4)}}
.cb-legend i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
.cb-legend i.tick{{width:2px;height:12px;border-radius:0}}
.cb-list{{border:1px solid var(--color-border-strong);border-radius:var(--radius-lg);background:var(--color-bg-surface);overflow:hidden}}
.cb-list .item{{display:flex;align-items:center;gap:var(--space-3);padding:10px var(--space-4)}}
.cb-list .item + .item{{border-top:1px solid var(--color-border-default)}}
.cb-list .item .txt{{flex:1;min-width:0}}
.cb-list .item .cost{{{ts('Body/Strong')}color:var(--color-text-primary);white-space:nowrap;text-align:right}}
/* compact list: one 44px row per item, chip centred (Overview Top alerts) */
.cb-list.compact .item{{padding:6px 12px;min-height:44px;gap:var(--space-2)}}
.cb-list.compact .item .cb-chip{{align-self:center}}
.cb-list.compact .item .txt{{flex:1;min-width:0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.cb-list.compact .item .cost .cb-cap{{font-weight:400}}
/* native elements, restyled to the tokens */
div[data-testid="stMainBlockContainer"]{{padding:48px var(--space-8) 64px !important}}
div[data-testid="stMetric"]{{min-height:118px;background:var(--color-bg-surface);border-radius:var(--radius-lg) !important;
  padding:3px 7px !important}}
div[data-testid="stMetricLabel"] p{{{ts('Caption/Strong')}color:var(--color-text-secondary)}}
div[class*="st-key-card_"]{{background:var(--color-bg-surface);border-radius:var(--radius-lg)}}
div[class*="st-key-figure_"]{{background:var(--color-bg-raised);border-radius:var(--radius-lg);padding:var(--space-4)}}
div.st-key-ctxbar{{border-bottom:1px solid var(--color-border-default);padding-bottom:10px;margin-bottom:4px}}
h1{{letter-spacing:-0.2px}}
div[data-testid="stHeading"]:has(h3){{margin-top:12px}}
div[class*="st-key-card_"] div[data-testid="stHeading"]:has(h3){{margin-top:0}}
section[data-testid="stSidebar"],div[data-testid="stSidebarCollapsedControl"]{{display:none}}
</style>"""


def inject_css() -> None:
    """Load the token CSS once per page run (app.py calls it)."""
    st.markdown(_css(), unsafe_allow_html=True)


def esc(s) -> str:
    return _html.escape(str(s))


def icon(name: str, size: int | None = None, color: str | None = None) -> str:
    """A Material Symbols glyph inside HTML (the font Streamlit already loads)."""
    style = (f"font-size:{size}px;" if size else "") + (f"color:{color};" if color else "")
    return f'<span class="cb-ms" style="{style}">{name}</span>'


def md(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def page_header(group: str, title: str, sub: str = "") -> None:
    """Overline (muted), H1 title, one-line purpose. `sub` is HTML."""
    md(f'<p class="cb-overline" style="margin-bottom:-12px">{esc(group)}</p>')
    st.title(title)
    if sub:
        md(f'<p class="cb-sub" style="margin-top:-12px">{sub}</p>')


def section(title: str, sub: str = "") -> None:
    st.subheader(title)
    if sub:
        md(f'<p class="cb-sub" style="margin-top:-12px">{sub}</p>')


#: status -> (chip class, icon). Status chips are for data status and target
#: pass/fail only; alerts use the neutral type_chip(). The icon is never optional.
_CHIP = {"info": ("info", "info"),
         "critical": ("critical", "error"), "warning": ("warning", "warning"),
         "ok": ("normal", "check_circle"), "normal": ("normal", "check_circle"),
         "good": ("normal", "check_circle"), "neutral": ("info", "info"),
         "offline": ("offline", "radio_button_unchecked")}
HOLLOW_STATES = {"Acknowledged", "In progress"}


def chip(status: str, label: str | None = None, hollow: bool = False) -> str:
    """Status chip (data status, target pass/fail). Hollow draws the outline only."""
    cls, ic = _CHIP.get(str(status).lower(), _CHIP["info"])
    return (f'<span class="cb-chip {cls}{" hollow" if hollow else ""}">{icon(ic)}'
            f'{esc(label if label is not None else str(status).capitalize())}</span>')


def state_chip(r) -> str:
    """An alert's neutral type chip, hollow once acknowledged or in progress."""
    return type_chip(r.rule, getattr(r, "state", "New"))


def dot(status: str, label: str, hollow: bool = False) -> str:
    """status: normal | warning | critical | offline (offline is always a ring)."""
    status = {"good": "normal"}.get(status, status)
    return f'<span class="cb-dot {status}{" hollow" if hollow else ""}"><i></i>{esc(label)}</span>'


def legend(items: list[tuple[str, str]], tick: bool = False) -> str:
    """[(label, hex)] swatches; tick=True draws the thin target marker instead."""
    return "".join(f'<span class="cb-legend"><i class="{"tick" if tick else ""}" '
                   f'style="background:{c}"></i>{esc(l)}</span>' for l, c in items)


def share_bar(pct: float, target: float | None = None) -> str:
    """A 0–100% blue magnitude with a target tick. Judgement goes in a chip beside it."""
    target = R.conv("overcooled_target_pct") if target is None else target
    w = max(0.0, min(100.0, pct)) if np.isfinite(pct) else 0.0
    return (f'<div class="cb-share"><div class="track"></div>'
            f'<div class="fill" style="width:{w:.1f}%"></div>'
            f'<div class="target" style="left:{target:.1f}%"></div></div>')


def tile(label: str, value: str, delta: str = "", sub: str = "", items: list[str] | None = None,
         help: str = "") -> None:
    """A KPI tile in HTML, for tiles that need a list or a sub-line st.metric can't hold."""
    li = "".join(f"<li>{i}</li>" for i in (items or []))
    md(f'<div class="cb-tile" title="{esc(help)}"><div class="lbl">{esc(label)}</div>'
       f'<div class="v">{value}</div>'
       + (f'<div class="d">{delta}</div>' if delta else "")
       + (f'<div class="s">{sub}</div>' if sub else "")
       + (f"<ul>{li}</ul>" if li else "") + "</div>")


# --------------------------------------------------------------------------- #
# data health — one plain caption line under the subtitle (Floor, Sensors)
# --------------------------------------------------------------------------- #

def data_health_line(floor: int, as_of) -> str:
    """'Point health 92/100 · 98% complete today' — or why there is none."""
    H = health(floor)
    if not (H.day == pd.Timestamp(as_of).normalize()).any():
        fe = R.floor_file_end(floor)
        if fe is not None and pd.Timestamp(as_of) >= fe.normalize():
            return f"No data — F{floor}'s file ends {fe.day} {fe:%b %Y}"
        return "No data for this date"
    phi = HL.point_health_index(H, as_of, floor)
    return f"Point health {phi['index']:.0f}/100 · {phi['completeness']:.0f}% complete today"


def health_caption(floor: int, as_of, extra: str = "") -> pd.DataFrame:
    """One plain line: the down points by name, point health and completeness
    today, plus `extra` (Floor's since-yesterday). Returns the point status."""
    ps = point_status(floor, as_of)
    down = ps[ps.is_down]
    names = ", ".join(f"{r.zone} {r.channel} {POINT_GLYPH[r.status][0]} {r.status.lower()}"
                      for r in down.head(3).itertuples())
    if len(down) > 3:
        names += f" +{len(down) - 3} more"
    lead = f"{len(down)} point(s) down: {names}" if len(down) else "No points down"
    parts = [esc(lead), esc(data_health_line(floor, as_of))] + ([extra] if extra else [])
    md(f'<p class="cb-health">{" · ".join(parts)}</p>')
    return ps


def status_notes(as_of, floor: int = 5) -> None:
    """Blackout / outage / change-point warnings for the chosen date. The file-end
    note is left to file_end_gate(), so a floor past its file end says so once."""
    for note in D.data_status(as_of, floor):
        if note["title"].startswith("Past the end"):
            continue
        fn = st.error if note["level"] == "error" else st.warning
        fn(f"**{note['title']}** — {note['detail']}",
           icon=":material/error:" if note["level"] == "error" else ":material/warning:")


def file_end_gate(floor: int, start, end) -> pd.Timestamp:
    """For a floor whose source file ends early: if the whole period lies after the
    file end, show one neutral message and stop the page; if it only overlaps,
    show one compact note. Returns the last day with data in the period."""
    fe = R.floor_file_end(floor)
    if fe is None or pd.Timestamp(end) < fe.normalize():
        return pd.Timestamp(end)
    if pd.Timestamp(start) > fe.normalize():
        st.info(f"No data — Floor {floor}'s file ends {fe.day} {fe:%b %Y}. Choose an earlier date.",
                icon=":material/info:")
        st.stop()
    st.caption(f"Floor {floor}'s file ends {fe.day} {fe:%b %Y} {fe:%H:%M}; this period shows data up to "
               "then. Later days are not metered, not zero.")
    return fe.normalize()


SINCE_LOOKBACK_DAYS = 90


@st.cache_data(show_spinner=False)
def _since_register(floor: int, as_of, tariff: float, lo: float) -> pd.DataFrame:
    s, e = D.window(as_of, SINCE_LOOKBACK_DAYS)
    return _raw_register(floor, s, e, as_of, tariff, lo)


def since_yesterday(floor: int, as_of, tariff: float, lo: float | None = None) -> dict:
    """New / cleared alerts on the as-of day, from a fixed 90-day look-back that
    ignores the period control (a one-day window would make every fault 'new')."""
    lo = float(R.conv("comfort_lo")) if lo is None else float(lo)
    reg = _since_register(floor, pd.Timestamp(as_of).normalize(), float(tariff), lo)
    if len(reg):
        reg = reg[~reg.planned]
    day = pd.Timestamp(as_of).normalize()
    if not len(reg):
        return {"new": [], "cleared": []}
    fs = pd.to_datetime(reg.first_seen).dt.normalize()
    ls = pd.to_datetime(reg.last_seen).dt.normalize()
    # Cleared today = it just passed the episode gap (as_of − last_seen = gap + 1 days).
    became = day - pd.Timedelta(days=int(R.conv("episode_gap_days")) + 1)
    return {"new": reg.id[fs == day].tolist(), "cleared": reg.id[ls == became].tolist()}


def comparison_guard(start, end, floor: int, show: bool = True) -> list:
    """Block a period-over-period comparison that straddles a change-point."""
    cps = [c for c in R.crosses_change_point(start, end, floor) if c.blocks_comparison]
    if show:
        for c in cps:
            st.error(f"**Comparison disabled — {c.title} ({c.date}).** {c.detail}",
                     icon=":material/block:")
    return cps


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def kwh(v, digits=0):
    return NOT_METERED if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:,.{digits}f}"


def thb(kwh_v: float, tariff: float, suffix: str = "") -> str:
    """฿ for an energy figure — or the prompt to set a rate, never ฿0."""
    if not tariff:
        return "set rate in Assumptions"
    if kwh_v is None or not np.isfinite(kwh_v):
        return DASH
    return f"฿{kwh_v * tariff:,.0f}{suffix}"


def money(v, tariff):
    return thb(v, tariff)


def round_numeric(df: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Round the number columns only (pandas 3 warns on .round over dates)."""
    return df.round({c: digits for c in df.select_dtypes("number").columns})


def signed_thb(v) -> str:
    """'-฿2,047' / '+฿310'. The sign must come first: st.metric reads the
    delta's direction from a leading '-'."""
    return f"{'-' if v < 0 else '+'}฿{abs(v):,.0f}"


def day_txt(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.day} {t:%b %Y}"


_GLYPH_KIND = {"●": "critical", "▲": "warning", "○": "offline", "◐": "warning",
               "✓": "normal", "ⓘ": "info", "⚠": "warning"}
_KIND_TEXT = {"critical": C["status/critical-text"], "warning": C["status/warning-text"],
              "offline": C["status/offline-text"], "normal": C["status/normal-text"],
              "info": INK_2}


def glyph_style(df: pd.DataFrame, cols: list[str], fmt: dict | None = None, highlight_row=None):
    """Colour glyph-led text cells ('● 3 crit', '○ OFFLINE') by their glyph, and
    show missing values as '—'. Returns a Styler for st.dataframe."""
    def f(v):
        s = str(v).strip()
        k = _GLYPH_KIND.get(s[:1]) if s else None
        return f"color:{_KIND_TEXT[k]}" if k else ""
    sub = [c for c in cols if c in df.columns]
    sty = df.style.map(f, subset=sub) if sub else df.style
    if highlight_row is not None:
        sty = sty.apply(lambda r: [f"color:{C['accent/hover']}" if r.name == highlight_row else ""
                                   for _ in r], axis=1, subset=df.columns[:1])
    return sty.format(fmt or {}, na_rep=DASH)


def is_stairwell(floor: int, zone: str) -> bool:
    """Only where the registry says so (F5 z3)."""
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


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #

def chart(c, height=280):
    """Chart chrome from the tokens: recessive grid and axes, muted labels, no frame."""
    if height is not None:                     # None for faceted charts: panels set their own size
        c = c.properties(height=height)
    return (c.configure(background="transparent").configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=C["border/strong"], tickColor=C["border/strong"],
                            labelColor=INK_3, titleColor=INK_2, labelFontSize=12, titleFontSize=12,
                            titleFontWeight=500)
            .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12)
            .configure_header(labelColor=INK_2, titleColor=INK_2, labelFontSize=12))


def expected_line(c: alt.Chart) -> alt.Chart:
    """Expected / baseline: secondary text colour, dashed."""
    return c.mark_line(color=EXPECTED, strokeWidth=2, strokeDash=[6, 4])


#: One sentence for the 2019-03-06 sensor reinstall, used on every floor (Comfort heat map,
#: Floor's cooled-hours tile, Energy's change-point marker).
REINSTALL_NOTE = ("Sensors were reinstalled on 6 Mar 2019; zones read 2–3 °C colder afterwards, possibly "
                  "partly a measurement change. Compare within 2019.")


def cps_rules(floor: int, start=None, end=None) -> alt.Chart | None:
    """Dotted muted rules for blocking change-points (and the sensor gap) inside a period."""
    rows = [{"date": pd.Timestamp(c.date), "label": c.title} for c in R.CHANGE_POINTS
            if c.blocks_comparison and floor in c.floors]
    lo, hi = (pd.Timestamp(x) for x in R.BLACKOUT)
    rows += [{"date": lo.normalize(), "label": "sensor gap starts"},
             {"date": hi.normalize(), "label": REINSTALL_NOTE}]
    df = pd.DataFrame(rows).drop_duplicates("date")
    if start is not None:
        df = df[df.date >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.date <= pd.Timestamp(end)]
    if not len(df):
        return None
    return alt.Chart(df).mark_rule(color=INK_3, strokeDash=[1, 3], strokeWidth=1.5).encode(
        x="date:T", tooltip=[alt.Tooltip("date:T"), "label:N"])


def overcool_heatmap(hm: pd.DataFrame, as_of, floor_sort: list[str], height: int = 230):
    """Floor × month overcooled share on the blue ramp, '—' cells for no data,
    labels in ink that clears 4.5:1 on each step."""
    hm = hm.copy()
    hm["label"] = np.where(hm.overcooled_pct.notna(),
                           hm.overcooled_pct.round(0).astype("Int64").astype(str), DASH)
    hm["cell"] = np.where(hm.overcooled_pct.notna(), "data", "no data")
    step = [seq_step(v) for v in hm.overcooled_pct.fillna(np.nan)]
    hm["ink"] = ["light" if s < 0 or SEQ_INK[s] == INK else "dark" for s in step]
    base = alt.Chart(hm).encode(
        x=alt.X("yearmonth(month):O", title=None,
                axis=alt.Axis(labelAngle=0, format="%b %y", orient="top", domain=False, ticks=False)),
        y=alt.Y("floor:N", title=None, sort=floor_sort, axis=alt.Axis(domain=False, ticks=False)))
    return alt.layer(
        base.transform_filter("datum.cell == 'no data'").mark_rect(
            color=NO_DATA, cornerRadius=4, stroke=SURFACE, strokeWidth=2),
        base.transform_filter("datum.cell == 'data'").mark_rect(
            cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
            color=alt.Color("overcooled_pct:Q", scale=OVERCOOL_COLOR, legend=None),
            tooltip=["floor", alt.Tooltip("month:T", format="%b %Y"),
                     alt.Tooltip("overcooled_pct:Q", format=".0f", title="cooled hours below band %"),
                     alt.Tooltip("hours:Q", title="cooled zone-hours")]),
        base.mark_text(fontSize=12, fontWeight=500).encode(
            text="label:N",
            color=alt.Color("ink:N", scale=alt.Scale(domain=["dark", "light"],
                                                     range=[C["text/on-accent"], INK]), legend=None)),
        alt.Chart(pd.DataFrame({"m": [pd.Timestamp(as_of).replace(day=1)]})).mark_rect(
            fill=None, stroke=INK, strokeWidth=2, cornerRadius=4).encode(x="yearmonth(m):O"),
    )


def seq_legend() -> str:
    """The five ramp steps with their ranges, plus the target and no-data keys."""
    edges = [0] + OVERCOOL_BINS + [100]
    sw = "".join(f'<span class="cb-legend"><i style="background:{SEQ[i]}"></i>'
                 f'{edges[i]}–{edges[i + 1]}%</span>' for i in range(5))
    return sw + legend([("No data", NO_DATA)])


# --------------------------------------------------------------------------- #
# zone map — schematic plans traced from CU-BEMS Fig. 2 (floors 1-2 and 3-7)
# --------------------------------------------------------------------------- #

with open(os.path.join(_HERE, "design", "floorplans.json"), encoding="utf-8") as _f:
    FLOORPLANS = json.load(_f)


def floorplan(floor: int) -> dict:
    """The traced layout for a floor: `f1_2` for floors 1-2, `f3_7` for 3-7."""
    return FLOORPLANS["layouts"][FLOORPLANS["floor_layout"][str(int(floor))]]


def _ring(pts: list, aspect: float) -> list:
    """Closed ring in plan units (x scaled by the aspect ratio, y down)."""
    ring = [[x * aspect, y] for x, y in pts]
    return ring + ring[:1]


def zone_map_reason(floor: int, zone: str) -> str:
    """Why a zone has no valid comfort value ('' when it can have one)."""
    has_ac = R.has_channel(floor, zone, "ac")
    tp = R.point_for(floor, zone, "temp")
    if tp is None and not has_ac:
        return "no AC / sensor"
    if tp is None:
        return "no sensor"
    if not has_ac:
        return "not AC-metered"          # a sensor but no cooling meter (F2 z3): no % below band
    if tp.role != R.ROLE_ZONE_TEMP:
        return "supply-air sensor" if tp.role == R.ROLE_SUPPLY_TEMP else "not a room sensor"
    return ""


def uncooled_zone_temps(floor: int, start, end) -> dict:
    """Mean occupied temperature for zones that have a room sensor but no AC meter
    (no % below band is possible there). {zone: °C}"""
    out = {}
    for z in R.ZONES.get(floor, []):
        tp = R.point_for(floor, z, "temp")
        if tp is None or tp.role != R.ROLE_ZONE_TEMP or R.has_channel(floor, z, "ac"):
            continue
        d = D.occupied(hourly(floor), zones=[z], require_temp=True)
        d = d[(d.ts >= pd.Timestamp(start)) & (d.ts < pd.Timestamp(end) + pd.Timedelta(days=1))]
        if len(d):
            out[z] = float(d.temp_c.mean())
    return out


SENSOR_CHANNELS = ("temp", "rh", "lux")


def zone_sensor_status(floor: int, as_of) -> dict:
    """zone -> (down: bool, word) for each zone's multi-sensor (temp / RH / lux),
    by ui.point_status: OFFLINE, FROZEN or BAD SAMPLES is down; PARTIAL works."""
    ps = point_status(floor, as_of)
    out = {}
    for z, g in ps[ps.channel.isin(SENSOR_CHANNELS)].groupby("zone"):
        down = g[g.is_down]
        out[z] = ((True, down.status.iloc[0]) if len(down)
                  else (False, "PARTIAL") if (g.status == "PARTIAL").any() else (False, "OK"))
    return out


def zone_map(floor: int, values: pd.DataFrame | None = None, value_col: str | None = None, as_of=None,
             label_fmt: str = "{:.0f}%", height: int = 340, tooltip_cols: tuple = (),
             label_extra: dict | None = None, sublabel: dict | None = None,
             zone_tips: dict | None = None, neutral: bool = False,
             zone_temps: dict | None = None) -> None:
    """Schematic plan of a wired floor (CU-BEMS Fig. 2), not interactive.

    Ramp mode: each zone filled by `value_col` on the blue data ramp (the overcooling
    heat map's bins); zones without a valid value are `bg/control` with a dashed edge
    and say why. Neutral mode (`neutral=True`): every zone `bg/raised`.
    `label_extra` {zone: text} is appended to the label; `sublabel` {zone: text} is a
    second line in status/critical-text (e.g. "● 2 meters down"); `zone_tips`
    {zone: text} adds a tooltip line. Sensor markers are status: a hollow green ring
    when the zone's multi-sensor works, a filled red dot when it is down."""
    fp = floorplan(floor)
    k = float(fp["aspect"])
    as_of = as_of if as_of is not None else ctx()["as_of"]
    v = values.set_index("zone") if values is not None and "zone" in values else values
    label_extra, sublabel, zone_tips = dict(label_extra or {}), sublabel or {}, zone_tips or {}
    zone_temps = zone_temps or {}
    zones = []
    for z, d in fp["zones"].items():
        if neutral:
            reason, val, valid = "", np.nan, False
            text = (z if R.point_for(floor, z, "temp") is not None or floor in REPORTING_ONLY
                    else f"{z} · no sensor")
        else:
            reason = zone_map_reason(floor, z)
            val = (float(v.loc[z, value_col]) if v is not None and z in v.index
                   and pd.notna(v.loc[z, value_col]) else np.nan)
            if not reason and not np.isfinite(val):
                reason = "no data"
            valid = not reason
            text = f"{z}  {label_fmt.format(val)}" if valid else f"{z} · {reason}"
            if reason == "not AC-metered" and np.isfinite(zone_temps.get(z, np.nan)):
                # no % below band without AC, but the room temperature is real: show it
                t_ = f"mean {zone_temps[z]:.1f} °C"
                label_extra[z] = f"{t_} · {label_extra[z]}" if label_extra.get(z) else t_
        step = seq_step(val) if valid else -1
        props = {"zone": z, "valid": valid, "value": val if valid else None,
                 "ink": "dark" if valid and SEQ_INK[step] != INK else "light", "label": text,
                 "note": zone_note(floor, z) or reason or "", "points": zone_tips.get(z, "")}
        for col in tooltip_cols:
            x = v.loc[z, col] if v is not None and z in v.index and col in v.columns else np.nan
            props[col] = None if pd.isna(x) else (round(float(x), 1) if isinstance(x, (int, float, np.floating)) else x)
        zones.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "MultiPolygon",
                                   "coordinates": [[_ring(p, k)] for p in d["polygons"]]}})
    base = alt.Chart(alt.Data(values=zones)).transform_calculate(valid="datum.properties.valid",
                                                                 value="datum.properties.value")
    tips = [alt.Tooltip("properties.zone:N", title="zone")] + [
        alt.Tooltip(f"properties.{c}:N", title=c) for c in tooltip_cols]
    tips += ([alt.Tooltip("properties.points:N", title="points")] if zone_tips
             else [alt.Tooltip("properties.note:N", title="sensor note")])
    edge = dict(stroke=C["border/strong"], strokeWidth=1.2)
    if neutral:
        layers = [base.mark_geoshape(fill=RAISED, **edge).encode(tooltip=tips)]
    else:
        layers = [base.transform_filter("datum.valid").mark_geoshape(**edge).encode(
                      color=alt.Color("value:Q", scale=OVERCOOL_COLOR, legend=None), tooltip=tips),
                  base.transform_filter("!datum.valid").mark_geoshape(
                      fill=NO_DATA, strokeDash=[5, 4], **edge).encode(tooltip=tips)]
    layers.append(alt.Chart(alt.Data(values=[{"type": "Feature", "properties": {},
                                              "geometry": {"type": "Polygon", "coordinates": [_ring(p, k)]}}
                                             for p in fp["core"]])).mark_geoshape(
        fill=RAISED if not neutral else C["bg/control"], stroke=GRID, strokeWidth=1))
    layers.append(alt.Chart(alt.Data(values=[{"type": "Feature", "properties": {},
                                              "geometry": {"type": "Polygon",
                                                           "coordinates": [_ring(fp["outline"], k)]}}])
                            ).mark_geoshape(filled=False, stroke=C["border/strong"], strokeWidth=2))

    # labels (and the red second line) in plan units
    sens = {s["zone"]: s["xy"] for s in fp["sensors"] if s.get("zone")}
    # A sensor the figure does not mark (F2 z3): draw it at the zone's label anchor.
    approx = set()
    for z in fp["zones"]:
        if z not in sens and R.point_for(floor, z, "temp") is not None:
            sens[z] = list(fp["zones"][z]["label"])
            approx.add(z)
    rows = []
    for f in zones:
        z = f["properties"]["zone"]
        lx, ly = fp["zones"][z]["label"]
        if z in sens and abs(sens[z][0] - lx) < 12 and abs(sens[z][1] - ly) < 9:
            ly = sens[z][1] + 9            # keep the label clear of the sensor marker
        rows.append({"x": lx * k, "y": ly, "label": f["properties"]["label"],
                     "ink": f["properties"]["ink"], "extra": label_extra.get(z, "") or "",
                     "sub": sublabel.get(z, ""), "zone": z})
    ldf = pd.DataFrame(rows)
    ink = alt.Color("ink:N", scale=alt.Scale(domain=["dark", "light"],
                                             range=[C["text/on-accent"], INK]), legend=None)
    layers.append(alt.Chart(ldf).mark_text(fontSize=12, fontWeight=600).encode(
        longitude="x:Q", latitude="y:Q", text="label:N", color=ink))
    if (ldf["extra"] != "").any():            # second line, same ink (e.g. "1 alert")
        layers.append(alt.Chart(ldf[ldf["extra"] != ""]).mark_text(fontSize=11, fontWeight=500, dy=15).encode(
            longitude="x:Q", latitude="y:Q", text="extra:N", color=ink))
    if (ldf["sub"] != "").any():              # status line in red (e.g. "● 2 meters down")
        layers.append(alt.Chart(ldf[ldf["sub"] != ""]).mark_text(
            fontSize=11, fontWeight=600, dy=15, color=C["status/critical-text"]).encode(
            longitude="x:Q", latitude="y:Q", text="sub:N"))

    # sensor markers: status colours, and the shape changes too
    st_ = zone_sensor_status(floor, as_of)
    srows = []
    for z, (sx, sy) in sens.items():
        if R.point_for(floor, z, "temp") is None or z not in st_:
            continue
        down, word = st_[z]
        srows.append({"x": sx * k, "y": sy, "zone": z, "down": down, "sensor": f"Sensor {word}",
                      "position": ("approximate (not marked in the paper's figure)" if z in approx
                                   else "as marked in CU-BEMS Fig. 2")})
    sdf = pd.DataFrame(srows)
    if len(sdf):
        stips = [alt.Tooltip("zone:N"), alt.Tooltip("sensor:N", title="multi-sensor"),
                 alt.Tooltip("position:N", title="position")]
        partial = ~sdf.down & sdf.sensor.eq("Sensor PARTIAL")
        ok = ~sdf.down & ~partial
        if ok.any():
            layers.append(alt.Chart(sdf[ok]).mark_point(
                shape="circle", size=90, filled=False, strokeWidth=2, color=OK).encode(
                longitude="x:Q", latitude="y:Q", tooltip=stips))
        if partial.any():
            # Degraded, not down: amber, and the half-filled ◐ the Sensors tables use,
            # so it differs from OK and down by shape as well as colour.
            layers.append(alt.Chart(sdf[partial]).mark_text(
                text=POINT_STATUS_MARK["PARTIAL"], fontSize=17, color=WARN,
                align="center", baseline="middle").encode(
                longitude="x:Q", latitude="y:Q", tooltip=stips))
        if sdf.down.any():
            layers.append(alt.Chart(sdf[sdf.down]).mark_point(
                shape="circle", size=110, filled=True, opacity=1, color=BAD).encode(
                longitude="x:Q", latitude="y:Q", tooltip=stips))
    chart = alt.layer(*layers).project(type="identity").properties(height=height)
    st.altair_chart(chart.configure(background="transparent").configure_view(strokeWidth=0),
                    width="stretch")
    sensor_key = (f'<span class="cb-legend"><span style="color:{OK}">○</span>sensor OK</span>'
                  f'<span class="cb-legend"><span style="color:{WARN}">◐</span>sensor partial</span>'
                  f'<span class="cb-legend"><span style="color:{BAD}">●</span>sensor down</span>')
    if not len(sdf):                     # no multi-sensors on this floor (F1): no sensor key
        sensor_key = ""
    if neutral:
        if sensor_key:
            md(f'<div style="margin-top:-6px">{sensor_key}</div>')
        return
    edges = [0] + OVERCOOL_BINS + [100]
    steps = "".join(f'<span class="cb-legend"><i style="background:{SEQ[i]}"></i>'
                    f'{edges[i]}–{edges[i + 1]}%</span>' for i in range(5))
    md(f'<div style="margin-top:-6px">{steps}'
       f'<span class="cb-legend"><i style="background:{NO_DATA};border:1px dashed {C["border/strong"]}">'
       f'</i>— no data / not metered</span>{sensor_key}'
       f'<span class="cb-legend cb-muted">target &lt; {R.conv("overcooled_target_pct"):.0f}% = '
       f'boundary of the 2nd and 3rd step</span></div>')


def file_end_note(floor: int, end=None) -> None:
    """Flag a floor whose source file ends early, wherever it appears."""
    fe = R.floor_file_end(floor)
    if fe is not None and (end is None or pd.Timestamp(end) >= fe.normalize()):
        st.warning(f"**Floor {floor}'s file ends {fe:%Y-%m-%d %H:%M}.** Everything after is "
                   "not metered — never compare it with an earlier period.", icon=":material/warning:")


def fmt_weeks(w) -> str:
    return "> 5 years" if w is None or not np.isfinite(w) else f"{w:.0f}"
