"""Shared UI helpers: cached loaders, design tokens, the global period, point
status, the alarm band, chips, formatting and chart theme.

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
                               range=[C["status/normal-bg"], C["status/warning-bg"], WARN, BAD, NO_DATA])
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


def ctx() -> dict:
    """Context-bar selections, set once in app.py and read by every page."""
    as_of = pd.Timestamp(st.session_state.get("as_of", D.DEFAULT_AS_OF))
    period = st.session_state.get("period", DEFAULT_PERIOD)
    start, end, label = period_bounds(as_of, period)
    return {
        "floor": st.session_state.get("floor", 5),
        "as_of": as_of,
        "tariff": st.session_state.get("tariff", R.conv("tariff_thb")),
        "area": st.session_state.get("area", R.conv("floor_area_m2")),
        "window": int((end - start).days) + 1,
        "ef": st.session_state.get("ef", R.conv("co2_kg_per_kwh")),   # set on Carbon; TGO default
        "demand_charge": st.session_state.get("demand_charge", R.conv("demand_charge_thb_per_kw")),
        "period": period, "start": start, "end": end, "period_label": label,
    }


def require_wired(floor: int) -> bool:
    """Render the placeholder panel for an unwired floor. True means go ahead."""
    meta = R.FLOORS.get(floor, {})
    if meta.get("wired"):
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
def _raw_register(floor: int, start, end, as_of, tariff: float) -> pd.DataFrame:
    return F.build_register(D.load_hourly(floor), D.load_health(floor), start=start, end=end,
                            floor=floor, tariff=tariff, as_of=as_of)


def with_states(reg: pd.DataFrame, show_planned: bool = False) -> pd.DataFrame:
    """A fault register as the FM team sees it: planned-maintenance items hidden,
    saved lifecycle states and session mutes applied (`state`, `is_open`)."""
    if not len(reg):
        return reg
    if not show_planned:
        reg = reg[~reg.planned]
    reg = W.apply_states(reg, FAULT_STATES)
    if len(reg):
        sm = reg.id.isin(st.session_state.get("muted", set()))
        reg.loc[sm, "state"], reg.loc[sm, "is_open"] = "Muted", False
    return reg


#: Offline / metering rules a file end would trigger by itself.
FILE_END_RULES = ("point_offline", "dead_meter", "metering_step_change")


def register(floor: int, c: dict | None = None, show_planned: bool = False) -> pd.DataFrame:
    """The floor's register over the global period, with states applied."""
    c = c or ctx()
    raw = _raw_register(floor, c["start"], c["end"], c["as_of"], float(c["tariff"]))
    fe = R.floor_file_end(floor)
    if fe is not None and len(raw):
        # Points "going offline" on or after the file end are the file ending, not
        # an outage. Outages that started before it stay.
        artefact = (raw.rule.isin(FILE_END_RULES)
                    & (pd.to_datetime(raw.first_seen).dt.normalize() >= fe.normalize()))
        raw = raw[~artefact]
    return with_states(raw.copy(), show_planned)


def building_register(c: dict | None = None) -> pd.DataFrame:
    c = c or ctx()
    regs = [register(f, c) for f in R.wired_floors()]
    regs = [r for r in regs if len(r)]
    return pd.concat(regs, ignore_index=True) if regs else pd.DataFrame()


PRIO_ORDER = {F.HIGH: 0, F.MED: 1, F.LOW: 2}
SEV = {F.HIGH: ("●", "critical", "Critical"), F.MED: ("▲", "warning", "Warning"),
       F.LOW: ("ⓘ", "info", "Info")}


def sev_glyph(priority: str) -> str:
    g, _, _ = SEV.get(priority, ("ⓘ", "info", "Info"))
    return f"{g} {str(priority).capitalize()}"


def sort_faults(reg: pd.DataFrame) -> pd.DataFrame:
    if not len(reg):
        return reg
    return reg.assign(_o=reg.priority.map(PRIO_ORDER)).sort_values(
        ["_o", "cost_per_day_thb"], ascending=[True, False]).drop(columns="_o")


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
/* alarm band: one quiet strip; only the counts carry colour */
.cb-band{{display:flex;align-items:center;flex-wrap:wrap;gap:var(--space-2) var(--space-6);padding:10px var(--space-4);
  border:1px solid var(--color-border-default);border-radius:var(--radius-md);background:var(--color-bg-surface)}}
.cb-band .n{{display:inline-flex;align-items:baseline;gap:6px;{ts('Body/Strong')}color:var(--color-text-secondary);white-space:nowrap}}
.cb-band .n b{{{ts('Heading/H3')}color:var(--color-text-primary)}}
.cb-band .n .g{{font-size:13px}}
.cb-band .n.critical .g,.cb-band .n.critical b{{color:var(--color-status-critical-text)}}
.cb-band .n.warning .g,.cb-band .n.warning b{{color:var(--color-status-warning-text)}}
.cb-band .n.offline .g,.cb-band .n.offline b{{color:var(--color-status-offline-text)}}
.cb-band .n.zero .g,.cb-band .n.zero b{{color:var(--color-text-muted)}}
.cb-band.inner{{border:0;background:transparent;padding:0}}
div.st-key-alarmband{{background:var(--color-bg-surface);border:1px solid var(--color-border-default);
  border-radius:var(--radius-md);padding:10px var(--space-4);overflow:visible}}
div.st-key-alarmband div[data-testid="stMarkdownContainer"]{{margin-bottom:0 !important}}
div.st-key-alarmband div[data-testid="stPageLink"]{{display:flex;justify-content:flex-end}}
.cb-band .sep{{width:1px;align-self:stretch;background:var(--color-border-default)}}
.cb-band .meta{{{ts('Caption/Regular')}color:var(--color-text-secondary)}}
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
/* native elements, restyled to the tokens */
div[data-testid="stMainBlockContainer"]{{padding:48px var(--space-8) 64px !important}}
div[data-testid="stMetric"]{{min-height:118px;background:var(--color-bg-surface);border-radius:var(--radius-lg) !important;
  padding:3px 7px !important}}
div[data-testid="stMetricLabel"] p{{{ts('Caption/Strong')}color:var(--color-text-secondary)}}
div[class*="st-key-card_"]{{background:var(--color-bg-surface);border-radius:var(--radius-lg)}}
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


#: priority/status -> (chip class, icon). The icon is never optional.
_CHIP = {"high": ("critical", "error"), "medium": ("warning", "warning"),
         "low": ("info", "info"), "info": ("info", "info"),
         "critical": ("critical", "error"), "warning": ("warning", "warning"),
         "ok": ("normal", "check_circle"), "normal": ("normal", "check_circle"),
         "good": ("normal", "check_circle"), "neutral": ("info", "info"),
         "offline": ("offline", "radio_button_unchecked")}
HOLLOW_STATES = {"Acknowledged", "In progress"}


def chip(priority: str, label: str | None = None, hollow: bool = False) -> str:
    """Severity chip. Filled for a new item; hollow (outline) once acknowledged."""
    cls, ic = _CHIP.get(str(priority).lower(), _CHIP["info"])
    return (f'<span class="cb-chip {cls}{" hollow" if hollow else ""}">{icon(ic)}'
            f'{esc(label if label is not None else str(priority).capitalize())}</span>')


def state_chip(r) -> str:
    """A fault's severity chip, hollow when it is acknowledged or in progress."""
    return chip(r.priority, hollow=getattr(r, "state", "New") in HOLLOW_STATES)


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
# the alarm band — top of every page
# --------------------------------------------------------------------------- #

def band_counts(reg: pd.DataFrame, ps: pd.DataFrame) -> dict:
    op = reg[reg.is_open] if len(reg) else reg
    return {
        "critical": int((op.priority == F.HIGH).sum()) if len(op) else 0,
        "warning": int((op.priority == F.MED).sum()) if len(op) else 0,
        "down": int(ps.is_down.sum()) if len(ps) else 0,
        "unack": int((op.state == "New").sum()) if len(op) else 0,
    }


def alarm_band(counts: dict, meta: str = "", link: bool = True) -> None:
    """● N Critical · ▲ N Warning · ○ N Points down · ◐ N Unacknowledged · meta · All →"""
    def n(kind, g, v, word):
        return (f'<span class="n {kind if v else "zero"}"><span class="g">{g}</span>'
                f'<b>{v}</b>{word}</span>')
    html = ('<div class="cb-band">'
            + n("critical", "●", counts["critical"], "Critical")
            + n("warning", "▲", counts["warning"], "Warning")
            + n("offline", "○", counts["down"], "Points down")
            + n("warning", "◐", counts["unack"], "Unacknowledged")
            + (f'<span class="sep"></span><span class="meta">{meta}</span>' if meta else "")
            + "</div>")
    if link:
        with st.container(key="alarmband"):
            cols = st.columns([12, 1], vertical_alignment="center", gap="small")
            with cols[0]:
                md(html.replace('class="cb-band"', 'class="cb-band inner"'))
            cols[1].page_link("views/5_faults.py", label="All", icon=":material/arrow_forward:",
                              icon_position="right")
    else:
        md(html)


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


def floor_band(floor: int, as_of, reg: pd.DataFrame | None = None, extra: str = "") -> pd.DataFrame:
    """The alarm band for one floor, with its down points named and one line of
    data health. Returns the floor's point status."""
    reg = register(floor) if reg is None else reg
    ps = point_status(floor, as_of)
    down = ps[ps.is_down]
    names = ", ".join(f"{r.zone} {r.channel} {POINT_GLYPH[r.status][0]} {r.status.lower()}"
                      for r in down.head(3).itertuples())
    if len(down) > 3:
        names += f" +{len(down) - 3} more"
    meta = " · ".join(x for x in [names, data_health_line(floor, as_of), extra] if x)
    alarm_band(band_counts(reg, ps), esc(meta).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>"))
    return ps


def status_notes(as_of, floor: int = 5) -> None:
    """Blackout / outage / change-point warnings for the chosen date."""
    for note in D.data_status(as_of, floor):
        fn = st.error if note["level"] == "error" else st.warning
        fn(f"**{note['title']}** — {note['detail']}",
           icon=":material/error:" if note["level"] == "error" else ":material/warning:")


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


def cps_rules(floor: int, start=None, end=None) -> alt.Chart | None:
    """Dotted muted rules for blocking change-points (and the sensor gap) inside a period."""
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


def file_end_note(floor: int, end=None) -> None:
    """Flag a floor whose source file ends early, wherever it appears."""
    fe = R.floor_file_end(floor)
    if fe is not None and (end is None or pd.Timestamp(end) >= fe.normalize()):
        st.warning(f"**Floor {floor}'s file ends {fe:%Y-%m-%d %H:%M}.** Everything after is "
                   "not metered — never compare it with an earlier period.", icon=":material/warning:")


def fmt_weeks(w) -> str:
    return "> 5 years" if w is None or not np.isfinite(w) else f"{w:.0f}"
