"""Shared UI helpers: cached loaders, design tokens, the trust band, cards, formatting,
chart theme.

Every colour, size and radius comes from design/tokens.json, which is exported
from the Figma file (see DESIGN.md). Pages use the names here (ui.C, ui.BAD,
ui.FLOOR_COLOR, ui.chip(), ...) and never a hex code of their own.
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
from sbanalytics import health as HL
from sbanalytics import registry as R
from sbanalytics import workorders as W

NOT_METERED = "—  not metered"

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "design", "tokens.json"), encoding="utf-8") as _f:
    TOKENS = json.load(_f)

#: Colour tokens by Figma name without the "color/" prefix: C["text/primary"].
C = TOKENS["color"]
FIGMA_URL = TOKENS["$figma"]

OK, WARN, BAD = C["status/good"], C["status/warning"], C["status/critical"]
INK, INK_2, INK_3 = C["text/primary"], C["text/secondary"], C["text/muted"]
SURFACE, RAISED, GRID = C["bg/surface"], C["bg/raised"], C["border/default"]
#: "No data" fill in charts: bg/control reads as empty on the surface without looking like a value.
NO_DATA = C["bg/control"]

ZONE_COLOR = alt.Scale(domain=["z1", "z2", "z3", "z4", "z5"],
                       range=[C[f"data/zone-{i}"] for i in range(1, 6)])
HEALTH_COLOR = alt.Scale(domain=[0, R.conv("health_warn"), R.conv("health_good"), 100],
                         range=[C["data/overcool-high"], C["status/warning"],
                                C["data/overcool-mid"], C["data/overcool-low"]])
END_USE_COLOR = alt.Scale(domain=["ac", "light", "plug"],
                          range=[C["data/enduse-ac"], C["data/enduse-light"], C["data/enduse-plug"]])
DAY_KIND_COLOR = alt.Scale(domain=["workday", "weekend", "holiday / closure"],
                           range=[C["data/floor-1"], C["data/floor-2"], C["status/neutral"]])
#: Overcooled share: green at or under the 40% target, through amber to red.
OVERCOOL_COLOR = alt.Scale(domain=[0, R.conv("overcooled_target_pct"), 100],
                           range=[C["data/overcool-low"], C["data/overcool-mid"],
                                  C["data/overcool-high"]])
#: Below the comfort band / above it.
COLD, WARM = C["data/cold"], C["data/warm"]
COLD_WARM = alt.Scale(range=[COLD, WARM])
ACTUAL, EXPECTED, MUTED = C["data/floor-1"], C["text/secondary"], C["border/strong"]
#: Floor N is always the same colour, as in the notebooks (dark-mode steps of
#: sbanalytics.viz.FLOOR_COLORS).
FLOOR_HEX = {f: C[f"data/floor-{f}"] for f in sorted(R.FLOORS)}
FLOOR_COLOR = alt.Scale(domain=[f"F{f}" for f in sorted(R.FLOORS)],
                        range=[FLOOR_HEX[f] for f in sorted(R.FLOORS)])


def overcool_hex(pct: float) -> str:
    """Bar colour for an overcooled share: at/under target, up to 70%, above."""
    if not np.isfinite(pct):
        return C["status/neutral"]
    if pct <= R.conv("overcooled_target_pct"):
        return C["data/overcool-low"]
    return C["data/overcool-mid"] if pct <= 70 else C["data/overcool-high"]

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
# page chrome and components (the Figma components, as HTML + CSS)
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
.cb-overline.accent{{color:var(--color-accent-default)}}
.cb-sub{{{ts('Body/Regular')}color:var(--color-text-secondary);margin:0}}
.cb-cap{{{ts('Caption/Regular')}color:var(--color-text-secondary);margin:0}}
.cb-muted{{color:var(--color-text-muted)}}
.cb-h3{{{ts('Heading/H3')}color:var(--color-text-primary);margin:0}}
.cb-metric{{{ts('Display/Metric')}color:var(--color-text-primary)}}
.cb-hero{{{ts('Display/Hero')}color:var(--color-text-primary)}}
.cb-h2{{{ts('Heading/H2')}color:var(--color-text-primary)}}
.cb-strong{{{ts('Body/Strong')}color:var(--color-text-primary)}}
.cb-body{{{ts('Body/Regular')}color:var(--color-text-primary)}}
.cb-row{{display:flex;align-items:center;gap:var(--space-2)}}
.cb-between{{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--space-3)}}
.cb-stack{{display:flex;flex-direction:column;gap:var(--space-1)}}
/* severity chip: icon + label, always both */
.cb-chip{{display:inline-flex;align-self:flex-start;align-items:center;gap:var(--space-1);padding:3px var(--space-2);
  border:1px solid;border-radius:var(--radius-full);{ts('Caption/Strong')}white-space:nowrap}}
.cb-chip .cb-ms{{font-size:14px}}
.cb-chip.critical{{background:var(--color-status-critical-bg);border-color:var(--color-status-critical);color:var(--color-status-critical-text)}}
.cb-chip.critical .cb-ms{{color:var(--color-status-critical)}}
.cb-chip.warning{{background:var(--color-status-warning-bg);border-color:var(--color-status-warning);color:var(--color-text-primary)}}
.cb-chip.warning .cb-ms{{color:var(--color-status-warning)}}
.cb-chip.neutral{{background:var(--color-status-neutral-bg);border-color:var(--color-status-neutral);color:var(--color-text-secondary)}}
.cb-chip.good{{background:var(--color-status-good-bg);border-color:var(--color-status-good);color:var(--color-status-good-text)}}
.cb-chip.good .cb-ms{{color:var(--color-status-good)}}
/* status dot */
.cb-dot{{display:inline-flex;align-items:center;gap:6px;{ts('Caption/Regular')}color:var(--color-text-secondary);white-space:nowrap}}
.cb-dot i{{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--color-status-good)}}
.cb-dot.warning i{{background:var(--color-status-warning)}}
.cb-dot.critical i{{background:var(--color-status-critical)}}
.cb-dot.offline i{{background:transparent;border:1.5px solid var(--color-status-neutral)}}
/* trust band */
.cb-band{{display:flex;align-items:center;gap:var(--space-8);padding:var(--space-5);border-radius:var(--radius-lg);
  border-left:3px solid var(--color-status-good);background:var(--color-status-good-bg);flex-wrap:wrap}}
.cb-band.warning{{border-left-color:var(--color-status-warning);background:var(--color-status-warning-bg)}}
.cb-band.critical{{border-left-color:var(--color-status-critical);background:var(--color-status-critical-bg)}}
.cb-band .verdict{{display:flex;align-items:center;gap:var(--space-3);flex:1 1 320px;min-width:0}}
.cb-band .verdict .cb-ms{{font-size:28px;color:var(--color-status-good)}}
.cb-band.warning .verdict .cb-ms{{color:var(--color-status-warning)}}
.cb-band.critical .verdict .cb-ms{{color:var(--color-status-critical)}}
.cb-band .stat b{{display:block;{ts('Heading/H2')}color:var(--color-text-primary)}}
.cb-band .stat span{{{ts('Caption/Regular')}color:var(--color-text-secondary)}}
/* card internals */
.cb-stats{{display:flex;justify-content:space-between;gap:var(--space-3)}}
.cb-stats b{{display:block;{ts('Body/Strong')}color:var(--color-text-primary)}}
.cb-stats span{{{ts('Caption/Regular')}color:var(--color-text-muted)}}
.cb-well{{display:flex;gap:var(--space-2);align-items:flex-start;background:var(--color-bg-raised);
  border-radius:var(--radius-md);padding:var(--space-2) var(--space-3);{ts('Caption/Regular')}color:var(--color-text-secondary)}}
.cb-well .cb-ms{{font-size:16px;color:var(--color-text-muted)}}
.cb-pills{{display:flex;gap:var(--space-4);align-items:center;background:var(--color-bg-raised);
  border-radius:var(--radius-md);padding:var(--space-2) var(--space-3);{ts('Caption/Strong')}color:var(--color-text-secondary)}}
.cb-pills .cb-ms{{font-size:14px;color:var(--color-text-muted);margin-right:6px}}
.cb-iconbox{{width:36px;height:36px;border-radius:50%;background:var(--color-bg-raised);display:flex;
  align-items:center;justify-content:center;flex:none}}
.cb-iconbox .cb-ms{{font-size:18px;color:var(--color-text-secondary)}}
.cb-share{{position:relative;height:12px;margin:6px 0 4px}}
.cb-share .track{{position:absolute;left:0;right:0;top:3px;height:6px;border-radius:3px;background:var(--color-bg-control)}}
.cb-share .fill{{position:absolute;left:0;top:3px;height:6px;border-radius:3px}}
.cb-share .target{{position:absolute;top:0;width:2px;height:12px;background:var(--color-text-primary)}}
.cb-legend{{display:inline-flex;align-items:center;gap:6px;{ts('Caption/Regular')}color:var(--color-text-secondary);margin-right:var(--space-4)}}
.cb-legend i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
.cb-legend i.tick{{width:2px;height:12px;border-radius:0}}
.cb-list{{border:1px solid var(--color-border-default);border-radius:var(--radius-lg);background:var(--color-bg-surface);overflow:hidden}}
.cb-list .item{{display:flex;align-items:center;gap:var(--space-4);padding:var(--space-4) var(--space-5)}}
.cb-list .item + .item{{border-top:1px solid var(--color-border-default)}}
.cb-list .item .txt{{flex:1;min-width:0}}
.cb-list .item .cost{{{ts('Heading/H3')}color:var(--color-text-primary);white-space:nowrap}}
/* native elements, restyled to the tokens */
div[data-testid="stMainBlockContainer"]{{padding:88px var(--space-8) 64px !important}}
/* st.metric(border=True) already pads 13px inside; this tops it up to the tile's 16 x 20 */
div[data-testid="stMetric"]{{background:var(--color-bg-surface);border-radius:var(--radius-lg) !important;
  padding:3px 7px !important}}
div[data-testid="stMetricLabel"] p{{{ts('Caption/Strong')}color:var(--color-text-secondary)}}
div[class*="st-key-card_"]{{background:var(--color-bg-surface);border-radius:var(--radius-lg)}}
h1{{letter-spacing:-0.2px}}
/* sections sit 32px apart, as in the Figma screens */
div[data-testid="stHeading"]:has(h3){{margin-top:16px}}
div[class*="st-key-card_"] div[data-testid="stHeading"]:has(h3){{margin-top:0}}
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
    """Overline (navigation group), H1 title, one-line purpose."""
    md(f'<p class="cb-overline accent" style="margin-bottom:-12px">{esc(group)}</p>')
    st.title(title)
    if sub:
        md(f'<p class="cb-sub" style="margin-top:-12px">{sub}</p>')


def section(title: str, sub: str = "") -> None:
    st.subheader(title)
    if sub:
        md(f'<p class="cb-sub" style="margin-top:-12px">{sub}</p>')


#: priority -> (chip class, icon). The icon is never optional: colour alone is
#: not a severity signal.
_CHIP = {"high": ("critical", "error"), "medium": ("warning", "warning"),
         "low": ("neutral", "info"), "info": ("neutral", "info"),
         "ok": ("good", "check_circle"), "critical": ("critical", "error"),
         "warning": ("warning", "warning"), "good": ("good", "check_circle"),
         "neutral": ("neutral", "info")}


def chip(priority: str, label: str | None = None) -> str:
    cls, ic = _CHIP.get(str(priority).lower(), _CHIP["info"])
    return (f'<span class="cb-chip {cls}">{icon(ic)}'
            f'{esc(label if label is not None else str(priority).capitalize())}</span>')


def dot(status: str, label: str) -> str:
    """status: good | warning | critical | offline."""
    return f'<span class="cb-dot {status}"><i></i>{esc(label)}</span>'


def legend(items: list[tuple[str, str]], tick: bool = False) -> str:
    """[(label, hex)] swatches; tick=True draws the thin target marker instead."""
    return "".join(f'<span class="cb-legend"><i class="{"tick" if tick else ""}" '
                   f'style="background:{c}"></i>{esc(l)}</span>' for l, c in items)


def share_bar(pct: float, target: float | None = None) -> str:
    """The overcooled-share bar with its target marker (Floor card)."""
    target = R.conv("overcooled_target_pct") if target is None else target
    w = max(0.0, min(100.0, pct)) if np.isfinite(pct) else 0.0
    return (f'<div class="cb-share"><div class="track"></div>'
            f'<div class="fill" style="width:{w:.1f}%;background:{overcool_hex(pct)}"></div>'
            f'<div class="target" style="left:{target:.1f}%"></div></div>')


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

    idx = phi["index"]
    stale = HL.last_good(H, as_of, floor)
    worst = int(stale.days_stale.max()) if len(stale) else 0
    if idx >= R.conv("health_good"):
        kind, ic, verdict = "good", "check_circle", "Data is trustworthy today"
    elif idx >= R.conv("health_warn"):
        kind, ic, verdict = "warning", "warning", "Use today's data with care"
    else:
        kind, ic, verdict = "critical", "error", "Today's data is not trustworthy"
    flagged = (f"{phi['faulted']} point-day(s) flagged — see Sensor &amp; data health."
               if phi["faulted"] else "No point-days flagged.")
    sub = ("" if compact else "Everything below is conditional on this band. ") + flagged
    stats = [
        (f"{idx:.0f}/100", "point health",
         "Mean per-point health for the selected day: completeness, less penalties for "
         "frozen, out-of-range and spiking points."),
        (f"{phi['points'] - phi['offline']}/{phi['points']}", "points reporting",
         f"{phi['offline']} offline" if phi["offline"] else "All points online."),
        (f"{phi['completeness']:.0f}%", "complete today",
         f"Target {R.conv('completeness_target'):.0f}% of expected samples per point per day."),
        (f"{worst} d", "longest stale", "Days since the stalest point last reported a good value."),
    ]
    md(f'<div class="cb-band {kind}"><div class="verdict">{icon(ic)}<div>'
       f'<div class="cb-h3">{verdict}</div><div class="cb-cap">{sub}</div></div></div>'
       + "".join(f'<div class="stat" title="{esc(h)}"><b>{v}</b><span>{l}</span></div>'
                 for v, l, h in stats) + "</div>")
    return phi


def stat_card(title: str, items: list[tuple[str, str]]) -> None:
    """A small card of big counts (Floor status' 'Since yesterday')."""
    cells = "".join(f'<div><div class="cb-metric">{esc(v)}</div><div class="cb-cap cb-muted">'
                    f'{esc(l)}</div></div>' for v, l in items)
    md(f'<div style="background:var(--color-bg-surface);border:1px solid var(--color-border-default);'
       f'border-radius:var(--radius-lg);padding:var(--space-4) var(--space-5);height:100%">'
       f'<div class="cb-cap" style="font-weight:500;margin-bottom:8px">{esc(title)}</div>'
       f'<div class="cb-between">{cells}</div></div>')


def card(key: str, edge: str | None = None):
    """A bordered surface card. `edge` paints the 4px identity edge (floor colour)."""
    if edge:
        md(f'<style>.st-key-card_{key}{{box-shadow:inset 4px 0 0 {edge}}}</style>')
    return st.container(border=True, key=f"card_{key}")


def floor_card_body(floor: int, row: dict, lever: str, kind: str = "office") -> str:
    """Floor card (Figma 7:263) without its button. row = one floor_league row."""
    if floor in R.FILE_END:
        status = dot("offline", f"File ends {R.floor_file_end(floor):%d %b %Y}")
    elif row.get("meter_pct", 100) < R.conv("completeness_target"):
        status = dot("warning", f"Meters {row['meter_pct']:.0f}% complete")
    else:
        status = dot("good", "Reporting")
    head = (f'<div class="cb-between" style="align-items:center"><div class="cb-h3">Floor {floor}'
            f'{" · non-office" if kind != "office" else ""}</div>{status}</div>')
    lo = R.conv("comfort_lo")
    if kind == "office":
        oc = row.get("overcooled_pct", np.nan)
        main = (f'<div class="cb-row" style="align-items:baseline"><span class="cb-metric">'
                f'{f"{oc:.0f}%" if np.isfinite(oc) else "—"}</span><span class="cb-cap">cooled '
                f'hours below {lo:g} °C</span></div>{share_bar(oc)}'
                f'<div class="cb-cap cb-muted">Target &lt; {R.conv("overcooled_target_pct"):.0f}%</div>')
    else:
        na = row.get("night_ac_pct", np.nan)
        main = (f'<div class="cb-row" style="align-items:baseline"><span class="cb-metric">'
                f'{f"{na:.0f}%" if np.isfinite(na) else "—"}</span><span class="cb-cap">overnight AC '
                f'vs daytime</span></div><div class="cb-cap cb-muted">Not ranked on overcooling '
                f'or schedule — the AC never switches off.</div>')

    def _f(v, fmt):
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"

    stats = (f'<div class="cb-stats"><div><b>{_f(row.get("eui"), "{:.0f}")}</b><span>kWh/m²·yr</span></div>'
             f'<div><b>{_f(row.get("after_hours_ac_pct"), "{:.1f}%")}</b><span>after-hours AC</span></div>'
             f'<div><b>{_f(row.get("deviation_pct"), "{:+.1f}%")}</b><span>vs baseline</span></div></div>')
    return (f'<div class="cb-stack" style="gap:16px">{head}<div class="cb-stack" style="gap:2px">{main}'
            f'</div>{stats}<div class="cb-well">{icon("lightbulb")}<span>{esc(lever)}</span></div></div>')


def zone_card_body(zone_title: str, status_html: str, chip_html: str, ic: str,
                   temp_label: str, temp: str, temp_big: bool, cold: str, cold_bad: bool,
                   ac: str, rh: str, rec: str) -> str:
    """Zone card (Figma 7:387) without its button."""
    tcls = "cb-hero" if temp_big else "cb-h2"
    tcol = "" if temp_big else "color:var(--color-text-secondary)"
    ccol = "var(--color-status-critical-text)" if cold_bad else "var(--color-text-muted)"
    return (
        f'<div class="cb-stack" style="gap:16px">'
        f'<div class="cb-between"><div class="cb-row" style="gap:10px"><div class="cb-iconbox">{icon(ic)}</div>'
        f'<div class="cb-stack" style="gap:2px"><div class="cb-h3">{esc(zone_title)}</div>{status_html}</div></div>'
        f'{chip_html}</div>'
        f'<div class="cb-between" style="align-items:flex-end"><div class="cb-stack" style="gap:2px">'
        f'<span class="cb-cap cb-muted">{esc(temp_label)}</span><span class="{tcls}" style="{tcol}">{esc(temp)}</span></div>'
        f'<div class="cb-stack" style="gap:2px;align-items:flex-end"><span class="cb-cap cb-muted">Cold hours</span>'
        f'<span class="cb-h2" style="color:{ccol}">{esc(cold)}</span></div></div>'
        f'<div class="cb-pills"><span>{icon("bolt")}{esc(ac)}</span><span>{icon("water_drop")}{esc(rh)}</span></div>'
        f'<div class="cb-stack" style="gap:6px"><p class="cb-overline">Recommended</p>'
        f'<p class="cb-body">{esc(rec)}</p></div></div>')


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


def round_numeric(df: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Round the number columns only (pandas 3 warns on .round over dates)."""
    return df.round({c: digits for c in df.select_dtypes("number").columns})


FAULTS_PERIOD_KEY = "faults_period"


def with_states(reg: pd.DataFrame, show_planned: bool = False) -> pd.DataFrame:
    """A fault register as the FM team sees it: planned-maintenance items hidden,
    saved lifecycle states and session mutes applied (`state`, `is_open`). Every
    page that counts open faults goes through here, so the counts agree."""
    if not len(reg):
        return reg
    if not show_planned:
        reg = reg[~reg.planned]
    reg = W.apply_states(reg, FAULT_STATES)
    if len(reg):
        sm = reg.id.isin(st.session_state.get("muted", set()))
        reg.loc[sm, "state"], reg.loc[sm, "is_open"] = "Muted", False
    return reg


def faults_link(n_open: int) -> None:
    """'All N open faults →', opening the Faults page on the same trailing
    window the count came from (its own default is 30 days)."""
    if st.button(f"All {n_open} open faults", icon=":material/arrow_forward:", type="secondary"):
        st.session_state[FAULTS_PERIOD_KEY] = "Trailing window"
        st.switch_page("views/5_faults.py")


def signed_thb(v) -> str:
    """'-฿2,047' / '+฿310'. The sign must come first: st.metric reads the
    delta's direction from a leading '-', so '฿-2,047' would show as a rise."""
    return f"{'-' if v < 0 else '+'}฿{abs(v):,.0f}"


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
    """Chart chrome from the tokens: recessive grid and axes, muted labels, no frame."""
    if height is not None:                     # None for faceted charts: panels set their own size
        c = c.properties(height=height)
    return (c.configure(background="transparent").configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=C["border/strong"], tickColor=C["border/strong"],
                            labelColor=INK_3, titleColor=INK_2, labelFontSize=12, titleFontSize=12,
                            titleFontWeight=500)
            .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12))


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
    return alt.Chart(df).mark_rule(color=WARN, strokeDash=[4, 3]).encode(
        x="date:T", tooltip=[alt.Tooltip("date:T"), "label:N"])


def file_end_note(floor: int, end=None) -> None:
    """Flag a floor whose source file ends early, wherever it appears."""
    fe = R.floor_file_end(floor)
    if fe is not None and (end is None or pd.Timestamp(end) >= fe.normalize()):
        st.warning(f"**Floor {floor}'s file ends {fe:%Y-%m-%d %H:%M}.** Everything after is "
                   "not metered — never compare it with an earlier period.")


def fmt_weeks(w) -> str:
    return "> 5 years" if w is None or not np.isfinite(w) else f"{w:.0f}"
