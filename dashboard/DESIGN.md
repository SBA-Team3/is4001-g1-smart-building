# Dashboard design system

The dashboard is organised around what the FM team must do, not what the
analysts proved. Its first questions are **energy, cost and which sensors are
down**, and its main screen is a 1440×900 laptop. Colours, type and components
are decided once here and used everywhere.

| What | Where |
|---|---|
| Tokens (colours, spacing, radii, type) | `dashboard/design/tokens.json` |
| Streamlit theme (widgets, alerts, metric deltas, fonts, toolbar) | `.streamlit/config.toml` (repo root) |
| Components as code (fault types, data-health caption, chips, dots, tiles, chart chrome, point status, period) | `dashboard/ui.py` |
| Logo | `dashboard/design/logo.svg`, `logo_mark.svg` |

Run the app from the repo root (`streamlit run dashboard/app.py`); Streamlit only
reads `.streamlit/config.toml` from the folder you start it in.

## Rules

1. **Dark only.** Light variants are specified below but not built. Don't add them
   to `tokens.json` or `config.toml`.
2. **No hex codes in `views/`.** Use the names in `ui.py`: `ui.C["text/primary"]`,
   `ui.PRIMARY`, `ui.CONTEXT`, `ui.END_USE_COLOR`, `ui.chip()`, …
3. **Cyan is for the interface, never for data.** `accent/default` (#2EB4C8) marks
   selection, focus and the primary button — nothing else. It is never decoration
   and never a chart mark. Page-header overlines are `text/muted`, not accent.
4. **Floors and zones are identified by label, not colour.** The seven floor hues and
   five zone hues are retired, so the dashboard no longer colour-matches the
   notebooks. In a chart, the selected floor or zone is `data/primary` and every
   other one `data/context`. A chart that used to stack floors shows the building
   total with the selected floor highlighted, or small multiples. One exception:
   the Overview's monthly energy chart, with **Bldg** selected, stacks F1–F7 in
   `data/floor-cat-1 … 7` (below).
5. **Status colours mean status, and always come with an icon and a word**
   (`ui.chip("ok")`, `ui.dot("offline", "…")`, glyph text like `● FROZEN`,
   `○ OFFLINE`). They mark data status and target pass/fail only — never alerts. Never use them for "series 4". `status/normal` appears on chips
   and legends only, never as a large fill. `status/offline` is always a hollow
   ring or a dashed outline.
6. **Normal is quiet.** Data health is one plain caption line (Floor, Sensors); there is
   no alert-count band.
7. **Text uses text tokens.** `text/primary` for values, `text/secondary` for
   labels and body copy, `text/muted` for units, footnotes and overlines.
8. **Missing data is shown as missing.** "—", "no data" or the `bg/control` cell
   fill, never 0. A ฿ figure with a zero tariff reads "set rate in Assumptions",
   never ฿0.
9. **One period.** Every page reads the context bar's period (`ui.ctx()` →
   `start`, `end`, `period_label`). A page never adds its own period control; a
   window a model fixes for itself (the baseline's training window) is shown as
   text.
10. **Subtitles carry facts, not explanations.** A page subtitle holds only what the
    reader needs to place the numbers: counts, the date range (`ui.ctx()["range"]` —
    the context bar already names the period), a band or a scope. Section headings and
    overlines are names ("Top alerts", "Floors"). Explanations go in help tooltips.
11. **Floor selector.** `Bldg` is offered only on Overview and Carbon; floor pages offer
    F3–F7 and fall back to the last wired floor (default F5).
12. **A floor past its file end** (`R.FILE_END`): if the whole period is after it, the
    page shows its header and one neutral "No data — …" message and stops
    (`ui.file_end_gate`); if the period overlaps it, one compact note.

## Alerts: fault type, not severity

Alerts are labelled, filtered and summarised by **fault type** (`ui.FAULT_TYPES`,
from the register's `rule`) and ranked by ฿/day, then days open. **Severity is not
shown anywhere**: the rule thresholds are not aligned with the 40% target and some
depend on the period length.

| Type | `rule` | Icon |
|---|---|---|
| Overcooling | `overcooling` | `ac_unit` |
| AC after hours | `after_hours_ac` | `schedule` |
| Lights overnight | `after_hours_light` | `lightbulb` |
| Short cycling | `short_cycling` | `autorenew` |
| Humidity | `humidity` | `water_drop` |
| Dead meter | `dead_meter` | `power_off` |
| Point offline | `point_offline` | `sensors_off` |
| Meter change | `step_change` | `swap_horiz` |

**States.** People set New → Acknowledged → In progress (a work order), or Mute. The
system sets **Cleared** (`ui.with_states`): last seen more than the episode gap (3 days)
before the as-of day. Open = not cleared and not muted; every count uses that. There is
no manual Resolve and no "fixed when" text in the UI; how each type clears is on Methods.

`ui.type_chip(rule, state)` is **neutral** — `bg/control` fill, `border/strong` edge,
`text/secondary`, icon + label — filled while New, an outline once Acknowledged or In
progress. Unmapped rules fall back to the title-cased rule name.

**Point down** = OFFLINE, FROZEN (`flag_flatline`) or BAD SAMPLES (out of range or
spikes) on the as-of day — `ui.point_status()`, the one definition every page uses.
PARTIAL (below the completeness target) is degraded, not down. Past a floor's file
end its points are NO FILE and are not counted.

## Tokens

Full list in `design/tokens.json`. `status/good*` and `status/neutral*` remain as
aliases of `status/normal*` and `status/offline*`.

| Token | Hex | Use |
|---|---|---|
| `bg/page` | #0F1113 | page background |
| `bg/surface` | #171A1D | cards, tiles |
| `bg/raised` | #202428 | wells, table headers (no inputs on it) |
| `bg/control` | #282D32 | inputs; "no data" cells |
| `bg/accent-subtle` | #12343A | selected-row wash |
| `border/default` | #30363C | decorative dividers only |
| `border/strong` | #626C77 | card and input edges |
| `text/primary` / `secondary` / `muted` | #ECEFF2 / #AEB6BF / #8E97A1 | rule 7 |
| `text/on-accent` | #0B0D0F | text on cyan and on filled chips |
| `accent/default` / `hover` | #2EB4C8 / #5CCBDB | rule 3; hover is also `linkColor` |

| Status | Mark | `-text` | `-bg` |
|---|---|---|---|
| `status/critical` | #F0524F | #FF8E8A | #3B1B1C |
| `status/warning` | #F2A516 | #FFC453 | #382B11 |
| `status/normal` | #3FAE6E | #72D198 | #15301F |
| `status/offline` | #8E97A1 | #B9C1CA | #23272C |

`status/normal-fill` #25613F: OK cells in status maps (the Sensors point × day map) —
quieter than `status/normal`, visible on `bg/surface` (2.4:1).

| Data | Hex | Use |
|---|---|---|
| `data/primary` | #4F8DF6 | AC, cold, actual, the selected floor/zone |
| `data/series-2` | #D8C08F | lighting, warm |
| `data/series-3` | #A6CBFB | plug |
| `data/context` | #5A636D | non-selected floors or zones; other configuration |
| `data/seq-1 … 5` | #24406E #3567B8 #4F8DF6 #86B4F8 #BBD7FD | overcooled share, 20-pp steps |
| `data/floor-cat-1 … 7` | #3D84EA #A9C8F8 #7258D2 #C9A6F5 #A3457F #D98CC4 #5568BC | **Overview monthly stack only** (F1 … F7) |

- Expected / baseline lines: `text/secondary`, dashed (`ui.expected_line`).
- Change-point annotations: `text/muted`, dotted (`ui.cps_rules`).
- The point × day health map *is* status: `status/normal-fill` (OK),
  warning (PARTIAL, BAD SAMPLES), critical (FROZEN), `bg/control` with "—" (OFFLINE).
- Overcooled share vs its 40% target is a blue magnitude plus a target tick; pass
  or fail goes in a chip beside it, never in the bar colour.
- Heatmap cell labels clear 4.5:1 on every ramp step: `text/primary` on seq-1/2,
  `text/on-accent` on seq-3/4/5 (`ui.SEQ_INK`).
- `chartCategoricalColors` = `[data/primary, data/series-2, data/series-3, data/context]`.
- `data/floor-cat-*` stays in the blue → violet → magenta arc: no red, amber, green or
  cyan, so it never reads as status or as the accent. A lightness ladder separates
  neighbours. Checked with the dataviz validator on `bg/surface` (dark): adjacent
  ΔE (OKLab×100) normal 20.6 / protan 15.9 / deutan 20.1 / tritan 19.4, all ≥ 3:1. It
  fails the validator's lightness band and chroma floor by design, because the ladder
  is what separates seven hues inside one arc. Non-adjacent pairs are closer (deutan
  ≥ 4.0), so the stack order, the legend and the tooltips carry identity. Min ΔE vs
  status: critical 13.9 (protan), warning 5.9 (tritan), normal 5.8 (tritan).

### Light variants (specified, not built)

| Token | Light |
|---|---|
| `bg/page` / `surface` / `raised` / `control` / `accent-subtle` | #F3F4F6 / #FFFFFF / #EDEFF2 / #E4E7EB / #DCF0F3 |
| `border/default` / `strong` | #D5DAE0 / #848D97 |
| `text/primary` / `secondary` / `muted` | #15181B / #474F58 / #5E6670 |
| `accent/default` / `hover` | #0B7F90 / #096A78 |
| `status/critical` (mark / text / bg) | #D12C2F / #A51C20 / #FCEBEB |
| `status/warning` | #C27C00 / #7F5000 / #FFF2D9 |
| `status/normal` | #1F8A4D / #16673A / #E4F3E9 |
| `status/offline` | #7A838D / #474F58 / #EDEFF2 |
| `data/primary` / `series-2` / `series-3` / `context` | #2F6FDB / #B0956A / #7EA6E6 / #B8C0C8 |

Spacing `space/1…10` = 4, 8, 12, 16, 20, 24, 32, 40 px. Radii `sm` 6, `md` 8 (controls),
`lg` 12 (cards), `full` (chips).

Type is **Inter**: Display/Metric 28/36 semibold (KPI values), H1 24/32, H2 18/26,
H3 15/22, Body 14/20, Caption 12/16, Overline 11/16 uppercase.

## Components → code

| Component | In code |
|---|---|
| Top nav (flat: Overview · Floor · Alerts · Sensors · Comfort · Energy · Carbon · Savings · Methods) | `st.navigation(position="top")` in `app.py` |
| Context bar (floor, as-of date, period, Assumptions popover: tariff, building floor area; on Carbon the popover is replaced by the grid emission factor, key `ef`) | `app.py`, keys `floor_select`, `as_of`, `period`; read via `ui.ctx()` |
| Data-health caption | `ui.health_caption(floor, as_of, extra)` (Floor, Sensors) |
| Page header | `ui.page_header(overline, title, sub_html)` |
| KPI tile | `st.metric(..., border=True)`, or `ui.tile()` when it needs a list or sub-line |
| Fault-type chip | `ui.type_chip(rule, state)`, `ui.state_chip(row)` |
| Status chip | `ui.chip(status, label, hollow=)` — data status and target pass/fail only |
| Status dot | `ui.dot(status, label, hollow=)` |
| Status text in tables | glyph text + `ui.glyph_style(df, cols)` |
| Row → detail | `st.dataframe(on_select=callback)` + `st.dialog` |
| Banner | `st.info` / `st.warning` / `st.error`, always with an `icon=`; `st.info` is a quiet grey note |
| Charts | Altair, wrapped in `ui.chart(c, height)` for axis and grid tokens |

## Changing a token

1. Update `dashboard/design/tokens.json`.
2. If it is a theme colour, update `.streamlit/config.toml` too — TOML can't read the
   JSON, so keep the two in step by hand.
3. Restart Streamlit; it caches `ui.py` and the theme.
