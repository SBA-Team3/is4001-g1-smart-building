# Dashboard design system

The dashboard follows one Figma file. Code and Figma share the same token names,
so a colour or size is decided once and used everywhere.

**Figma:** <https://www.figma.com/design/gDQN41sGJmdXJTwwI6vzTd> (pages: *Read me*,
*Foundations*, *Components*, *Screens*)

| What | Where |
|---|---|
| Tokens (colours, spacing, radii, type) | `dashboard/design/tokens.json`, exported from the Figma variables |
| Streamlit theme (widgets, alerts, metric deltas, fonts) | `.streamlit/config.toml` (repo root) |
| Components as code (chips, dots, cards, trust band, chart chrome) | `dashboard/ui.py` |
| Logo | `dashboard/design/logo.svg`, `logo_mark.svg` (icon from the Figma top bar) |

Run the app from the repo root (`streamlit run dashboard/app.py`); Streamlit only
reads `.streamlit/config.toml` from the folder you start it in.

## Rules

1. **Dark only.** There is no light theme yet. Don't add light colours.
2. **No hex codes in `views/`.** Use the names in `ui.py`: `ui.C["text/primary"]`,
   `ui.INK`, `ui.BAD`, `ui.FLOOR_COLOR`, `ui.END_USE_COLOR`, … `tests/test_design_tokens.py`
   fails if a page hard-codes a colour.
3. **Pink is for the interface, never for data.** `accent/default` (#DA5F8E) marks
   what is selected or clickable: the active nav section, primary buttons, the
   selected trailing window, links. It never colours a chart mark or a floor.
4. **Floor N always has the same colour** (`data/floor-1` … `data/floor-7`), in the
   dashboard and the notebooks. Floor 5's pink is a *data* colour that happens to sit
   near the accent, so a selected floor is shown with a white outline, never a pink one.
5. **Status colours mean status.** `status/critical`, `warning`, `good`, `neutral`
   are only for severity and health, and always come with an icon and a word
   (`ui.chip("high")`, `ui.dot("warning", "…")`). Never use them for "series 4".
6. **Text uses text tokens.** `text/primary` for values, `text/secondary` for
   labels and body copy, `text/muted` for units and footnotes. A coloured mark next
   to the text carries identity, not the text itself.
7. **Missing data is shown as missing.** "—", "no data" or the `NO_DATA` cell fill,
   never 0.

## Tokens

Colours (dark mode). Full list in `design/tokens.json`.

| Token | Hex | Use |
|---|---|---|
| `bg/page` | #121417 | page background |
| `bg/surface` | #1B1E22 | cards, KPI tiles, sidebar |
| `bg/raised` | #24282D | wells inside cards, table headers |
| `bg/control` | #2C3137 | inputs; "no data" cells in charts |
| `border/default` / `border/strong` | #33383F / #4A5058 | card borders / axis lines, secondary buttons |
| `text/primary` / `secondary` / `muted` | #F2F4F5 / #A9B0B8 / #8A929B | see rule 6 |
| `accent/default` / `hover` / `bg/accent-subtle` | #DA5F8E / #E57AA3 / #3A2230 | see rule 3 |
| `status/critical` (+ `-text`, `-bg`) | #D03B3B (#F07A7A, #3A1F21) | High priority, failing |
| `status/warning` (+ `-bg`) | #FAB219 (#3A2F14) | Medium priority, use with care |
| `status/good` (+ `-text`, `-bg`) | #0CA30C (#3FC56B, #173222) | OK, trustworthy |
| `status/neutral` (+ `-bg`) | #8A929B (#24282D) | Low / info, offline |
| `data/floor-1 … 7` | #3987E5 #D95926 #199E70 #C98500 #D55181 #008300 #9085E9 | one colour per floor |
| `data/enduse-ac / light / plug` | = floor-1 / 2 / 3 | AC, lighting, plug load |
| `data/overcool-low / mid / high` | #2E8B57 / #D9C24A / #B0453A | overcooled share, 0 → 40% target → 100% |
| `data/cold` / `data/warm` | = floor-1 / floor-2 | below / above the comfort band |
| `data/zone-1 … 5` | #D55181 #9085E9 #8A929B #199E70 #C98500 | zones (a chart with more than three zones uses small multiples) |

The categorical sets were checked for colour-blind separation on the dark surface
with the dataviz palette validator.

Spacing `space/1…10` = 4, 8, 12, 16, 20, 24, 32, 40 px. Radii `sm` 6, `md` 8 (controls),
`lg` 12 (cards), `full` (chips).

Type is **Inter**: Display/Metric 28/36 semibold (KPI values), H1 24/32, H2 18/26
(section titles, `st.subheader`), H3 15/22 (card titles), Body 14/20, Caption 12/16,
Overline 11/16 uppercase.

## Components → code

| Figma component | In code |
|---|---|
| Top bar | `st.navigation(position="top")` + `st.logo` in `app.py` |
| Context sidebar | `st.sidebar` in `app.py` (floor, as-of date, trailing window, assumptions, data status) |
| Page header | `ui.page_header(group, title, sub)` |
| Section header | `ui.section(title, sub)` |
| KPI tile | `st.metric(..., border=True)`; `delta_color="inverse"` when lower is better |
| Banner | `st.info` / `st.warning` / `st.error`, always with an `icon=` |
| Severity chip | `ui.chip(priority, label)` |
| Status dot | `ui.dot(status, label)` |
| Trust band | `ui.trust_band(floor, as_of)` |
| Floor card | `ui.card(key, edge=ui.FLOOR_HEX[f])` + `ui.floor_card_body(...)` + a tertiary `st.button` |
| Zone card | `ui.card(...)` + `ui.zone_card_body(...)` |
| Fault card | `ui.card(...)` + `st.columns([5, 4, 2])` in `views/5_faults.py` |
| Button | `st.button` / `st.download_button`; `type="primary" / "secondary" / "tertiary"` |
| Select field | `st.selectbox` / `st.date_input` |
| Charts | Altair, wrapped in `ui.chart(c, height)` for the axis and grid tokens |
| Table progress bars | `st.column_config.ProgressColumn(color=ui.ACTUAL, ...)` — the default is the pink accent, which rule 3 forbids for data |

## Changing a token

1. Change the variable in Figma (*Foundations* page).
2. Update `dashboard/design/tokens.json` to match.
3. If it is a theme colour (accent, backgrounds, text, borders, status), update
   `.streamlit/config.toml` too. TOML can't read the JSON, so
   `tests/test_design_tokens.py` checks the two agree.
4. Restart Streamlit; it caches `ui.py` and the theme.

## Where the build differs from Figma

- Streamlit draws its own widgets (selects, segmented control, toggles), so they
  are close to the Figma components but not identical.
- Floor status keeps a "How is the floor running?" KPI row (energy today, peak
  demand, AC share, zones outside band, open faults) that the Figma frame leaves out.
- Every card grid has a table view in an expander ("All floors as a table",
  "Zones as a table").
- Zones, Sensor health, Energy, Opportunities and Methods use the theme, page
  header and chart tokens, but not yet the card layouts from their Figma frames.
