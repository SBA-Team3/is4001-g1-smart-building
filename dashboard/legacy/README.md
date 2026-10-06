# Superseded prototype

The original 8-tab analyst's what-if app (`streamlit_app.py` + `f5_core.py`).
It is kept for reference only and is **not** wired into the dashboard.

Everything that survived was ported into `dashboard/core/` and the Opportunities
and Methods pages, with two corrections applied on the way — see the Methods page:

- the baseline training window now stops before the 2019-09-27 metering change
- the occupied window follows the *measured* AC schedule, not an assumed 08:00-18:00

To run the old prototype you would need to restore the cache paths it expects
(`floor_5_Terence/cache/`); the current pipeline writes to `dashboard/cache/`.
