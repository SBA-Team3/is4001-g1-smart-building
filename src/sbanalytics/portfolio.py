"""
Building-level (portfolio) view: every floor side by side, read-only.

This is the page an FM lead opens first, so it reads the whole building
whether or not a floor is wired into the operational pages. Two rules keep it
honest:

* Floor 1 is not an office floor. It stays in building totals (its energy is
  real) but is left out of every schedule or overcooling ranking.
* Floor 6's file ends on 2019-10-22. After that date it is "not metered", so
  building totals for those days exclude it and say so.

Cross-floor structural KPIs come from data/derived/ (notebooks 01-02), so the
dashboard and the notebooks quote the same numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import analysis as A
from . import anomaly as AN
from . import baseline as B
from . import config as C
from . import data as D
from . import kpis as K
from . import registry as R


def derived(name: str) -> pd.DataFrame:
    """A cross-floor table the notebooks wrote to data/derived/."""
    return pd.read_csv(C.DATA_DERIVED / f"{name}.csv")


def office_floors() -> list[int]:
    return [f for f in sorted(R.FLOORS) if f not in R.NON_OFFICE_FLOORS]


# --------------------------------------------------------------------------- #
# window KPIs
# --------------------------------------------------------------------------- #

def drop_after_file_end(d: pd.DataFrame) -> pd.DataFrame:
    """Rows past a floor's file end are "no file", not a meter outage."""
    keep = pd.Series(True, index=d.index)
    for f in R.FILE_END:
        keep &= ~((d.floor == f) & (d.ts >= R.floor_file_end(f)))
    return d[keep]


def completeness(h: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Per floor: share of zone-hours with usable metering, and with usable
    indoor temperature on sensored zones. Hours past a floor's file end are left
    out — they are "no file", flagged separately, not a logger outage."""
    d = drop_after_file_end(D.slice_hourly(h, start, end))
    rows = []
    for f, g in d.groupby("floor"):
        sens = g[g.zone.isin(R.sensored_zones(int(f)))]
        rows.append({"floor": int(f), "meter_pct": g.kwh_ok.mean() * 100,
                     "sensor_pct": sens.temp_ok.mean() * 100 if len(sens) else np.nan})
    return pd.DataFrame(rows).set_index("floor")


def building_kpis(h: pd.DataFrame, start=None, end=None, tariff: float | None = None,
                  ef: float | None = None) -> dict:
    """Headline numbers for the whole building over a window."""
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    ef = R.conv("co2_kg_per_kwh") if ef is None else ef
    d = D.slice_hourly(h, start, end)
    kwh = float(d[["ac_kwh", "light_kwh", "plug_kwh"]].sum().sum())
    oc = K.overcooled_share(K.cooled_occupied(d[d.floor.isin(office_floors())]), by=())
    comp = completeness(h, start, end)
    truncated = [f for f in R.FILE_END
                 if end is None or pd.Timestamp(end) >= R.floor_file_end(f).normalize()]
    return {
        "kwh": kwh, "thb": kwh * tariff, "tco2e": kwh * ef / 1000,
        "overcooled_pct": float(oc.share_below.iloc[0] * 100) if len(oc) else np.nan,
        "overcooled_hours": int(oc.hours.iloc[0]) if len(oc) else 0,
        "meter_pct": float(drop_after_file_end(d).kwh_ok.mean() * 100) if len(d) else np.nan,
        "sensor_pct": float(comp.sensor_pct.mean()) if len(comp) else np.nan,
        "days": d.date.nunique(), "truncated_floors": truncated,
    }


def monthly_energy(h: pd.DataFrame) -> pd.DataFrame:
    """kWh per floor per month. Months with no metered hours stay out — a floor
    whose file has ended has no bar, not a zero bar."""
    mo = (h.assign(month=h.ts.dt.to_period("M").dt.to_timestamp())
          .groupby(["floor", "month"])[["ac_kwh", "light_kwh", "plug_kwh"]].sum(min_count=1)
          .sum(axis=1, min_count=1).rename("kWh").reset_index())
    return mo.dropna(subset=["kWh"])


# --------------------------------------------------------------------------- #
# floor league
# --------------------------------------------------------------------------- #

def weather_normalised_deviation() -> pd.DataFrame:
    """Jul-Dec 2019 actual vs the 2018-trained daily baseline, per floor.

    Read from notebook 02's chosen model (total kWh, test period). G14's NMBE is
    (actual - expected) / actual; the FM-facing number is relative to expected.
    """
    c = derived("baseline_chosen")
    c = c[c.target == "total_kwh"].set_index("floor")
    nmbe = c.nmbe / 100
    return pd.DataFrame({"deviation_pct": nmbe / (1 - nmbe) * 100,
                         "baseline_meets_g14": c.meets_g14.astype(bool),
                         "baseline_cv_rmse": c.cv_rmse})


def meter_outages(H: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Floor-level meter outage episodes from the point-day health table.

    A day counts when at least `min_share` of a floor's energy points received
    nothing; days closer together than `episode_gap_days` join one episode, and
    only episodes longer than that gap are kept, so a one-day logger blip is not
    a status note.
    """
    gap = R.conv("episode_gap_days")
    e = H[H.channel.isin(["ac", "light", "plug"])]
    day = e.assign(off=e.received == 0).groupby(["floor", "day"]).off.mean()
    day = day[day >= min_share].reset_index()
    rows = []
    for f, g in day.groupby("floor"):
        g = g.sort_values("day")
        ep = (g.day.diff().dt.days.fillna(1) > gap).cumsum()
        for _, x in g.groupby(ep):
            n = x.day.nunique()
            fe = R.floor_file_end(int(f))
            if n <= gap or (fe is not None and x.day.min() >= fe.normalize()):
                continue                       # past the file end is "no file", not an outage
            rows.append({"floor": int(f), "start": x.day.min(), "end": x.day.max(), "days": n,
                         "share_offline": x.off.mean() * 100})
    return pd.DataFrame(rows, columns=["floor", "start", "end", "days", "share_offline"])


def status_note(floor: int, comp_row=None, outages: pd.DataFrame | None = None) -> str:
    """The caveat an FM needs next to a floor's row, generated from the registry
    and the data rather than typed per floor."""
    notes = []
    if floor in R.NON_OFFICE_FLOORS:
        notes.append("Not an office floor — in building totals, excluded from schedule and "
                     "overcooling rankings")
    fe = R.floor_file_end(floor)
    if fe is not None:
        notes.append(f"File ends {fe.date()} — no data after; never compare F{floor} past it")
    if outages is not None and len(outages):
        o = outages[outages.floor == floor]
        if len(o):
            notes.append("Meter outages " + ", ".join(
                f"{r.start:%d %b %y}–{r.end:%d %b %y} ({r.days} d)" for r in o.itertuples()))
    if comp_row is not None and comp_row.get("meter_pct", 100) < R.conv("completeness_target"):
        notes.append(f"Meters {comp_row['meter_pct']:.0f}% complete over the record")
    if not R.sensored_zones(floor):
        notes.append("No indoor sensors")
    return "; ".join(notes) if notes else "—"


def night_ac_ratio(h: pd.DataFrame) -> pd.Series:
    """Per floor: mean AC in 01-04 h as % of mean AC in workday 08-16 h. Near 100%
    means the AC never switches off — the signature of a non-office floor."""
    night = h[h.hour.between(1, 4)].groupby(["floor", "ts"]).ac_kwh.sum().groupby("floor").mean()
    day = h[h.is_workday & h.hour.between(8, 16)].groupby(["floor", "ts"]).ac_kwh.sum().groupby("floor").mean()
    return night / day * 100


def floor_league(h: pd.DataFrame, area_m2: float | None = None,
                 H: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per floor, whole record. Office floors sorted by overcooled share,
    worst first; non-office floors last."""
    area_m2 = R.conv("floor_area_m2") if area_m2 is None else area_m2
    k = derived("floor_kpis").set_index("floor")
    dev = weather_normalised_deviation()
    comp = completeness(h)
    t = pd.DataFrame(index=sorted(R.FLOORS))
    t.index.name = "floor"
    t["eui"] = k.kwh_per_day * 365 / area_m2
    t["ac_share_pct"] = k.ac_share * 100
    t["overcooled_pct"] = k.overcooled_share * 100
    t["after_hours_ac_pct"] = k.after_hours_ac_share * 100
    t["deviation_pct"] = dev.deviation_pct
    t["baseline_meets_g14"] = dev.baseline_meets_g14
    t["meter_pct"] = comp.meter_pct
    t["sensor_pct"] = comp.sensor_pct
    t["night_ac_pct"] = night_ac_ratio(h)
    t["office"] = [f not in R.NON_OFFICE_FLOORS for f in t.index]
    for f in R.NON_OFFICE_FLOORS:          # never rank a non-office floor on office KPIs
        t.loc[f, ["overcooled_pct", "after_hours_ac_pct"]] = np.nan
    out = meter_outages(H) if H is not None else None
    t["status"] = [status_note(f, comp.loc[f].to_dict() if f in comp.index else None, out)
                   for f in t.index]
    t = t.reset_index()
    return t.sort_values(["office", "overcooled_pct"], ascending=[False, False],
                         na_position="last").reset_index(drop=True)


def overcooled_by_month(h: pd.DataFrame) -> pd.DataFrame:
    """Floor x month overcooled share — the 'where do I send the team' view.

    Office floors with sensors only. Months with no valid cooled hours (the
    building-wide sensor gap, F6 after its file ends) come back as NaN rows so
    the heatmap shows them as gaps, not as zero.
    """
    d = K.cooled_occupied(h[h.floor.isin(office_floors())])
    d = d.assign(month=d.ts.dt.to_period("M").dt.to_timestamp())
    s = K.overcooled_share(d, by=("floor", "month"))
    months = pd.period_range(D.RECORD_START, D.RECORD_END, freq="M").to_timestamp()
    floors = [f for f in office_floors() if R.sensored_zones(f)]
    grid = pd.MultiIndex.from_product([floors, months], names=["floor", "month"]).to_frame(index=False)
    out = grid.merge(s, on=["floor", "month"], how="left")
    out["overcooled_pct"] = out.share_below * 100
    return out


# --------------------------------------------------------------------------- #
# the one-sentence "what to look at"
# --------------------------------------------------------------------------- #

def setpoint_range(est: pd.DataFrame) -> tuple[float, float, int] | None:
    """(low, high, n) %/degC saving magnitude from the n estimators that pass
    notebook 06's reliability rule. None when no estimator is usable."""
    keep = []
    for _, e in est[est.estimator != "Midpoint"].iterrows():
        width = abs(e.hi - e.lo)
        if np.isfinite(width) and e.pct_per_c < 0 and width <= R.conv("estimator_ci_max_ratio") * abs(e.pct_per_c):
            keep.append(-e.pct_per_c)
    return (min(keep), max(keep), len(keep)) if keep else None


def floor_levers(h: pd.DataFrame) -> pd.DataFrame:
    """Annualised kWh for the levers this data can size, per office floor.

    * +1 degC zone-temperature lift — notebook 06's estimator range
    * weekend/holiday AC lockout — metered excess above each floor's night base
    * workday AC run-on outside the schedule — metered excess
    Plus the worst comfort zone, so the sentence can name a place to walk to.
    """
    fh = B.floor_hourly(h)
    fd = B.floor_daily(fh)
    year = 365 / fd[fd.ok].groupby("floor").size()
    ah = AN.after_hours_ac(fh)
    if len(ah):
        ah["off_day"] = ah.day_type.isin(["weekend", "holiday", "closure"])
    co = K.cooled_occupied(h)
    zs = K.overcooled_share(co, by=("floor", "zone"))
    rows = []
    for f in office_floors():
        hf = h[h.floor == f]
        if not R.comfort_zones(f):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = A.estimator_range(A.fit_physics(hf, floor=f), A.run_regression(hf, floor=f))
        rng = setpoint_range(est)
        base = A.setpoint_saving(hf, pct_per_c=-1, d_temp=1, floor=f).kwh_year.sum()
        a = ah[ah.floor == f] if len(ah) else ah
        z = zs[zs.floor == f].sort_values("share_below", ascending=False)
        rows.append({
            "floor": f,
            "setpoint_kwh_lo": base * rng[0] / 100 if rng else np.nan,
            "setpoint_kwh_hi": base * rng[1] / 100 if rng else np.nan,
            "n_estimators": rng[2] if rng else 0,
            "lockout_kwh": a[a.off_day].excess_kwh.sum() * year.get(f, np.nan) if len(a) else 0.0,
            "runon_kwh": a[~a.off_day].excess_kwh.sum() * year.get(f, np.nan) if len(a) else 0.0,
            "worst_zone": z.zone.iloc[0] if len(z) else None,
            "worst_zone_pct": z.share_below.iloc[0] * 100 if len(z) else np.nan,
        })
    return pd.DataFrame(rows).set_index("floor")


def what_to_look_at(floor: int, league_row: dict, levers: pd.DataFrame,
                    tariff: float | None = None) -> str:
    """One generated sentence per floor. Data problems outrank energy levers,
    because a lever sized on a broken meter is not a lever."""
    tariff = R.conv("tariff_thb") if tariff is None else tariff
    if floor in R.NON_OFFICE_FLOORS:
        return (f"Not an office profile — overnight AC runs at {league_row['night_ac_pct']:.0f}% of its "
                "daytime level, so there is no schedule to trim: confirm what the space is and whether "
                "24/7 cooling is required before setting any target.")
    lead = ""
    if league_row.get("meter_pct", 100) < R.conv("completeness_target"):
        lead = f"Data first — meters only {league_row['meter_pct']:.0f}% complete. "
    if floor not in levers.index:
        return lead + "No comfort-zone data to size a lever."
    lv = levers.loc[floor]
    cands = {
        "the AC calendar (weekend/holiday runs)": lv.lockout_kwh,
        "AC run-on outside the workday schedule": lv.runon_kwh,
        "a +1 °C zone-temperature lift": np.nanmean([lv.setpoint_kwh_lo, lv.setpoint_kwh_hi]),
    }
    cands = {k: v for k, v in cands.items() if np.isfinite(v) and v > 0}
    if not cands:
        return lead + "No lever this data can size."
    best = max(cands, key=cands.get)
    kwh = cands[best]
    if best.startswith("a +1") and lv.n_estimators > 1:
        size = (f"≈{lv.setpoint_kwh_lo:,.0f}–{lv.setpoint_kwh_hi:,.0f} kWh/yr, "
                f"฿{lv.setpoint_kwh_lo * tariff:,.0f}–{lv.setpoint_kwh_hi * tariff:,.0f}")
    elif best.startswith("a +1"):
        size = (f"≈{kwh:,.0f} kWh/yr from a single usable estimator, ฿{kwh * tariff:,.0f} — "
                "treat as indicative")
    else:
        size = f"≈{kwh:,.0f} kWh/yr metered, ฿{kwh * tariff:,.0f}"
    where = (f"; start in {lv.worst_zone} ({lv.worst_zone_pct:.0f}% of its cooled hours below "
             f"{R.conv('comfort_lo'):g} °C)" if lv.worst_zone else "")
    tail = ""
    fe = R.floor_file_end(floor)
    if fe is not None:
        tail = f" Sized on data to {fe.date()} only."
    return f"{lead}Biggest lever: {best} ({size}){where}.{tail}"
