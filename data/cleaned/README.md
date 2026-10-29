# data/cleaned — the shared, cleaned dataset

Built from `data/raw/` + `data/external/` by:

```bash
python -m sbanalytics.pipeline          # ~2 min, all 7 floors, deterministic
```

Do not edit these files by hand — change the rules in `src/sbanalytics/` and rebuild.
Load them with `sbanalytics.data` (e.g. `D.load_hourly(floor=None)` for the whole building).

## Files

| File | Grain | Rows (approx.) | Committed |
|---|---|---|---|
| `hourly_zone.parquet` | floor × zone × hour | 435k | yes |
| `hourly_point.parquet` | raw meter column × hour (long) | 2.5M | yes |
| `daily_zone.parquet` | floor × zone × day | 18k | yes |
| `point_day_health.parquet` | raw meter column × day, computed **before** cleaning | 105k | yes |
| `weather_hourly.parquet` | hour (ERA5, Bangkok local time) | 13k | yes |
| `day_types.parquet` | floor × day | 3.8k | yes |
| `cleaning_log.csv` | floor × column × rule: every minute changed | — | yes |
| `build_summary.csv` | floor: coverage, raw vs clean energy by end use | 7 | yes |
| `column_registry.csv` | every raw column: zone, channel, unit, role, bounds, note | 192 | yes |
| `minute/floor{N}.parquet` | cleaned 1-minute grid | 790k per floor | **no** (gitignored, ~115 MB) |

## `hourly_zone.parquet`

| Column | Meaning |
|---|---|
| `ts` | hour start, Bangkok local time (tz-naive) |
| `floor`, `zone` | 1–7, `z1`…`z5` |
| `ac_kwh`, `light_kwh`, `plug_kwh` | energy in the hour (Σ kW / 60 over received minutes). NaN = not metered or nothing received — **never** a silent 0 |
| `temp_c`, `rh_pct` | hourly mean of the zone's S1 sensor |
| `lux_idx` | hourly mean of the "lux" channel — **a scaled index, not illuminance** |
| `minutes_present` | minutes in which all of the zone's primary meters reported |
| `kwh_ok` | `minutes_present ≥ 48` — use for models |
| `temp_minutes`, `temp_ok`, `sensor_alive` | sensor minutes; `temp_ok` = ≥ 30 min (quality gate for means) |
| `hour`, `dow`, `month`, `date`, `is_weekend` | calendar |
| `is_holiday`, `holiday_name` | from `data/external/thai_holidays_2018_2019.csv` |
| `day_type` | `workday` / `weekend` / `holiday` / `closure` / `holiday_worked` (closures and worked holidays detected from each floor's load) |
| `is_workday` | `workday` or `holiday_worked` |
| `out_temp`, `out_rh`, `out_dew`, `out_solar`, `out_cloud` | ERA5 weather, converted UTC → local; solar shifted to hour-start labelling |
| `out_temp_3h`, `out_temp_24h` | rolling means of outdoor temperature |

## What is and is not in the data (read before analysing)

* **Indoor sensors (temp/RH/lux) are missing on every floor from ~12 Sep 2018 to 6 Mar 2019.** Comfort analysis is possible only for Jul–mid-Sep 2018 and Mar–Dec 2019.
* **F6 ends on 2019-10-22 13:27** (truncated source file). Its later hours are NaN.
* **F3 meters are offline 11 Oct – 21 Nov 2018, F4 29 Oct – 21 Nov 2018**; many AC meters are missing 23 Jan – 12 Feb 2019.
* **F1 is not an office floor** (evening-peaking, AC never off).
* Dead / residual / constant meters are tagged by `role` in `column_registry.csv` (e.g. F2 `z2_AC11`, F7 `z1_AC1` are dead; on F3/F4/F5/F7 zone 1's real AC load is on `AC4`).
