# qc (processing/qc.py, task T3.3)

## What it does
Sensors sometimes give readings that are clearly wrong: a light sensor that suddenly reads 2 lux in the middle of 88-lux readings, or a thermal frame that jumps 10 °C because a hand passed in front of the camera. This module looks at every row of a sensor log and writes short **flags** into a `qc_flags` column that say what looks wrong. It **never deletes or changes a row**. Later steps (T3.4 measurement windows) decide what to do with flagged rows, and the raw file on disk is never touched.

## How it works
1. **`add_qc_flags(df, cfg)`** takes a table (pandas DataFrame) of sensor rows and returns a **copy** with a `qc_flags` column. Several flags on one row are joined with `;`, for example `range_lux;duplicate_timestamp`. An empty string means the row is clean. Flags that were already in the column are kept.
2. It runs four kinds of checks, each only on the columns that exist in the file (so the old 8-column debug log and the new `sensor_log.csv` both work):
   1. **Range checks** → `range_<column>`
   2. **Spike checks** (Hampel filter) → `spike_<column>`
   3. **Overflow** → `sensor_overflow`
   4. **Duplicate timestamps** → `duplicate_timestamp`
3. **`hampel_flags(x, window, n_sigma, mad_to_sigma, min_abs, min_rel)`** is the spike check for one column. It returns True for each row that is a spike.

## Every rule and its threshold

All numbers live in `config.yaml → qc`. None is written inside the code.

### 1. Range checks
A value outside these limits is impossible, or impossible for this experiment. The limits are **inclusive**: exactly 0 % RH or exactly 50 °C passes.

| Column | Allowed range | Why this range | Source |
| --- | --- | --- | --- |
| air_temp_c | −10 to 50 °C | An indoor or greenhouse soybean experiment never leaves this range; outside it the sensor is faulty. | spec T3.3 |
| humidity_pct | 0 to 100 % | Relative humidity can't be negative or above 100 %. | spec T3.3 |
| pressure_hpa | 900 to 1100 hPa | Air pressure near sea level stays well inside this (the debug log reads about 995). | spec T3.3 |
| lux | 0 or more | Light can't be negative. No upper limit (sunlight is very bright). | spec T3.3 |
| thermal columns: thermal_min_c, thermal_max_c, thermal_mean_c, leaf_temp_c, leaf_temp_median_c, card_temp_c, wet_ref_temp_c | −10 to 80 °C | A living leaf, a card or the room can't be outside this; a camera value outside it is a read error. | spec T3.3 |

- **Infinity** (`inf`) counts as out of range.
- **A missing value (NaN) is not a range failure.** Missing data is a different problem, handled by the next steps (as `missing_input` in T1.4), not hidden under a range flag.

### 2. Spike checks (Hampel filter)
A **spike** is a single reading (or two or three) that is far from its neighbours and then the readings go back to normal. It is checked only on these columns: `air_temp_c`, `humidity_pct`, `pressure_hpa`, `lux`, `thermal_min_c`, `thermal_max_c`, `thermal_mean_c`.

For each row:
1. Take a **window** of 7 rows centred on it (3 before, the row, 3 after). At the start and end of the file the window is shorter; it needs at least 4 real values.
2. Find the window's **median** m (the middle value).
3. Find the **MAD**, the median absolute deviation: the median of |each value − m|. It measures the normal spread, but one or two wild values barely change it (unlike the standard deviation).
4. Turn the MAD into a standard-deviation-like spread: σ ≈ 1.4826 × MAD.
5. **Flag the row if |value − m| is larger than all three of:**
   - 3 × σ (the usual "3-sigma" rule), and
   - a **minimum absolute size** for that column, and
   - a **minimum relative size** (a fraction of m), used for lux.

| Setting | Value | Why |
| --- | --- | --- |
| hampel_window | 7 rows (35 s at 5 s per row) | spec T3.3. Short enough that a real change in the plant (minutes) is not mistaken for a spike, long enough to have a stable median. |
| hampel_n_sigma | 3 | spec T3.3. Normal noise goes past 3σ only about 0.3 % of the time. |
| mad_to_sigma | 1.4826 | Derived below. |
| spike_min_abs: air_temp_c | 0.5 °C | **Placeholder** (owner, 2026-11-09). About the BME280's stated accuracy class; smaller jumps are not worth flagging. |
| spike_min_abs: humidity_pct | 3 % RH | **Placeholder**. Same as the humidity sigma placeholder (uncertainty.sigma_rh_pct). |
| spike_min_abs: pressure_hpa | 1 hPa | **Placeholder**. Pressure barely matters to g_s (only through γ); 1 hPa changes γ by 0.1 %. |
| spike_min_abs: thermal_min_c / max / mean | 1 °C | **Placeholder**. Same as the camera sigma placeholder (uncertainty.sigma_leaf_temp_c). |
| spike_min_rel: lux | 0.2 (20 % of the median) | **Placeholder**. Light readings scale with brightness, so a fixed number of lux makes no sense; 20 % is well above normal flicker. |

**Why the minimum sizes are needed.** When readings barely change, the MAD is tiny, so 3σ is tiny too, and ordinary noise gets flagged. In the debug log, pressure moves in steps of 0.01–0.05 hPa (about the sensor's resolution); the MAD there is 0.01 hPa, so a 0.05 hPa step was flagged as a "spike". The same happened to a 2 % lux change and a 0.5 °C change in thermal_max. Since T3.4 leaves flagged rows out of the measurement windows, those false flags would throw away good data. The minimum size stops that.

**Why there are no spike checks on leaf_temp_c, card_temp_c or wet_ref_temp_c** (owner decision, 2026-11-09). These are the measured results. Removing rows because the leaf or card temperature itself looks unusual would be **selecting data on the measured result**, which can bias g_s and r_b (for the card this is exactly what T1.3 forbids). An object in front of the camera shows up in the whole-frame summaries (thermal_min/max/mean), which are checked; the range check still applies to the ROI temperatures.

**Why a step is not a spike.** When a lamp is switched on, the leaf warms and stays warm. Once more than half the window is at the new level, the median moves to the new level, so the step itself is not flagged (tested).

#### Where 1.4826 comes from (derivation, not recalled)
For normally distributed data with spread σ, half the values lie within ±q·σ of the middle, where q is the 75th percentile of the standard normal distribution: Φ(q) = 0.75, so q = Φ⁻¹(0.75) = 0.6745. The median of |x − m| is therefore 0.6745 σ, so

  σ = MAD / 0.6745 = **1.4826 × MAD**.

Units: the MAD has the units of the data (°C, %, hPa, lux), and 1.4826 has none, so σ has the data's units ✓. A test checks the config value against `1 / scipy.stats.norm.ppf(0.75)`.

### 3. Overflow → `sensor_overflow`
The TSL2591 light sensor counts light in two raw channels (`tsl_full_spectrum`, `tsl_infrared`). If a channel is at its maximum count, the sensor is saturated and the lux value is wrong (too low). Flag if either channel is **≥ 65535** (the largest 16-bit number), or if lux is ±infinity.

**Open item (owner, 2026-11-09):** the real saturation count depends on the integration time and may be lower than 65535. Check the limit used by the installed `adafruit_tsl2591` library on the Pi and put that value in `qc.overflow_count`. The old debug log has no raw channels, so overflow can't be checked there; whether the two 1.98-lux readings were an overflow glitch is unknown (they are caught as spikes).

### 4. Duplicate timestamps → `duplicate_timestamp`
Two rows with exactly the same timestamp mean the logger wrote a row twice, or the clock jumped. **Every copy** is flagged, because there is no way to know which one is right. The column checked is the first of `timestamp_iso` (new log) or `timestamp` (old log) that exists.

## Why it is done this way
- **Flag, never delete** (spec B12). Raw data stays as it was recorded. Every decision to leave a row out can be seen, counted and reversed.
- **Hampel instead of mean ± 3 SD.** One wild value inflates the standard deviation so much that it can hide itself. The median and MAD are not pulled by it.
- **Centred window.** A row is compared with readings before and after it, so a short dip in the middle is caught, and a lasting change is not.

## Worked example (the first debug log)
File `data/raw/debug_logs/first_sensor_log_(old_combined_logger).csv`, 10 rows. File line numbers (the header is line 1):

| Line | What happened | Check | Numbers | Flag |
| --- | --- | --- | --- | --- |
| 6 | thermal frame jumped (mean 37.61 °C) | spike, thermal_mean_c | window median 27.47, MAD 0.15 → 3σ = 0.67; limit max(0.67, 1.0) = 1.0; deviation 10.14 > 1.0 | `spike_thermal_min_c;spike_thermal_max_c;spike_thermal_mean_c` |
| 8 | humidity 68.63 % (probably breath) | spike, humidity_pct | median 62.22, MAD 0.86 → 3σ = 3.82; limit max(3.82, 3.0) = 3.82; deviation 6.41 > 3.82 | `spike_humidity_pct` |
| 9 | lux 1.98 among ≈ 88–90 | spike, lux | median 89.01, MAD 0.96 → 3σ = 4.27; limit max(4.27, 0.2 × 89.01 = 17.8) = 17.8; deviation 87.0 > 17.8 | `spike_lux` |
| 10 | lux 1.98 again | spike, lux | window shrinks at the end (5 values): median 89.56; limit 17.9; deviation 87.6 | `spike_lux` |
| 8 | pressure 994.83 | spike, pressure_hpa | 3σ = 0.044 hPa but minimum 1 hPa; deviation 0.05 | none (would be a false flag without the minimum) |
| 5 | lux 90.49 | spike, lux | 3σ = 1.16 but minimum 17.7; deviation 1.87 | none |

The other six rows get no flag. All 10 rows and every value come back unchanged (tested).

## Methods paragraph (draft)
Sensor logs were screened with automated quality-control flags; no observation was deleted. Values outside physically plausible ranges (air temperature −10 to 50 °C, relative humidity 0–100 %, pressure 900–1100 hPa, illuminance ≥ 0, thermal-camera temperatures −10 to 80 °C) were flagged. Spikes in the environmental sensors and in whole-frame thermal statistics were flagged with a centred Hampel filter (7 samples, 35 s), marking samples deviating from the window median by more than three robust standard deviations (1.4826 × median absolute deviation) and by more than a minimum size per variable ([0.5 °C, 3 % RH, 1 hPa, 20 % of illuminance, 1 °C]). Spike screening was not applied to leaf or reference-card temperatures, to avoid selecting data on the measured response. Light-sensor saturation and duplicate timestamps were also flagged. Software: Python 3.13, NumPy, pandas [versions].

## Limitations
1. **The minimum spike sizes are placeholders.** They should be set from the measured noise of each sensor after calibration (T7.5); until then they are educated guesses.
2. **A real fast change can be flagged.** The humidity jump at line 8 may be real (someone breathing near the sensor). It is flagged on purpose (owner decision), because it would distort VPD, but a genuine short event is treated the same as a glitch.
3. **More than 3 bad rows in a row are not spikes.** In a 7-row window, 4 or more wrong values move the median, so a long fault (for example a sensor stuck for a minute) is not caught by the spike check. Range checks and later steady-state checks (T3.4) must catch it.
4. **The overflow limit 65535 may be too high** (see the open item). Until it is checked, a saturated light sensor could go unflagged.
5. **Leaf and card temperatures are only range-checked.** A short glitch in the leaf ROI temperature is not flagged; T3.4's steady-state check and the thermal-frame spike flags are the protection.
6. **Window in rows, not seconds.** If rows are missing (gaps), the 7-row window covers more than 35 s.
