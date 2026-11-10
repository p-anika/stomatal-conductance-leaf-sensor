# detection_limit (analysis/detection_limit.py, task T5.4)

## What it does
Every instrument has a floor: below some value it can no longer tell a small signal from "nothing". This script finds that floor for g_s. It asks: **"If a leaf's true g_s were X, would our setup give an answer precise enough to trust?"** It asks this for 60 values of X from 0.001 to 2.0 mol m⁻² s⁻¹.

The answer is a **resolved band**:
- The **lowest** trusted value is the **detection limit**.
- There is also a **highest** trusted value, because very open stomata run into the top of the measurable range.

It then checks the answer against two real objects that should both fail:
- a leaf coated in petroleum jelly, whose stomata are sealed, so g_s ≈ 0;
- a wet reference, which is pure evaporation with no stomata, so r_s ≈ 0.

It writes a figure and prints one summary line.

## How it works
Run it with `python -m analysis.detection_limit` after `aggregate_results`. Add `--synthetic` for test data.

1. **Representative conditions** (`representative_reading`). These are the medians of air temperature, RH, pressure, lux and r_bH over all **valid card_direct rows** in `leaf_gs_results.csv`.
   - Censored rows are included. Their conditions are as real as any other window's, and leaving them out would pick conditions by their result.
   - The r_bH spread is the median over those rows of σ_rb = ln(r_bh_hi / r_bh_lo) / 3.92, the same formula as in aggregate_results.
2. **Sweep** (`sweep`). For each true g_s on the grid (`uncertainty.dl_grid_min_mol` to `g_s_cap_mol`, `dl_grid_points` steps, log spaced):
   1. `forward_leaf_temp` finds the leaf temperature that this true g_s would produce under those conditions. This is the energy balance solved the other way round.
   2. `resolution_at` adds the usual random errors 2000 times and re-solves for g_s each time, exactly as for a real window.
   3. It records the 2.5 / 50 / 97.5 % values, the fraction of usable draws (λE > 0 and r_s > 0), and the relative width = (p97.5 − p2.5) / median.
3. **Resolved** (rule A33). All three conditions must hold:
   - at least 90 % of draws are usable;
   - the relative width is below 1.0;
   - p97.5 is below the cap of 2.0 mol m⁻² s⁻¹, so the upper bound is a real bound.

   `resolved_band` returns the lowest and highest resolved true g_s.

   **This is the same code** that stores `g_s_mol_dl` on every window in T5.1. A test checks that the two agree. The T5.4 number is the limit at *typical* conditions; each window's own limit is at *its* conditions.
4. **Brackets** (`bracket_table`). Each bracket's temperature is solved as if it were a leaf, and should come out **not resolved**:
   - **Real data** (`session_brackets`): the vaseline leaf temperature comes from `card_temp_c` in sessions with `dry_reference_type = vaseline_leaf`. The wet reference comes from `wet_ref_temp_c`. Each is solved at its own window's air, RH, pressure and lux, with the representative r_bH.
   - **Synthetic**: the temperatures that a true g_s of 10⁻⁶ (`synthetic.bracket_dry_g_s_mol`) and of 1000 mol m⁻² s⁻¹ (`bracket_wet_g_s_mol`) would give.
5. **Outputs:**
   - `data/processed/detection_limit_sweep.csv` (one row per grid value);
   - `detection_limit_brackets.csv`;
   - `docs/figures/detection_limit.png`, in three panels: estimate with 95 % band against true g_s, usable fraction, and relative width. The thresholds are dashed and the resolved band is shaded;
   - one printed line. With `--synthetic`, every file name ends in `_synthetic`.

## Why it is done this way
- **Simulate, don't guess:** the limit depends on how errors in leaf temperature, air temperature, humidity, light and r_bH combine through the curved g_s = 1/r_s formula. Simulation handles that curve exactly. A formula for error propagation would only be a straight-line approximation.
- **A band, not one number:** near the top, many draws give r_s ≤ 0 and are held at the cap. The interval then *looks* narrow only because it was cut off at the cap (see the right end of the bottom panel). Rule 3 stops that from counting as precise. So the setup also has an upper limit, and the paper should state both ends.
- **Rule A33 instead of the shorter T5.4 wording:** the T5.4 text names only rules 1 and 2. Without rule 3, values near the cap would be wrongly called resolved (owner decision E, 2026-11-10).
- **Brackets:** they check the simulation against reality. If a real vaseline leaf came out "resolved", the error model would be too optimistic.

## Worked example
Synthetic campaign (invented numbers; test sigmas σ_leaf 0.2 °C, σ_air 0.1 °C, σ_RH 1 %, σ_lux 5 %). Conditions are the medians of 11 card_direct windows: air ≈ 24.0 °C, RH ≈ 50 %, 995 hPa, E_s ≈ 120 W m⁻², r_bH ≈ 35 s m⁻¹.

| True g_s (mol m⁻² s⁻¹) | Leaf temperature (°C) | 2.5 %–97.5 % | Usable draws | Relative width | Resolved |
| --- | --- | --- | --- | --- | --- |
| 0.0799 | 24.97 | 0.040–0.123 | 1.000 | (0.1227 − 0.0401) / 0.0812 = 1.017 | no (width ≥ 1) |
| **0.0908** | 24.83 | 0.051–0.136 | 1.000 | (0.1356 − 0.0506) / 0.0923 = **0.921** | **yes** |
| 1.359 | 20.41 | 1.02–**2.0** | 1.000 | 0.709 | no (upper bound at the cap) |

Result: `lowest resolved g_s 0.0908 mol m-2 s-1; resolved band 0.0908 to 1.19; brackets: petroleum_jelly_leaf unresolved in 1/1, wet_reference unresolved in 1/1`.

Here the width rule is the one that sets the limit. The leaf at 0.09 is only 0.8 °C warmer than at 0.25, so a 0.2 °C camera error is already a large share of the signal.

**Units check:** g_s is in mol m⁻² s⁻¹ throughout. Relative width is (mol m⁻² s⁻¹ − mol m⁻² s⁻¹) / mol m⁻² s⁻¹, which has no units. ✓

**With the current placeholder sigmas** (σ_leaf 1.0 °C, σ_lux 15 %, σ_rb 0.2), **no** true g_s anywhere on the grid is resolved. At r_bH = 40 s m⁻¹ the relative width stays at 2.6 or more wherever the upper bound is below the cap. Wherever the width is smaller, the upper bound has hit the cap, so the interval is open above. The summary line then says "no true g_s … is resolved". This is not a bug. It says that a ±1 °C thermal camera cannot measure g_s with a 95 % interval narrower than the value itself. The real limit can only be stated after the thermal calibration (T7.5).

## Methods paragraph (draft)
The detection limit was estimated by simulation at representative conditions (the medians of all valid primary-method windows). For 60 log-spaced true conductances between 0.001 and 2.0 mol m⁻² s⁻¹, the leaf temperature implied by the energy balance was computed. Measurement errors were then propagated with 2000 Monte Carlo draws, using the calibrated sensor uncertainties. A true conductance was considered resolved when at least 90 % of draws gave a positive latent heat flux and stomatal resistance, the 95 % interval was narrower than the median, and its upper bound lay below 2.0 mol m⁻² s⁻¹. The detection limit was the lowest resolved value. A petroleum-jelly-coated leaf and a wet reference were measured as empirical brackets and were expected to be unresolved. Analyses used Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1 and Matplotlib 3.11.2.

## Limitations
- **Depends on the sigmas.** The limit is only as real as the error sizes put in. Until T7.5 they are placeholders, and nothing is resolved.
- **One set of conditions.** A cooler, more humid or dimmer session has a different limit. That is why each window also carries its own `g_s_mol_dl`.
- **Brackets are solved with the leaf's r_bH and absorptivity.** A wet paper or a coated leaf may absorb light differently and have a different boundary layer (see T5.5), so a bracket that comes out "resolved" may point to that rather than to the error model.
- **Correlated errors:** the shared-camera correlation between card and leaf (A43) is not simulated. That makes card_direct intervals, and so this limit, probably conservative (too high).
- **Grid resolution:** the limit is a grid value. The true threshold lies between it and the grid value below it, about 12 % lower with 60 points.
