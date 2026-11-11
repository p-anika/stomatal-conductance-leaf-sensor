# aggregate_results (analysis/aggregate_results.py, task T5.1)

## What it does
This is the script that makes the main results table. For every one-minute measurement window on every leaf, it works out stomatal conductance (g_s) **three times**, once for each way of getting the boundary-layer resistance r_bH:

| Method | Role | Where r_bH comes from |
| --- | --- | --- |
| card_direct | **primary**: all headline results | the dry reference card in the same photo frame (eq. 9) |
| ml_corrected | secondary: reported only if it passes validation | flat plate × the machine-learning correction f (eq. 11) |
| flat_plate | baseline: a comparison | textbook flat-plate formula from leaf size and wind (eq. 10) |

Each g_s gets a 95 % interval, a "resolved" yes/no and a detection limit. When a wet reference and a petroleum-jelly leaf are in view, it also adds the Jones index. Everything goes into `data/processed/leaf_gs_results.csv`, one row per window per method.

**Two rules matter most:**
- **No row is ever silently missing.** If a method can't run, its row is still written, marked invalid, with a reason code (for example `no_wind`).
- **Censored results stay in.** "Closed stomata" (`below_detection`) and "too open to measure" (`above_range`) rows are kept with their label and their detection limit. Dropping them would make drought-stressed plants look like they transpire more than they do.

**The method roles are fixed now, before any real data exist** (owner decision 2026-11-10). card_direct is the primary method. It must not be swapped for another method after seeing which one gives the "nicer" result, because that would be choosing the analysis by its outcome.

## How it works
Run it with `python -m analysis.aggregate_results`. Add `--synthetic` to use only synthetic test sessions and write `leaf_gs_results_synthetic.csv` instead. Without `--synthetic`, synthetic sessions are skipped, so the two kinds can never mix.

1. **`aggregate(sessions, processed_dir, cards, cfg, cal, model)`** loops over sessions and calls `session_results` on each. Rows are sorted by session, then window, then role (primary, secondary, baseline).
2. **Leaf windows.** `build_windows` (T3.4) cuts the sensor log into 60 s windows and judges steady state on the leaf temperature. It also joins photo geometry (d_min_mm, solidity, …) by time from `data/processed/<session>/photo_features.csv`.
3. **Card windows** (`card_windows`). These are only made when the session's `dry_reference_type` is `card` and it has a `card_id`. The same windowing is applied to `card_temp_c`, with steady state judged on the card. Each steady card window goes through `solve_card_rbh` (eq. 9, with the exogenous light gate) and `card_rbh_uncertainty` (a Monte Carlo giving the 2.5 / 97.5 % r_bH interval).
4. **Matching** (`match_card`). For each leaf window, the code picks the nearest *valid* card window whose centre is within `windows.card_match_minutes` (10 min). If none is valid, the nearest card window's reason is copied (for example `low_card_signal`). If there is no card at all, the reason is `missing_input`.
5. **Per method, in this order of checks:**
   - **card_direct.** Takes the window's reason, except `no_photo_in_window`, which is ignored because card_direct does not need a photo. Then the card match. It does **not** need wind.
   - **flat_plate.** Window reason → `no_wind` if wind is missing → `no_photo_in_window` if there is no leaf size → r_bH = flat-plate value.
   - **ml_corrected.** Same checks as flat_plate. Then, if no model file exists, `missing_input` plus flag `ml_model_missing`. Otherwise `predict_rbh` gives r_bH with its interval.
6. **The r_bH spread for the Monte Carlo** (owner decision A). r_bH is treated as log-normal. Its spread σ_rb comes from the method's own 95 % r_bH interval:

   σ_rb = ln(r_bH,hi / r_bH,lo) / (2 × 1.96)

   - **Why 1.96:** for a normal variable, 95 % of values lie within ±1.96 σ. A log-normal r_bH means ln r_bH is normal. So ln(hi) − ln(lo) = 2 × 1.96 × σ_rb.
   - **Units check:** hi/lo is s m⁻¹ ÷ s m⁻¹, so it has no units. Its log has no units, and σ_rb has no units, which matches `uncertainty.sigma_rb_fraction`.
   - **flat_plate** uses `uncertainty.sigma_rb_fraction` (0.2). **This is too small.** Decision 20 says the flat-plate constant C may be anywhere from 104 to 313 s^0.5 m⁻¹, so r_bH could be off by up to 3×. The flat-plate r_bh_lo/hi are r_bH × exp(∓1.96 × σ_rb), the same log-normal.
   - If a method's interval has a lower end ≤ 0 (possible for the ML interval), the ratio cannot be taken. The config value is used instead, and the row is flagged `marginal`.
7. **g_s.** A `LeafReading` is built from the window means, with surroundings temperature = air temperature (A4). It goes through `solve_gs` (eqs. 1–8), then `gs_uncertainty` with the method's σ_rb and seed 0. Seed 0 is used for every row, so the three methods see identical random draws, and differences between methods come from r_bH alone. `attach_uncertainty` copies on the interval and the detection limit.
8. **Convection regime.** The flat-plate regime (forced, mixed or free) comes from the leaf's size, the wind and leaf − air temperature.
   - For card_direct with no leaf photo, the card's size from `reference_cards.csv` is used instead. With no wind, the regime is `unknown` with flag `regime_unknown`.
   - Mixed or free convection adds `card_transfer_unreliable` to card_direct (A17) and `baseline_unreliable` to flat_plate.
9. **Jones index** (owner decision B), steady windows only.
   - The dry reference must be a **petroleum-jelly (vaseline) leaf**. Its temperature is read from the `card_temp_c` column in sessions with `dry_reference_type = vaseline_leaf`. A card is not used: it has a different absorptivity and size from a leaf, so a "card I_g" would be on a different scale.
   - A wet reference without a vaseline leaf gives blank I_g and CWSI with flag `jones_no_leaf_dry_reference`.
   - When the leaf is almost as cool as the wet reference, I_g is blank with `jones_ig_censor = above_range` (owner decision C). Both I_g and CWSI are the same for all three method rows of a window, because they don't use r_bH.

## Why it is done this way
- **Three rows per window, always:** later scripts (T5.2, T5.3) must count how many windows were excluded and why. A missing row hides that.
- **Non-steady windows are written too** (as invalid rows with `not_steady_state`), for the same reason. The spec only asked for steady windows; this adds rows and never removes any.
- **card_direct needs no wind or photo:** until the camera and wind sensor arrive, it is the only method that can run on real data.
- **Each method's own r_bH spread:** the alternative, one 0.2 for every method, would give card_direct and flat_plate the same interval even though their r_bH are known with very different confidence.
- **Not using the card as the Jones dry reference:** the Jones method assumes the dry reference is a leaf, with the same light absorption and boundary layer as the measured leaf.

## Worked example
Synthetic session (invented numbers, test sigmas σ_leaf 0.2 °C, σ_air 0.1 °C, σ_RH 1 %, σ_lux 5 %). True g_s = 0.300 mol m⁻² s⁻¹, wind 2.0 m s⁻¹. The leaf shape has hidden correction f = 1.53 (d_min 57.9 mm), and the card in view is the same shape. One steady window:

| Quantity | Value |
| --- | --- |
| Leaf, air | 23.045 °C, 24.012 °C |
| RH, pressure | 49.96 %, 995 hPa |
| Lux → E_s | 12 041 lux × k 0.01 → 120.4 W m⁻² |
| card r_bH (2.5–97.5 %) | 33.89 s m⁻¹ (25.03–45.52) |
| σ_rb | ln(45.52 / 25.03) / 3.92 = 0.598 / 3.92 = **0.153** |

Following eqs. 3–8 for card_direct:
- H = 1210 × (23.045 − 24.012) / 33.89 = 1210 × (−0.967) / 33.89 = **−34.5 W m⁻²**
- λE = Rn − H = 113.45 + 34.5 = **147.97 W m⁻²**
- VPD = 1325.3 Pa; γ = 0.0665 × 995 = 66.17 Pa K⁻¹
- r_s = 1210 × 1325.3 / (66.17 × 147.97) − 0.93 × 33.89 = 163.8 − 31.5 = **132.3 s m⁻¹**
- g_s = (1 / 132.3) × 99 500 / (8.314 × 297.16) = 0.00756 × 40.27 = **0.304 mol m⁻² s⁻¹**
- 95 % interval 0.239–0.395, resolved; detection limit for this window 0.091 mol m⁻² s⁻¹.

**Units check:** (J m⁻³ K⁻¹ × K) / (s m⁻¹) = W m⁻² for H. (J m⁻³ K⁻¹ × Pa) / (Pa K⁻¹ × W m⁻²) = s m⁻¹ for r_s. (m s⁻¹) × (mol m⁻³) = mol m⁻² s⁻¹ for g_s. ✓

The flat_plate row for the same window has r_bH 23.35 s m⁻¹ (too small by about f, because the plate ignores the leaf's shape), with interval 19.19–28.41 = 23.35 × exp(∓1.96 × 0.1). It gives g_s 0.318.

Here the flat-plate error in r_bH (−31 %) moves g_s by only +5 %. The leaf is just 1 °C from the air, so H is small, and r_bH matters less. Closer to the air temperature, r_bH errors matter less. Further from it, they matter more.

## Methods paragraph (draft)
For each 60 s steady-state window, stomatal conductance was computed from the leaf energy balance (eqs. 1–8) using three estimates of the boundary-layer resistance:
1. the primary estimate, from a dry reference card imaged in the same frame within 10 min (eq. 9);
2. a machine-learning correction of the flat-plate value (eq. 11), reported only if it passed validation;
3. a flat-plate baseline (eq. 10).

The roles of the three methods were fixed before data collection. Uncertainty was propagated by Monte Carlo (2000 draws per window). The boundary-layer resistance was perturbed log-normally, with a spread taken from each method's own 95 % interval. Windows where the latent heat flux was ≤ 0 or the stomatal resistance was ≤ 0 were retained as left- or right-censored values, with a per-window detection limit, and were never removed. Analyses used Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1 and pandas 3.1.1.

## Interval details added 2026-11-11 (review fixes)
- **New column `n_draws_unusable`:** how many Monte Carlo draws had no valid r_bH. They are
  kept and count as not usable (flags `draws_unusable`, `draws_all_unusable`).
- **Flag `interval_diagnostic_only`:** the usable fraction is below 90 %, so the row is
  unresolved and its interval is for diagnosis only.
- **ml_corrected:** the r_bH spread comes from the model's own interval level and is drawn
  *normal*, because the ML interval is Gaussian in f. If the interval's lower end f_lo ≤ 0, the
  flat-plate placeholder spread is used and the row is flagged `rb_sigma_fallback`.

## card_direct uses one card r_bH per session (changed 2026-11-11)
- **What changed.** card_direct no longer takes the nearest valid card window. It uses one card
  r_bH per session: ρc_p × Σ(T_card − T_air) / Σ Rn_card, over every steady, light-gated card
  window, *including* windows where the card read at or below air temperature.
- **Why.** The old rule skipped card windows that read low. The card and the leaf share the
  camera's offset, so that quietly chose data on a measured result.
- **If the pooled value is ≤ 0,** the session's card_direct rows are kept, labelled
  `suspect_card_rbh`, and counted per group in the T5.3 table.
- **Flag `wind_varies_in_session`:** the session's wind changed by more than 20 % (placeholder),
  so one r_bH per session is doubtful.
- **Details and the synthetic test** are in `card_selection_bias.md`.

## Limitations
- **Shared camera error (open item, A43).** The leaf and the card are read by the same thermal camera. Part of their temperature error is therefore shared, and in card_direct it partly cancels. The Monte Carlo treats the card's r_bH interval and the leaf temperature as independent, so card_direct intervals are **probably too wide (conservative)**. Fix: draw the card and leaf temperature errors together (correlated draws).
- **flat_plate interval too small:** σ_rb = 0.2 against a possible 3× error (decision 20).
- **card_direct assumes the card's r_bH equals the leaf's.** That holds best under forced convection with a card cut to a typical leaflet shape (A17, decision 18). The `card_transfer_unreliable` flag marks where it may not.
- **With the current placeholder sigmas, almost nothing is resolved** (see the detection_limit explainer), so `g_s_mol_dl` is blank on most rows until the thermal and light calibrations (T7.5) replace them. A valid row with a blank detection limit always carries one of two flags, so the two cases can be told apart:
  - `dl_not_resolved`: every grid value was searched and none was resolved at that window's conditions;
  - `dl_search_failed`: for at least one grid value, the leaf temperature could not be found (the root search failed or errored), so "nothing resolved" cannot be claimed.

  Invalid rows never get either flag, because their reason code already explains the blank.
- **Surroundings = air temperature** (A4).
- **A missing model gives `missing_input`** (plus `ml_model_missing`), because there is no reason code for "no model".
- **One sample of draws:** seed 0 is the same for every row, so Monte Carlo noise is shared across rows. It is not averaged away by looking at many windows.
