# validate_rb_vaseline (analysis/validate_rb_vaseline.py)

## What it does
It checks the boundary-layer model on a real leaf shape. A soybean leaflet is coated with
petroleum jelly (vaseline), which blocks its stomata, so it can barely lose water. With no water
loss, all the absorbed light energy has to leave as heat, exactly like a dry card. So we can
**solve its boundary-layer resistance r_bH from its temperature**, just as we do for a card. We
then ask: how close are the two predictions for that same leaf shape and wind?

- **flat_plate:** the textbook formula, Appendix A eq. 10.
- **ml_corrected:** the formula times the ML correction factor f, eq. 11.

Run it with `python -m analysis.validate_rb_vaseline` (add `--synthetic` for synthetic data).

## How it works
1. **Pick sessions** whose metadata say `dry_reference_type = vaseline_leaf`. The coated leaf is
   held in the card position, so its temperature is `card_temp_c` and its light factor is `c_card`.
2. **Get its geometry** from the `reference_cards.csv` row named by the session's `card_id`
   (source `traced_leaf`, coating `vaseline`).
3. **Use leaf optics, not card optics** (`leaf_optics`). Absorptivity is the light source's
   `a_s_leaf` (0.85 for the LED lamp); emissivity is the leaf's 0.97. The coating may change both,
   so every row carries the flag `coated_leaf_optics_assumed`.
4. **Choose windows** with the same gates as the card training set (`session_rows`): steady
   state, wind measured, and enough absorbed light. None of these look at the temperature
   difference.
5. **Solve r_bH like a card** (eq. 9, λE = 0):
   - Rn = a_s × E_s + n × ε × σ × (T_sur⁴ − T_coated⁴) [W m⁻²]
   - r_bH,solved = ρc_p (T_coated − T_air) / Rn [s m⁻¹]
6. **Predict r_bH** for the same window: flat_plate from the leaf's d, the wind and ΔT;
   ml_corrected = f̂ × flat_plate (`predict_rbh`). With no model file the ML value is blank and
   flagged `ml_model_missing`.
7. **Compare.** ratio = predicted / solved (1 = perfect); difference = predicted − solved
   [s m⁻¹]. One coated leaf is one **case** (the mean over its windows). Case-level numbers are
   primary; window-level numbers are secondary. There are no p-values: with a few coated leaves
   this is a consistency check, not a test.

## Why it is done this way
- **A real leaf shape.** Cards are idealised shapes. A coated leaf keeps the real outline,
  serration and curl, so it tests the flat-plate formula and the ML correction where it matters.
- **Same gates as the cards**, so the comparison is not tuned to look good.
- **Leaf optics, not card optics.** The coated leaf absorbs light like a leaf, not like matte
  black paint. Using a card's 0.95 would overstate Rn and understate r_bH.
- **Rejected alternatives:**
  - Assuming the coated leaf has exactly zero conductance and fitting r_bH inside the full leaf
    model. That needs a second unknown and the same temperature, so it adds nothing.
  - p-values on a handful of leaves.

## Worked example
A synthetic coated leaf (d_min = 45 mm) with a planted true r_bH = 1.3 × flat plate.
LED lamp at 12 000 lux, k = 0.01 W m⁻² per lux, air 24.0 °C, surroundings 24.0 °C, wind 0.5 m s⁻¹.

1. E_s = 0.01 × 12 000 = 120 W m⁻²; absorbed = 0.85 × 120 = **102 W m⁻²**.
2. The leaf settles at 26.81 °C (ΔT = 2.81 °C).
3. Longwave = 2 × 0.97 × 5.67×10⁻⁸ × (297.15⁴ − 299.96⁴) = **−32.9 W m⁻²**, so Rn = 102 − 32.9 =
   **69.1 W m⁻²**.
4. r_bH,solved = 1210 × 2.81 / 69.1 = **49.2 s m⁻¹**.
5. Flat plate for d = 45 mm, U = 0.5 m s⁻¹, ΔT = 2.81 °C: **37.9 s m⁻¹**.
6. ratio = 37.9 / 49.2 = **0.77** (= 1/1.3, the planted value). The flat plate underestimates
   this leaf's resistance by 23 %.

Units check: (J m⁻³ K⁻¹ × K) / (W m⁻²) = (J m⁻³) / (J s⁻¹ m⁻²) = s m⁻¹. ✓

## Methods paragraph (draft)
The boundary-layer model was checked against petroleum-jelly-coated soybean leaflets. Coating
blocks transpiration, so the leaflet's heat-transfer resistance was solved from its energy
balance with zero latent heat flux, r_bH = ρc_p(T_leaf − T_air)/R_n. R_n was computed with the
leaf's shortwave absorptivity and longwave emissivity, and windows were selected with the same
exogenous criteria as the reference cards. The solved r_bH was compared with the flat-plate
prediction and the ML-corrected prediction for the same leaflet geometry and measured wind
speed. Agreement is reported per leaflet as the ratio predicted/solved and the difference in
s m⁻¹, without hypothesis tests. Analyses used Python 3.13.12 with NumPy, pandas and the
project's processing modules.

## Limitations
- **Residual water loss.** Petroleum jelly may not seal every stoma, and the cuticle still
  loses a little water. Any latent heat cools the leaf, so with ΔT > 0 the solved r_bH is too
  low, and both predictions would then look biased high.
- **A consistency check, not an exact truth.** The coated leaf tests whether the boundary-layer
  model is plausible for a real shape. Its own r_bH carries the thermal-camera and light errors.
- **Optics assumed.** The coating may change a_s and emissivity (flag
  `coated_leaf_optics_assumed`); the leaf values are used unchanged.
- **Position.** The coated leaf sits in the card position, not the leaf position. Any light or
  wind difference between the two positions (A16) is not corrected here.
- **Few cases.** Only a few coated leaves are practical, so the results describe those leaves,
  not every shape.
