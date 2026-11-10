# ML boundary-layer correction (T4.3 build_training_set, T4.4 train_correction_model, T4.5 predict_correction)

## What it does
To turn a leaf temperature into stomatal conductance, we need the leaf's **boundary-layer resistance r_bH**, which measures how hard it is for heat to leave through the thin layer of still air around the leaf.

The textbook gives a value for an ideal **flat plate** (Appendix A eq. 10). A real leaflet has a shape, notches, texture and tilt that may make it behave differently. The **correction factor f** says how different:

  r_bH,leaf = f × r_bH,flat        (eq. 11)

f = 1 means "the flat plate was right", f = 1.3 means "30 % more resistance than the flat plate".

We can't measure f on a living leaf, because its evaporation hides the boundary-layer signal. So we measure it on **dry cut-out cards** of many shapes, where the energy balance has no evaporation term (eq. 9). Then we teach a model to predict f from shape features and wind. Three programs do this:
1. **`ml/build_training_set.py`** turns card sessions into one row per good 60-second window, each with its f.
2. **`ml/train_correction_model.py`** tries three models, tests them fairly, and saves the winner only if it truly beats "f = 1".
3. **`ml/predict_correction.py`** gives f (with an uncertainty range) and r_bH for a new leaf.

## How it works

### 1. Building the training set (T4.3)
`python -m ml.build_training_set --out data/processed/rb_training_set.csv`
1. Load every session that has a `card_id`. Synthetic sessions are skipped unless `--include-synthetic` is given.
2. Cut it into 60 s windows judged steady on the **card** temperature (`build_windows(..., temp_column="card_temp_c")`, T3.4 rules).
3. **Gates, only on conditions we set, never on the result:**
   - steady state, else `not_steady_state`;
   - wind measured, else `no_wind`;
   - card geometry measured by the Arducam (`geometry_source = camera`), else `missing_input`;
   - absorbed light ≥ 40 W m⁻², else `low_card_signal`;
   - |Rn| ≥ 1 W m⁻², a guard against dividing by zero.

   The card's temperature rise above air is never used to choose rows.
4. Solve the card's r_bH (eq. 9): r_bH,card = ρc_p (T_card − T_air) / Rn_card.
5. **One exception: rows with r_bH ≤ 0 are rejected** as `suspect_card_rbh`. A negative resistance is impossible, and it means the temperatures are off. Dropping them is a selection on the result, so the program measures what it costs (step 7).
6. Flat plate at the card's d_min, the wind and the measured ΔT. **f = r_bH,card / r_bH,flat.** Label uncertainty r_bh_card_sigma = 0.1 × r_bH,card is stored and printed, but **not** used as a training weight (see "Why").
7. **Report:**
   - rows kept, rejects by reason, number of shapes, wind and light ranges;
   - suspect count by shape and by wind (it should depend on neither);
   - **truncation bias** = mean f of kept rows − mean f of kept + suspect rows;
   - a warning if more than 2 % of gated windows are suspect.
8. **Fewer than 10 shapes:** the program stops, unless `--force` is given, which prints an overfitting warning.
9. **Card tilt:** `incidence_angle_deg` = the session's `card_angle_deg`. If that is blank, 0 is used and the flag `card_angle_default` is set.

### 2. Training (T4.4)
`python -m ml.train_correction_model`
1. **Features:** d_min, d_eq, aspect ratio, eccentricity, solidity, circularity, edge density, incidence angle, wind. **Target:** f on a normal (linear) scale.
2. **Three candidates:**
   - **null:** always predicts f = 1, i.e. "just use the flat plate";
   - **ridge:** a straight-line formula with a small penalty that stops it chasing noise;
   - **forest:** a small random forest, an average of 100 shallow decision trees that can bend to curved patterns.
3. **Leave-one-shape-out cross-validation (LOGO).** Hide every window of one shape, train on all the others, and predict the hidden shape. Repeat for every shape. Each model is scored by how far its predicted r_bH (= f × r_bH,flat) lands from the card's measured r_bH: RMSE and MAE in s m⁻¹.
4. **Nested LOGO**, for the go/no-go decision. For each hidden shape, the choice between ridge and forest is made by a second LOGO on the *other* shapes only. Then that choice is scored on the hidden shape. This scores the whole "pick a model" procedure honestly.
5. **Decision:** improvement = (RMSE_null − RMSE_nested) / RMSE_null.
   - At least **5 %** → status `ok`; save the chosen model, fitted on all rows.
   - Otherwise → status **`no_improvement`**; save the null model.
6. **Bootstrap:** refit 200 times on resampled sets of *whole shapes* (rows of one shape stay together). The spread of their predictions shows how much the answer depends on which cards we happened to make.
7. **Files saved:**
   - `data/models/correction_model.joblib`;
   - `correction_model_card.json`, holding the status, both the inner (selection) and outer (honest) scores, the training-data SHA-256 fingerprint, the number of shapes, library versions, the date, the feature ranges and the interval method.

### 3. Prediction (T4.5)
- **`predict_f(features)`** returns f_hat, f_lo, f_hi, valid, reason and flags.
  - **Interval:** f_hat ± 1.96 × √(SD_bootstrap² + residual_RMS²). The residual RMS is how far the out-of-fold predictions missed during cross-validation.
  - **Flags:** `ml_no_improvement` (f = 1 returned) and `outside_training_range` (the leaf's features are outside what the cards covered, so the model is guessing beyond its data).
  - **Reasons:** missing wind → `no_wind`; any other missing feature → `missing_input`.
- **`predict_rbh(features, wind_ms, delta_t_c, air_temp_c, cfg)`** returns f_hat × the flat plate for this leaf, with the same interval and the flat plate's convection flag. ΔT and air temperature are needed because warm air rising off the leaf (free convection) changes the flat-plate value.

## Why the model is only trusted if it beats the null model on held-out shapes
Imagine a student who memorises the answers to last year's test. They score 100 % on that test, but that says nothing about this year's. A model trained on cards can do the same: it can "memorise" each card, including its random measurement noise, and look excellent on the very cards it learned from.

So we judge it only on **shapes it has never seen** (leave-one-shape-out). Hiding single windows wouldn't be enough: windows of the same card are near-copies of each other, so the model would still have seen the answer.

And we always compare it with the **dumbest possible model, f = 1**. If a clever model can't beat "do nothing" on unseen shapes by at least 5 %, it has learned nothing useful about shape. Using it would only add noise and false confidence, so the program says `no_improvement` and keeps the flat plate.

**Nested** cross-validation fixes a quieter cheat. If you try two models and keep whichever scored better, the winner's score is lucky by construction: you picked it *because* it got lucky. The nested score repeats the picking inside each fold, so the luck is scored too. The model card shows both numbers: the inner one (optimistic) and the outer one (honest). Only the honest one decides.

The **null card set** (every true f = 1) tests the whole system. There is nothing to find, so a correct pipeline must answer `no_improvement`. It did: on the null set the nested score was 2.1 % *worse* than f = 1.

## Why the other choices
- **Linear f, not log f:** the rejected r_bH ≤ 0 rows have f ≤ 0, which a logarithm can't hold. A linear target lets the bias check include them.
- **No sample weights:** the label uncertainty is 10 % *of the measured r_bH*. Weighting by 1/σ² would give rows that happen to come out low more weight, and pull f down. Weights must not depend on the measured result.
- **Bootstrap whole shapes:** the real uncertainty is "which shapes did we cut", not "which windows".
- **Interval = bootstrap and residual in quadrature:** the bootstrap alone gives only the uncertainty of the *average* f for that shape. A single new leaf also scatters around it, by about the cross-validation residual.

## Worked example (synthetic; proves the code, not the plants)
Synthetic set: 20 shapes × 3 winds (0.3, 0.8, 1.5 m s⁻¹), 6-minute sessions.
1. **Training set:** 236 rows kept. The 124 rejects are all `not_steady_state`: 120 lack 3 minutes of history, and 4 are chance slope failures, about the 1 % the T3.4 explainer predicts. 20 shapes; no suspect rows.
2. **LOGO RMSE of r_bH:** null 13.16 s m⁻¹, ridge 2.19, forest 2.78. Ridge wins: the planted formula is a straight line, which ridge matches.
3. **Nested (honest):** 2.19 s m⁻¹ against null 13.16, so improvement = (13.16 − 2.19) / 13.16 = **83 %** ≥ 5 %, and status is `ok`. Ridge was picked in all 20 outer folds.
4. **Held-out shapes** (seed 1, never seen): RMSE of f against the planted formula is **0.026**, against 0.395 for f = 1. Units check: f is dimensionless; r_bH = f × (s m⁻¹) = s m⁻¹ ✓.
5. **Null set** (true f = 1): LOGO RMSE null 0.595, ridge 0.606, forest 0.602 s m⁻¹. Nested 0.607, i.e. **−2.1 %**, so `no_improvement`, and prediction returns f = 1 with flag `ml_no_improvement`.
6. **Truncation stress test:**
   - Setup: weak light (60 W m⁻²) and strong wind (2 m s⁻¹), so the card is only about 1 °C above air, plus a per-session calibration offset (thermal σ 1.0 °C, air σ 0.5 °C).
   - Some windows came out with r_bH ≤ 0 and were rejected. Mean f of the kept rows was **0.365 higher** than with them included.
   - Against the known truth, the kept rows were +0.79 too high and all gated rows +0.42.
   - So rejecting suspect rows adds a large upward bias when the card's ΔT is small compared with the calibration error. Even keeping them leaves a bias, because the calibration offsets themselves don't average away over a few dozen sessions. A smaller part comes from curvature: r_bH = ρc_p·ΔT / Rn(ΔT) is a curved function of the measured ΔT, so noise pushes its average up (A42).
   - **Lesson for real data:** run cards with a temperature rise of several °C (strong light, modest wind), and fix the thermal calibration first.

## Methods paragraph (draft)
The boundary-layer correction factor f = r_bH,card / r_bH,flat was estimated from steady-state windows of dry reference cards. Windows were selected only by exogenous conditions: card-temperature steady state, measured wind, camera-measured card geometry, and absorbed shortwave ≥ 40 W m⁻². Windows with a non-positive card r_bH were rejected and counted, and the resulting truncation bias in mean f was reported. Three models predicted f from nine geometric and wind features: a null model (f = 1), ridge regression on standardised features (α = 1), and a random forest (100 trees, depth 3). Models were evaluated by leave-one-shape-out cross-validation, using the RMSE of the implied r_bH. The decision to use a learned model was based on nested leave-one-shape-out cross-validation, in which model selection was repeated within each training fold. A learned model was used only if its nested RMSE was at least 5 % below that of the null model; otherwise f = 1 was used. Prediction intervals combined the spread of 200 shape-level bootstrap refits with the cross-validated residual RMS in quadrature. Software: Python 3.13, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1. The pipeline was verified on synthetic data with a planted correction and with a null correction; those checks verify the software only.

## Limitations
1. **Everything so far is synthetic.** The 83 % improvement shows the code can find a planted straight-line pattern. It says nothing about whether real card shapes change r_bH.
2. **The interval is approximate.** It assumes the same residual spread for every shape and roughly normal errors.
3. **Cards are not leaves.** Dry cards run warmer than transpiring leaves, so free convection differs (A17). Rows in mixed or free convection carry `baseline_unreliable`.
4. **Few shapes.** With 10–20 shapes even nested cross-validation is noisy. A 5 % threshold is a minimum, not proof.
5. **Truncation and calibration bias** can be large when the card's ΔT is small (worked example step 6, A42). Watch the report's suspect fraction and truncation bias.
6. **Extrapolation.** Leaves outside the cards' feature ranges get `outside_training_range`. Their f should not be trusted.
7. **Model settings** (ridge α, forest size and depth) are untuned placeholders in config `ml`.
8. **Edge density in the synthetic formula** is not evidence that texture changes r_bH (T4.2 explainer).
