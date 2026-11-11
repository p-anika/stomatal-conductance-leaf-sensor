# pipeline_dry_run (processing/pipeline_dry_run.py)

## What it does
It runs the **whole project once on made-up (synthetic) data** where the right answers are
known, and checks that the answers come back. Run it before every merge to main:

```
python -m processing.pipeline_dry_run            # default: seed 0, target sigmas, fast ML settings
python -m processing.pipeline_dry_run --seed 3
python -m processing.pipeline_dry_run --sigmas placeholder   # today's placeholder errors
python -m processing.pipeline_dry_run --settings full        # full ML settings (not time-limited)
python -m processing.pipeline_dry_run --k-error 0.10         # one shared 10 % light-calibration error
python -m processing.pipeline_dry_run --air-rh target       # air 0.2 C / RH 2 % (also: placeholder, intermediate,
                                                            #   relaxed = leaf 0.4 C, air 0.3 C, RH 3 %)
```

Every output says **SYNTHETIC: if this calibration is achieved (leaf temperature 0.2 °C, light
5 %, air .. °C, RH .. %), not a prediction**, with the air and RH values of the chosen scenario
(default: the placeholders 0.5 °C and 3 %). These numbers describe what the device *could* do if the thermal and
light calibrations reach those targets. They are not results.

## How it works
1. **Cards and the ML model.** The 20 training card shapes (with a hidden true correction f) are
   simulated at two winds (0.8 and 2.0 m s⁻¹). Card r_bH is solved (eq. 9), the training set is
   built and the ML correction model is trained.
2. **Leaves.** 2 genotypes × 2 treatments × 5 plants = 20 leaf sessions. Each plant gets one
   held-out leaf shape and a planted true g_s:
   - first genotype: 0.40 well-watered, **0.004** drought (near zero)
   - second genotype: 0.40 well-watered, 0.24 drought (a 40 % drop)
   - each plant varies by ±10 %.

   The most typical card is in every frame. Every session gets one shared error per sensor
   (`synthetic.session_bias`): thermal, air, humidity and light.
3. **g_s three ways with uncertainty** (T5.1): card_direct, ml_corrected and flat_plate, each
   with its Monte Carlo interval and detection limit. The dry run switches on
   `uncertainty.card_leaf_shared_draws`: card and leaf share the thermal and air draws (A43).
   It uses the full `n_draws` (2000) and grid (60 points), because the detection limit depends
   on n_draws (see Findings).
4. **Statistics:**
   - the T5.4 campaign detection limit, which is the fallback for blank row limits
   - then genotype_comparison (T5.3): plant-level analysis with 0, ½ and full detection-limit
     substitutions
5. **Figure:** device vs true g_s for each method, with atypical leaf shapes highlighted
   (`docs/figures/pipeline_dry_run_synthetic.png`).
6. **Checks.** Each check is PASS, FAIL or NOT EVALUATED, and the exit code is 0 only if nothing
   FAILS:

| Check | What it asks (current wording) |
| --- | --- |
| a | the pipeline completes with rows for all three methods |
| b | **ML's r_bH error is smaller than flat-plate's on atypical leaves** (top third of shape deviation). Plant-level g_s RMSE is reported alongside, with the session offsets on and off. *Changed after run 1, see below.* |
| c | the planted genotype × treatment interaction is recovered: card_direct, half-DL substitution, the bootstrap CI of the difference of drops excludes 0, and the first genotype's drop is ≥ 85 % |
| d | the change in the first genotype's drop when censored windows are dropped is **reported** (size and direction, or "group lost"). The direction is not asserted. *Reworded after run 1.* |
| e | runtime under 300 s (fast settings only; was 120 s, see run 3; machine-dependent; always reported) |
| f | exit code 0 (no check failed) |
| g | placeholder sigmas only: completes, nothing resolved, no claims made, and the statistics refuse or flag |

If no window is resolved at all, b–d are NOT EVALUATED: the run makes no claim it cannot support.

## Why it is done this way
- **Known answers.** Only synthetic data has a known true g_s, r_bH and genotype effect, so
  only it can show whether the chain gets them back.
- **Fixed seeds.** Every result can be reproduced, and seeds 0 to 4 show how much one dataset
  can vary.
- **Failures are recorded, not hidden.** The first run's failures are kept below exactly as
  they came out. Two checks were then changed, and the reason is written down. The
  pre-registered real-data analysis (decision log) is not affected.

## Run history

### Run 1 (2026-11-10): first run, before any fix, recorded as it came out
Target sigmas (leaf temperature 0.2 °C, light 5 %): **if this calibration is achieved, not a
prediction**. Per-session offsets on. Fast settings unless stated.

| Run | a completes | b ML beats flat on atypical (g_s) | c interaction recovered | d dropping censored shrinks it | e < 120 s | exit | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed 0 | PASS | FAIL (0.0264 vs 0.0202) | FAIL (drop 59.8 %) | FAIL (99.6 % vs 59.8 %) | PASS | 1 | 17.1 s |
| seed 1 | PASS | PASS (0.0497 vs 0.0609) | FAIL (drop 69.7 %) | FAIL (98.5 %) | PASS | 1 | 18.8 s |
| seed 2 | PASS | PASS (0.0919 vs 0.1056) | FAIL (no detection limit: refused) | FAIL | PASS | 1 | 26.6 s |
| seed 3 | PASS | PASS (0.1163 vs 0.1195) | FAIL (drop 70.2 %) | PASS (group lost) | PASS | 1 | 17.7 s |
| seed 4 | PASS | PASS (0.0743 vs 0.0945) | FAIL (drop 67.5 %) | PASS (group lost) | PASS | 1 | 18.8 s |
| placeholder sigmas, seed 0 | PASS | not evaluated | not evaluated | not evaluated | PASS | 0 | 26.4 s |
| k error 0.10, seed 0 | PASS | FAIL (0.0346 vs 0.0341) | FAIL (drop 62.2 %) | FAIL | PASS | 1 | 17.1 s |
| full settings, seed 0 | PASS | FAIL | FAIL (no detection limit: refused) | FAIL | not limited | 1 | 32.6 s |

g_s values are plant-level RMSE on atypical leaves, ml_corrected vs flat_plate, mol m⁻² s⁻¹.
The planted first-genotype drop is 99 %; the check needed ≥ 85 %.

The r_bH relative RMSE on atypical leaves (ml_corrected vs flat_plate) was 0.158 vs 0.300,
0.082 vs 0.298, 0.068 vs 0.307, 0.045 vs 0.302 and 0.071 vs 0.295 for seeds 0 to 4.

What went wrong:
- **(b)** The 0.2 °C per-session thermal offsets dominate the g_s error. With the offsets
  switched off (seed 0), ml_corrected g_s RMSE was 0.0034 against 0.0075 for flat_plate.
- **(c)** The campaign detection limit (0.32–0.42 mol m⁻² s⁻¹, or none) comes from
  card_direct's wide card r_bH interval. Half of it is far above the true 0.004, so substitution
  inflates the near-zero group.
- **(d)** The spec assumed dropping censored windows shrinks the effect. That only holds when
  the substituted value is below the uncensored values.

### Changes after run 1 (owner, 2026-11-10)
1. **Check b** now judges the r_bH error on atypical leaves, because that is what ML corrects.
   In run 1, g_s was dominated by the per-session thermal offsets. This was changed *after*
   seeing the results; the reason is above.
2. **Correlated card and leaf draws** (A43, `uncertainty.card_leaf_shared_draws`) are
   implemented and switched on in every dry run. The real-pipeline default stays false.
3. **Check d** is reworded to "report the size and direction".
4. **The dry run uses the full n_draws (2000) and grid (60)** for the detection limit.

### Run 2 (2026-11-10): after the changes above
| Run | a | b (r_bH) | c | d | e | f | runtime | card_direct rows with a detection limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 0 | PASS | PASS (0.158 vs 0.300) | FAIL: statistics refused | FAIL: no estimate | PASS | 1 | 84.7 s | 11 of 118 |
| seed 1 | PASS | PASS (0.082 vs 0.298) | FAIL: refused | FAIL | PASS | 1 | 86.8 s | 15 of 118 |
| seed 2 | PASS | PASS (0.068 vs 0.307) | FAIL: refused | FAIL | PASS | 1 | 118.9 s | 5 of 119 |
| seed 3 | PASS | PASS (0.045 vs 0.302) | FAIL: refused | FAIL | PASS | 1 | 47.6 s | 24 of 118 |
| seed 4 | PASS | PASS (0.071 vs 0.295) | FAIL: refused | FAIL | PASS | 1 | 42.2 s | 23 of 118 |
| placeholder sigmas, seed 0 | PASS | not evaluated | not evaluated | not evaluated | PASS | **0** (g PASS) | 60.0 s | 0 |
| full settings, seed 0 | PASS | PASS | FAIL: refused | FAIL | not limited | 1 | 129.6 s | 11 of 118 |

Plant-level g_s RMSE on atypical leaves, ml_corrected vs flat_plate (mol m⁻² s⁻¹):

| Seed | offsets on | offsets off |
| --- | --- | --- |
| 0 | 0.0264 vs 0.0202 | **0.0023 vs 0.0075** |
| 1 | 0.0497 vs 0.0609 | **0.0030 vs 0.0108** |
| 2 | 0.0919 vs 0.1056 | **0.0015 vs 0.0121** |
| 3 | 0.1163 vs 0.1195 | **0.0042 vs 0.0071** |
| 4 | 0.0743 vs 0.0945 | **0.0037 vs 0.0112** |

**Why check c still fails:**
- Every censored card_direct window needs a detection limit. In run 2, only 5 to 24 of about
  118 valid card_direct windows have one; the rest are `dl_not_resolved`, and the search never
  failed (`dl_search_failed` 0).
- The campaign fallback (T5.4, still lognormal r_bH) found no limit at 2000 draws in any seed,
  and its grid search found a root at every point (`campaign_dl_grid_no_root` = 0).
- So the statistics refuse, as designed, and check d has no estimate to report.
- **The cause is the air-temperature and humidity errors**, which are still placeholders
  (0.5 °C, 3 %). The target only sets leaf temperature and light. With a 0.5 °C air error, the
  shared air draw works *against* the leaf at high g_s (the leaf is cooler than the air, the
  card warmer), so sharing does not resolve anything.
- In a unit test with air 0.2 °C and RH 2 %, sharing halves the card_direct detection limit
  (0.054 vs 0.118 mol m⁻² s⁻¹).

### Changes before run 3 (owner, 2026-11-11)
1. **Three air-temperature / RH scenarios** (`--air-rh`), each added to the target leaf
   (0.2 °C) and light (5 %) sigmas. Every one is labelled **"if this calibration is achieved,
   not a prediction"**:
   - placeholder: air 0.5 °C, RH 3 % (the current placeholders)
   - intermediate: air 0.3 °C, RH 2.5 %
   - target: air 0.2 °C, RH 2 %

   **2 % RH may not be achievable**: the BME280's own humidity accuracy is about ±3 %
   (manufacturer figure, not checked here). The default scenario is placeholder; it was not
   chosen by which one passes.
2. **Check e's limit is now 300 s.** The original 120 s was set for the old fast settings
   (reduced detection-limit draws). Timing depends on the machine, and runtime is always
   reported.

### Run 3 (2026-11-11): three air/RH scenarios, seeds 0–4
| Scenario | Seed | a | b (r_bH) | c | d | e | f | runtime | campaign DL | card_direct rows with a DL | first-genotype drop | difference of drops (CI excludes 0?) | change when censored dropped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| placeholder | 0 | PASS | PASS | FAIL (refused) | FAIL (no estimate) | PASS | 1 | 93.1 s | none | 9 % | – | – | – |
| placeholder | 1 | PASS | PASS | FAIL (refused) | FAIL | PASS | 1 | 43.3 s | none | 13 % | – | – | – |
| placeholder | 2 | PASS | PASS | FAIL (refused) | FAIL | PASS | 1 | 134.4 s | none | 4 % | – | – | – |
| placeholder | 3 | PASS | PASS | FAIL (refused) | FAIL | PASS | 1 | 96.3 s | none | 20 % | – | – | – |
| placeholder | 4 | PASS | PASS | FAIL (refused) | FAIL | PASS | 1 | 42.4 s | none | 19 % | – | – | – |
| intermediate | 0 | PASS | PASS | **FAIL** (drop 81.7 % < 85 %) | PASS | PASS | 1 | 58.5 s | 0.152 | 81 % | 81.7 % | −49.8 (yes) | +18.2 pts |
| intermediate | 1 | PASS | PASS | PASS | PASS | PASS | 0 | 61.3 s | 0.152 | 69 % | 88.8 % | −39.1 (yes) | +10.6 pts |
| intermediate | 2 | PASS | PASS | PASS | PASS | PASS | 0 | 62.0 s | 0.173 | 71 % | 89.0 % | −48.8 (yes) | +10.6 pts |
| intermediate | 3 | PASS | PASS | **FAIL** (drop 83.8 % < 85 %) | PASS | PASS | 1 | 40.0 s | 0.152 | 73 % | 83.8 % | −41.0 (yes) | group lost |
| intermediate | 4 | PASS | PASS | PASS | PASS | PASS | 0 | 43.3 s | 0.134 | 81 % | 89.8 % | −31.9 (yes) | group lost |
| target | 0 | PASS | PASS | PASS | PASS | PASS | 0 | 75.1 s | 0.118 | 100 % | 90.9 % | −59.1 (yes) | group lost |
| target | 1 | PASS | PASS | PASS | PASS | PASS | 0 | 100.4 s | 0.118 | 92 % | 93.4 % | −43.7 (yes) | +6.3 pts |
| target | 2 | PASS | PASS | PASS | PASS | PASS | 0 | 104.9 s | 0.118 | 100 % | 91.9 % | −51.2 (yes) | +7.7 pts |
| target | 3 | PASS | PASS | PASS | PASS | PASS | 0 | 96.7 s | 0.118 | 95 % | 91.1 % | −50.9 (yes) | group lost |
| target | 4 | PASS | PASS | PASS | PASS | PASS | 0 | 39.4 s | 0.103 | 95 % | 92.3 % | −39.3 (yes) | group lost |

The planted first-genotype drop is 99 % and the planted difference of drops is about −59 points
(second genotype 40 %).
- r_bH relative RMSE on atypical leaves, ml_corrected vs flat_plate: 0.05–0.17 vs about 0.30 in
  every run.
- Campaign detection-limit grid points with no root: 0 in every run.
- "Change when censored dropped" means the first-genotype drop with censored windows dropped
  minus the drop with them kept at half the detection limit. "Group lost" means every drought
  window of that genotype was censored.

What this shows:
- **Placeholder air/RH:** nothing works for card_direct. The statistics refuse on every seed.
- **Intermediate:** the interaction is found on every seed (CI excludes 0). The first
  genotype's drop reaches 85 % on 3 of 5 seeds; on the other two, substituting half the
  detection limit (about 0.076) for a true 0.004 keeps the drop at 82–84 %.
- **Target:** every check passes on all 5 seeds.
- Dropping censored windows always made the drop *larger* in these runs (+6 to +18 points),
  because half the detection limit is far above the true near-zero g_s.

### card_direct detection limit vs air and RH sigma
One fixed window from `tests/test_correlated_draws.py`:
- card 26.5 °C, leaf 24.2 °C, air 24.0 °C, r_bH 35.4 s m⁻¹
- leaf 0.2 °C, light 5 %, 2000 draws

| Air σ (°C) | RH σ (%) | DL, independent draws | DL, shared card/leaf draws |
| --- | --- | --- | --- |
| 0.5 | 3.0 | 0.290 | none resolved |
| 0.5 | 2.0 | 0.255 | none resolved |
| 0.3 | 2.5 | 0.134 | 0.062 |
| 0.2 | 3.0 | 0.118 | 0.054 |
| 0.2 | 2.0 | 0.118 | 0.054 |
| 0.1 | 1.0 | 0.103 | 0.048 |

(mol m⁻² s⁻¹; `dl_grid_no_root` = 0 everywhere.)

**The air-temperature error decides the detection limit; RH barely matters here:** air 0.2 °C
gives the same limit at RH 2 % and 3 %. So the air sensor's calibration matters most, and an
unachievable 2 % RH is not needed for this result.

### Relative width vs true g_s at the campaign conditions (seed 0)
This is the 95 % interval width divided by the median, from `pipeline_dry_run_dl_sweep_*.csv`.
The detection-limit rule needs it below 1.0.

| True g_s (mol m⁻² s⁻¹) | 0.062 | 0.103 | 0.173 | 0.290 | 0.485 | 0.812 |
| --- | --- | --- | --- | --- | --- | --- |
| placeholder air/RH | 2.20 | 1.68 | 1.25 | 1.07 | 1.04 | 1.24 |
| intermediate | 1.84 | 1.21 | 0.89 | 0.74 | 0.74 | 0.85 |
| target | 1.55 | 1.01 | 0.73 | 0.62 | 0.61 | 0.68 |

Above about 1.4 mol m⁻² s⁻¹ the upper bound reaches the cap, so those points are never resolved.

**Open item: the threshold is knife-edge.** With the placeholder air/RH (and in run 1) the
width sits at about 0.96–1.07 across the middle of the range, right at the 1.0 threshold. So
small changes (n_draws, one seed) flip a whole range between resolved and not resolved.
Report this width-versus-g_s curve with every detection limit, and review the threshold after
real calibration.

### Run 4 (2026-11-11): fourth scenario "relaxed", seeds 0–4
Owner request: leaf 0.4 °C, air 0.3 °C, RH 3 % (light stays 5 %), labelled **"if this
calibration is achieved, not a prediction"**. No threshold and no other scenario was changed.
`--air-rh relaxed` also sets the leaf sigma; the other scenarios keep leaf 0.2 °C.

| Seed | b (r_bH, ML vs flat) | c | d | campaign DL | card_direct rows with a DL | resolved rows | runtime | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | PASS (0.203 vs 0.301) | FAIL (statistics refused) | FAIL (no estimate) | none | 2 % | 18 of 354 | 146.0 s | 1 |
| 1 | PASS (0.119 vs 0.298) | FAIL (refused) | FAIL | none | 0 % | 5 of 354 | 109.8 s | 1 |
| 2 | not evaluated | not evaluated | not evaluated | none | 0 % | **0 of 357** | 60.2 s | **0** |
| 3 | PASS (0.080 vs 0.300) | FAIL (refused) | FAIL | none | 9 % | 8 of 354 | 51.9 s | 1 |
| 4 | PASS (0.076 vs 0.294) | FAIL (refused) | FAIL | none | 0 % | 7 of 357 | 75.2 s | 1 |

- **Seed 2 resolved nothing.** By design, checks b–d are then NOT EVALUATED and the run makes no
  claim, so its exit code is 0. **That 0 is not a pass**: it means the run had nothing to judge.
- **The width threshold is not met anywhere.** At the campaign conditions (seed 0) the
  relative width is 1.84, 1.33, 1.15, 1.17 and 1.38 at true g_s 0.10, 0.17, 0.29, 0.49 and
  0.81 mol m⁻² s⁻¹. It never falls below 1.0, so no campaign detection limit exists.
- **Comparison with the intermediate scenario** (same air 0.3 °C): RH 3 % instead of 2.5 %
  barely matters (run 3 table). The change that breaks card_direct here is the leaf sigma,
  0.4 °C instead of 0.2 °C.
- Check e passed on every seed (limit 300 s); runtime 52–146 s.

### Run 4b (2026-11-11): do the rank tests find the planted ordering? (diagnostic, not a check)
**Question:** when the detection limit is missing and the numeric statistics refuse, can the
censoring-free tools still see the planted pattern? The planted pattern is:
- genotype 1 drought **lowest** (true g_s 0.004)
- genotype 2 drought **reduced by about 40 %** (0.24 vs 0.40)

These rows are labelled DIAG in the checks file. They never change a check or the exit code.

**Definition** (written into the decision log before computing):
- **Data:** card_direct, valid windows. Below-detection windows are tied below every measured
  value. Each plant gets the median of its windows, and each group the median of its plants.
- **"Genotype 1 drought lowest" = yes** only if both hold:
  - its group value is strictly lower than all three other groups (a tie is "no")
  - its share of below-detection windows is the highest and above 0
- **Genotype 2 reduction** = 100 × (1 − drought / well-watered) of the group values. It is "not
  computable" if a group value is set by below-detection windows (below the smallest measured
  g_s). This is reported, not judged.
- **Rank tests:**
  - exact Mann–Whitney on plant medians within each genotype (two-sided, with direction)
  - Kruskal–Wallis over the four groups

  p-values are reported, not thresholded.

**Note: the below-detection label comes from the point estimate (λE ≤ 0), not from the
detection limit.** So the share table and the rank tests measure *censoring*, not a resolved g_s.
This is why they still work when no detection limit exists.

All runs are seeds 0–4. "Leaf 1.0 placeholder" is `--sigmas placeholder` (leaf 1.0 °C, air 0.5 °C,
RH 3 %, light 15 %); seeds 1–4 were run for this task, and seed 0 was re-run. The other scenarios
use the saved results of runs 3 and 4.

| Scenario | Seed | G1 drought lowest | G1 drought share below detection | G2 reduction (planted ≈ 40 %) | Mann–Whitney G1 p | Mann–Whitney G2 p | Kruskal–Wallis p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf 1.0 placeholder | 0 | yes | 0.79 | 14.4 % | 0.0079 | 0.69 | 0.0115 |
| leaf 1.0 placeholder | 1 | yes | 0.96 | 61.8 % | 0.029 | 0.016 | 0.0019 |
| leaf 1.0 placeholder | 2 | yes | 0.76 | 59.3 % | 0.0079 | 0.42 | 0.0072 |
| leaf 1.0 placeholder | 3 | yes | 0.93 | 43.9 % | 0.0079 | 0.22 | 0.0078 |
| leaf 1.0 placeholder | 4 | yes | 1.00 | 44.8 % | 0.0079 | 0.29 | 0.0081 |
| air/RH placeholder (leaf 0.2) | 0 | yes | 0.93 | 31.5 % | 0.0079 | 0.0079 | 0.0013 |
| air/RH placeholder | 1 | yes | 0.77 | 33.7 % | 0.0079 | 0.0079 | 0.0010 |
| air/RH placeholder | 2 | yes | 0.90 | 32.1 % | 0.0079 | 0.0079 | 0.0010 |
| air/RH placeholder | 3 | yes | 1.00 | 44.1 % | 0.0079 | 0.016 | 0.0012 |
| air/RH placeholder | 4 | yes | 1.00 | 43.6 % | 0.0079 | 0.032 | 0.0024 |
| intermediate | 0 | yes | 0.97 | 35.0 % | 0.0079 | 0.0079 | 0.0010 |
| intermediate | 1 | yes | 0.80 | 41.1 % | 0.0079 | 0.0079 | 0.0010 |
| intermediate | 2 | yes | 0.90 | 37.5 % | 0.0079 | 0.0079 | 0.0010 |
| intermediate | 3 | yes | 1.00 | 42.4 % | 0.0079 | 0.0079 | 0.0010 |
| intermediate | 4 | yes | 1.00 | 50.7 % | 0.0079 | 0.016 | 0.0018 |
| target | 0 | yes | 1.00 | 37.6 % | 0.0079 | 0.016 | 0.0012 |
| target | 1 | yes | 0.83 | 44.0 % | 0.0079 | 0.0079 | 0.0010 |
| target | 2 | yes | 0.90 | 40.1 % | 0.0079 | 0.0079 | 0.0010 |
| target | 3 | yes | 1.00 | 41.1 % | 0.0079 | 0.0079 | 0.0010 |
| target | 4 | yes | 1.00 | 53.1 % | 0.0079 | 0.016 | 0.0015 |
| relaxed | 0 | yes | 0.97 | 32.7 % | 0.0079 | 0.032 | 0.0024 |
| relaxed | 1 | yes | 0.77 | 49.2 % | 0.0079 | 0.0079 | 0.0008 |
| relaxed | 2 | yes | 0.83 | 44.4 % | 0.0079 | 0.032 | 0.0022 |
| relaxed | 3 | yes | 1.00 | 41.2 % | 0.0079 | 0.016 | 0.0012 |
| relaxed | 4 | yes | 1.00 | 57.5 % | 0.0079 | 0.032 | 0.0024 |

In every run, both Mann–Whitney tests put the drought group lower.
- The genotype-1 drought group is at the below-detection tie value in every run.
- 0.0079 = 2/252 is the smallest p the exact test can give with 5 vs 5 plants.

The leaf-1.0 placeholder runs took 134.6, 115.4, 110.7, 109.1 and 100.9 s, and exited 0 (check g
PASS, nothing resolved).

What this shows:
- **Genotype 1's collapse is found in all 25 runs,** in every scenario, even where the numeric
  statistics refused (placeholder and relaxed).
- **Genotype 2's 40 % drop is seen as a direction in every run.** Its Mann–Whitney p is below
  0.05 in 20 of 25 runs; the 5 that are not are all in the leaf-1.0 placeholder scenario.
- **The size of the reduction comes from medians** and ranges 31–58 % in the four leaf-0.2 or
  leaf-0.4 scenarios, and 14–62 % with the 1.0 °C leaf placeholder.
- **The rank tests cannot give an effect size with a confidence interval,** and they say nothing
  about whether single windows are resolved.

**No-effect check (reported, not asserted).** 200 synthetic card_direct tables with four
identical groups (5 plants × 6 windows, true g_s 0.40, plant SD 0.03, window SD 0.03):
- Mann–Whitney p < 0.05 in **3.5 %** (genotype 1) and **3.5 %** (genotype 2) of tables
- Kruskal–Wallis p < 0.05 in **4.0 %**

The exact 5-vs-5 test can only reach discrete p-values, so its true rate below 0.05 is at most
12/252 = 4.8 %. With 200 tables the standard error is about 1.5 percentage points, so 3.5 % is
consistent with that. The tests do not find effects that are not there more often than they
should.

## Findings
1. **ML corrects r_bH on unusual leaf shapes.** On seeds 0–4 its r_bH error is 5–16 %, against
   about 30 % for the flat plate. Without per-session offsets it also has 1.7–8 times lower g_s
   error.
2. **Per-session calibration offsets dominate g_s error.** With the 0.2 °C thermal offset per
   session, the g_s difference between ML and flat plate shrinks to almost nothing (seed 0:
   ML worse).
3. **The detection limit depends on n_draws.** At the representative conditions the relative
   95 % width sits right at the 1.0 threshold (about 0.96–1.03 over g_s 0.3–0.5). At 500
   draws, percentile noise dips it below 1.0 ("resolved"); at 2000 draws it is about 1.03
   (not resolved). The root search did not fail (`dl_grid_no_root` = 0).
   **Every earlier detection-limit number made with the fast settings, including the T5.4
   sweep, must be rerun with full draws and real calibration.**
4. **Correlated card and leaf draws help only when the air-temperature error is small.**
   Sharing the thermal and air draws halves the card_direct detection limit at air 0.2 °C /
   RH 2 %. At the 0.5 °C air placeholder it makes the limit worse, because the shared air error
   pushes card r_bH and leaf H the same way when the leaf is cooler than the air and the card
   warmer.
5. **A shared 10 % k error barely matters** (run 1, seed 0):
   - The first genotype's drop goes from 59.8 % to 62.2 % and the difference of drops from
     −27.5 to −30.1 points.
   - ML's r_bH error is unchanged (0.158 → 0.158). The model is trained on cards that carry the
     same k error, so that part cancels.
   - card_direct's r_bH error rises from 0.317 to 0.362.
   - Plant g_s RMSE on atypical leaves rises for both methods (ML 0.026 → 0.035, flat plate
     0.020 → 0.034).
6. **Dropping censored windows can push the estimate either way.** It shrinks the effect when
   the substituted value is below the uncensored ones, and enlarges it when the detection limit
   is far above the true value (run 1: 99.6 % vs 59.8 %). So it is reported, not asserted.
7. **Runtime:** with full detection-limit settings, runs took 39–134 s on this laptop; the limit
   is now 300 s.
8. **Air/RH scenarios (run 3):**
   - placeholder: c and d fail on every seed (statistics refused)
   - intermediate: the interaction is recovered on 5 of 5 seeds, but the 85 % drop threshold
     is met on only 3 of 5
   - target: every check passes on 5 of 5
   
   The air-temperature error decides the card_direct detection limit; RH barely matters.
9. **The detection-limit threshold is knife-edge** with the placeholder air/RH: the relative
   width sits at 0.96–1.07 (open item).
10. **The rank tests and the share-below-detection table recover the planted ordering in all 25
    runs** (run 4b), even where the numeric statistics refuse. They measure censoring from the point
    estimate, not resolved g_s, and give direction, not size.

## Worked example
Seed 0, run 2, one atypical plant:
- **Planted true g_s** = 0.40 × (1 + plant offset) mol m⁻² s⁻¹; true leaf r_bH = f_leaf ×
  flat plate.
- **r_bH error.** The flat plate underestimates r_bH by about 30 %, because the hidden f is
  above 1 for these shapes. ML gets within about 16 %. So check b passes: 0.158 < 0.300.
- **g_s with offsets on.** The same plant's g_s is also hit by the session's thermal offset
  (one draw from N(0, 0.2 °C)). Under the lamp, 0.2 °C on the leaf changes λE by roughly
  ρc_p × 0.2 / r_bH ≈ 1210 × 0.2 / 40 ≈ 6 W m⁻² out of about 70 W m⁻² (about 9 %), which is as
  large as the r_bH effect.
- Units check: (J m⁻³ K⁻¹ × K) / (s m⁻¹) = J m⁻² s⁻¹ = W m⁻². ✓

## Methods paragraph (draft)
The full analysis chain was tested end to end on synthetic data with known boundary-layer
corrections, known stomatal conductances and a planted genotype × treatment interaction,
including a stressed group with near-zero conductance. Each session received one random offset
per sensor drawn from the target calibration uncertainties (leaf temperature 0.2 °C, light 5 %).
Card and leaf errors were drawn jointly, since both are measured by the same thermal camera and
air sensor. Five random seeds were run. The ML correction reduced the r_bH error on atypical leaf
shapes from about 30 % (flat plate) to 5–16 %. The planted interaction could not be tested on
these settings because the card-direct detection limit was not resolved with the current
air-temperature and humidity uncertainty placeholders.

## Limitations
- **Synthetic physics only.** The data are made with the same equations used to analyse them,
  so model-form errors cannot show up.
- **Air temperature and RH errors are still placeholders** (0.5 °C, 3 %). The air error decides
  whether card_direct resolves anything (findings 4 and 8). The intermediate and target
  scenarios are what *could* happen if those calibrations are achieved; 2 % RH may not be.
- **The ML settings are reduced** (50 bootstraps, 50 trees) to meet the time limit. The
  detection-limit settings are full.
- **Runtime depends on the machine** (39–134 s here, limit 300 s).
- **The T5.4 campaign limit** still uses lognormal r_bH, not the correlated draws.
