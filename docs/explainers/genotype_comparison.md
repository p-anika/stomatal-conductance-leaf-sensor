# genotype_comparison (analysis/genotype_comparison.py)

## What it does
This is the main science result. It answers: **does drought lower stomatal conductance (g_s), and
does it lower it more in one soybean genotype than in the other?** It takes every leaf window in
`leaf_gs_results.csv`, turns them into one number per plant, and compares the four groups
(2 genotypes × well-watered / drought-stressed). It reports the percent drop under drought for
each genotype, with 95 % confidence intervals.

- **Primary analysis:** card_direct g_s.
- **Sensitivity analyses:** the same analysis repeated with ml_corrected g_s, flat_plate g_s, the
  Jones index I_g and CWSI. If the answer changes with the method, the headline is less certain.

Run it with `python -m analysis.genotype_comparison` (add `--synthetic` for synthetic data).

## Why one window is not one replicate
A "window" is one minute of readings. One plant gives dozens of windows, but they are **not**
independent repeats of the experiment:

- **Same plant.** Windows from one plant share its roots, soil water, leaf age and genetics. If
  that plant happens to have high g_s, all of its windows are high together.
- **Same setup.** They share the same leaf position, thermal-camera offset, lamp distance and
  day. A positioning error affects them all in the same direction.
- **Close in time.** Stomata change over minutes, so neighbouring windows are almost copies of
  each other.

The treatment (watering) was given to **plants**, not to windows. So the number of independent
replicates is the number of plants. Treating 300 windows from 20 plants as 300 replicates would
make the p-values far too small and the confidence intervals far too narrow. This is called
**pseudo-replication**. Here, every primary result is computed on **plant means**. The residual
degrees of freedom of the ANOVA are (number of plants − 4), not (number of windows − 4).

## How it works
1. **Choose windows** (`prepare`). Only valid windows are kept: for I_g and CWSI, steady windows
   that have a value. Every excluded window is counted by its reason in `genotype_counts.csv`.
   **Censored windows are kept** (A15): dropping "below detection" windows would remove exactly
   the most drought-stressed plants and hide the effect we are looking for.
2. **Give censored windows a number** (`substitute`). The proper censoring-aware model (Tobit)
   is **deferred**. Until then:
   - A below_detection window gets **fraction × detection limit**, where fraction = 0, ½ and 1
     (`genotype_comparison.censor_substitution`). The whole analysis is run three times, once
     per fraction. ½ is the headline (`primary_substitution`).
   - An above_range window gets the top of the measurable range, `uncertainty.g_s_cap_mol`.
   - The detection limit is the window's own `g_s_mol_dl`. If that is blank, the T5.4 campaign
     limit is used (the lowest resolved g_s in `detection_limit_sweep.csv`) and the window is
     flagged `dl_campaign`. If neither exists, the mixed model, ANOVA and bootstrap **refuse**
     with a message in `genotype_notes.csv`. Every case is counted (`dl_row`, `dl_campaign`,
     `dl_none`).
   - A conclusion is called **robust** only if it is the same under all three fractions
     (`genotype_robustness.csv`).
3. **Plant means** (`plant_means`): the mean of each plant's windows (mol m⁻² s⁻¹).
4. **Two-way ANOVA on plant means** (`anova_plant_means`, type II): mean ~ genotype ×
   treatment (+ block if blocks were recorded). It gives a p-value for genotype, for treatment
   and for their **interaction**: does the drought effect depend on genotype?
5. **Effect sizes with 95 % confidence intervals** (`effect_sizes`), all from plant means:
   - `drop_under_drought:<genotype>`: difference = mean_drought − mean_watered, and
     pct_drop = 100 × (1 − mean_drought / mean_watered).
   - `genotype_difference:<treatment>`: second genotype − first genotype.
   - `difference_of_drops`: pct_drop(second genotype) − pct_drop(first genotype).
   - **CIs come from a bootstrap.** Each group's plants are resampled with replacement 2000
     times, and the middle 95 % of the recomputed effects is kept. No bell-curve assumption is
     needed.
   - **The bootstrap interval for difference_of_drops is the distribution-free interaction
     result.** If it excludes 0, the genotypes respond differently to drought, without assuming
     normal data or equal spread.
   - **Permutation p-values:** the drought / watered labels are shuffled among one genotype's
     plants, and we count how often a shuffled difference is at least as big as the real one.
     When there are few plants every arrangement is listed (an exact test).
6. **Mixed model (supportive)** (`mixed_model`): statsmodels MixedLM on windows, value ~
   genotype × treatment, plus a random intercept for each plant (each plant gets its own
   baseline). This is the textbook model for repeated measures. With only 4–5 plants per group,
   though, its Wald intervals tend to be too narrow and it may fail to converge. So it is
   **supportive only**. Non-convergence is written to `genotype_notes.csv`.
7. **Rank tests** (`rank_tests`). These need no detection-limit number:
   - below_detection windows are tied **below** every measured value, and above_range windows
     tied **above**
   - each plant's median is taken
   - an exact Mann–Whitney test compares watered vs drought within each genotype
   - a Kruskal–Wallis test compares all four groups.

   A rank test cannot test the interaction; that comes from the ANOVA and the bootstrap.
8. **Share below detection** (`censor_share`): the fraction of censored windows per genotype ×
   treatment. A high share in one group is itself evidence of low g_s.
9. **Forced-convection only:** the primary analysis is repeated on forced-convection windows,
   because card_direct is least reliable at low wind (A17). Until the wind sensor exists this
   gives a note "no forced-convection windows".

**CWSI** rises under stress (0 = fully transpiring, 1 = closed). It is reported as a signed
difference with its direction written in the `direction` column, and it has no percent drop.

### How small can a rank-test p-value be?
With n plants per group, an exact two-sided Mann–Whitney test can never give a p-value below
**2 / C(2n, n)**, where C(2n, n) is the number of ways to choose which n of the 2n plants were
watered. The most extreme split is one of only 2 out of those C(2n, n) arrangements.

| Plants per group | C(2n, n) | Smallest p |
| --- | --- | --- |
| 4 | 70 | 2/70 = **0.029** |
| 5 | 252 | 2/252 = **0.008** |

Ties (for example several plants all below detection) make the smallest p larger still. So
p = 0.029 with 4 plants per group is the strongest result possible, not a weak one.

### Design rule for measurement days
The model has **no random effect for day**. Instead, the design removes the problem: **all four
groups are measured on every measurement day, in alternating order.** A warm or bright day then
affects every group equally and cancels out of the comparisons. If a day is missing a group, the
day effect can no longer be separated from the group effect.

## Why it is done this way
- **Primary inference at plant level** (pre-registered in the decision log on 2026-11-10,
  before any real data). It means the plant-mean ANOVA, plus the bootstrap and permutation
  results. These respect the unit of replication and do not rely on a mixed model fitting well
  with few plants.
- **Substitution with three fractions** instead of a single guess. It shows how much the answer
  depends on the unknown values below the detection limit. A Tobit model would be better but is
  deferred until Tier 1 works.
- **Rejected alternatives:**
  - Window-level t-tests or ANOVA (pseudo-replication).
  - Dropping censored windows (biased toward finding no drought effect).
  - The mixed model as the primary result (fragile with 16–20 plants).
  - A log scale (cannot take the zero substitution).

## Worked example
**Censored windows → one plant mean.** One drought plant has three windows: 0.06, below
detection (limit 0.05), 0.07 mol m⁻² s⁻¹.

| Substitution | Values | Plant mean |
| --- | --- | --- |
| zero | 0.06, 0, 0.07 | 0.0433 |
| half_dl | 0.06, 0.025, 0.07 | 0.0517 |
| dl | 0.06, 0.05, 0.07 | 0.0600 |

**Percent drop.** Two plants per group (plant means):

| Group | Plants | Mean |
| --- | --- | --- |
| G1 watered | 0.48, 0.52 | 0.50 |
| G1 drought | 0.28, 0.32 | 0.30 |
| G2 watered | 0.49, 0.51 | 0.50 |
| G2 drought | 0.44, 0.46 | 0.45 |

- G1 pct_drop = 100 × (1 − 0.30 / 0.50) = **40 %**.
- G2 pct_drop = 100 × (1 − 0.45 / 0.50) = **10 %**.
- difference_of_drops = 10 − 40 = **−30 percentage points**: G2 drops 30 points less.
- In mol units: (0.45 − 0.50) − (0.30 − 0.50) = **+0.15 mol m⁻² s⁻¹**. This is the same number
  as the mixed model's interaction term.
- Units check: pct_drop is a ratio of two conductances, so it has no units. Differences stay in
  mol m⁻² s⁻¹.

## Methods paragraph (draft)
Stomatal conductance was compared between genotypes and water treatments with the plant as the
experimental unit. Window-level g_s values were averaged per plant. The primary inference,
pre-registered before data collection, was a two-way ANOVA (type II) on plant means
(genotype × treatment, with block as a covariate when recorded), together with percent reductions
under drought for each genotype and their difference, each with 95 % bootstrap confidence
intervals (2000 resamples of plants within groups) and exact or Monte Carlo permutation p-values.
A linear mixed-effects model (statsmodels MixedLM, REML; genotype × treatment fixed effects, a
random intercept for plant) was fitted to window-level data as a supportive analysis. Windows
below the detection limit were retained. Their values were substituted with 0, one half and the
full detection limit, and conclusions were considered robust only when they held under all
three. Rank tests on plant medians, with censored windows tied below all measured values, and the
fraction of below-detection windows per group were reported as censoring-agnostic checks. The
card-direct method was primary; the ML-corrected and flat-plate methods, the Jones index I_g and
CWSI were sensitivity analyses. Analyses used Python 3.13.12, statsmodels 0.15.0, SciPy 1.18.1,
NumPy 2.5.4 and pandas 3.1.1.

### Windows excluded as not steady, by group (added 2026-11-11)
The steady-state test keeps a window only if the leaf temperature is not drifting. That is a
choice based on the *measured* leaf temperature. A leaf whose stomata are closing warms up and
drifts, so drought-stressed windows may be thrown out more often than watered ones. If that
happened, the stressed group would look less stressed than it is.

`genotype_excluded_by_group.csv` shows, for every outcome and every genotype × treatment group:
- how many windows went in
- how many were excluded as `not_steady_state`
- that share

A clearly higher share in one group is a warning sign for the comparison. It is reported, not
corrected.

## Limitations
- **No day effect in the model.** Day-to-day changes in the room are handled only by the design
  rule above.
- **Substitution is not a censoring model.** The three substitutions bracket the answer but do
  not estimate the true values below the limit. The Tobit fit is deferred.
- **Raw scale.** The model assumes equal spread in all groups. Drought groups near zero usually
  have less spread, which mostly affects the mixed model and the ANOVA. The bootstrap and
  permutation results do not assume equal spread.
- **Few plants.** With 4–5 plants per group, the bootstrap intervals are rough (a bootstrap
  re-uses the same few plants) and only large effects can be detected.
- **Campaign detection limit.** It comes from the card_direct sweep at typical conditions, so for
  other methods and unusual windows it is only an approximation (flag dl_campaign).
- **I_g above_range windows** (leaf as cool as the wet reference) have no number. They are left
  out of the numeric I_g analyses (counted) and tied at the top in the rank tests.
