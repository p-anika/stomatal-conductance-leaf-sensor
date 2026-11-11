# validation (analysis/validate_gs_ground_truth.py and analysis/baseline_benchmark.py)

## What it does
These two scripts answer the question "how close is the device to the truth?". The "truth" is a
borrowed porometer: a hand-held instrument that measures stomatal conductance (g_s) directly on a
leaf. Readings are typed into `data/validation/ground_truth_gs.csv`.

- `validate_gs_ground_truth.py` (T5.6) pairs every porometer value with the device's value for
  the same plant at about the same time. For each r_b method (card_direct, ml_corrected,
  flat_plate) it reports how far apart they are, and draws a Bland–Altman plot.
- `baseline_benchmark.py` (T5.2) uses the same pairs. It splits the results by leaf shape, g_s
  range and wind regime, and tests whether the ML method beats the flat-plate method.

Run them with `python -m analysis.validate_gs_ground_truth` and `python -m analysis.baseline_benchmark`.
Add `--synthetic` to read the synthetic files instead. Every output from a synthetic run is
stamped SYNTHETIC.

## How it works

### Step 1: put the porometer on the same footing as the device (change E4)
Soybean has stomata on **both** faces of the leaf: the upper face (adaxial) and the lower face
(abaxial). The device sees the whole leaf, so with `physics.faces: 2` its g_s is both faces
together. Most porometers clamp onto one face only.

- One-face readings are therefore added: **g_total = g_adaxial + g_abaxial**. Conductances in
  parallel add, like two open doors letting people out of a room.
- An upper and a lower reading are added only if they are from the same plant and leaf and were
  taken within `validation.surface_pair_minutes` (5 min) of each other. The closest pairs are
  matched first, and each reading is used once.
- A reading with no partner face goes into the "one_surface" table with reason
  `one_surface_only`. It is **never** compared with the device. Comparing one face with two
  faces would make the device look roughly 50 % to 200 % too high.
- If `physics.faces` is not 2, the script stops with an error.

**Weighed (gravimetric) rows** are kept out of every error table. The spec does not say whether
they describe one leaf or the whole plant (open item A45). They go into a separate
**trend-consistency table** instead. For each plant it lists the gravimetric value next to the
device's mean value within `validation.gravimetric_match_hours` (1 h). Across plants it reports
only the **Spearman rank correlation**: does the device put the plants in the same order as the
scale does? It gives no bias or RMSE, because those need both numbers to mean the same thing.

### Step 2: pair by plant and time
`pair_ground_truth(whole, results, pair_max_minutes)` handles one porometer value and one method
at a time:

- It looks for device windows of the same plant (and the same leaf_position when both are filled
  in) whose **window midpoint** is within `validation.pair_max_minutes` (10 min).
- It picks the nearest *valid* window. Validity does not depend on the g_s value, so this is not
  cherry-picking.
- Each pair gets one status:
  - `paired`: valid and not censored. Used in the statistics.
  - `device_censored`: the device said below_detection or above_range.
  - `device_invalid`: only invalid windows were found.
  - `no_window`: nothing within 10 min.
- Nothing is silently dropped: the summary counts each status.

### Step 3: the statistics (all in mol m⁻² s⁻¹)
Call the difference for one pair **d = device − truth**. A positive d means the device read high.

| Statistic | Formula | What it tells you |
| --- | --- | --- |
| **Bias** | mean of d | The *average* offset. +0.03 means the device reads 0.03 too high on average. Positive and negative errors cancel, so a bias of 0 does **not** mean "accurate". |
| **MAE** (mean absolute error) | mean of \|d\| | The typical size of an error, ignoring its sign. |
| **RMSE** (root-mean-square error) | √(mean of d²) | Like MAE, but squaring makes big errors count more. RMSE ≥ MAE always; if RMSE is much bigger than MAE, a few pairs are badly off. This is the headline accuracy number. |
| **SD of differences** | standard deviation of d (n − 1) | How much the error varies from pair to pair (the random part). RMSE² ≈ bias² + SD². |
| **Slope and intercept** | least-squares line: device = slope × truth + intercept | Slope 1 and intercept 0 would be perfect. Slope < 1 means the device squashes the range (reads high when g_s is low and low when g_s is high). The intercept is the device reading when the truth is 0. |
| **R²** | 1 − (sum of squared distances from the line) / (sum of squared distances from the mean) | How tightly the points sit around the fitted line (0 to 1). **Careful:** a device that always reads exactly double the truth has R² = 1. R² measures *correlation*, not *agreement*. That is why we also need bias and RMSE. |
| **Bland–Altman limits of agreement** | bias ± 1.96 × SD | About 95 % of single differences should fall between these two lines, if the differences are roughly bell-shaped. 1.96 is the z-value that leaves 2.5 % in each tail of a normal distribution. It is in config as `validation.loa_z`. |

**The Bland–Altman plot** puts the *average* of device and truth on the x-axis and their
*difference* on the y-axis. It adds a line at the bias and dashed lines at the limits of
agreement. It shows things a scatter plot hides, such as errors that grow as g_s grows (a funnel
shape). The dotted vertical line is the device's median **detection limit**: points left of it
are near where the device cannot tell g_s from zero.

**Plant means are the primary result.** Several windows from one plant are not independent: the
same leaf, the same day and the same positioning errors. Treating them as separate samples makes
the results look more certain than they are. So each plant becomes one point (mean device, mean
truth), and the statistics on those points are the headline (`plant_mean (primary)`). The
window-level numbers are reported too, labelled `window (secondary, non-independent)`.

### Step 4: the detection table
For every porometer value with a valid device window and a known detection limit (`g_s_mol_dl`):

- `n_truth_below_dl`: how many true values were below the device's detection limit.
- `n_device_below_detection` and `fraction_correct`: how often the device correctly said
  below_detection in those cases.
- `n_false_below_detection`: how often the device said below_detection when the truth was
  *above* the limit (a false alarm).

This matters because censored pairs are left out of bias and RMSE. Leaving them out makes the
device look better at low g_s, so this table shows what happened to them.

### Step 5: the benchmark splits (baseline_benchmark.py)
1. **Leaf shape** comes from the photos, using the same photo-to-window code as the rest of the
   pipeline (`attach_photo_features`). No photo means `shape_unknown`.
2. **Shape deviation** measures how unusual a leaf is: deviation = √(z_solidity² + z_aspect²),
   where z = (value − median) / (1.4826 × MAD).
   - MAD is the median absolute deviation from the median. It ignores a few strange leaves
     better than the standard deviation does.
   - Why the scaling? Aspect ratio varies far more than solidity, so without it aspect ratio
     would swamp solidity.
   - Where 1.4826 comes from: for a normal distribution, half the values lie within 0.6745 SD of
     the median (the 75th percentile of a standard normal is 0.6745). So MAD = 0.6745 × SD, and
     SD = MAD / 0.6745 = 1.4826 × MAD. Units check: solidity and aspect ratio have no units, so z
     and the deviation have none either.
3. **Splits.** Pairs are split into tertiles of shape deviation (q1 = most typical, q3 = most
   unusual), tertiles of true g_s (q1 = lowest), and convection regime (forced, mixed, free,
   unknown). These tables are **descriptive only**: counts and statistics, no p-values. Testing
   many slices would turn up "significant" results by chance.
4. **The flat-plate method is judged on forced-convection windows only.** Its formula is only
   meant for wind-driven flow (spec T5.2). The by-regime table still shows it in every regime so
   you can see how it fails.
5. **Pre-registered primary analysis.** This was fixed in the decision log on 2026-11-10, before
   any real data existed, and must not change after seeing results:
   - **Plant-level RMSE per method on forced-convection windows.**
   - **A paired Wilcoxon signed-rank test, ML vs flat plate**, on each plant's mean absolute
     error (forced windows, same porometer values for both methods). It is two-sided.
   - In plain words: for each plant, take ML's error minus the flat plate's error. The Wilcoxon
     test ranks those differences by size and asks whether the negative ones (ML better)
     outweigh the positive ones more than chance would allow.
   - Wilcoxon rather than a paired t-test because it does not assume the differences are
     bell-shaped, which is safer with 10 to 20 plants.
   - A window-level Wilcoxon is also reported, labelled secondary and non-independent.

## Why it is done this way
- **Whole-leaf basis (E4).** A one-face porometer reading is only part of the leaf's
  conductance. Comparing it with the device would build a large, fake bias into the headline
  result.
- **Plant is the unit (E7).** Statistics on plant means avoid pseudo-replication.
- **Bias, MAE, RMSE and Bland–Altman, not just R².** R² can be high even when the device is
  consistently wrong.
- **Rejected alternatives:**
  - Deming regression, which allows for error in the porometer as well as the device. It is
    better in principle but not coded yet (see Limitations).
  - Choosing the nearest window regardless of validity. That would throw away good pairs when an
    invalid window happens to be a few seconds closer.
  - Putting gravimetric values in the error tables. Their basis is undefined.

## Worked example
One method and four plants. Each plant has one pair (so plant means = pairs). Values are in
mol m⁻² s⁻¹. These numbers come from running `agreement_stats` itself.

| Plant | Truth (adaxial + abaxial) | Device | d = device − truth |
| --- | --- | --- | --- |
| A | 0.07 + 0.13 = 0.20 | 0.24 | +0.04 |
| B | 0.30 | 0.31 | +0.01 |
| C | 0.40 | 0.47 | +0.07 |
| D | 0.50 | 0.52 | +0.02 |

- Bias = (0.04 + 0.01 + 0.07 + 0.02) / 4 = **0.035**.
- MAE = the same here, 0.035, because every d is positive.
- RMSE = √((0.0016 + 0.0001 + 0.0049 + 0.0004) / 4) = √0.0175 = **0.0418**.
- SD of d = **0.0265**. Check: bias² + SD² × 3/4 = 0.001225 + 0.000525 = 0.00175 = RMSE². ✓
- Limits of agreement = 0.035 ± 1.96 × 0.0265 = **−0.017 to +0.087**.
- Least-squares line: slope **1.00**, intercept **0.035**, R² **0.96**.

In words: the device reads about 0.035 mol m⁻² s⁻¹ high at every g_s (slope 1, so the error does
not grow with g_s). A single reading could be off by between −0.017 and +0.087. If plant A had
had only the adaxial 0.07, it would be listed as one-surface and left out. It would not be
compared with 0.24.

## Methods paragraph (draft)
Device g_s was compared with porometer measurements on a whole-leaf basis. Because soybean leaves
have stomata on both surfaces, adaxial and abaxial readings taken within 5 min on the same leaf
were summed (g_total = g_adaxial + g_abaxial). Single-surface readings without a partner were
excluded from comparison and reported separately. Each porometer value was paired with the
nearest valid device window of the same plant whose midpoint was within 10 min. Agreement was
summarised as bias, mean absolute error, root-mean-square error (RMSE), the ordinary
least-squares slope, intercept and R², and Bland–Altman 95 % limits of agreement (bias ± 1.96 SD
of the differences) [Bland & Altman 1986, UNVERIFIED citation]. The plant was the unit of
analysis: statistics were computed on per-plant means, with window-level values reported as
secondary. The pre-registered primary analysis was the plant-level RMSE of each method on
forced-convection windows, and a two-sided paired Wilcoxon signed-rank test comparing plant-level
mean absolute error of the ML-corrected and flat-plate methods. Results split by leaf-shape
deviation (robust distance of solidity and aspect ratio from the median leaf, in tertiles), g_s
range and convection regime are descriptive. Analyses used Python 3.13.12, NumPy 2.5.4,
pandas 3.1.1, SciPy 1.18.1 and Matplotlib 3.11.2.

## Limitations
- **Porometer error is ignored in the fit.** Least squares assumes the x-axis (porometer) is
  exact. Porometers have their own error, which pulls the fitted slope toward 0 ("regression
  dilution"). Deming regression would correct this; it is not implemented.
- **Censored pairs are left out of bias and RMSE.** This flatters the device at low g_s; the
  detection table reports what happened to them.
- **Faces with different stomata.** The device solves one bulk r_s with one r_b for the whole
  leaf. The real leaf is two faces in parallel, each with its own stomatal and boundary-layer
  resistance, and soybean's lower face usually conducts more than its upper face. The bulk value
  is not exactly g_ad + g_ab; the difference is expected to be small but has not been measured.
- **Plant-level limits of agreement describe plant means, not single readings.** Single-window
  limits will be wider.
- **Few plants means low power.** With 10 to 20 plants, the Wilcoxon test can only detect large
  differences, and tertiles hold only a few plants each.
- **Placeholders.** surface_pair_minutes (5) and gravimetric_match_hours (1) are settings, not
  measured values. The gravimetric basis is undefined (A45). Which porometer surface label
  matches the device must be settled before real use (A40).
