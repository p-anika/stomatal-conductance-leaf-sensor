# card_selection_bias (analysis/card_selection_bias.py) and the session-pooled card r_bH

## What it does
The card_direct method borrows the reference card's boundary-layer resistance r_bH and uses it
for the leaf. The card and the leaf are seen by **one** thermal camera, so a camera offset δ
makes *both* read too warm or too cold together.

The old code picked, for each leaf window, the nearest card window that gave a valid r_bH. A card
window that read low (card ΔT ≤ 0, so r_bH ≤ 0) was skipped, and if no valid card window was near,
the leaf window was thrown out. That is choosing data on a measured result, and the measurement
shares the leaf's error.

The new code (**option C**) uses **one card r_bH per session**, pooled over every steady,
light-gated card window, including the ones that read low:

r_bH,session = ρc_p × Σ(T_card − T_air) / Σ Rn_card [s m⁻¹]

If this pooled value is ≤ 0, the session's card_direct rows are **kept**, labelled
`suspect_card_rbh`, and counted per genotype × treatment in the T5.3 table
`genotype_excluded_by_group`. They are never silently dropped.

`python -m analysis.card_selection_bias` runs a synthetic experiment that compares the old rule
(A) and the new rule (C) on the same simulated sessions, with a planted shared offset.

## How it works
1. **Pooling** (`aggregate_results.pooled_card_rbh`). Only exogenous rules choose the windows:
   steady state, and absorbed light at least `physics.min_card_absorbed_sw_wm2`. The card's own
   temperature difference never decides.
2. **Interval and shared draws** (`session_card`). The interval comes from the card Monte Carlo
   at the session-mean card reading. Calibration errors do not average out over windows, so the
   one-window sigmas are kept. The session-mean reading is also what the shared card/leaf draws
   use (A43).
3. **Wind-change flag.** One r_bH per session assumes one wind setting. If the steady windows'
   wind range divided by the median wind is above `windows.wind_variation_max_frac` (0.2,
   placeholder), card_direct rows are flagged `wind_varies_in_session`.
4. **Old rule kept for comparison.** `windows.card_source: nearest_window` brings it back.
5. **The experiment.** Its design was written into the decision log before anything was computed:
   - A shared offset δ is added to both the leaf and the card temperature of every row in a
     session.
   - Light is low (60 W m⁻²), so the card is only about 1.2 °C above air.
   - 2 groups × 10 plants.
   - Designs: random δ ~ N(0, σ) per session, or one fixed δ for every session, with σ = 0.2,
     0.5 and 1.0 °C. Plus a light-difference design and a planted-effect design.

## Results (seed 0; reported, not thresholded)
Bias = mean over plants of (plant-mean card_direct g_s − true g_s); true g_s = 0.30 mol m⁻² s⁻¹.

| Design | A: bias (mol m⁻² s⁻¹) | C: bias | Rows invalid A / C | Plants lost A / C |
| --- | --- | --- | --- | --- |
| random σ 0.2 | +0.062 (+21 %) | +0.061 (+20 %) | 0 / 0 % | 0 / 0 |
| random σ 0.5 | **+7.40** (+2467 %) | **+0.103** (+34 %) | 0 / 5 % | 0 / 1 |
| random σ 1.0 | +0.152 (+51 %) | +0.151 (+50 %) | 15 / 15 % | 3 / 3 |
| fixed δ −0.2 | +0.122 (+40 %) | +0.121 (+40 %) | 0 / 0 % | 0 / 0 |
| fixed δ +0.2 | −0.023 (−8 %) | −0.024 (−8 %) | 0 / 0 % | 0 / 0 |
| fixed δ −0.5 | +0.355 (+118 %) | +0.353 (+118 %) | 0 / 0 % | 0 / 0 |
| fixed δ +0.5 | −0.084 (−28 %) | −0.084 (−28 %) | 0 / 0 % | 0 / 0 |
| fixed δ −1.0 | +3.45 | +3.21 | 0 / 0 % | 0 / 0 |
| fixed δ +1.0 | −0.145 (−48 %) | −0.145 (−48 %) | 0 / 0 % | 0 / 0 |
| light difference (60 vs 150 W m⁻², σ 0.5) | +0.059; group difference −0.037 | +0.059; −0.036 | 0 / 0 % | 0 / 0 |
| planted effect (50 % drop, σ 0.5) | +3.68; drop −1868 % | +0.069; drop **44 %** | 0 / 5 % | 0 / 1 |

Correlation between a session's lost rows and how low it read (Spearman with −δ): 0.38 (C, σ 0.5)
and 0.62 (both A and C, σ 1.0).

**Option E sensitivity (reported only):** substituting flat_plate g_s for invalid card_direct
rows gave a bias of +0.096 (A) and +0.111 (C) at σ 0.5 random, and +0.18 at σ 1.0 (both). That
is no better than C.

## What this means
1. **Where C helps, it is for a different reason than planned.** In A, one session close to the
   boundary (δ ≈ −card ΔT) kept a few card windows with a tiny positive r_bH. That made the leaf's
   sensible heat, and so g_s, blow up (+7.4 mol m⁻² s⁻¹). C pools those windows, finds the
   session's r_bH ≤ 0, and labels the session `suspect_card_rbh` instead. One plant is lost, but
   counted. The same happened in the planted-effect design, where C recovers a 44 % drop (planted
   50 %).
2. **C does not remove session-level selection.** When the offset is larger than the card's
   temperature rise, the whole session is unusable under either rule (σ 1.0: 15 % of rows, 3
   plants, strongly linked to reading low).
3. **The offset itself is the bigger problem, and neither rule can fix it.** At low light, one
   fixed −0.2 °C offset biases g_s by +40 %, and +0.2 °C by −8 %. The response is lopsided,
   because a negative offset pushes card ΔT toward 0, where r_bH collapses.
   - With random offsets, the average bias is positive even though the offsets average zero.
   - C did no harm with fixed offsets: the same as A within noise.
4. **So card_direct needs a large card temperature rise** (strong light) and a good thermal
   calibration. Low light makes it fragile.

## Why it is done this way
- **Selection on exogenous criteria only.** It avoids choosing data on a value that shares the
  leaf's error.
- **Kept and counted, not dropped.** A lost session is visible in the group table.
- **Rejected:**
  - D, a time-local pool around each leaf window (more fragile at low light; noted to revisit
    with real data).
  - E, a fallback method (reported only).

## Worked example
One session at 60 W m⁻²:
- The card is 1.2 °C above air on average, with Rn_card about 45 W m⁻².
- r_bH = 1210 × 1.2 / 45 ≈ 32 s m⁻¹.
- A −1.0 °C shared offset gives a pooled ΔT of 0.2 °C, so r_bH ≈ 1210 × 0.2 / 45 ≈ 5 s m⁻¹.
- For a leaf 0.3 °C below air (read 1.3 °C below), H = 1210 × (−1.3) / 5 ≈ −315 W m⁻² instead
  of −11 W m⁻². λE and g_s explode, which is why fixed δ −1.0 gives a bias of about +3 mol m⁻² s⁻¹.
- Units check: (J m⁻³ K⁻¹ × K) / (W m⁻²) = s m⁻¹, and (J m⁻³ K⁻¹ × K) / (s m⁻¹) = W m⁻². ✓

## Methods paragraph (draft)
For the card-direct method, one boundary-layer resistance per session was computed from the
reference card as ρc_p Σ(T_card − T_air) / Σ R_n,card over all steady, sufficiently illuminated
card windows, without excluding windows on the card's measured temperature. Sessions whose pooled
value was not positive were retained, labelled and counted per group. A synthetic experiment with
planted thermal offsets shared by card and leaf showed that this removes extreme errors from
near-zero card resistances. It does not remove the bias caused by the offset itself, which at low
irradiance reached +40 % for a −0.2 °C offset.

## Limitations
- **One r_bH per session** assumes constant wind and light within the session (flag
  `wind_varies_in_session`).
- **The training set (T4.3) still rejects card windows with r_bH ≤ 0,** so the truncation bias
  (A42) remains there (open item).
- **Seed 0 only, 10 plants per group,** synthetic physics only.
