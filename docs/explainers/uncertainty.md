# uncertainty (processing/uncertainty.py, task T1.5)

## What it does
Every sensor is a little wrong. This module asks: **"If each reading could be off by about its typical error, how far off could g_s be?"** It answers by re-running the g_s calculation 2000 times. Each time it adds a different random error to every input. The spread of the 2000 answers is the uncertainty. From the same idea it also finds the **detection limit**: the smallest g_s this setup can tell apart from "stomata closed" under this window's conditions. A second helper does the same for the card's r_b, which becomes the error bar on each ML training label.

## How it works
1. **`get_sigmas(cfg, cal, light)`** collects one "sigma" (typical 1-σ error) per input. Each comes from `calibration.json → uncertainty` if measured, otherwise from `config.yaml` (a **pre-calibration placeholder**). The result records the source of every value, and `sigma_placeholder=True` if any one is a placeholder.
2. **`gs_draws(...)`** makes n draws with a fixed random seed (`numpy.random.default_rng(seed)`, default seed 0), so the same window always gives the same answer. Each draw does this:
   - Leaf temperature: measured + σ_leaf × z₁
   - Air temperature: measured + σ_air × (ρ z₁ + √(1−ρ²) z₂). ρ is the leaf–air error correlation, because both are calibrated against one reference thermometer, so part of their error is shared. Default ρ = 0.
   - Humidity: measured + σ_RH × z₃, then **clipped to 0–100 %**. This clips the possible *true* humidity, not the result.
   - Light: E_s × exp(σ_lux × z₄), and r_bH: r_bH × exp(σ_rb × z₅). These are **lognormal**, so they can never go negative (see "Why").
   - Then it re-solves eqs. 1–8 for that draw (`solve_core`, a fast array version of `gs_calculator.solve_gs`; a test checks they agree to 10⁻⁹).
3. Each draw is sorted into one of four outcomes:

   | Draw outcome | Value used in the percentiles |
   | --- | --- |
   | OK (λE > 0, r_s > 0) | g_s (mol), but no more than the cap |
   | λE ≤ 0 | **0** (closed stomata; kept) |
   | r_s ≤ 0 | **the cap**, g_s_cap_mol = 2.0 (kept) |
   | VPD ≤ 0 | left out of the percentiles; its fraction is reported, and it counts as *not resolved* |

4. **`gs_uncertainty(...)`** returns `GsUncertainty`: the median, 2.5th and 97.5th percentiles; the fraction of draws with λE ≤ 0, r_s ≤ 0 and VPD ≤ 0; `frac_resolved`; `relative_width`; `resolved`; `hi_at_cap`; `g_s_mol_dl`; plus n, seed and the sigmas used.
5. **Resolved label** (a label in the table, never a filter). A window is resolved when **all three** hold:
   - at least 90 % of *all* draws are OK (`dl_min_resolved_fraction`)
   - (p97.5 − p2.5) / median < 1.0 (`max_relative_width`)
   - p97.5 < cap. An upper bound at the cap means "not bounded above", so it can't count as resolved.
6. **Detection limit `g_s_mol_dl`.** On a log-spaced grid of 60 "true" g_s values (0.001 to 2.0 mol m⁻² s⁻¹), the module:
   1. works out the leaf temperature each true g_s *would* produce under this window's air, humidity, light, r_b and pressure (`forward_leaf_temp`, root finding);
   2. adds the same kind of random errors;
   3. checks the resolved rule.

   The smallest true g_s that passes is the detection limit. If none passes, it is `None`, meaning the setup can't resolve anything in these conditions.
7. **`attach_uncertainty(result, unc)`** copies `g_s_mol_dl`, `g_s_lo` and `g_s_hi` (all in mol m⁻² s⁻¹) onto the T1.4 result. Nothing is dropped, and the censor label is unchanged.
8. **`card_rbh_uncertainty(...)`** handles the card. It draws card temperature (σ_leaf, same camera), air temperature (σ_air, correlated by ρ) and an Rn multiplier (1 + σ_rn_card × z, Gaussian as in the spec). It recomputes r_bH = ρcₚ(T_card − T_air)/Rn and reports the median, p2.5, p97.5, the **robust sigma = (p84.13 − p15.87)/2**, and the fraction of draws with r_bH ≤ 0. Draws are kept even when r_bH ≤ 0. Draws with |Rn| < 1 W m⁻² are left out of the percentiles and counted.

## How the sigmas are chosen, and where each value comes from
**Every value below is a pre-calibration placeholder.** None of them has been measured on this equipment yet. They are listed in `docs/decision_log.md` (2026-11-08, T1.5) so you can replace them. To replace one, put the measured value in `calibration.json` under `"uncertainty"` with the same key. The code then uses it automatically and marks it "calibration".

| Key | Placeholder | Where it comes from | How to measure it |
| --- | --- | --- | --- |
| sigma_leaf_temp_c | 1.0 °C | Spec Appendix B. Roughly the MLX90640's stated accuracy class (UNVERIFIED; check the datasheet) | T7.5: thermal camera vs reference thermometer on a target; SD of residuals after offset correction, on **window means** |
| sigma_air_temp_c | 0.5 °C | Spec Appendix B (BME280 class; UNVERIFIED) | Same reference thermometer, SD of residuals |
| sigma_rh_pct | 3 % RH | Spec Appendix B (BME280 class; UNVERIFIED) | Salt-solution check or a reference hygrometer |
| sigma_lux_fraction | 0.15 (15 %) | Spec Appendix B; used when k is calibrated | Light calibration (E2): scatter of reference-meter W m⁻² around k × lux |
| sigma_lux_fraction_uncalibrated | 0.5 (50 %) | Spec Appendix B; used when k is the sunlight fallback (flag k_uncalibrated) | Replaced by calibrating k |
| sigma_rb_fraction | 0.2 (20 %) | Spec Appendix B | Card r_b robust sigma / r_b (from `card_rbh_uncertainty`), or card-vs-Vaseline-leaf validation |
| sigma_rn_card_fraction | 0.1 (10 %) | Spec Appendix B; card–leaf irradiance mismatch (A16) | Light-field map (T7.8): spread of c_card / c_leaf |
| rho_leaf_air_temp | 0.0 | Owner decision (T1.5). 0 is the cautious choice: it gives the widest interval | Correlation of the leaf and air calibration residuals from the same reference session |

The sigmas should describe the error of a **60-s window mean**, not of a single 5-s sample, because g_s is solved on window means. Random noise shrinks when averaged, but a calibration offset does not.

## Why it is done this way
- **Monte Carlo instead of a formula.** The g_s equation is very non-linear: λE can cross zero, and r_s can cross zero. A simple "add the errors in quadrature" formula breaks down exactly where we need it most (stressed leaves). Re-solving 2000 times handles that directly.
- **Censored draws are kept** (A15). A draw where λE ≤ 0 means "this could be closed stomata", so it goes in as g_s = 0. A draw with r_s ≤ 0 means "this could be extremely open", so it goes in as the cap. Dropping them would make the interval look narrower and more confident than it really is.
- **Lognormal light and r_b** (owner decision; deviation from the spec's Gaussian). With σ = 0.5, a Gaussian fractional error gives *negative* light in about 2.3 % of draws (P(z < −2) ≈ 0.023), which is impossible. A lognormal error multiplies by exp(σz), which is always positive, has median 1, and for small σ is almost the same as the Gaussian (exp(0.15z) ≈ 1 + 0.15z).
- **Correlated temperatures.** g_s depends mostly on the *difference* T_leaf − T_air. If both sensors share an offset from the same reference thermometer, part of it cancels in the difference. ρ > 0 captures that; the default of 0 ignores it on purpose (cautious).
- **The detection limit is computed, not assumed.** It depends on the conditions. In bright light with a big VPD, a small g_s still cools the leaf measurably; in dim, humid conditions it doesn't.

## Worked example
Spec worked-example window: leaf 23.0, air 24.0, surroundings 24.0 °C, RH 50 %, 995 hPa, E_s 120 W m⁻², a_s 0.85, faces 1, r_bH 70. Point estimate g_s = 0.316 mol m⁻² s⁻¹. 2000 draws, seed 0.

**Each sigma on its own** (the others set to 0):

| Error source (placeholder) | 95 % interval, mol m⁻² s⁻¹ | Relative width |
| --- | --- | --- |
| leaf temperature 1.0 °C | 0.130 – 0.930 | 2.49 |
| light 15 % | 0.223 – 0.471 | 0.78 |
| air temperature 0.5 °C | 0.236 – 0.435 | 0.63 |
| humidity 3 % | 0.264 – 0.396 | 0.42 |
| r_b 20 % | 0.294 – 0.377 | 0.26 |
| **all together** | **0.114 – 1.388** (median 0.329) | **3.87** → not resolved |

**Check against the spec (T1.7 item 3).** With only σ_leaf = 0.5 °C, the 15.87th and 84.13th percentiles of g_s equal g_s at T_leaf + 0.5 and − 0.5 °C (g_s falls steadily as T_leaf rises), giving −21 % and +27 %. The test confirms this within 3 points.

**Detection limit vs thermal noise** (only leaf/air temperature errors, same window):

| σ_leaf / σ_air (°C) | g_s_mol_dl | worked-example window |
| --- | --- | --- |
| 0.1 / 0.05 | 0.012 | 0.287 – 0.349, resolved |
| 0.2 / 0.1 | 0.025 | 0.262 – 0.388, resolved |
| 0.3 / 0.2 | 0.048 | 0.236 – 0.436, resolved |
| 0.4 / 0.2 | 0.080 | 0.218 – 0.482, resolved |
| 0.5 / 0.3 | None | width 1.12, not resolved |
| 1.0 / 0.5 (placeholders) | None | width 2.49+, not resolved |

**Important:** even with very good thermal sensors (0.2 / 0.1 °C), adding the placeholder light (15 %), humidity (3 %) and r_b (20 %) errors gives a width of 1.14. The window is then not resolved and the detection limit is `None`. With the current placeholders, **no g_s can pass the 100 %-width rule.** Measured sigmas are needed before the detection limit means anything.

**Card example** (card 28.0 °C, air 24.0 °C, 120 W m⁻², a_s_card 0.95, faces 2): r_bH = 71.3 s m⁻¹.
- Rn error (10 %) only: 59.5 – 88.1, robust sigma **7.1 s m⁻¹ (10 %)**.
- All placeholders (camera 1.0 °C, air 0.5 °C, Rn 10 %): 24.3 – 165.2, robust sigma **32.9 s m⁻¹ (47 %)**.

So the temperature errors dominate the card label, which is why they are included (owner decision).

## Methods paragraph (draft)
Measurement uncertainty was propagated by Monte Carlo simulation (2000 draws per window, fixed seed). Leaf and air temperature errors were drawn as correlated normal deviates (σ = [x] and [y] °C, correlation [ρ], from calibration against a common reference thermometer); relative humidity errors were normal (σ = [z] %) with draws limited to 0–100 %; irradiance and boundary-layer resistance errors were lognormal (log-SD [a] and [b]). Each draw was re-solved with the full energy-balance model. Draws with non-positive latent heat flux were assigned g_s = 0 and draws with non-positive stomatal resistance were assigned the upper limit of the measurable range (2.0 mol m⁻² s⁻¹), so that intervals include closed- and fully-open-stomata outcomes; draws with non-positive VPD were excluded and their fraction reported. Intervals are the 2.5th–97.5th percentiles. A window was labelled resolved when ≥ 90 % of draws gave a positive, finite g_s and the relative 95 % interval width was below 1.0. The per-window detection limit was the smallest true g_s, on a logarithmic grid, meeting that criterion under the window's conditions. Resolved status was reported, not used to exclude data. Software: Python 3.13, NumPy, SciPy [versions].

## Limitations
1. **All sigmas are placeholders.** With them, nothing is resolved and the detection limit is undefined (see the worked example). Thermal calibration (T7.5) and light calibration are needed first.
2. **Pressure and surroundings temperature are not perturbed** (C3). If T_sur is taken from the air sensor, its error is really shared with T_air. The longwave effect is about 5.7 W m⁻² K⁻¹ per face, small next to H (≈ 17 W m⁻² K⁻¹ at r_b = 70).
3. **Errors are assumed independent between inputs** (except the leaf–air ρ) and **normal or lognormal**. A calibration offset that is the same in every window is not random within a session; it shifts all windows together.
4. **The detection limit uses a grid** with steps of about 12 % in g_s, so it is accurate to about one grid step.
5. **The resolved band has an upper end too.** At very high g_s the leaf is close to the wet limit and r_s ≤ 0 draws appear, so high values can also be unresolved. Only the lower limit is reported.
6. **The cap (2.0 mol m⁻² s⁻¹) is UNVERIFIED.** It is chosen to be above typical soybean maxima, which have not been checked here.
7. **Card Rn error is Gaussian** (spec), not lognormal. At the card gate Rn ≥ about 24 W m⁻², so a 10 % error never makes it negative.
