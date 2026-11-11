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
3. Each draw is sorted into one of three outcomes. **Every draw is used in the percentiles; none is left out.**

   | Draw outcome | Value used in the percentiles |
   | --- | --- |
   | OK (λE > 0 and r_s > 0, which also means VPD > 0) | g_s (mol), but no more than the cap |
   | λE ≤ 0 | **0** (closed stomata; kept) |
   | r_s ≤ 0, including VPD ≤ 0 with λE > 0 | **the cap**, g_s_cap_mol = 2.0 (kept) |

   A draw with VPD ≤ 0 and λE > 0 always has r_s < 0 (eq. 8), so it is held at the cap like any other r_s ≤ 0 draw (spec, Session 8 review). An earlier version left these draws out, which cut off the high-g_s end of the interval.
4. **`gs_uncertainty(...)`** returns `GsUncertainty`: the median, 2.5th and 97.5th percentiles; the fraction of draws with λE ≤ 0, with r_s ≤ 0, and (as a diagnostic only) with VPD ≤ 0; `frac_resolved`; `relative_width`; `resolved`; `hi_at_cap`; `g_s_mol_dl`; `dl_grid_points` and `dl_grid_no_root` (see step 6); plus n, seed and the sigmas used.
5. **Resolved label** (a label in the table, never a filter). A window is resolved only when **all three** hold:
   1. at least 90 % of *all* draws are OK, meaning λE > 0, r_s > 0 and VPD > 0 (`dl_min_resolved_fraction`);
   2. (p97.5 − p2.5) / median is **strictly below** 1.0 (`max_relative_width`); exactly 1.0 does not pass;
   3. p97.5 is **strictly below** the cap. An upper bound at the cap means "not bounded above", so it can't count as resolved.
6. **Detection limit `g_s_mol_dl`.** On a log-spaced grid of 60 "true" g_s values (0.001 to 2.0 mol m⁻² s⁻¹), the module:
   1. works out the leaf temperature each true g_s *would* produce under this window's air, humidity, light, r_b and pressure (`forward_leaf_temp`, root finding);
   2. adds the same kind of random errors;
   3. checks the resolved rule.

   The smallest true g_s that passes is the detection limit. If none passes, it is `None`. Step 1 can fail: if no leaf temperature within ±40 °C of air balances the energy budget for some true g_s, that grid point is skipped. The result counts these skips in `dl_grid_no_root` (out of `dl_grid_points`, 60). So there are two different kinds of `None`:
   - `dl_grid_no_root = 0`: every grid point was tested and none was resolved. The setup can't resolve anything in these conditions.
   - `dl_grid_no_root > 0`: some (or all) grid points could not be tested. `None` may then be a search failure, not a measurement fact; check the inputs.
7. **`attach_uncertainty(result, unc)`** copies `g_s_mol_dl`, `g_s_lo` and `g_s_hi` (all in mol m⁻² s⁻¹) onto the T1.4 result. Nothing is dropped, and the censor label is unchanged.
8. **`card_rbh_uncertainty(...)`** handles the card. It draws card temperature (σ_leaf, same camera), air temperature (σ_air, correlated by ρ) and an Rn multiplier (1 + σ_rn_card × z, Gaussian as in the spec). It recomputes r_bH = ρcₚ(T_card − T_air)/Rn and reports the median, p2.5, p97.5, the **robust sigma = (p84.13 − p15.87)/2**, and the fraction of draws with r_bH ≤ 0. Draws are kept even when r_bH ≤ 0. Draws with |Rn| < 1 W m⁻² are left out of the percentiles (r_bH would divide by almost zero) and counted; if **every** draw is like that, the result is `valid=False, reason=rn_near_zero`. A card reading that `solve_card_rbh` already rejects (`low_card_signal`, `rn_near_zero`, `suspect_card_rbh`, `missing_input`) comes back `valid=False` with the same reason, so a suspect card can never be used as a training label by mistake. (Its r_bH value is still kept by `solve_card_rbh` for the rejects file and the count.)

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

## Pooled estimate for groups of windows (pooled_gs_mol, T1.7)

**What it does.** A drought-stressed leaf can have g_s so low that sensor noise pushes some windows to λE ≤ 0. Those windows are labelled below_detection and have no g_s value. If you average only the windows that do have a g_s, the answer comes out too high, because you threw away exactly the windows where the leaf looked most closed. This is called *truncation bias*. `pooled_gs_mol(readings, results, cfg)` gives one g_s for a group of windows without that bias.

**How it works.**
1. Keep every valid window, censored ones included, and that includes windows with VPD ≤ 0 (censored above_range, flag vpd_nonpositive). λE and VPD are always computed, even when they are ≤ 0, so no window has a "missing" value. Only invalid windows (missing_input) are left out and counted in `n_excluded`.
2. Average λE [W m⁻²], VPD [Pa], r_bH [s m⁻¹], pressure [hPa] and air temperature [°C] over those windows.
3. Apply eq. 8 once to the averages: r_s = ρcₚ·VPD / (γ·λE) − 0.93·r_bH, then g_s,mol = (1/r_s)·P / (R·T_a).
4. If the mean λE ≤ 0 the group is below_detection; if r_s ≤ 0 it is above_range. Both are valid results, as for a single window. If the mean VPD ≤ 0, the result also gets the flag vpd_nonpositive.

**Why the VPD ≤ 0 windows must stay in.** In humid air, noise can push a cool, fast-transpiring leaf's reading below the dew point, so its VPD comes out ≤ 0. Those are the high-g_s windows. In a simulation (RH 97 %, E_s 30 W m⁻², true g_s 0.3 mol m⁻² s⁻¹, σ_leaf 0.5 °C, σ_air 0.25 °C, 200 windows × 20 seeds), about 5 % of windows had VPD ≤ 0:

| Pooled estimate | Mean error over 20 seeds |
| --- | --- |
| all windows kept (the code) | about +6 % |
| VPD ≤ 0 windows dropped (old behaviour) | about −15 % |

At RH 98 % (13 % of windows flagged) dropping them gave about −47 %. The test `test_vpd_nonpositive_windows_kept_and_dropping_them_biases_low` checks the direction. The per-seed error is large at these conditions (worst seed about 35 %), so the 10 % target is not claimed for humid, dim conditions.

Units check for step 3: ρcₚ [J m⁻³ K⁻¹] × VPD [Pa] / (γ [Pa K⁻¹] × λE [W m⁻² = J s⁻¹ m⁻²]) = J m⁻³ / (J s⁻¹ m⁻²) = s m⁻¹, the same units as r_bH. ✓

**Why flux space.** λE = Rn − H changes almost in a straight line with a temperature error (about 23 W m⁻² per °C at r_b = 70 s m⁻¹). g_s = 1/r_s is strongly curved: in the worked example −0.5 °C gives +27 % but +0.5 °C gives only −21 %. Averaging a curved quantity over noisy inputs gives a biased average even when nothing is censored. Averaging the straight-line quantity first and converting once avoids both problems. The alternative tested and rejected was a Tobit fit (a normal distribution cut off at g_s = 0). It still came out 15 % high on average at 0.02 mol m⁻² s⁻¹, because g_s errors are not normal.

**Worked example (the T1.7 truncation test).** True g_s 0.05 mol m⁻² s⁻¹, worked-example conditions (air 24 °C, RH 50 %, 995 hPa, E_s 120 W m⁻², r_bH 70 s m⁻¹, faces 1). The true leaf temperature from the energy balance is 26.60 °C. Each of 200 windows gets independent normal errors on leaf temperature (σ_leaf) and air temperature (σ_air = σ_leaf / 2). This is repeated with 20 random seeds. Error is the estimate divided by the truth, minus 1.

| σ_leaf (°C) | σ_air (°C) | Pooled: mean error | Pooled: worst of 20 seeds | Truncated mean: mean error | Windows censored |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.125 | +0.5 % | 2.9 % | +1.2 % | 0.0 % |
| **0.5** | **0.25** | **+0.9 %** | **5.7 %** | **+3.9 %** | 0.0 % |
| 1.0 | 0.5 | +1.6 % | 11.4 % | +20.3 % | 4.3 % |

Same at true g_s 0.02 mol m⁻² s⁻¹ (true leaf 27.57 °C):

| σ_leaf (°C) | Pooled: mean error | Pooled: worst of 20 seeds | Truncated mean: mean error | Windows censored |
| --- | --- | --- | --- | --- |
| 0.25 | +0.9 % | 5.4 % | +2.2 % | 0.1 % |
| 0.5 | +1.7 % | 10.8 % | +14.2 % | 5.4 % |
| 1.0 | +3.3 % | 21.6 % | +71.6 % | 20.6 % |

The test asserts only the bold row: the pooled estimate is within 10 % on all 20 seeds, and the truncated mean is high on average. A second test at 0.02 mol checks that censored windows really occur and that the truncated mean is high; it does not assert the pooled 10 %. The other rows are reported, not asserted.

**Plainly: the 10 % target holds only when the thermal sensors are calibrated to about σ_leaf ≤ 0.5 °C (and σ_air ≤ 0.25 °C) for the window means.** At the current placeholder (1.0 °C) it fails on some seeds even at 0.05 mol m⁻² s⁻¹, and at 0.02 mol it can be about 20 % off.

Note: at 0.05 mol and σ_leaf 0.5 °C almost no window is censored, so the +3.9 % of the truncated mean there comes from the curve of 1/r_s, not from dropped windows. The real truncation effect shows up in the rows with censored windows (+20 % and +72 %).

**Methods sentence (draft).** Where several windows on the same leaf or plant shared conditions, g_s was estimated by averaging latent heat flux (including non-positive values from censored windows) and VPD across windows and solving Eq. 8 once, which avoids the upward bias of averaging only uncensored g_s values.

**Limitations of the pooled estimate.**
1. **Valid only for windows that share one true g_s and one set of conditions** (replicate windows on one leaf or plant; register A37). Pooling windows with different light, VPD or r_b mixes different leaves' physics.
2. **It removes random window-to-window noise only.** A calibration offset that is the same in every window (for example a thermal camera reading 0.5 °C high all session) does not average out; pooling does nothing for it.
3. **Errors were simulated as independent normal errors on window means.** Real errors may be correlated in time.
4. It does not give an interval. Bootstrapping over windows would; not done yet.

## Correlated card and leaf draws (A43, added in T6.1)
The card and the leaf are seen by **one** thermal camera and share **one** air thermometer. With
`uncertainty.card_leaf_shared_draws: true`, every card_direct draw uses:
1. **One thermal offset** for both: T_leaf = T_leaf,meas + σ_leaf z₁ and T_card = T_card,meas + σ_leaf z₁.
2. **One air temperature** for both (the same T_air draw).
3. **r_bH re-solved from the card in that draw** (eq. 9):
   r_bH = ρc_p (T_card − T_air) / (Rn_card × (1 + σ_rn,card z₆)).
   Draws whose card r_bH ≤ 0 or |Rn_card| < min_abs_rn_wm2 are dropped.

With the key false, r_bH is drawn as before: r_bH,meas × exp(σ_rb z₅), with σ_rb from the card's
own interval.

**Shared errors do not always cancel.**
- *Thermal offset:* it raises T_card and T_leaf together, so it partly cancels in the card's
  r_bH and the leaf's H.
- *Air temperature:* the leaf is often cooler than the air while the card is warmer. A shared
  air error then makes the card's ΔT smaller (r_bH smaller) and the leaf's ΔT more negative, so
  both push λE the same way.

So with the 0.5 °C air placeholder, sharing makes the detection limit worse (nothing resolved
vs 0.29 mol m⁻² s⁻¹). With air 0.2 °C and RH 2 % it halves it (0.054 vs 0.118 mol m⁻² s⁻¹,
tests/test_correlated_draws.py).

**The detection limit also depends on n_draws** when the relative width sits near the 1.0
threshold. Use the full 2000 draws for every reported limit (T6.1 finding).

## Methods paragraph (draft)
Measurement uncertainty was propagated by Monte Carlo simulation (2000 draws per window, fixed seed). Leaf and air temperature errors were drawn as correlated normal deviates (σ = [x] and [y] °C, correlation [ρ], from calibration against a common reference thermometer); relative humidity errors were normal (σ = [z] %) with draws limited to 0–100 %; irradiance and boundary-layer resistance errors were lognormal (log-SD [a] and [b]). Each draw was re-solved with the full energy-balance model. Draws with non-positive latent heat flux were assigned g_s = 0 and draws with non-positive stomatal resistance were assigned the upper limit of the measurable range (2.0 mol m⁻² s⁻¹), so that intervals include closed- and fully-open-stomata outcomes; draws with non-positive VPD and positive latent heat flux have non-positive r_s and were therefore also assigned the upper limit. No draw was excluded. Intervals are the 2.5th–97.5th percentiles. A window was labelled resolved only when all three of the following held: at least 90 % of draws had positive latent heat flux, positive r_s and positive VPD; the relative 95 % interval width, (p97.5 − p2.5)/median, was strictly below 1.0; and the 97.5th percentile was strictly below the upper limit of the measurable range. The per-window detection limit was the smallest true g_s, on a logarithmic grid, meeting all three criteria under the window's conditions; grid points for which no leaf temperature satisfied the energy balance were counted and reported. Resolved status was reported, not used to exclude data. Software: Python 3.13, NumPy, SciPy [versions].

## Limitations
1. **All sigmas are placeholders.** With them, nothing is resolved and the detection limit is undefined (see the worked example). Thermal calibration (T7.5) and light calibration are needed first.
2. **Pressure and surroundings temperature are not perturbed** (C3). If T_sur is taken from the air sensor, its error is really shared with T_air. The longwave effect is about 5.7 W m⁻² K⁻¹ per face, small next to H (≈ 17 W m⁻² K⁻¹ at r_b = 70).
3. **Errors are assumed independent between inputs** (except the leaf–air ρ) and **normal or lognormal**. A calibration offset that is the same in every window is not random within a session; it shifts all windows together.
   - **The card and the leaf share the air thermometer** (register A38). With card_direct, one T_air error enters both the card's r_bH = ρcₚ(T_card − T_air)/Rn_card and the leaf's H = ρcₚ(T_leaf − T_air)/r_bH. Here the r_bH error (σ_rb) is drawn separately from the T_air error, so this correlation is ignored and the interval may be too wide or too narrow. Since T6.1 this can be switched on with `uncertainty.card_leaf_shared_draws: true` (see below); the default is still false.
4. **The detection limit uses a grid** with steps of about 12 % in g_s, so it is accurate to about one grid step.
5. **The resolved band has an upper end too.** At very high g_s the leaf is close to the wet limit and r_s ≤ 0 draws appear, so high values can also be unresolved. Only the lower limit is reported.
6. **The cap (2.0 mol m⁻² s⁻¹) is UNVERIFIED.** It is chosen to be above typical soybean maxima, which have not been checked here.
7. **Card Rn error is Gaussian** (spec), not lognormal. At the card gate Rn ≥ about 24 W m⁻², so a 10 % error never makes it negative.
