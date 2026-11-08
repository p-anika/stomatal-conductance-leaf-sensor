# physics_baseline (processing/physics_baseline.py, task T1.3)

## What it does
A leaf's temperature depends on two unknowns at once: how fast it loses water (which is what we want, stomatal conductance) and how easily heat leaves its surface. That second one is the **boundary-layer resistance, r_bH**: the thin layer of still air on a surface slows heat on its way out. This module measures r_bH with a **dry reference card**, a leaf-shaped piece of painted material that cannot evaporate water. The card sits next to the leaf in the same light and air, and its r_bH is then used for the leaf.

## How it works
1. **`ReferenceCardReading`** holds the inputs for one measurement window. These are window means (one average per window), not single 5-second samples: card temperature `card_temp_c` (°C), air temperature `air_temp_c` (°C), surroundings temperature `surroundings_temp_c` (°C), light `lux`, the light source name, and the card position's light factor `c_card`.
2. **`card_absorptivity(cfg, cal)`** finds the card's shortwave absorptivity a_s_card, checking calibration.json first and then config.yaml. If neither has a value, the code stops with `CalibrationMissing`. Only if you set `card.allow_default_a_s_card: true` does it use the placeholder (0.95, UNVERIFIED), and then it adds the `a_s_default` flag.
3. **`card_emissivity(cfg, cal)`** finds the card's longwave emissivity in the same order.
4. **`solve_card_rbh(reading, cfg, cal)`** does the following:
   - Converts light: E_s,card = c_card × k × lux, in W m⁻² (eq. 1). k comes from the light calibration. If `c_card` is not given, the card position factor from calibration is used, or 1.0 with the flag `c_position_default` if it hasn't been measured.
   - Computes net radiation: Rn = a_s_card × E_s + n × ε × σ × (T_sur⁴ − T_card⁴), with temperatures in kelvin (eq. 2).
   - Runs the checks in this order: `missing_input`, then `low_card_signal`, then `rn_near_zero`, then `suspect_card_rbh`, otherwise `ok`.
   - Returns r_bH = ρcₚ (T_card − T_air) / Rn in s m⁻¹ (eq. 9), packed into a `CardResult(r_bh, rn_wm2, valid, reason, flags)`.
5. **`count_reasons(results)`** counts the results by reason. The training-set report (T4.3) will use it to print how many `suspect_card_rbh` readings there were.

## Why it is done this way

### Why the card gives r_bH directly
Energy coming into a surface must go somewhere. For a leaf:

  **Rn = H + λE**: net radiation in = heat carried off by air (H) + energy used to evaporate water (λE).

On a leaf there are two unknowns in that sentence, H (which depends on r_bH) and λE (which depends on the stomata). The card has no water, so **λE = 0**. The balance becomes simply **Rn = H**. H is written as

  H = ρcₚ (T_card − T_air) / r_bH

which says: heat flows from warm card to cooler air, faster when the temperature gap is bigger and slower when the resistance is bigger. Put Rn = H and solve for the one unknown left:

  **r_bH = ρcₚ (T_card − T_air) / Rn**

Everything on the right is measured: temperatures from the thermal camera and sensors, and Rn from the light reading and the card's absorptivity.

**Units check:** ρcₚ is J m⁻³ K⁻¹. (T_card − T_air) is K. Rn is W m⁻² = J s⁻¹ m⁻². So (J m⁻³ K⁻¹ × K) / (J s⁻¹ m⁻²) = (J m⁻³) × (s m² J⁻¹) = **s m⁻¹** ✓, the unit of a resistance.

ρcₚ = 1.204 kg m⁻³ × 1005 J kg⁻¹ K⁻¹ = 1210.02 J m⁻³ K⁻¹ (config values, assumption A2).

The card must use **its own** absorptivity a_s_card, not the leaf's. A matte black card absorbs much more light than a leaf, so using the leaf's value would make Rn, and therefore r_bH, wrong.

### Why the gate looks only at the light (selection bias)
Some readings are too weak to use: if there is little light, the card hardly warms, and r_bH becomes a small, noisy number divided by a small number. We must throw those away. The question is **what we look at when deciding**.

- **Wrong way:** drop readings where the measured T_card − T_air is small, or where r_bH comes out negative. The measured difference = true difference + camera noise. When noise happens to push it *up*, the reading is kept. When noise pushes it *down*, it is dropped. The kept readings therefore have, on average, a temperature difference that is too big, so r_bH is **biased upward**. These r_bH values become the training labels for the ML correction, so the whole model would learn values that are too high.
- **Right way (used here):** decide using things that are fixed *before* we look at the card's temperature. Here that means the absorbed light a_s_card × E_s,card. Light is not affected by camera noise on the card, so this cut keeps high-noise and low-noise readings equally and does not tilt the average. This is called an **exogenous** gate.

The rule is: absorbed shortwave below `physics.min_card_absorbed_sw_wm2` (40 W m⁻², a placeholder) gives `low_card_signal`. The test `test_gate_ignores_card_temperature` checks that changing only T_card never changes the gate decision.

**The |Rn| check is only a divide-by-zero guard.** Rn includes the card's own longwave loss, which depends on the measured T_card, so this check is not purely exogenous. Once the light gate passes, Rn can only fall below 1 W m⁻² if the card is about 3.5 °C hotter than its surroundings or the surroundings are very cold, so in practice it almost never triggers. It is there so the code never divides by nearly zero.

**r_bH ≤ 0 is kept, labelled `suspect_card_rbh`, and counted.** The value is returned so it can go to the rejects file. The spec excludes these readings from training, and that exclusion is a cut on the sign of r_bH, so it does carry some upward bias (see Limitations). The count is the warning light: if it keeps rising, the temperature calibration is probably off, rather than the noise being unlucky.

### How big is the signal at the gate? (about 1.4 °C, not 2 °C)
The spec's "about 2 °C at 40 W m⁻²" ignores longwave radiation. A card warmer than its surroundings also *radiates* heat away. Per °C of warming, both faces together lose

  n × ε × 4σT³ = 2 × 0.95 × 4 × 5.67×10⁻⁸ × 298.15³ ≈ **11.4 W m⁻² K⁻¹**

(units: W m⁻² K⁻⁴ × K³ = W m⁻² K⁻¹ ✓). At steady state, with r_b = 70 s m⁻¹:

  ΔT = Rn × r_b / ρcₚ  and  Rn = 40 − 11.4 ΔT
  → Rn = 40 / (1 + 11.4 × 70 / 1210) = 40 / 1.66 ≈ 24.1 W m⁻²
  → ΔT ≈ 24.1 × 70 / 1210 ≈ **1.39 °C** (2.31 °C if longwave is ignored)

With the placeholder camera noise σ = 1 °C, that is only about 1.4 σ. If the error were normal, about **8 %** of good readings at the gate would come out ≤ 0 by noise alone (P(z < −1.39) ≈ 0.082). So 40 W m⁻² is a **placeholder**. After thermal calibration (T7.5), set the gate so the expected signal is at least about 5 times the calibrated noise of a window mean. Card sessions need enough light to clear that gate.

## Worked example
Inputs: LED lamp with k = 0.01 W m⁻² per lux (example calibration), lux = 12000, c_card = 1.0, a_s_card = 0.95, card emissivity 0.95, faces n = 2, T_card = 28.0 °C, T_air = 24.0 °C, T_sur = 24.0 °C.

1. E_s,card = 1.0 × 0.01 × 12000 = **120 W m⁻²**
2. Absorbed shortwave = 0.95 × 120 = **114 W m⁻²**. That is at least 40, so the gate passes.
3. Longwave: T_sur = 297.15 K, T_card = 301.15 K
   2 × 0.95 × σ × 297.15⁴ = 839.98 W m⁻²; 2 × 0.95 × σ × 301.15⁴ = 886.13 W m⁻²
   Longwave = 839.98 − 886.13 = **−46.15 W m⁻²** (the card loses heat by radiation)
4. Rn = 114 − 46.15 = **67.85 W m⁻²**. |Rn| ≥ 1, so the guard passes.
5. r_bH = 1210.02 × (28.0 − 24.0) / 67.85 = 4840.1 / 67.85 = **71.3 s m⁻¹**. This is > 0, so the reason is `ok`.

Sensitivity: T_card = 27.0 °C gives 45.6 s m⁻¹ and 29.0 °C gives 108.0 s m⁻¹. A ±1 °C card-temperature error moves r_bH by about −36 % / +51 %, which is why calibrated thermal noise matters (T7.5).

The same numbers come out of the code:
`solve_card_rbh(ReferenceCardReading("round_1", 28.0, 24.0, 24.0, 12000.0, "led_lamp", 1.0), cfg, cal)` → r_bh 71.34, rn_wm2 67.85.

## Methods paragraph (draft)
The boundary-layer resistance to heat transfer (r_bH) was measured with a dry reference card placed beside the leaf. Because the card does not transpire, its latent heat flux is zero and its energy balance reduces to Rn = H, giving r_bH = ρcₚ (T_card − T_air) / Rn (Appendix A eq. 9). Net radiation used the card's own shortwave absorptivity, its longwave emissivity, and a calibrated lux-to-irradiance factor with a position-specific irradiance factor. Calculations used 60-s window means. To avoid selection bias, readings were screened only on exogenous conditions: windows with absorbed shortwave below [threshold, set from calibrated thermal noise] W m⁻² were excluded before card temperatures were examined, and a guard excluded |Rn| < 1 W m⁻². Windows yielding r_bH ≤ 0 were retained in a rejects file, and their count and fraction are reported. Software: Python 3.13 [versions from scripts/snapshot_versions.py].

## Limitations
1. **Card = leaf assumption (A16, A17).** The card's r_bH is used for the leaf. This fails if the card gets different light or airflow, or at low wind, where a warmer card has stronger buoyant flow than a transpiring leaf.
2. **Placeholder gate.** At 40 W m⁻² the expected signal is only about 1.4 °C (1.4 σ). Until T7.5 sets the threshold from measured noise, readings near the gate are noisy.
3. **Remaining sign-cut bias.** Dropping r_bH ≤ 0 from training is a cut on the outcome and biases labels upward. The bias is small only if the gate makes noise-driven negatives rare (about 5σ). At the placeholder gate, about 8 % of good readings could go negative by noise. T4.3 must print the fraction of negative r_bH values before any are dropped.
4. **a_s_card not measured.** It is null in config. The code refuses to run unless calibrated or explicitly allowed to use the UNVERIFIED 0.95 placeholder (flag `a_s_default`). An a_s_card error of x % changes Rn's shortwave part, and so r_bH, by about the same amount.
5. **Window means are not enforced here.** The function cannot tell a single 5-s sample from a window mean. T3.4 and T4.3 must enforce window means.
6. **ρcₚ fixed at 20 °C, 1013 hPa** (A18).
