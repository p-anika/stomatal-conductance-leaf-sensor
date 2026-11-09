<!-- Copy of docs/spec/08_appendix_A_equations.md, re-copied 2026-11-08 (T0.6, after T1.7). If the spec changes, re-copy it; do not edit equations here. -->

## Appendix A. Final equation set

All temperatures in °C except in the radiation term (kelvin). Symbols: T\_s leaf (or card) surface temperature, T\_a air temperature, T\_sur surroundings temperature, RH relative humidity (%), P pressure (hPa).

```
1  Shortwave on the leaf        E_s = c_pos x k x lux                                   W m-2
2  Net radiation                Rn = a_s x E_s + n x e_L x sigma x (T_sur^4 - T_s^4)
                                n = faces (1 or 2); e_L = 0.97; sigma = 5.670374419e-8
3  Sensible heat                H = rho cp (T_s - T_a) / r_bH                   rho cp = 1.204 x 1005 = 1210
4  Energy balance               Rn = H + lambdaE   so   lambdaE = Rn - H
5  Vapor pressure               e_s(T) = 610.78 exp(17.27 T / (T + 237.3))      Pa
                                e_a = e_s(T_a) x RH / 100
                                VPD = e_s(T_s) - e_a
6  Psychrometric constant       gamma = 0.0665 x P_hPa                          Pa K-1
7  Latent heat                  lambdaE = (rho cp / gamma) x VPD / (r_s + r_bw)
                                r_bw = ratio x r_bH   (ratio = 0.93)
8  Solve for g_s                r_s = rho cp x VPD / (gamma x lambdaE) - r_bw
                                g_s = 1 / r_s                                   m s-1
                                g_s,mol = g_s x P_Pa / (R x T_a_K)              R = 8.314
9  Card (lambdaE = 0)           r_bH,card = rho cp (T_card - T_a) / Rn_card     uses a_s,card
10 Flat plate baseline          r_bH,flat = 1 / g_bH;  g_bH = s x D_H x (Nu_forced + Nu_free) / d                 Nu_forced = b x 0.66 x Pr^(1/3) x Re^(1/2); Nu_free = 0.54 x Pr^(1/4) x Gr^(1/4); s and b to be verified. Shortcut r = C sqrt(d/U) has C in s^0.5 m-1 (C = 313 / (s x b): 156 for s = 2, b = 1; unverified)
11 ML correction                r_bH,leaf = f_hat(geometry, wind) x r_bH,flat
                                training target f = r_bH,card / r_bH,flat
12 Jones conductance index      I_g = (T_dry - T_leaf) / (T_leaf - T_wet)
13 Water stress index           CWSI = (T_leaf - T_wet) / (T_dry - T_wet)
   c_pos = card or leaf position factor from light_field_map.py (default 1)
```

**Validity rules:** g\_s has a point estimate when lambdaE > 0 and r\_s > 0. lambdaE ≤ 0 means g\_s is at or below the detection limit (censored, below\_detection) and r\_s ≤ 0 means it is above the measurable range (above\_range); both are kept, never dropped. Inputs must be usable: steady state reached and wind measured. For cards the gate is exogenous: absorbed shortwave of at least 40 W m⁻² and |Rn| ≥ 1 W m⁻²; the measured temperature difference is never used to select data.

**Detection-limit rule (added in Session 6, decision A33).** A true g\_s counts as resolved only if all three hold: (1) in at least 90% of the Monte Carlo draws, latent heat flux is above zero, r\_s is above zero and VPD is above zero; (2) the relative 95% interval width is at most uncertainty.max\_relative\_width (config, 1.0); and (3) the 97.5th percentile is below uncertainty.g\_s\_cap\_mol (config, 2.0 mol m-2 s-1, UNVERIFIED). Draws with r\_s at or below zero are held at the cap, so an interval that reaches the cap is not bounded above and must not count as precise.

**Worked example (test fixture):** leaf 23.0, air 24.0, surroundings 24.0, RH 50%, 995 hPa, E\_s 120 W m⁻², a\_s 0.85, faces 1 (set explicitly; the config default is 2), r\_bH 70 s m⁻¹. Result: Rn 107.7, H −17.3, λE 125.0 W m⁻², VPD 1317 Pa, r\_s 127.6 s m⁻¹, g\_s 0.00784 m s⁻¹ = 0.316 mol m⁻² s⁻¹. A ±0.5 °C leaf-temperature error gives 0.402 and 0.250 mol m⁻² s⁻¹.
