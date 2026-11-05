## 9. Decisions only the project owner can make

Each has a default in the code so work can continue. Revisit when the stated input is available.

| # | Decision | Default used | Resolve by |
| --- | --- | --- | --- |
| 1 | Include longwave radiation | Yes, surroundings = air temperature unless measured | Compare with a measured wall or window temperature |
| 2 | Sidedness convention | faces = 2 (both leaf faces, per projected area) | Verify in Campbell & Norman (1998) and Guilioni et al. (2008) |
| 3 | Vapor-to-heat boundary-layer ratio | 0.93 | Verify the value in the same references |
| 4 | Flat-plate convection parameters | Nusselt form, enhancement 1.0 (unverified), two faces, forced plus free (C ≈ 156 s^0.5 m⁻¹; 104 to 313 across conventions) | Check against Campbell & Norman; do not count sidedness twice |
| 5 | Characteristic dimension d | d\_min | Test sensitivity; justify in write-up |
| 6 | Leaf absorptivity per light source | Window 0.5, LED lamp 0.85 | Literature or a measurement |
| 7 | Card coating and its absorptivity | Dark-green matte, a\_s\_card measured (black would be about 0.95) | Measure or justify before building the full card set |
| 8 | Measurement light | LED lamp with the window covered | Buy a lamp; calibrate k (required: no LED fallback) |
| 9 | Wet reference material | Damp filter paper or felt | Choose and test |
| 10 | Ground-truth instrument | Porometer if loaned, else weighed transpiration | Ask university labs, extension offices, teachers |
| 11 | Wind noise floor | 0.1 m/s placeholder | wind\_check.py |
| 12 | Number of card shapes | 15–20 or more, including traced leaflets | Cut from one plastic sheet |
| 13 | Fair date, forms and timeline limits | Unknown | Confirm with the fair’s review committee and Society for Science rules |
| 14 | Second plant cohort | Start now | Seeds and a light source |

**Added after external review (defaults in the code):** (15) Clock fix \[DEFER until Tier 1 works\]: for now only record the clock status and refuse to start on a clearly wrong clock; the typed-in time offset is the later default; buy a $5–15 RTC module if sessions will run without network, for example at a fair demonstration. (16) Lamp uniformity target: irradiance varies by no more than 5% across the card and leaf areas, and c\_card and c\_leaf differ by no more than 0.05. (17) Censoring analysis: default is a rank test plus a sensitivity table, with a Tobit-style plant-level fit added later (DEFER until Tier 1 works); a Bayesian censored model is an alternative if you are comfortable with one.

**Added after second review (defaults in the code):** (18) Primary reference card: a card cut to the shape and size of a typical soybean leaflet, traced from your own leaves, used for card\_direct (config methods.primary\_card\_id). (19) Convection combination rule: sum of the forced and free Nusselt numbers by default; switch to larger-of if Campbell & Norman says so. (20) Flat-plate parameters: enhancement factor, sidedness and characteristic width are still to be verified against Campbell & Norman (1998) and cited. The plausible range of C is about 104 to 313 s^0.5 m⁻¹ depending on those choices, so r\_b could be off by up to about 3×; treat the baseline as provisional until then.

**Added after third review (defaults in the code):** (21) Measurement airflow: run every plant and card session with the fan on low so convection is forced and card\_direct transfers (A17); use fan off only for the ML sub-dataset and checks. (22) Sidedness: faces = 2 drives r\_b, longwave and the g\_s convention together; the porometer comparison is whole-leaf, so ask whoever lends the porometer whether it can read both faces. (23) Card gate threshold (min\_card\_absorbed\_sw\_wm2) is a placeholder until the thermal calibration gives a real noise figure.

**Owner tasks that are not code:** update the handwritten notes to match Appendix A; confirm the TSL2591 faces the same way as the leaf’s upper surface (toward the light), not toward the leaf; check that the sensor plate does not shade the leaf; obtain a reference thermometer and a light meter; buy petroleum jelly and a lamp; build the cards, a lamp jig and a neutral matte backdrop.

## 10. Definition of done

- pytest passes on a laptop, and the end-to-end dry run passes.
- A real 30-minute session on the Pi produces a complete, loadable session folder with sensible leaf, card and reference temperatures and no crashes.
- Thermal, air-temperature and light calibrations are stored with dates, and the uncertainty sigmas come from them.
- Every result row has method, valid, reason and an uncertainty range, and a search of the code finds no hard-coded physical constants.
- Validation scripts have been run on real paired data and report bias and error; the detection limit is documented.
- The ML model was trained on at least the minimum number of distinct shapes, leave-one-shape-out results are reported, and the model card states whether it beats the null model.
- Genotype and treatment comparisons use plant-level mixed models with confidence intervals.
- One command regenerates all figures and tables from raw data.
- README and docs are current and the handwritten notes match Appendix A.
- The camera and wind sensor are integrated, and real sessions fill the image and wind columns.

* Windows with closed stomata are kept as censored values, the share censored per group is reported, and the genotype and treatment comparison does not depend on dropping them.
* Every session has a trustworthy clock record (synchronized, or recorded as unsynchronized), and a session timestamp matches an external record such as a phone; typed-offset and RTC support are deferred.
* Lamp irradiance and wind at the card and leaf positions have been mapped, and the card position-swap check passes.
* Photos are paired to measurement windows by timestamp, and thermal frames are stored in one file per session.

- card\_direct, the primary method, has been validated against ground truth with its error and detection limit reported; ml\_corrected is reported only if it passes validation, and flat\_plate is judged on forced-convection windows only.
- No g\_s is computed for an LED lamp without a calibrated light factor.
