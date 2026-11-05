## Appendix C. Glossary

- **Stomata / g\_s:** pores on the leaf surface and how open they are; g\_s is the conductance of water vapor through them.
- **Boundary layer / r\_b:** the thin film of still air on a surface; r\_b is how strongly it resists heat (r\_bH) or vapor (r\_bw) flow.
- **r\_s:** stomatal resistance, the inverse of g\_s.
- **VPD:** vapor pressure deficit, how much drier the air is than the leaf’s interior.
- **Rn, H, λE:** absorbed net radiation, heat carried away by air, and heat carried away by evaporation.
- **ROI:** region of interest, the group of thermal pixels that cover only the leaf, card or wet reference.
- **Dry and wet reference:** surfaces that cannot transpire (dry) or evaporate freely (wet), used to bracket leaf temperature.
- **Petroleum-jelly leaf:** a real leaf coated so it cannot transpire; it keeps its real shape, so its r\_b can be measured by the same algebra as a card.
- **Jones conductance index:** a relative g\_s measure from dry, leaf and wet temperatures.
- **Leave-one-shape-out validation:** testing the model on a card shape it never saw during training.
- **Bland–Altman plot:** a plot of device-minus-reference difference against the average, showing bias and spread.
- **Mixed-effects model:** a statistical model that treats repeated readings from the same plant correctly.
- **Steady state:** leaf temperature has stopped drifting, so the energy-balance equation applies.
- **I2C, STEMMA QT / Qwiic:** the two-wire sensor bus and its plug-in connectors.
- **CSI:** the Raspberry Pi camera connector.

* **Censored value:** a measurement known only to lie above or below a limit (here, g\_s below the detection limit). It is kept and analyzed as such, never deleted.
* **Detection limit:** the smallest g\_s the device can resolve under given conditions, from the uncertainty analysis.

## Appendix D. Revision log: external reviews

Three outside reviews were checked against the physics and this document. Every point is listed with its verdict and where the plan changed.

### Review 1 (five points)

- **λE ≤ 0 truncation: valid and important.** Not fatal as stated, but dropping those windows removes the closed-stomata data and biases stressed means upward, which shrinks the apparent drought effect. Changed: A15; T1.3 to T1.7, T5.1, T5.3, T5.4, T6.1; Appendix A validity rules; truncation test.
- **Pi clock without a network: valid in substance, partly overstated.** Raspberry Pi OS usually restores the last saved time, so expect a stale time rather than 1970 (verify), and sessions at home normally have Wi-Fi. Changed: B14; T7.2, T7.7; decision 15; manifest fields; config. Third review: the typed-offset and RTC parts are deferred.
- **Card and leaf radiative and airflow mismatch: valid.** The error is proportional (5–10% in Rn gives 5–10% in r\_b), not massive, but it is systematic across every training label. Changed: A16, E9, T7.8, T5.8; metadata fields; wind-mapping step; config.
- **Photos every 60 s versus sensor rows every 5 s: valid gap.** It also exposed a missing batch step for photo features. Changed: B15, T3.5; T3.4, T5.1; config.
- **SD-card load from many .npy files: partly valid.** Write bandwidth is only about 0.6 kB/s, so a bottleneck is unlikely, but hundreds of tiny files and power-loss corruption are real. Changed: B13; T0.1, T7.2; layout; schema.

### Review 2 (four points)

- **C is in s^0.5 m⁻¹, not s m⁻¹: correct on units.** My first fix was itself wrong (see review 3). Changed: A10, Appendix A equation 10, config, decisions 4 and 20.
- **The sunlight lux fallback must fail loudly for LED: correct.** Efficacy differs by roughly 2–3×, in either direction depending on the lamp spectrum. Changed: A7, T1.1, T7.6, config, decision 8, definition of done.
- **Add a free-convection term or flag low-wind windows: correct.** The earlier max(U, floor) rule was a crude patch. Changed: A11, T1.2, T1.7, T4.3, T5.2, config, flags, schemas, Section 8.3, decisions 11 and 19.
- **State that card\_direct is primary and ML secondary: correct.** It also fits the schedule, because card\_direct needs neither the wind sensor nor the camera. Changed: Sections 1 and 2, A9, T5.1 to T5.3, config methods, Section 8.4, definition of done, decision 18.

### Review 3 (six points)

- **The convection numbers contradict their own formula: correct.** C = √ν / (s × b × 0.66 × D\_H × Pr^(1/3)) ≈ 313 / (s × b), giving 156 for two faces with no enhancement, 104 for two faces with b = 1.5 and 208 for one face with b = 1.5. The earlier text gave 210 for two faces, and the config mixed s = 1 with faces = 2, which could put r\_b about 2× off. Changed: A10, config (one switch, physics.faces; enhancement factor 1.0 and unverified), T1.2, T1.7, Appendix A, decisions 4 and 20.
- **The card gate was oversold: correct.** At 40 W m⁻² the expected signal is about 2 °C, only about twice the placeholder sigma of 1 °C. It is now a placeholder set from calibrated noise. Changed: T1.3, config, decision 23.
- **Card versus leaf buoyancy: correct, with one change to the suggested test.** A coated leaf warms like the card, so it cannot detect a buoyancy mismatch alone; the test is solved r\_b against temperature difference at fixed wind. Changed: A17, E10, T5.9, T5.3, flags, config, decision 21.
- **Porometer sidedness: correct.** Changed: E4, T5.6, ground-truth schema, decision 22.
- **Deferred items were not marked: correct.** New \[DEFER\] tag on the Tobit fit, the typed-offset clock and RTC logic, fsync tuning and the position-swap check; the cheap versions stay (censored windows are kept, the clock is recorded, the light-field map comes first). Changed: tags in Section 0, T5.3, T5.8, B13, B14, T7.2, decisions 15 and 17, definition of done.
- **Cosmetic: correct on faces = 1, layout not reproducible.** T1.7 and Appendix A now say faces = 1 is set explicitly. Both former appendix tables were intact in the editor (header plus five and four rows), so the split was most likely a page break in an export; the logs are now lists under one heading, which cannot split mid-table.
