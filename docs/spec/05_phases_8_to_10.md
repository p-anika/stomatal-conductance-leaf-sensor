### Phases 8 and 9: Camera and wind sensor \[BLOCKED\]

See Section 8 for the complete task lists, what to write now, and what to do when each part arrives.

### Phase 10: Real-data integration and final analysis \[OWNER plus code; needs real data\]

- **T10.1** Run the calibrations (T7.4–T7.6) and the light-field map (T7.8) and store them.
- **T10.2 Card campaign.** Each card in the manifest, at three or more wind conditions (after the FS3000 arrives) and three or more light levels, with repeats; sessions marked dry\_reference\_type = card. Include traced soybean-leaflet cards and at least one petroleum-jelly leaf session.
- **T10.3 Wet-reference and coated-leaf sessions** for the detection-limit brackets and the Jones index.
- **T10.4 Plant sessions.** Two genotypes × two treatments × 4–5 plants, on several days, in randomized blocks; porometer or weighed-transpiration pairs on a subset, typed into data/validation/ground\_truth\_gs.csv.
- **T10.5 Run the pipeline:** build windows, build training set, train, aggregate results, benchmark, validate, statistics, figures.
- **T10.6 Review.** Inspect QC flags and exclusion counts, confirm the detection limit, state honestly which tier of claim the data supports, and finalize docs/assumptions\_and\_limitations.md.
