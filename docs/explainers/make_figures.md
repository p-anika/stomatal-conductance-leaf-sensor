# make_figures (analysis/make_figures.py)

## What it does
One command rebuilds every figure and table for the paper from the files in `data/`:

```
python -m analysis.make_figures            # real data
python -m analysis.make_figures --synthetic
python -m analysis.make_figures --figures-only   # redraw from existing tables, do not re-run analyses
```

It writes figures to `docs/figures/`, tables to `docs/tables/`, and a checklist,
`docs/tables/index.md` (or `index_synthetic.md`). The checklist says which items were built and
which are missing, and why.

## How it works
1. **Run the analyses in order.** Each step uses its own command:
   1. aggregate_results (T5.1)
   2. detection_limit (T5.4)
   3. validate_gs_ground_truth (T5.6)
   4. baseline_benchmark (T5.2)
   5. genotype_comparison (T5.3)
   6. validate_rb_vaseline (T5.5)
   7. card_buoyancy_check (T5.9)

   What happens when a step can't complete:
   - If its input files are missing, it is **skipped** and the reason is recorded. Older outputs
     on disk are still used.
   - If it crashes for any other reason, it is **failed** and the command exits with code 1.
   - The ML training set and model (T4.3–T4.5) are not re-made here. They are inputs.
2. **Draw the figures:**

   | Figure | From |
   | --- | --- |
   | `gs_by_group` | `leaf_gs_results.csv`: plant means per genotype × treatment, group mean with 95 % bootstrap CI, censored windows at half the detection limit |
   | `method_comparison` | T5.6 plant-level bias and RMSE per method |
   | `bland_altman` | drawn by T5.6 |
   | `detection_limit` | drawn by T5.4 |
   | `ml_cv` | `correction_model_card.json`: cross-validated RMSE per candidate, and the honest nested estimate vs f = 1 |
   | `calibration_residuals` | missing until the thermal and light calibration scripts (T7.x) produce a residual table |
   | `below_detection` | T5.3 share of below-detection windows per group |
   | `vaseline_rb` | T5.5 predicted / solved r_bH per coated leaf |
   | `card_buoyancy` | T5.9 fractional change per card and wind, with the ±0.15 threshold |
3. **Copy the tables** (every analysis CSV) into `docs/tables/`.
4. **Write the checklist.** With `--synthetic`, every file name ends in `_synthetic` and every
   title and the checklist say SYNTHETIC.

## Why it is done this way
- **Nothing in the paper should be made by hand.** One command from raw data means any number
  can be traced and rebuilt after a fix.
- **Skip instead of crash.** Most inputs (porometer data, wind, coated leaves) arrive at
  different times; the checklist shows exactly what is still missing.

## Worked example
On a fresh folder with only `leaf_gs_results_synthetic.csv`, the T5.3 and T5.6 tables and a model
card, `--synthetic --figures-only` builds `gs_by_group`, `method_comparison`, `ml_cv` and
`below_detection`. It lists `calibration_residuals` as missing ("T7.x not run") and
`vaseline_rb` and `card_buoyancy` as missing (no inputs).

## Methods paragraph (draft)
All figures and tables were generated from the stored data by a single script
(`python -m analysis.make_figures`), which re-runs every analysis step in order and records any
missing inputs.

## Limitations
- It does not retrain the ML model; re-run `python -m ml.build_training_set` and
  `python -m ml.train_correction_model` first when card data change.
- The `ml_cv` figure has no `_synthetic` source of its own: it reads whichever model card is in
  `data/models/`.
