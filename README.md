# Stomatal Conductance Leaf Sensor

## Structure
- `hardware_tests/` -- one-off per-sensor diagnostics (unchanged from before)
- `acquisition/` -- field scripts: session metadata (module 8), combined logger (modules 2+3)
- `processing/` -- physics: reference-card baseline (module 5), g_s calculator (module 7),
  shared constants/helpers in `energy_balance_utils.py`
- `ml/` -- correction model training/inference (module 6) -- not yet implemented
- `analysis/` -- end-of-campaign comparison scripts (module 9) -- not yet implemented
- `data/raw/` -- one subfolder per session, plus `session_metadata.csv`
- `data/processed/`, `data/models/` -- created as ml/ and analysis/ scripts are added

## Running things
All scripts assume you're in the repo root, and import each other as
packages (e.g. `acquisition.session_metadata`). Run them as modules:

```bash
python -m acquisition.combined_logger      # field session, on the Pi
python -m processing.physics_baseline      # sanity check, synthetic data
python -m processing.gs_calculator         # sanity check, synthetic data
```

Hardware tests still run directly, no change:
```bash
python hardware_tests/test_bme280.py
```

## Status
- session_metadata.py, physics_baseline.py, gs_calculator.py: implemented,
  unit-testable with synthetic data (see each file's `__main__` block).
- camera_capture.py, leaf_segmentation.py, ml/*, analysis/*: not yet written --
  next once calipers/plate/Arducam mounting are sorted.
