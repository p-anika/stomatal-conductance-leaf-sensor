"""Predict the boundary-layer correction f and the corrected r_bH for a leaf (T4.5, eq. 11).

    f_hat           = model(features)                       (-)    (1 when status is no_improvement)
    sd              = sqrt(sd_bootstrap^2 + residual_rms_f^2)       (approximate, see T4.4)
    f_lo, f_hi      = f_hat -/+ z sd,  z = Phi^-1(0.5 + interval_level / 2)   (1.96 for 0.95)
    r_bH,ml         = f_hat x r_bH,flat(d, wind, delta_t, T_air)    s m-1
Flags: ml_no_improvement (model did not beat f = 1; f = 1 returned), outside_training_range
(a feature lies outside the training min-max: the model is extrapolating), plus the flat-plate
convection flag (baseline_unreliable) in predict_rbh.
Missing wind -> reason no_wind; any other missing feature -> missing_input. Never raises on data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from scipy.stats import norm

from processing import energy_balance_utils as ebu

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "correction_model.joblib"
MM_PER_M = 1000.0
WIND = "wind_ms"
_BUNDLE_KEYS = ("model", "bootstrap_models", "model_name", "status", "features", "residual_sd_f",
                "feature_ranges", "interval_level")


@dataclass(frozen=True)
class FPrediction:
    f_hat: float | None
    f_lo: float | None
    f_hi: float | None
    valid: bool
    reason: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class RbPrediction:
    """r_bH values in s m-1 (None when invalid)."""
    r_bh: float | None
    r_bh_lo: float | None
    r_bh_hi: float | None
    r_bh_flat: float | None
    convection_regime: str | None
    f_hat: float | None
    f_lo: float | None
    f_hi: float | None
    valid: bool
    reason: str
    flags: tuple[str, ...]


def load_model(path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Load a bundle saved by train_correction_model.save_model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"correction model not found: {path} (run python -m ml.train_correction_model)")
    bundle = joblib.load(path)
    missing = [k for k in _BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"{path} is not a correction-model bundle (missing {missing})")
    return bundle


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def predict_f(features: Mapping[str, Any], model: dict[str, Any] | None = None) -> FPrediction:
    """f_hat with an approximate interval for one leaf (features named as in ml.features)."""
    b = load_model() if model is None else model
    names = b["features"]
    if not _finite(features.get(WIND)):
        return FPrediction(None, None, None, False, "no_wind", ())
    if not all(_finite(features.get(n)) for n in names):
        return FPrediction(None, None, None, False, "missing_input", ())
    x = np.array([[float(features[n]) for n in names]])
    flags: list[str] = []
    if any(not (lo <= float(features[n]) <= hi) for n, (lo, hi) in b["feature_ranges"].items()):
        flags.append("outside_training_range")
    if b["status"] != "ok":
        flags.insert(0, "ml_no_improvement")
    f_hat = float(b["model"].predict(x)[0])
    boots = [float(m.predict(x)[0]) for m in b["bootstrap_models"]]
    sd_boot = float(np.std(boots, ddof=1)) if len(boots) > 1 else 0.0
    sd = math.sqrt(sd_boot ** 2 + float(b["residual_sd_f"]) ** 2)
    z = float(norm.ppf(0.5 + float(b["interval_level"]) / 2.0))
    return FPrediction(f_hat, f_hat - z * sd, f_hat + z * sd, True, "ok", tuple(flags))


def predict_rbh(features: Mapping[str, Any], wind_ms: float, delta_t_c: float, air_temp_c: float,
                cfg: dict[str, Any], model: dict[str, Any] | None = None) -> RbPrediction:
    """f_hat x flat-plate r_bH for this leaf. delta_t_c = T_leaf - T_air (deg C) enters the
    free-convection term; d = the leaf's d_min_mm or d_eq_mm (physics.characteristic_dimension)."""
    def invalid(reason: str, flags: tuple[str, ...] = ()) -> RbPrediction:
        return RbPrediction(None, None, None, None, None, None, None, None, False, reason, flags)

    if not _finite(wind_ms):
        return invalid("no_wind")
    feats = dict(features)
    feats[WIND] = float(wind_ms)
    pf = predict_f(feats, model)
    if not pf.valid:
        return invalid(pf.reason, pf.flags)
    d_key = f"{cfg['physics']['characteristic_dimension']}_mm"
    if not (_finite(feats.get(d_key)) and _finite(delta_t_c) and _finite(air_temp_c)):
        return invalid("missing_input", pf.flags)
    flat, regime = ebu.flat_plate_rbh(float(feats[d_key]) / MM_PER_M, float(wind_ms), float(delta_t_c),
                                      float(air_temp_c), cfg)
    flags = pf.flags + tuple(f for f in ebu.convection_flags(regime) if f not in pf.flags)
    return RbPrediction(pf.f_hat * flat, pf.f_lo * flat, pf.f_hi * flat, flat, regime,
                        pf.f_hat, pf.f_lo, pf.f_hi, True, "ok", flags)
