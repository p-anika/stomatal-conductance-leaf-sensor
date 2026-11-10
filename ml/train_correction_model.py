"""Train the boundary-layer correction model f(geometry, wind) (T4.4, Appendix A eq. 11).

    python -m ml.train_correction_model [--data data/processed/rb_training_set.csv] [--out-dir data/models]

Target f = r_bH,card / r_bH,flat on a linear scale (owner decision 2026-11-09), unweighted.
Candidates: null (f = 1, always included), ridge (standardised features), small random forest.
Scores are errors in r_bH, s m-1:  r_bH,pred = f_pred x r_bH,flat;  RMSE and MAE vs r_bH,card.

Cross-validation is always leave-one-shape-out (LeaveOneGroupOut on shape_group), so no shape is
ever in both the training and the test part.
  inner (selection): LOGO on all rows, ridge vs forest -> chosen model (optimistic: it was picked
                     on these same scores)
  outer (honest):    nested LOGO; for each held-out shape the model is chosen by an inner LOGO on
                     the other shapes only, then scored on the held-out shape
Go / no-go: improvement = (RMSE_null - RMSE_nested) / RMSE_null must be >= ml.min_improvement,
else status no_improvement and the saved model is the null model (prediction f = 1 + flag).

Prediction interval (T4.5) = bootstrap spread (ml.n_bootstrap resamples of whole shape groups)
and the out-of-fold residual RMS of f combined in quadrature. Approximate: assumes the residual
spread is the same for every shape and errors are roughly normal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from processing.config import load_config

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = "correction_model.joblib"
CARD_FILE = "correction_model_card.json"
TARGET = "f_ratio"
NULL, RIDGE, FOREST = "null", "ridge", "forest"
LEARNED = (RIDGE, FOREST)
INTERVAL_NOTE = ("f_lo/f_hi = f_hat -/+ z x sqrt(sd_bootstrap^2 + residual_rms_f^2); approximate: assumes a "
                 "constant residual spread across shapes and roughly normal errors")


class NullModel:
    """f = 1 for every input (the flat plate unchanged). Has fit/predict like a scikit-learn model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X))


def make_candidates(cfg: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    m = cfg["ml"]
    return {
        NULL: NullModel,
        RIDGE: lambda: make_pipeline(StandardScaler(), Ridge(alpha=float(m["ridge_alpha"]))),
        FOREST: lambda: RandomForestRegressor(n_estimators=int(m["forest_n_estimators"]),
                                              max_depth=int(m["forest_max_depth"]),
                                              random_state=int(m["random_state"])),
    }


def logo_splits(df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-shape-out splits (row positions); each test part is exactly one shape_group."""
    yield from LeaveOneGroupOut().split(np.zeros(len(df)), groups=df["shape_group"].to_numpy())


def _check(df: pd.DataFrame, features: list[str]) -> None:
    need = features + [TARGET, "r_bh_flat", "r_bh_card", "shape_group"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"training data is missing column(s) {missing}")
    bad = [c for c in features + [TARGET, "r_bh_flat", "r_bh_card"] if df[c].isna().any()]
    if bad:
        raise ValueError(f"training data has blank values in {bad}")
    if df["shape_group"].nunique() < 2:
        raise ValueError("leave-one-shape-out needs at least 2 shape groups")


def _oof_f(df: pd.DataFrame, factory: Callable[[], Any], features: list[str]) -> np.ndarray:
    """Out-of-fold f predictions by LOGO."""
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(float)
    out = np.empty(len(df))
    for tr, te in logo_splits(df):
        out[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return out


def rb_errors(df: pd.DataFrame, f_pred: np.ndarray) -> dict[str, float]:
    """RMSE and MAE of r_bH (s m-1) with r_bH,pred = f_pred x r_bH,flat."""
    err = f_pred * df["r_bh_flat"].to_numpy(float) - df["r_bh_card"].to_numpy(float)
    return {"rmse_s_m": float(math.sqrt(np.mean(err ** 2))), "mae_s_m": float(np.mean(np.abs(err)))}


def logo_scores(df: pd.DataFrame, cfg: dict[str, Any], names: tuple[str, ...]) -> tuple[dict, dict]:
    """({name: errors}, {name: out-of-fold f}) for the named candidates."""
    cands = make_candidates(cfg)
    features = list(cfg["ml"]["features"])
    oof = {n: _oof_f(df, cands[n], features) for n in names}
    return {n: rb_errors(df, oof[n]) for n in names}, oof


def nested_logo(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Honest score of 'pick ridge or forest by inner LOGO, then fit': outer LOGO predictions."""
    cands = make_candidates(cfg)
    features = list(cfg["ml"]["features"])
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(float)
    pred = np.empty(len(df))
    choices: dict[str, int] = {n: 0 for n in LEARNED}
    for tr, te in logo_splits(df):
        inner = df.iloc[tr].reset_index(drop=True)
        if inner["shape_group"].nunique() >= 2:
            scores, _ = logo_scores(inner, cfg, LEARNED)
            pick = min(LEARNED, key=lambda n: scores[n]["rmse_s_m"])
        else:
            pick = RIDGE
        choices[pick] += 1
        pred[te] = cands[pick]().fit(X[tr], y[tr]).predict(X[te])
    outer = rb_errors(df, pred)
    null = rb_errors(df, np.ones(len(df)))
    return {"outer_rmse_s_m": outer["rmse_s_m"], "outer_mae_s_m": outer["mae_s_m"],
            "null_rmse_s_m": null["rmse_s_m"], "null_mae_s_m": null["mae_s_m"],
            "improvement_vs_null": (null["rmse_s_m"] - outer["rmse_s_m"]) / null["rmse_s_m"],
            "fold_choices": choices}


@dataclass(frozen=True)
class TrainedModel:
    bundle: dict[str, Any]     # what predict_correction needs (saved with joblib)
    card: dict[str, Any]       # model card (saved as JSON)


def _bootstrap(df: pd.DataFrame, factory: Callable[[], Any], features: list[str], n: int,
               seed: int) -> list[Any]:
    """Refit on n resamples of whole shape groups (rows of one shape stay together)."""
    rng = np.random.default_rng(seed)
    groups = df["shape_group"].unique()
    by_group = {g: np.flatnonzero(df["shape_group"].to_numpy() == g) for g in groups}
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(float)
    models = []
    for _ in range(n):
        idx = np.concatenate([by_group[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        models.append(factory().fit(X[idx], y[idx]))
    return models


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}


def train_correction_model(df: pd.DataFrame, cfg: dict[str, Any],
                           data_sha256: str | None = None) -> TrainedModel:
    """Score the candidates, decide against the null model, fit and bootstrap (module docstring)."""
    m = cfg["ml"]
    features = list(m["features"])
    df = df.reset_index(drop=True)
    _check(df, features)
    if data_sha256 is None:
        data_sha256 = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()

    logo, oof = logo_scores(df, cfg, (NULL,) + LEARNED)
    chosen = min(LEARNED, key=lambda n: logo[n]["rmse_s_m"])
    nested = nested_logo(df, cfg)
    ok = nested["improvement_vs_null"] >= float(m["min_improvement"])
    final_name = chosen if ok else NULL
    status = "ok" if ok else "no_improvement"
    logger.info("chosen %s; nested improvement over null %.1f%% -> %s", chosen,
                100 * nested["improvement_vs_null"], status)

    cands = make_candidates(cfg)
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(float)
    final = cands[final_name]().fit(X, y)
    boots = [] if final_name == NULL else _bootstrap(df, cands[final_name], features, int(m["n_bootstrap"]),
                                                     int(m["random_state"]))
    resid = df[TARGET].to_numpy(float) - oof[final_name]
    residual_rms_f = float(math.sqrt(np.mean(resid ** 2)))
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in features}

    bundle = {"model": final, "bootstrap_models": boots, "model_name": final_name, "status": status,
              "features": features, "residual_sd_f": residual_rms_f, "feature_ranges": ranges,
              "interval_level": float(m["interval_level"])}
    card = {
        "status": status, "chosen_model": chosen, "saved_model": final_name, "target": "f (linear scale)",
        "features": features,
        "metrics": {
            "logo": logo,
            "inner_selection": {"candidates": {n: logo[n] for n in LEARNED}, "chosen": chosen,
                                "improvement_vs_null": (logo[NULL]["rmse_s_m"] - logo[chosen]["rmse_s_m"])
                                / logo[NULL]["rmse_s_m"],
                                "note": "optimistic: the model was chosen on these same scores"},
            "nested": nested,
            "min_improvement": float(m["min_improvement"]),
        },
        "training_data_sha256": data_sha256, "n_rows": int(len(df)),
        "n_shapes": int(df["shape_group"].nunique()), "library_versions": _versions(),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_bootstrap": len(boots), "residual_sd_f": residual_rms_f, "feature_ranges": ranges,
        "interval_level": float(m["interval_level"]), "interval_note": INTERVAL_NOTE,
        "settings": {k: m[k] for k in ("ridge_alpha", "forest_n_estimators", "forest_max_depth",
                                       "random_state")},
    }
    return TrainedModel(bundle, card)


def save_model(trained: TrainedModel, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path, card_path = out / MODEL_FILE, out / CARD_FILE
    joblib.dump(trained.bundle, model_path)
    card_path.write_text(json.dumps(trained.card, indent=2), encoding="utf-8")
    return model_path, card_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ml.train_correction_model",
                                description="Train the r_bH correction model with leave-one-shape-out CV (T4.4).")
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "processed" / "rb_training_set.csv"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "models"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"training data not found: {data} (run python -m ml.build_training_set first)")
    df = pd.read_csv(data)
    trained = train_correction_model(df, load_config(args.config),
                                     data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    model_path, card_path = save_model(trained, args.out_dir)
    n = trained.card["metrics"]["nested"]
    print(f"status: {trained.card['status']}  chosen: {trained.card['chosen_model']}  "
          f"nested RMSE {n['outer_rmse_s_m']:.2f} s/m vs null {n['null_rmse_s_m']:.2f} s/m "
          f"({100 * n['improvement_vs_null']:.1f}% better; need {100 * trained.card['metrics']['min_improvement']:.0f}%)")
    print(f"wrote {model_path} and {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
