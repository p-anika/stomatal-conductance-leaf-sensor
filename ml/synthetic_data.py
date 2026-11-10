"""Synthetic cards and sessions with a known answer, in exactly the real file schemas (T4.2).

SYNTHETIC RESULTS PROVE THE CODE WORKS. THEY SAY NOTHING ABOUT REAL PLANTS.
Every number used here comes from config.yaml section `synthetic` and is invented.

Hidden correction (what the ML step must later recover), per card or leaf shape:
    f = 1 + a (1 - solidity) + b (aspect_ratio - 1) + c edge_density + N(0, noise_sd)
    true r_bH = f x flat-plate r_bH(d_min, wind, T_s - T_a)                        s m-1  (eq. 10, 11)

Forward physics (the inverse solvers are never called here; only energy_balance_utils helpers):
    card: find T_card with   Rn(T) - rho cp (T - T_a) / r_bH(T) = 0                       (eq. 2, 3, 9)
    leaf: find T_leaf with   Rn(T) - rho cp (T - T_a) / r_bH(T)
                               - (rho cp / gamma) VPD(T) / (r_s + 0.93 r_bH(T)) = 0         (eq. 2-7)
          r_s = molar density / g_s,mol                                                    (eq. 8 inverted)
Then per-sample noise (synthetic.noise) is added, plus, if synthetic.session_bias is true, one
offset per session drawn from the uncertainty sigmas (one thermal offset shared by leaf and card,
since one camera sees both). Surroundings temperature = air temperature (assumption A4/A41).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from acquisition.session_metadata import SessionMetadata, make_session
from processing import energy_balance_utils as ebu
from processing.config import load_config
from processing.data_io import CARD_COLUMNS_ORDER, EVENTS_COLUMNS, MANIFEST_FIELDS, SENSOR_LOG_COLUMNS
from processing.extract_photo_features import OUTPUT_COLUMNS as PHOTO_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
LIGHT_SOURCE = "led_lamp"            # synthetic sessions use the lamp; k comes from synthetic config
MM_PER_M = 1000.0
SECONDS_PER_MINUTE = 60.0


def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return load_config(REPO_ROOT / "config.yaml") if cfg is None else cfg


# ---------------------------------------------------------------- shapes and the hidden correction

def hidden_correction(geometry: pd.DataFrame | pd.Series, cfg: dict[str, Any]) -> Any:
    """Deterministic part of f (dimensionless): 1 + a(1 - solidity) + b(aspect - 1) + c edge_density."""
    s = cfg["synthetic"]
    return (1.0 + s["correction_a"] * (1.0 - geometry["solidity"])
            + s["correction_b"] * (geometry["aspect_ratio"] - 1.0)
            + s["correction_c"] * geometry["edge_density"])


def _draw_geometry(rng: np.random.Generator, s: dict[str, Any], n: int) -> pd.DataFrame:
    """Ellipse-like shapes whose features agree with each other (plausible, not exact):
    d_max = aspect x d_min; area = solidity x pi/4 x d_min x d_max (a dented ellipse);
    d_eq = sqrt(4A/pi); eccentricity = sqrt(1 - 1/aspect^2) (ellipse with that aspect);
    perimeter = Ramanujan ellipse perimeter x (1 + 2 (1 - solidity)) (dents lengthen the edge);
    circularity = 4 pi A / P^2."""
    d_min = rng.uniform(*s["d_min_mm_range"], n)
    aspect = rng.uniform(*s["aspect_ratio_range"], n)
    solidity = rng.uniform(*s["solidity_range"], n)
    edge = rng.uniform(*s["edge_density_range"], n)
    d_max = aspect * d_min
    area = solidity * math.pi / 4.0 * d_min * d_max
    a, b = d_max / 2.0, d_min / 2.0
    p_ellipse = math.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    perimeter = p_ellipse * (1.0 + 2.0 * (1.0 - solidity))
    return pd.DataFrame({
        "area_mm2": area, "d_min_mm": d_min, "d_eq_mm": np.sqrt(4.0 * area / math.pi),
        "aspect_ratio": aspect, "eccentricity": np.sqrt(1.0 - 1.0 / aspect ** 2),
        "solidity": solidity, "circularity": np.minimum(4.0 * math.pi * area / perimeter ** 2, 1.0),
        "edge_density": edge, "d_max_mm": d_max, "perimeter_mm": perimeter,
    })


def make_synthetic_cards(n_shapes: int = 20, seed: int = 0, cfg: dict[str, Any] | None = None,
                         null: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cards, truth). cards has exactly the data/reference_cards.csv columns (Appendix B), one
    card per shape_group. truth (never written into the schema table): card_id, shape_group,
    f_deterministic, f_true, d_max_mm, perimeter_mm. null=True gives f_true = 1 for every card
    (a set with no real correction, to check that later ML does not invent one)."""
    cfg = _cfg(cfg)
    s = cfg["synthetic"]
    rng = np.random.default_rng(seed)
    geo = _draw_geometry(rng, s, n_shapes)
    prefix = f"{'NULL' if null else 'SYN'}{seed}"
    ids = [f"{prefix}_C{i:02d}" for i in range(n_shapes)]
    groups = [f"{prefix}_G{i:02d}" for i in range(n_shapes)]
    cards = pd.DataFrame({
        "card_id": ids, "shape_group": groups, "source": "designed", "material": "synthetic",
        "thickness_mm": s["card_thickness_mm"], "coating": "matte_black", "a_s_card": s["a_s_card"],
        "emissivity": cfg["card"]["emissivity"], **{c: geo[c] for c in
        ("area_mm2", "d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity", "circularity",
         "edge_density")},
        "geometry_source": "camera", "photo_path": "", "notes": "SYNTHETIC (ml/synthetic_data.py)",
    })[list(CARD_COLUMNS_ORDER)]
    f_det = hidden_correction(geo, cfg)
    noise = rng.normal(0.0, s["correction_noise_sd"], n_shapes)
    truth = pd.DataFrame({"card_id": ids, "shape_group": groups,
                          "f_deterministic": np.ones(n_shapes) if null else f_det,
                          "f_true": np.ones(n_shapes) if null else f_det + noise,
                          "d_max_mm": geo["d_max_mm"], "perimeter_mm": geo["perimeter_mm"]})
    return cards, truth


def make_card_sets(cfg: dict[str, Any] | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """train (synthetic.train_seed), held-out test (synthetic.test_seed) and null (f_true = 1)."""
    cfg = _cfg(cfg)
    s = cfg["synthetic"]
    n = int(s["n_shapes"])
    return {"train": make_synthetic_cards(n, s["train_seed"], cfg),
            "test": make_synthetic_cards(n, s["test_seed"], cfg),
            "null": make_synthetic_cards(n, s["train_seed"], cfg, null=True)}


def write_reference_cards(cards: pd.DataFrame, path: str | Path) -> Path:
    """Write a cards table as a reference_cards.csv-format file (for loader checks)."""
    path = Path(path)
    cards[list(CARD_COLUMNS_ORDER)].to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- forward physics

def card_rbh_of_t(f: float, d_min_mm: float, wind_ms: float, air_c: float, cfg: dict[str, Any]):
    """r_bH(T) = f x flat-plate r_bH at this card's d_min, s m-1 (T enters through Grashof)."""
    d_m = d_min_mm / MM_PER_M
    return lambda t: f * ebu.flat_plate_rbh(d_m, wind_ms, t - air_c, air_c, cfg)[0]


def forward_card_temp_c(r_bh_of_t, air_c: float, sur_c: float, e_s_wm2: float, a_s: float,
                        emissivity: float, cfg: dict[str, Any]) -> float:
    """Root of Rn_card(T) - rho cp (T - T_a) / r_bH(T) = 0 (eq. 9 with lambdaE = 0), deg C."""
    p = cfg["physics"]
    rc = ebu.rho_cp_j_m3_k(cfg)

    def residual(t: float) -> float:
        rn = ebu.net_radiation_wm2(e_s_wm2, a_s, t, sur_c, emissivity, p["faces"],
                                   p["include_longwave"], cfg)
        return rn - rc * (t - air_c) / r_bh_of_t(t)

    lo, hi = cfg["synthetic"]["root_bracket_c"]
    return brentq(residual, air_c + lo, air_c + hi, xtol=1e-12)


def forward_leaf_temp_c(g_s_mol: float, r_bh_of_t, air_c: float, sur_c: float, rh_pct: float,
                        p_hpa: float, e_s_wm2: float, a_s: float, cfg: dict[str, Any]) -> float:
    """Root of Rn(T) - H(T) - lambdaE(T) = 0 for the leaf (eqs. 2-7), deg C."""
    if not g_s_mol > 0:
        raise ValueError(f"g_s_mol must be positive, got {g_s_mol}")
    p = cfg["physics"]
    rc = ebu.rho_cp_j_m3_k(cfg)
    gamma = ebu.psychrometric_constant_pa(p_hpa, cfg)
    e_a = ebu.saturation_vapor_pressure_pa(air_c, cfg) * rh_pct / 100.0
    r_s = ebu.molar_density_mol_m3(p_hpa, air_c, cfg) / g_s_mol

    def residual(t: float) -> float:
        r_bh = r_bh_of_t(t)
        rn = ebu.net_radiation_wm2(e_s_wm2, a_s, t, sur_c, p["leaf_emissivity_longwave"],
                                   p["faces"], p["include_longwave"], cfg)
        vpd = ebu.saturation_vapor_pressure_pa(t, cfg) - e_a
        return rn - rc * (t - air_c) / r_bh - rc / gamma * vpd / (r_s + p["rbw_to_rbh_ratio"] * r_bh)

    lo, hi = cfg["synthetic"]["root_bracket_c"]
    return brentq(residual, air_c + lo, air_c + hi, xtol=1e-12)


# ---------------------------------------------------------------- sessions

@dataclass(frozen=True)
class SimulatedSession:
    """Tables in the real schemas plus the hidden truth. sensor_log: SENSOR_LOG_COLUMNS;
    events: EVENTS_COLUMNS; metadata: validated SessionMetadata; manifest: MANIFEST_FIELDS;
    photo_features: photo_features.csv columns. truth and cal are synthetic-only."""
    session_id: str
    sensor_log: pd.DataFrame
    events: pd.DataFrame
    metadata: SessionMetadata
    manifest: dict[str, Any]
    photo_features: pd.DataFrame
    truth: dict[str, Any]
    cal: dict[str, Any]


def synthetic_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    """calibration.json-shaped dict for synthetic data only. The real config is not changed."""
    s = cfg["synthetic"]
    return {"light_sources": {LIGHT_SOURCE: {"k_wm2_per_lux": s["k_wm2_per_lux"]}},
            "spatial": {"irradiance_factor_leaf": 1.0, "irradiance_factor_card": 1.0},
            "card": {"a_s_card": s["a_s_card"]}}


def _bias(cfg: dict[str, Any], seed: int, active: bool) -> dict[str, float]:
    """One offset per session from the uncertainty sigmas (zero unless active)."""
    if not active:
        return {"thermal_c": 0.0, "air_temp_c": 0.0, "humidity_pct": 0.0, "lux_factor": 1.0}
    u = cfg["uncertainty"]
    rng = np.random.default_rng([seed, 1])          # own stream: switching bias on keeps the noise
    return {"thermal_c": float(rng.normal(0, u["sigma_leaf_temp_c"])),
            "air_temp_c": float(rng.normal(0, u["sigma_air_temp_c"])),
            "humidity_pct": float(rng.normal(0, u["sigma_rh_pct"])),
            "lux_factor": float(1.0 + rng.normal(0, u["sigma_lux_fraction"]))}


def _metadata(cfg: dict[str, Any], start: datetime, plant_id: str, card_id: str | None,
              acclimation_min: float, overrides: dict[str, Any] | None) -> SessionMetadata:
    e = cfg["enumerations"]
    raw: dict[str, Any] = {
        "plant_id": plant_id, "genotype": e["genotype"][0], "water_treatment": e["water_treatment"][0],
        "reproductive_stage": "V3", "block": "1", "position": "synthetic", "leaf_position": "synthetic",
        "light_source": LIGHT_SOURCE, "lamp_distance_mm": "300", "window_blocked": "true",
        "acclimation_minutes": str(acclimation_min), "fan_state": "low", "working_distance_mm": "300",
        "incidence_angle_deg": "0", "card_angle_deg": "", "lamp_angle_deg": "0", "lamp_to_card_mm": "300",
        "background": "neutral_matte", "card_id": card_id or "",
        "dry_reference_type": "card" if card_id else "none", "wet_reference_present": "false",
        "operator": "SYN", "porometer_gs_mol": "", "notes": "SYNTHETIC (ml/synthetic_data.py)",
    }
    raw.update(overrides or {})
    return make_session(raw, cfg, now=start)


def _photo_rows(times: list[str], images: list[str], geometry: pd.Series, extra: pd.Series | None,
                angle: float) -> pd.DataFrame:
    rows = []
    for t, img in zip(times, images):
        row = {c: "" for c in PHOTO_COLUMNS}
        row.update(timestamp=t, image_file=Path(img).name, ok=True, reason="ok",
                   flags="foreshortening_corrected", mm_per_pixel="", incidence_angle_deg=angle,
                   n_components=1)
        for c in ("area_mm2", "d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity",
                  "circularity", "edge_density"):
            row[c] = float(geometry[c])
        row["area_projected_mm2"] = float(geometry["area_mm2"])
        if extra is not None:
            row["d_max_mm"] = float(extra["d_max_mm"])
            row["perimeter_mm"] = float(extra["perimeter_mm"])
        rows.append(row)
    return pd.DataFrame(rows, columns=PHOTO_COLUMNS)


def _simulate(cfg: dict[str, Any], seed: int, noise: bool, conditions: dict[str, Any] | None,
              card: pd.Series | None, card_truth: pd.Series | None,
              leaf: pd.Series | None, leaf_truth: pd.Series | None, g_s_mol: float | None,
              plant_id: str, metadata: dict[str, Any] | None) -> SimulatedSession:
    cfg = copy.deepcopy(cfg)            # never touch the caller's config
    s = cfg["synthetic"]
    cond = {**s["conditions"], **(conditions or {})}
    air, rh, p_hpa = float(cond["air_temp_c"]), float(cond["humidity_pct"]), float(cond["pressure_hpa"])
    e_s, wind = float(cond["irradiance_wm2"]), float(cond["wind_ms"])
    sur = air                           # assumption A41: surroundings at air temperature
    cal = synthetic_calibration(cfg)

    truth: dict[str, Any] = {"seed": seed, "noise": noise, "air_temp_true_c": air, "humidity_true_pct": rh,
                             "pressure_true_hpa": p_hpa, "irradiance_true_wm2": e_s, "wind_true_ms": wind,
                             "surroundings_true_c": sur, "g_s_true_mol": g_s_mol}
    t_card = t_leaf = math.nan
    if card is not None:
        f = float(card_truth["f_true"])
        rb = card_rbh_of_t(f, float(card["d_min_mm"]), wind, air, cfg)
        t_card = forward_card_temp_c(rb, air, sur, e_s, float(card["a_s_card"]), float(card["emissivity"]), cfg)
        truth.update(card_id=card["card_id"], f_card_true=f, card_temp_true_c=t_card, r_bh_card_true=rb(t_card),
                     r_bh_card_flat=rb(t_card) / f)
    if leaf is not None:
        f = float(leaf_truth["f_true"])
        rb = card_rbh_of_t(f, float(leaf["d_min_mm"]), wind, air, cfg)
        t_leaf = forward_leaf_temp_c(g_s_mol, rb, air, sur, rh, p_hpa, e_s,
                                     cfg["light_sources"][LIGHT_SOURCE]["a_s_leaf"], cfg)
        truth.update(leaf_shape_id=leaf["card_id"], f_leaf_true=f, leaf_temp_true_c=t_leaf,
                     r_bh_leaf_true=rb(t_leaf), r_bh_leaf_flat=rb(t_leaf) / f)

    acq = cfg["acquisition"]
    dt = float(acq["log_interval_sec"])
    n = int(round(float(cond["duration_min"]) * SECONDS_PER_MINUTE / dt))
    photo_every = max(1, int(round(float(acq["photo_interval_sec"]) / dt)))
    start = datetime.fromisoformat(s["start_iso"])
    meta = _metadata(cfg, start, plant_id, card["card_id"] if card is not None else None,
                     float(cond["acclimation_minutes"]), metadata)
    sid = meta.session_id
    bias = _bias(cfg, seed, bool(s["session_bias"]) and noise)
    truth.update(session_id=sid, bias=bias)

    rng = np.random.default_rng(seed)
    sd = s["noise"]

    def noisy(value: float, sigma: float, offset: float = 0.0) -> np.ndarray:
        if math.isnan(value):
            return np.full(n, np.nan)
        return value + offset + (rng.normal(0.0, sigma, n) if noise else np.zeros(n))

    times = [(pd.Timestamp(start) + pd.Timedelta(seconds=i * dt)).isoformat() for i in range(n)]
    leaf_c = noisy(t_leaf, sd["leaf_temp_c"], bias["thermal_c"])
    card_c = noisy(t_card, sd["card_temp_c"], bias["thermal_c"])
    air_c = noisy(air, sd["air_temp_c"], bias["air_temp_c"])
    lux_frac = rng.normal(0.0, sd["lux_fraction"], n) if noise else np.zeros(n)
    k = s["k_wm2_per_lux"]
    images = [f"images/synthetic_{i // photo_every:04d}.png" if i % photo_every == 0 else "" for i in range(n)]
    stack = np.vstack([leaf_c, card_c, air_c])
    n_px = s["roi_n_px"]
    log = pd.DataFrame({c: [""] * n for c in SENSOR_LOG_COLUMNS})
    log = log.assign(
        session_id=sid, timestamp_iso=times, monotonic_s=np.arange(n) * dt,
        air_temp_c=air_c, humidity_pct=noisy(rh, sd["humidity_pct"], bias["humidity_pct"]),
        pressure_hpa=noisy(p_hpa, sd["pressure_hpa"]),
        lux=e_s / k * bias["lux_factor"] * (1.0 + lux_frac),
        wind_ms=noisy(wind, sd["wind_ms"]), wind_status="ok", wind_source="simulated",
        thermal_frame_idx=np.arange(n), thermal_min_c=np.nanmin(stack, axis=0),
        thermal_max_c=np.nanmax(stack, axis=0), thermal_mean_c=np.nanmean(stack, axis=0),
        leaf_temp_c=leaf_c, leaf_temp_median_c=leaf_c,
        leaf_temp_std_c=np.full(n, sd["leaf_temp_c"] if leaf is not None else np.nan),
        leaf_n_px=np.full(n, n_px if leaf is not None else np.nan),
        card_temp_c=card_c, card_temp_std_c=np.full(n, sd["card_temp_c"] if card is not None else np.nan),
        card_n_px=np.full(n, n_px if card is not None else np.nan),
        wet_ref_temp_c=np.nan, wet_ref_n_px=np.nan, tsl_visible=np.nan, tsl_infrared=np.nan,
        tsl_full_spectrum=np.nan, tsl_integration_ms=np.nan, wind_raw=np.nan, image_file=images,
    )[list(SENSOR_LOG_COLUMNS)]

    events = pd.DataFrame({"timestamp_iso": [times[0]], "label": ["measurement_start"],
                           "note": ["synthetic session"]})[list(EVENTS_COLUMNS)]
    shape, extra = (leaf, leaf_truth) if leaf is not None else (card, card_truth)
    photo_idx = [i for i in range(n) if images[i]]
    photos = _photo_rows([times[i] for i in photo_idx], [images[i] for i in photo_idx], shape, extra,
                         float(meta.incidence_angle_deg))
    end = (pd.Timestamp(start) + pd.Timedelta(seconds=(n - 1) * dt)).isoformat()
    manifest = {
        "session_id": sid, "start_iso": times[0], "end_iso": end, "clock_status": "ntp_synced",
        "clock_offset_s": 0.0, "n_samples": n, "git_commit": "synthetic", "config_snapshot": cfg,
        "calibration_sha256": "", "package_versions": {},
        "sensor_settings": {"synthetic": True, "generator": "ml.synthetic_data", "seed": seed},
        "throttled_flags": [],
    }
    assert tuple(manifest) == MANIFEST_FIELDS
    return SimulatedSession(sid, log, events, meta, manifest, photos, truth, cal)


def simulate_card_session(card: pd.Series, card_truth: pd.Series, cfg: dict[str, Any] | None = None,
                          seed: int = 0, noise: bool = True, conditions: dict[str, Any] | None = None,
                          metadata: dict[str, Any] | None = None) -> SimulatedSession:
    """One card alone under steady conditions; true r_bH = f_true x flat plate (see module docstring)."""
    return _simulate(_cfg(cfg), seed, noise, conditions, card, card_truth, None, None, None,
                     "SYN-CARD", metadata)


def simulate_leaf_session(g_s_mol: float, leaf: pd.Series, leaf_truth: pd.Series,
                          card: pd.Series | None = None, card_truth: pd.Series | None = None,
                          cfg: dict[str, Any] | None = None, seed: int = 0, noise: bool = True,
                          conditions: dict[str, Any] | None = None, plant_id: str = "SYN-01",
                          metadata: dict[str, Any] | None = None) -> SimulatedSession:
    """A leaf with known g_s (mol m-2 s-1) and shape (a row of a synthetic cards table plus its
    truth row), optionally with a reference card in the same frame."""
    if (card is None) != (card_truth is None):
        raise ValueError("give both card and card_truth, or neither")
    return _simulate(_cfg(cfg), seed, noise, conditions, card, card_truth, leaf, leaf_truth,
                     g_s_mol, plant_id, metadata)


def write_session(sim: SimulatedSession, root: str | Path) -> Path:
    """Write a session in the real layout under root: data/raw/<id>/ (session.json, sensor_log.csv,
    events.csv, run_manifest.json) and data/processed/<id>/ (photo_features.csv,
    synthetic_truth.json). Refuses to overwrite an existing session folder (raw is append-only)."""
    root = Path(root)
    raw = root / "data" / "raw" / sim.session_id
    if raw.exists():
        raise FileExistsError(f"{raw} already exists; raw session folders are never overwritten")
    (raw / "images").mkdir(parents=True)
    (raw / "session.json").write_text(json.dumps(asdict(sim.metadata), indent=2), encoding="utf-8")
    sim.sensor_log.to_csv(raw / "sensor_log.csv", index=False)
    sim.events.to_csv(raw / "events.csv", index=False)
    (raw / "run_manifest.json").write_text(json.dumps(sim.manifest, indent=2, default=str), encoding="utf-8")
    processed = root / "data" / "processed" / sim.session_id
    processed.mkdir(parents=True, exist_ok=True)
    sim.photo_features.to_csv(processed / "photo_features.csv", index=False)
    (processed / "synthetic_truth.json").write_text(json.dumps(sim.truth, indent=2, default=str),
                                                    encoding="utf-8")
    return raw
