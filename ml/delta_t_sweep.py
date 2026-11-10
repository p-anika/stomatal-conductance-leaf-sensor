"""How does the error in mean card f depend on the card's temperature rise delta T? (synthetic)

    python -m ml.delta_t_sweep [--out data/processed/delta_t_sweep.csv]

SYNTHETIC ONLY, with PLACEHOLDER calibration sigmas: the table shows that the code behaves as
expected and how the bias scales; the real numbers must be redone after T7.5.

For each lamp level (incident irradiance E_s), wind and offset scale s (s x the uncertainty
sigmas used as one calibration offset per session; s = 0 means per-sample noise only), every
synthetic card is simulated `repeats` times with the same random draws at every setting, then run
through the real build_training_set. Per window the relative error is
    e = f_measured / f_true - 1                                         (-)
and per session the mean of e over its windows. Reported (sessions weigh equally):
    bias = mean over sessions of the session mean e;   SE = SD over sessions / sqrt(n)
for three row sets:
    kept         the pipeline as it is: absorbed-shortwave gate on, r_bH <= 0 rejected
    gated_all    gate on, r_bH <= 0 rows kept (no truncation)
    nogate_all   gate off, r_bH <= 0 rows kept (no selection except steady state)
The gate is not changed: the sweep runs build_training_set once on a config COPY with the gate
at 0 and then applies the real gate as a filter, a_s,card x c_card x k x lux >= threshold,
the same test solve_card_rbh uses.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.build_training_set import build_training_set
from ml.synthetic_data import make_synthetic_cards, simulate_card_session, synthetic_calibration
from processing.config import load_config
from processing.data_io import CARD_COLUMNS_ORDER, ReferenceCard, Session

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_COLUMNS = ("irradiance_incident_wm2", "absorbed_sw_wm2", "wind_ms", "offset_scale", "delta_t_true_c",
                 "n_sessions", "n_sessions_pass_gate", "low_signal_fraction", "suspect_fraction",
                 "bias_kept", "se_kept", "spread_kept", "bias_gated_all", "se_gated_all",
                 "bias_nogate_all", "se_nogate_all")
SIGMA_KEYS = ("sigma_leaf_temp_c", "sigma_air_temp_c", "sigma_rh_pct", "sigma_lux_fraction")
TABLE_HEADER = ("SYNTHETIC sweep with PLACEHOLDER calibration sigmas (uncertainty section); it proves the code "
                "works and must be redone after T7.5. Irradiance: incident on the card and absorbed "
                "(x a_s,card); delta T = true card - air temperature. Bias = mean relative error of f "
                "(f_measured / f_true - 1) +/- standard error over sessions.")


def _session(sim) -> Session:
    return Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)


def _cards(table: pd.DataFrame) -> dict[str, ReferenceCard]:
    out = {}
    for _, r in table.iterrows():
        vals = {c: r[c] for c in CARD_COLUMNS_ORDER}
        vals["photo_path"] = vals["photo_path"] or None
        out[r["card_id"]] = ReferenceCard(**vals)
    return out


def _stats(rows: pd.DataFrame, f_true: dict[str, float]) -> tuple[float, float, float, int]:
    """(bias, SE, spread, n sessions) of the session-mean relative error of f."""
    if rows.empty:
        return math.nan, math.nan, math.nan, 0
    e = rows["f_ratio"] / rows["session_id"].map(f_true) - 1.0
    per_session = e.groupby(rows["session_id"]).mean()
    n = len(per_session)
    sd = float(per_session.std(ddof=1)) if n > 1 else math.nan
    return float(per_session.mean()), sd / math.sqrt(n) if n > 1 else math.nan, sd, n


def sweep_point(cfg: dict[str, Any], cards: pd.DataFrame, truth: pd.DataFrame, irradiance: float,
                wind: float, scale: float, repeats: int, duration_min: float) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["synthetic"]["session_bias"] = True
    for k in SIGMA_KEYS:
        c["uncertainty"][k] = cfg["uncertainty"][k] * scale
    gate = float(cfg["physics"]["min_card_absorbed_sw_wm2"])
    c["physics"]["min_card_absorbed_sw_wm2"] = 0.0              # copy only; real gate applied below
    cal = synthetic_calibration(c)
    sessions, f_true, dts = [], {}, []
    for i in range(len(cards)):
        for r in range(repeats):
            seed = i * repeats + r                               # same draws at every setting
            sim = simulate_card_session(cards.iloc[i], truth.iloc[i], cfg=c, seed=seed,
                                        conditions={"irradiance_wm2": irradiance, "wind_ms": wind,
                                                    "duration_min": duration_min},
                                        metadata={"plant_id": f"SWEEP-{i}-{r}"})
            sessions.append(_session(sim))
            f_true[sim.session_id] = sim.truth["f_card_true"]
            dts.append(sim.truth["card_temp_true_c"] - sim.truth["air_temp_true_c"])
    ts = build_training_set(sessions, _cards(cards), c, cal, force=True)

    a_s = float(cal["card"]["a_s_card"])
    k = float(cal["light_sources"]["led_lamp"]["k_wm2_per_lux"])
    def passes(df: pd.DataFrame) -> pd.Series:
        return a_s * df["c_card"] * k * df["lux"] >= gate
    sus = ts.rejects[ts.rejects["reason"] == "suspect_card_rbh"]
    everyone = pd.concat([ts.kept, sus])
    gated_all = everyone[passes(everyone)]
    kept = ts.kept[passes(ts.kept)]
    reached_gate = pd.concat([everyone, ts.rejects[ts.rejects["reason"] == "rn_near_zero"]])

    b_k, se_k, sd_k, _ = _stats(kept, f_true)
    b_g, se_g, _, n_pass = _stats(gated_all, f_true)
    b_n, se_n, _, n_all = _stats(everyone, f_true)
    n_gated = len(gated_all)
    return {
        "irradiance_incident_wm2": irradiance, "absorbed_sw_wm2": a_s * irradiance, "wind_ms": wind,
        "offset_scale": scale, "delta_t_true_c": float(np.mean(dts)), "n_sessions": len(sessions),
        "n_sessions_pass_gate": n_pass,
        "low_signal_fraction": float((~passes(reached_gate)).mean()) if len(reached_gate) else math.nan,
        "suspect_fraction": float((~gated_all["valid"].astype(bool)).sum() / n_gated) if n_gated else math.nan,
        "bias_kept": b_k, "se_kept": se_k, "spread_kept": sd_k,
        "bias_gated_all": b_g, "se_gated_all": se_g, "bias_nogate_all": b_n, "se_nogate_all": se_n,
    }


def run_sweep(cfg: dict[str, Any], irradiances=None, winds=None, scales=None, repeats: int | None = None,
              n_cards: int | None = None) -> pd.DataFrame:
    sw = cfg["synthetic"]["delta_t_sweep"]
    irradiances = sw["irradiance_wm2"] if irradiances is None else irradiances
    winds = sw["wind_ms"] if winds is None else winds
    scales = sw["offset_scales"] if scales is None else scales
    repeats = int(sw["repeats"]) if repeats is None else repeats
    cards, truth = make_synthetic_cards(n_cards or int(cfg["synthetic"]["n_shapes"]),
                                        int(cfg["synthetic"]["train_seed"]), cfg)
    rows = [sweep_point(cfg, cards, truth, float(e), float(w), float(s), repeats, float(sw["duration_min"]))
            for w in winds for s in scales for e in irradiances]
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def _pm(b: float, se: float) -> str:
    return "n/a" if math.isnan(b) else f"{100 * b:+.1f} ± {100 * se:.1f} %"


def format_table(df: pd.DataFrame) -> str:
    lines = [TABLE_HEADER, "",
             "| E incident (W m⁻²) | E absorbed (W m⁻²) | wind (m s⁻¹) | offset scale | true ΔT (°C) | "
             "sessions passing gate | r_bH ≤ 0 fraction | bias kept (pipeline) | spread kept | "
             "bias gate on, r_bH ≤ 0 kept | bias gate off, all kept |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.irradiance_incident_wm2:.0f} | {r.absorbed_sw_wm2:.0f} | {r.wind_ms:.1f} | {r.offset_scale:.1f} | "
            f"{r.delta_t_true_c:.2f} | {int(r.n_sessions_pass_gate)}/{int(r.n_sessions)} | {r.suspect_fraction:.1%} | "
            f"{_pm(r.bias_kept, r.se_kept)} | {100 * r.spread_kept:.1f} % | "
            f"{_pm(r.bias_gated_all, r.se_gated_all)} | {_pm(r.bias_nogate_all, r.se_nogate_all)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ml.delta_t_sweep",
                                description="Synthetic sweep: bias of mean card f versus card delta T.")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "delta_t_sweep.csv"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--irradiance", type=float, nargs="+", default=None, help="incident W m-2 (default: config)")
    p.add_argument("--wind", type=float, nargs="+", default=None)
    p.add_argument("--scales", type=float, nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--n-cards", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)        # build_training_set warns per point; table says it all
    df = run_sweep(load_config(args.config), args.irradiance, args.wind, args.scales, args.repeats, args.n_cards)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    text = format_table(df)
    md = out.with_suffix(".md")
    md.write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    print(text)
    print(f"\nwrote {out} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
