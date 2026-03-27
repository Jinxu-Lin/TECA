#!/usr/bin/env python3
"""Phase 3: Full-Scale TECS Core Measurement + 5 Null Baselines (200 facts).

Loads precomputed ROME deltas and TDA gradients, computes:
  - TECS_real for each fact
  - 5 null baselines (A: random-fact, B: wrong-layer, C: shuffled, D: random-direction, E: test-gradient)
  - Cohen's d with 10k bootstrap CI, Bonferroni correction

Depends: full_rome_200, full_tda_200

Usage:
    python -m experiments.full_scale.tecs_core_200 --config configs/phase_3_full_scale.yaml
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, cohens_d,
    bootstrap_ci, paired_test, get_results_dir, get_data_dir,
)
from core.config import load_config


def run_tecs_core_200(cfg: dict) -> dict:
    """Run full-scale TECS measurement with all null baselines."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "rome_deltas_200")
    grad_dir = os.path.join(data_dir, "tda_gradients_200")
    os.makedirs(results_dir, exist_ok=True)

    null_cfg = cfg.get("null_baselines", {})
    n_null_repeats = null_cfg.get("null_a_num", 10)
    stats_cfg = cfg.get("statistics", {})
    bootstrap_n = stats_cfg.get("bootstrap_n", 10000)

    print("=" * 60)
    print("Phase 3: Full-Scale TECS Core (200 Facts)")
    print("=" * 60)

    start_time = time.time()

    # Discover common case IDs
    rome_files = {f.replace("delta_", "").replace(".pt", "")
                  for f in os.listdir(rome_dir) if f.startswith("delta_") and f.endswith(".pt")}
    grad_files = {f.replace("g_M_", "").replace(".pt", "")
                  for f in os.listdir(grad_dir) if f.startswith("g_M_") and f.endswith(".pt")}
    common_ids = sorted(rome_files & grad_files, key=lambda x: int(x))
    print(f"  Common case IDs: {len(common_ids)}")

    if len(common_ids) < 10:
        raise RuntimeError(f"Only {len(common_ids)} valid facts, need >= 10. "
                           "Run rome_200 and tda_gradients_200 first.")

    # Load all tensors
    print("\nLoading tensors...")
    deltas = {}
    gradients = {}
    for cid in common_ids:
        d = torch.load(os.path.join(rome_dir, f"delta_{cid}.pt"), map_location="cpu", weights_only=False)
        deltas[cid] = d["delta_weight"].float()
        gradients[cid] = torch.load(
            os.path.join(grad_dir, f"g_M_{cid}.pt"), map_location="cpu", weights_only=False
        ).float()

    # Compute TECS + null baselines
    print("\nComputing TECS and null baselines...")
    tecs_real = []
    null_a_means = []
    null_c_means = []
    null_d_means = []
    per_fact = []

    rng_py = random.Random(seed)

    for fi, cid in enumerate(common_ids):
        delta = deltas[cid]
        gm = gradients[cid]

        # TECS_real
        tv = cosine_similarity_flat(delta, gm)
        tecs_real.append(tv)

        # Null-A: random fact swap
        others = [x for x in common_ids if x != cid]
        swaps = rng_py.sample(others, min(n_null_repeats, len(others)))
        na = [cosine_similarity_flat(deltas[s], gm) for s in swaps]
        null_a_means.append(np.mean(na))

        # Null-C: shuffled gradient
        gm_flat = gm.reshape(-1)
        nc = []
        for _ in range(n_null_repeats):
            perm = torch.randperm(gm_flat.shape[0])
            gs = gm_flat[perm].reshape(gm.shape)
            nc.append(cosine_similarity_flat(delta, gs))
        null_c_means.append(np.mean(nc))

        # Null-D: random direction
        nd = []
        for _ in range(n_null_repeats):
            rG = torch.randn_like(gm)
            nd.append(cosine_similarity_flat(delta, rG))
        null_d_means.append(np.mean(nd))

        per_fact.append({
            "case_id": cid,
            "tecs_real": tv,
            "null_a_mean": float(np.mean(na)),
            "null_c_mean": float(np.mean(nc)),
            "null_d_mean": float(np.mean(nd)),
        })

        if (fi + 1) % 20 == 0:
            print(f"  [{fi + 1}/{len(common_ids)}] TECS={tv:.6f}")

    tecs_arr = np.array(tecs_real)
    na_arr = np.array(null_a_means)
    nc_arr = np.array(null_c_means)
    nd_arr = np.array(null_d_means)

    # Statistical tests
    print("\nStatistical analysis...")
    comparisons = {}
    comp_a = paired_test(tecs_arr, na_arr, "TECS vs Null-A (random fact)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_a"] = comp_a
    comp_c = paired_test(tecs_arr, nc_arr, "TECS vs Null-C (shuffled)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_c"] = comp_c
    comp_d = paired_test(tecs_arr, nd_arr, "TECS vs Null-D (random)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_d"] = comp_d

    for name, comp in comparisons.items():
        print(f"  {name}: d={comp['cohens_d']:.4f}, p={comp['p_value']:.2e}")

    tecs_ci = bootstrap_ci(tecs_arr, n_boot=bootstrap_n, seed=seed)

    # Decision gate
    primary_d = comp_a["cohens_d"]
    decision = "POSITIVE" if primary_d > 0.2 else "NEGATIVE"

    # Bonferroni
    n_comp = len(comparisons)
    bonf_alpha = 0.05 / n_comp

    elapsed = time.time() - start_time

    results = {
        "experiment": "full_tecs_200",
        "phase": "3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "n_facts": len(common_ids),
            "null_repeats": n_null_repeats,
            "bootstrap_n": bootstrap_n,
            "seed": seed,
        },
        "tecs_distribution": {
            "mean": float(tecs_arr.mean()),
            "std": float(tecs_arr.std()),
            "median": float(np.median(tecs_arr)),
            "ci_95": list(tecs_ci),
            "n_positive": int((tecs_arr > 0).sum()),
            "n_negative": int((tecs_arr < 0).sum()),
        },
        "null_distributions": {
            "null_a_mean": float(na_arr.mean()),
            "null_c_mean": float(nc_arr.mean()),
            "null_d_mean": float(nd_arr.mean()),
        },
        "statistical_tests": comparisons,
        "bonferroni": {
            "alpha": bonf_alpha,
            "n_comparisons": n_comp,
            "significant": {k: v["p_value"] < bonf_alpha for k, v in comparisons.items()},
        },
        "decision_gate": {
            "decision": decision,
            "primary_d": primary_d,
            "threshold": 0.2,
        },
        "per_fact_results": per_fact,
    }

    print(f"\n{'=' * 60}")
    print(f"  TECS Core 200 Result:")
    print(f"  TECS_real: {tecs_arr.mean():.6f} +/- {tecs_arr.std():.6f}")
    print(f"  95% CI: [{tecs_ci[0]:.6f}, {tecs_ci[1]:.6f}]")
    print(f"  Cohen's d (vs Null-A): {primary_d:.4f}")
    print(f"  Decision: {decision}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "full_tecs_200.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: TECS Core 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_tecs_core_200(cfg)


if __name__ == "__main__":
    main()
