#!/usr/bin/env python3
"""Phase 6: GPT-J Core TECS + Subspace Geometry.

Depends: cross_gptj_rome, cross_gptj_tda

Usage:
    python -m experiments.cross_model.gptj_tecs --config configs/phase_6_cross_model.yaml
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


def run_gptj_tecs(cfg: dict) -> dict:
    """Run TECS + subspace geometry on GPT-J data."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "gptj_rome_deltas")
    grad_dir = os.path.join(data_dir, "gptj_tda_gradients")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 6: GPT-J TECS + Subspace Geometry")
    print("=" * 60)

    start_time = time.time()

    # Find common IDs
    rome_files = {f.replace("delta_", "").replace(".pt", "")
                  for f in os.listdir(rome_dir) if f.startswith("delta_") and f.endswith(".pt")}
    grad_files = {f.replace("g_M_", "").replace(".pt", "")
                  for f in os.listdir(grad_dir) if f.startswith("g_M_") and f.endswith(".pt")}
    common_ids = sorted(rome_files & grad_files, key=lambda x: int(x))
    print(f"  Common case IDs: {len(common_ids)}")

    # Load tensors
    deltas = {}
    gradients = {}
    for cid in common_ids:
        d = torch.load(os.path.join(rome_dir, f"delta_{cid}.pt"), map_location="cpu", weights_only=False)
        deltas[cid] = d["delta_weight"].float()
        gradients[cid] = torch.load(
            os.path.join(grad_dir, f"g_M_{cid}.pt"), map_location="cpu", weights_only=False
        ).float()

    # TECS computation
    tecs_real = []
    null_a_means = []
    rng_py = random.Random(seed)

    for cid in common_ids:
        tv = cosine_similarity_flat(deltas[cid], gradients[cid])
        tecs_real.append(tv)

        others = [x for x in common_ids if x != cid]
        swaps = rng_py.sample(others, min(10, len(others)))
        na = [cosine_similarity_flat(deltas[s], gradients[cid]) for s in swaps]
        null_a_means.append(np.mean(na))

    tecs_arr = np.array(tecs_real)
    na_arr = np.array(null_a_means)

    test = paired_test(tecs_arr, na_arr, "GPT-J TECS vs Null-A", seed=seed)
    primary_d = test["cohens_d"]

    # Subspace geometry (simplified)
    D_rows = [deltas[cid].flatten() for cid in common_ids]
    G_rows = [gradients[cid].flatten() for cid in common_ids]
    D = torch.stack(D_rows)
    G = torch.stack(G_rows)

    # Effective dimensionality
    _, S_D, _ = torch.svd_lowrank(D - D.mean(0, keepdim=True), q=min(len(common_ids), 50), niter=5)
    _, S_G, _ = torch.svd_lowrank(G - G.mean(0, keepdim=True), q=min(len(common_ids), 50), niter=5)

    def eff_dim(S):
        s2 = S ** 2
        p = s2 / s2.sum()
        p = p[p > 1e-12]
        return float(torch.exp(-(p * torch.log(p)).sum()))

    eff_dim_D = eff_dim(S_D)
    eff_dim_G = eff_dim(S_G)

    elapsed = time.time() - start_time

    results = {
        "experiment": "cross_gptj_tecs",
        "phase": "6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": "gpt-j-6b",
            "n_facts": len(common_ids),
            "seed": seed,
        },
        "tecs": {
            "mean": float(tecs_arr.mean()),
            "std": float(tecs_arr.std()),
            "cohens_d_vs_null_a": primary_d,
        },
        "statistical_test": test,
        "subspace": {
            "D_eff_dim": eff_dim_D,
            "G_eff_dim": eff_dim_G,
        },
        "decision": {
            "tecs_near_zero": abs(primary_d) < 0.2,
            "replicates_gpt2xl": abs(primary_d) < 0.2,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  GPT-J TECS Result:")
    print(f"  TECS: {tecs_arr.mean():.6f}, d={primary_d:.4f}")
    print(f"  D eff-dim: {eff_dim_D:.1f}, G eff-dim: {eff_dim_G:.1f}")
    print(f"  Replicates GPT-2-XL: {abs(primary_d) < 0.2}")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "cross_gptj_tecs.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: GPT-J TECS")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_gptj_tecs(cfg)


if __name__ == "__main__":
    main()
