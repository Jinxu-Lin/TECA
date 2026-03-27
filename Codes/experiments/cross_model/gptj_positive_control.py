#!/usr/bin/env python3
"""Phase 6: GPT-J Positive Control (ROME vs self + related facts).

Depends: cross_gptj_rome

Usage:
    python -m experiments.cross_model.gptj_positive_control --config configs/phase_6_cross_model.yaml
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, bootstrap_ci,
    get_results_dir, get_data_dir,
)
from core.config import load_config


def run_gptj_positive_control(cfg: dict) -> dict:
    """Run positive control experiments on GPT-J."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "gptj_rome_deltas")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 6: GPT-J Positive Control")
    print("=" * 60)

    start_time = time.time()

    # Load GPT-J ROME deltas
    deltas = {}
    for f in os.listdir(rome_dir):
        if f.startswith("delta_") and f.endswith(".pt"):
            cid = f.replace("delta_", "").replace(".pt", "")
            d = torch.load(os.path.join(rome_dir, f), map_location="cpu", weights_only=False)
            deltas[cid] = d["delta_weight"].float()

    case_ids = sorted(deltas.keys(), key=lambda x: int(x))
    print(f"  Loaded {len(case_ids)} ROME deltas")

    # ROME vs Self
    sigmas = [0.0, 0.01, 0.1, 0.5, 1.0]
    sigma_results = {}

    for sigma in sigmas:
        tecs_values = []
        for cid in case_ids:
            delta = deltas[cid]
            if sigma == 0.0:
                noisy = delta
            else:
                noise = torch.randn_like(delta) * sigma * delta.norm()
                noisy = delta + noise
            tecs = cosine_similarity_flat(delta, noisy)
            tecs_values.append(tecs)

        arr = np.array(tecs_values)
        sigma_results[str(sigma)] = {
            "sigma": sigma,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        print(f"  sigma={sigma:.2f}: TECS={arr.mean():.6f}")

    # Monotonic check
    means = [sigma_results[str(s)]["mean"] for s in sigmas]
    is_monotonic = all(means[i] >= means[i + 1] - 0.01 for i in range(len(means) - 1))

    elapsed = time.time() - start_time

    results = {
        "experiment": "cross_gptj_positive",
        "phase": "6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": "gpt-j-6b",
            "n_deltas": len(case_ids),
            "sigmas": sigmas,
            "seed": seed,
        },
        "rome_vs_self": {
            "sigma_results": sigma_results,
            "monotonic": is_monotonic,
            "pass": abs(sigma_results["0.0"]["mean"] - 1.0) < 1e-6 and is_monotonic,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  GPT-J Positive Control:")
    print(f"  ROME vs Self monotonic: {is_monotonic}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "cross_gptj_positive.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: GPT-J Positive Control")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_gptj_positive_control(cfg)


if __name__ == "__main__":
    main()
