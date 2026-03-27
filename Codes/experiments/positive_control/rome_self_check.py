#!/usr/bin/env python3
"""Phase 1a: ROME vs Self (Trivial Positive Control).

Compute TECS(delta_W, delta_W + epsilon) for varying noise levels sigma.
Validates that the TECS metric pipeline works correctly:
  - TECS(sigma=0) should be exactly 1.0
  - TECS should decrease monotonically with increasing sigma

Uses precomputed ROME deltas from pilot data if available,
otherwise computes fresh ones.

Usage:
    python -m experiments.positive_control.rome_self_check --config configs/phase_1_positive_control.yaml
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, paired_test,
    bootstrap_ci, load_experiment_config, get_results_dir,
)
from core.config import load_config


def run_rome_self_check(cfg: dict) -> dict:
    """Run ROME vs self positive control experiment."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    pc_cfg = cfg.get("positive_control", {})
    sigmas = pc_cfg.get("rome_self_sigmas", [0.0, 0.01, 0.1, 0.5, 1.0])
    num_facts = pc_cfg.get("rome_self_num_facts", 100)
    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    print("=" * 60)
    print("Phase 1a: ROME vs Self Positive Control")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Num facts: {num_facts}")
    print(f"  Seed: {seed}")

    start_time = time.time()

    # Load model and compute ROME deltas
    from core.model_utils import load_model_and_tokenizer
    from core.retrieval import load_counterfact
    from core.rome_utils import compute_rome_edit

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact(counterfact_path, num_facts=num_facts, seed=seed)

    print(f"\nComputing ROME deltas for {len(facts)} facts...")

    # Collect deltas
    deltas = []
    for i, fact in enumerate(facts):
        try:
            result = compute_rome_edit(
                model, tokenizer,
                subject=fact["subject"],
                prompt=fact["prompt"],
                target_new=fact["target_new"],
                target_old=fact["target_old"],
                edit_layer=edit_layer,
                device=device,
            )
            deltas.append(result.delta_weight.cpu())
        except Exception as e:
            print(f"  [WARN] Fact {i} ({fact['subject']}): {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}] deltas computed")

    print(f"  {len(deltas)} valid deltas")

    # Free model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute TECS for each sigma
    print("\nComputing TECS across noise levels...")
    sigma_results = {}
    rng = np.random.RandomState(seed)

    for sigma in sigmas:
        tecs_values = []
        for delta in deltas:
            if sigma == 0.0:
                noisy_delta = delta
            else:
                noise = torch.randn_like(delta) * sigma * delta.norm()
                noisy_delta = delta + noise

            tecs = cosine_similarity_flat(delta, noisy_delta)
            tecs_values.append(tecs)

        tecs_arr = np.array(tecs_values)
        ci = bootstrap_ci(tecs_arr, seed=seed)

        sigma_results[str(sigma)] = {
            "sigma": sigma,
            "mean": float(tecs_arr.mean()),
            "std": float(tecs_arr.std()),
            "median": float(np.median(tecs_arr)),
            "min": float(tecs_arr.min()),
            "max": float(tecs_arr.max()),
            "ci_95": list(ci),
            "n": len(tecs_values),
        }
        print(f"  sigma={sigma:.2f}: TECS={tecs_arr.mean():.6f} +/- {tecs_arr.std():.6f}")

    # Validate monotonic decrease
    means = [sigma_results[str(s)]["mean"] for s in sigmas]
    is_monotonic = all(means[i] >= means[i + 1] - 0.01 for i in range(len(means) - 1))

    # Pass criteria
    tecs_zero = sigma_results[str(0.0)]["mean"] if "0.0" in sigma_results else sigma_results["0"]["mean"]
    tecs_one = sigma_results[str(1.0)]["mean"] if "1.0" in sigma_results else sigma_results["1"]["mean"]
    pass_zero = abs(tecs_zero - 1.0) < 1e-6
    pass_one = tecs_one > 0.3

    elapsed = time.time() - start_time

    results = {
        "experiment": "pc_rome_self",
        "phase": "1a",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": len(deltas),
            "sigmas": sigmas,
            "seed": seed,
        },
        "sigma_results": sigma_results,
        "validation": {
            "monotonic_decrease": is_monotonic,
            "tecs_sigma_0": tecs_zero,
            "tecs_sigma_1": tecs_one,
            "pass_sigma_0_eq_1": pass_zero,
            "pass_sigma_1_gt_03": pass_one,
            "overall_pass": pass_zero and pass_one and is_monotonic,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  ROME vs Self Result:")
    print(f"  TECS(sigma=0) = {tecs_zero:.6f} (expected: 1.0)")
    print(f"  TECS(sigma=1) = {tecs_one:.6f} (expected: >0.3)")
    print(f"  Monotonic: {is_monotonic}")
    print(f"  Overall: {'PASS' if results['validation']['overall_pass'] else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    output_path = os.path.join(results_dir, "pc_rome_self.json")
    save_results(results, output_path)

    return results


def main():
    args = None
    # Support both --config and direct config path
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1a: ROME vs Self Positive Control")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_rome_self_check(cfg)


if __name__ == "__main__":
    main()
