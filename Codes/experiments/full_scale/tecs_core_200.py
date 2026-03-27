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
import json
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
    load_counterfact_facts,
)
from core.config import load_config


def _compute_test_gradient(model, tokenizer, prompt_text, layer_idx, device):
    """Compute gradient of next-token prediction loss on a single prompt.

    This is the 'test gradient': grad_W L(prompt; theta) for the CounterFact
    prompt itself, without BM25 retrieval. Used by Null-E to test whether
    gradient source matters.
    """
    from core.gradient_utils import compute_gradient_at_layer
    return compute_gradient_at_layer(model, tokenizer, prompt_text, layer_idx, device)


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

    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17
    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    num_facts = data_cfg.get("num_facts", 200)

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

    # Load CounterFact prompts for Null-E (test-gradient baseline)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)
    cid_to_prompt = {str(f["case_id"]): f["prompt"] for f in facts}

    # Load model for Null-E test-gradient computation
    print("\nLoading model for Null-E test-gradient computation...")
    from core.model_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    # Freeze all parameters to save memory (only need grad on target param)
    for p in model.parameters():
        p.requires_grad_(False)

    # Check for precomputed Null-B (wrong-layer) gradients
    placebo_offsets = null_cfg.get("placebo_offsets", [-5, 5])
    placebo_grad_dirs = {}
    for offset in placebo_offsets:
        layer = 17 + offset  # edit_layer + offset
        pdir = os.path.join(data_dir, f"tda_gradients_200_L{layer}")
        if os.path.exists(pdir):
            placebo_grad_dirs[offset] = pdir
    has_null_b = len(placebo_grad_dirs) > 0

    # Compute TECS + null baselines
    print("\nComputing TECS and null baselines...")
    tecs_real = []
    null_a_means = []
    null_b_means = []  # Null-B: wrong-layer placebo
    null_c_means = []
    null_d_means = []
    null_e_means = []  # Null-E: test-gradient (gradient of test prompt, not training)
    per_fact = []

    rng_py = random.Random(seed)
    # S1: Explicit seed generator for reproducible null baselines across environments
    torch_gen = torch.Generator().manual_seed(seed)

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

        # Null-B: wrong-layer TECS (use precomputed if available, else skip)
        if has_null_b:
            nb = []
            for offset, pdir in placebo_grad_dirs.items():
                placebo_file = os.path.join(pdir, f"g_M_{cid}.pt")
                if os.path.exists(placebo_file):
                    g_placebo = torch.load(placebo_file, map_location="cpu", weights_only=False).float()
                    nb.append(cosine_similarity_flat(delta, g_placebo))
            null_b_means.append(np.mean(nb) if nb else 0.0)
        else:
            null_b_means.append(0.0)

        # Null-C: shuffled gradient
        gm_flat = gm.reshape(-1)
        nc = []
        for _ in range(n_null_repeats):
            perm = torch.randperm(gm_flat.shape[0], generator=torch_gen)
            gs = gm_flat[perm].reshape(gm.shape)
            nc.append(cosine_similarity_flat(delta, gs))
        null_c_means.append(np.mean(nc))

        # Null-D: random direction
        nd = []
        for _ in range(n_null_repeats):
            rG = torch.randn(gm.shape, generator=torch_gen)
            nd.append(cosine_similarity_flat(delta, rG))
        null_d_means.append(np.mean(nd))

        # Null-E: test-gradient TECS (gradient-source control)
        # Use the CounterFact prompt itself to compute a "test gradient"
        # grad_W L(prompt; theta), then compute TECS with that instead of g_M.
        # This tests whether the gradient SOURCE (training data vs test prompt) matters.
        prompt_text = cid_to_prompt.get(cid, None)
        if prompt_text is not None:
            g_test = _compute_test_gradient(model, tokenizer, prompt_text, edit_layer, device)
            ne_val = cosine_similarity_flat(delta, g_test)
            null_e_means.append(ne_val)
        else:
            # Fallback: if prompt not found, record 0.0
            null_e_means.append(0.0)

        # NOTE: Previous implementation used random sign flip (commented out below).
        # That was a Null-C variant, not the intended test-gradient control.
        # --- Old sign-flip implementation (Null-C variant, NOT Null-E) ---
        # ne = []
        # for _ in range(n_null_repeats):
        #     signs = torch.sign(torch.randn_like(gm))
        #     g_sign_flipped = gm * signs
        #     ne.append(cosine_similarity_flat(delta, g_sign_flipped))
        # null_e_means.append(np.mean(ne))

        per_fact.append({
            "case_id": cid,
            "tecs_real": tv,
            "null_a_mean": float(np.mean(na)),
            "null_b_mean": float(null_b_means[-1]),
            "null_c_mean": float(np.mean(nc)),
            "null_d_mean": float(np.mean(nd)),
            "null_e_test_grad": float(null_e_means[-1]),
        })

        if (fi + 1) % 20 == 0:
            print(f"  [{fi + 1}/{len(common_ids)}] TECS={tv:.6f}")

    # Clean up model used for Null-E
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tecs_arr = np.array(tecs_real)
    na_arr = np.array(null_a_means)
    nb_arr = np.array(null_b_means)
    nc_arr = np.array(null_c_means)
    nd_arr = np.array(null_d_means)
    ne_arr = np.array(null_e_means)

    # Statistical tests
    print("\nStatistical analysis...")
    comparisons = {}
    comp_a = paired_test(tecs_arr, na_arr, "TECS vs Null-A (random fact)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_a"] = comp_a
    if has_null_b:
        comp_b = paired_test(tecs_arr, nb_arr, "TECS vs Null-B (wrong layer)", bootstrap_n=bootstrap_n, seed=seed)
        comparisons["vs_null_b"] = comp_b
    comp_c = paired_test(tecs_arr, nc_arr, "TECS vs Null-C (shuffled)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_c"] = comp_c
    comp_d = paired_test(tecs_arr, nd_arr, "TECS vs Null-D (random)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_d"] = comp_d
    comp_e = paired_test(tecs_arr, ne_arr, "TECS vs Null-E (test-gradient)", bootstrap_n=bootstrap_n, seed=seed)
    comparisons["vs_null_e"] = comp_e

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
            "null_b_mean": float(nb_arr.mean()) if has_null_b else None,
            "null_b_available": has_null_b,
            "null_c_mean": float(nc_arr.mean()),
            "null_d_mean": float(nd_arr.mean()),
            "null_e_mean": float(ne_arr.mean()),
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
