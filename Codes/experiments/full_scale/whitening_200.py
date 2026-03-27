#!/usr/bin/env python3
"""Phase 5: Whitening Decomposition on 200 Facts.

Compare TECS_whitened (standard) vs TECS_unwhitened (without C^{-1}).
Tests whether ROME's statistical whitening is the primary source of
geometric incommensurability.

Follows pilot code pattern from teca_pilot/negative_whitening.py.

Usage:
    python -m experiments.full_scale.whitening_200 --config configs/phase_5_extended.yaml
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, paired_test,
    get_results_dir, get_data_dir,
)
from core.config import load_config


def compute_covariance_matrix(model_name, edit_layer, device, n_samples=100, max_seq_len=256):
    """Compute C = E[k k^T] where k is the MLP c_fc input at the edit layer."""
    from core.model_utils import load_model_and_tokenizer
    from datasets import load_dataset

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    activations = []

    def hook_fn(module, input, output):
        inp = input[0].detach().float()
        for b in range(inp.shape[0]):
            activations.append(inp[b, ::4, :].cpu())

    block = model.transformer.h[edit_layer]
    handle = block.mlp.c_fc.register_forward_hook(hook_fn)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item["text"].strip()
        if len(t) > 100:
            texts.append(t)
        if len(texts) >= n_samples:
            break

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", max_length=max_seq_len,
                               truncation=True).to(device)
            model(**tokens)

    handle.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_acts = torch.cat(activations, dim=0)
    mean_act = all_acts.mean(dim=0)
    centered = all_acts - mean_act
    C = (centered.T @ centered) / centered.shape[0]

    return C


def run_whitening_200(cfg: dict) -> dict:
    """Run whitening decomposition analysis on 200 facts."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "rome_deltas_200")
    grad_dir = os.path.join(data_dir, "tda_gradients_200")
    os.makedirs(results_dir, exist_ok=True)

    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    print("=" * 60)
    print("Phase 5: Whitening Decomposition (200 Facts)")
    print("=" * 60)

    start_time = time.time()

    # Find common case IDs
    rome_files = {f.replace("delta_", "").replace(".pt", "")
                  for f in os.listdir(rome_dir) if f.startswith("delta_") and f.endswith(".pt")}
    grad_files = {f.replace("g_M_", "").replace(".pt", "")
                  for f in os.listdir(grad_dir) if f.startswith("g_M_") and f.endswith(".pt")}
    common_ids = sorted(rome_files & grad_files, key=lambda x: int(x))
    print(f"  Common case IDs: {len(common_ids)}")

    # Compute or load covariance matrix
    cov_cache = os.path.join(data_dir, "cov_matrix_layer17.pt")
    if os.path.exists(cov_cache):
        print("  Loading cached covariance matrix...")
        C = torch.load(cov_cache, map_location="cpu", weights_only=True).float()
    else:
        print("  Computing covariance matrix from WikiText...")
        C = compute_covariance_matrix(model_name, edit_layer, device)
        torch.save(C, cov_cache)

    # Eigendecomposition for C^{-1}
    print("  Computing eigendecomposition...")
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    top_k = min(512, len(eigenvalues))
    eigenvalues = eigenvalues[-top_k:]
    eigenvectors = eigenvectors[:, -top_k:]
    threshold = eigenvalues.max() * 1e-6
    valid = eigenvalues > threshold
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    # Process each fact
    print(f"\nComputing whitening variants for {len(common_ids)} facts...")
    tecs_raw = []
    tecs_unwhitened = []
    tecs_whitened = []

    for i, cid in enumerate(common_ids):
        d = torch.load(os.path.join(rome_dir, f"delta_{cid}.pt"), map_location="cpu", weights_only=False)
        delta_W = d["delta_weight"].float()
        g_M = torch.load(os.path.join(grad_dir, f"g_M_{cid}.pt"), map_location="cpu", weights_only=False).float()

        # Raw TECS
        t_raw = cosine_similarity_flat(delta_W, g_M)
        tecs_raw.append(t_raw)

        # For whitening operations, we need to identify the key/value structure.
        # delta_W has shape matching the c_proj weight. For GPT-2 Conv1D: (6400, 1600).
        # The 1600-dim axis is the key (output) space where C operates.
        # C is (1600, 1600).

        # Unwhitened: apply C to the 1600-dim columns of delta_W -> delta_W @ C
        # (6400, 1600) @ (1600, 1600) = (6400, 1600)
        if delta_W.shape[1] == C.shape[0]:
            delta_unwhitened = delta_W @ C
        else:
            # Shape mismatch, try transpose
            delta_unwhitened = delta_W.T @ C
            delta_unwhitened = delta_unwhitened.T

        t_unwhitened = cosine_similarity_flat(delta_unwhitened, g_M)
        tecs_unwhitened.append(t_unwhitened)

        # Whitened: apply C^{-1} to g_M columns -> g_M @ C^{-1}
        # C^{-1} @ x = V diag(1/lambda) V^T @ x
        if g_M.shape[1] == eigenvectors.shape[0]:
            g_M_T = g_M.T  # (1600, 6400)
            proj = eigenvectors.T @ g_M_T
            scaled = proj / eigenvalues.unsqueeze(1)
            g_whitened_T = eigenvectors @ scaled
            g_whitened = g_whitened_T.T
        else:
            g_whitened = g_M  # fallback

        t_whitened = cosine_similarity_flat(delta_W, g_whitened)
        tecs_whitened.append(t_whitened)

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(common_ids)}]")

    raw_arr = np.array(tecs_raw)
    unwhitened_arr = np.array(tecs_unwhitened)
    whitened_arr = np.array(tecs_whitened)

    # Statistical tests
    test_unwhitened = paired_test(unwhitened_arr, raw_arr, "TECS_unwhitened vs TECS_raw", seed=seed)
    test_whitened = paired_test(whitened_arr, raw_arr, "TECS_whitened vs TECS_raw", seed=seed)

    h6_supported = (
        (test_unwhitened["p_value"] < 0.05 and test_unwhitened["cohens_d"] > 0.2) or
        (test_whitened["p_value"] < 0.05 and test_whitened["cohens_d"] > 0.2)
    )

    elapsed = time.time() - start_time

    results = {
        "experiment": "ext_whitening_200",
        "phase": "5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "n_facts": len(common_ids),
            "model": model_name,
            "edit_layer": edit_layer,
            "seed": seed,
        },
        "distributions": {
            "raw": {"mean": float(raw_arr.mean()), "std": float(raw_arr.std())},
            "unwhitened": {"mean": float(unwhitened_arr.mean()), "std": float(unwhitened_arr.std())},
            "whitened": {"mean": float(whitened_arr.mean()), "std": float(whitened_arr.std())},
        },
        "statistical_tests": {
            "unwhitened_vs_raw": test_unwhitened,
            "whitened_vs_raw": test_whitened,
        },
        "decision": {
            "h6_whitening_explains_gap": h6_supported,
            "interpretation": (
                "SUPPORTED: Whitening correction significantly improves alignment"
                if h6_supported else
                "NOT SUPPORTED: Whitening is not the primary source of the gap"
            ),
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  Whitening 200 Result:")
    print(f"  Raw: {raw_arr.mean():.6f}, Unwhitened: {unwhitened_arr.mean():.6f}, Whitened: {whitened_arr.mean():.6f}")
    print(f"  H6: {'SUPPORTED' if h6_supported else 'NOT SUPPORTED'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "ext_whitening_200.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5: Whitening 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_whitening_200(cfg)


if __name__ == "__main__":
    main()
