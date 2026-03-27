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


def compute_covariance_matrix(model_name, edit_layer, device, n_samples=100, max_seq_len=256,
                              low_rank_k=512):
    """Compute low-rank approximation of C = E[k k^T] where k is the MLP c_proj
    input (d_ff-dimensional) at the edit layer.

    For GPT-2-XL, c_proj input is 6400-dimensional (d_ff). A full (6400, 6400)
    covariance would be expensive, so we use torch.svd_lowrank to get the top-k
    eigenvectors/eigenvalues directly from the centered activation matrix.

    Returns:
        eigenvalues: (low_rank_k,) tensor of top eigenvalues (descending)
        eigenvectors: (d_ff, low_rank_k) tensor of corresponding eigenvectors
    """
    from core.model_utils import load_model_and_tokenizer
    from datasets import load_dataset

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    activations = []

    def hook_fn(module, input, output):
        # Capture c_proj input: shape (batch, seq_len, d_ff)
        inp = input[0].detach().float()
        for b in range(inp.shape[0]):
            # Subsample every 4th token to reduce memory
            activations.append(inp[b, ::4, :].cpu())

    block = model.transformer.h[edit_layer]
    # Hook on c_proj (NOT c_fc) to capture d_ff-dimensional activations
    handle = block.mlp.c_proj.register_forward_hook(hook_fn)

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

    all_acts = torch.cat(activations, dim=0)  # (N_tokens, d_ff)
    mean_act = all_acts.mean(dim=0)
    centered = all_acts - mean_act  # (N_tokens, d_ff)

    # Low-rank SVD: centered = U @ diag(S) @ Vh
    # The covariance C = centered^T @ centered / N has eigenvalues S^2/N
    # and eigenvectors are the columns of Vh^T (= V).
    k = min(low_rank_k, centered.shape[0], centered.shape[1])
    U, S, V = torch.svd_lowrank(centered, q=k, niter=5)
    # Eigenvalues of C = S^2 / N (descending because svd_lowrank returns sorted)
    eigenvalues = (S ** 2) / centered.shape[0]
    eigenvectors = V  # (d_ff, k)

    return eigenvalues, eigenvectors


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

    # Compute or load covariance low-rank approximation
    # Cache uses a different name since we changed from c_fc (d_model) to c_proj (d_ff)
    cov_cache = os.path.join(data_dir, "cov_lowrank_cproj_layer17.pt")
    if os.path.exists(cov_cache):
        print("  Loading cached covariance low-rank factors...")
        cached = torch.load(cov_cache, map_location="cpu", weights_only=True)
        eigenvalues = cached["eigenvalues"].float()
        eigenvectors = cached["eigenvectors"].float()
    else:
        print("  Computing covariance matrix from WikiText (c_proj input, d_ff space)...")
        eigenvalues, eigenvectors = compute_covariance_matrix(model_name, edit_layer, device)
        torch.save({"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}, cov_cache)

    # Filter near-zero eigenvalues
    threshold = eigenvalues.max() * 1e-6
    valid = eigenvalues > threshold
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]
    print(f"  Covariance: {eigenvectors.shape[0]}-dim space, {len(eigenvalues)} valid eigenvalues")

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

        # Whitening operations in the key space (d_ff = 6400 for GPT-2-XL c_proj).
        # delta_W shape: (d_ff, d_model) = (6400, 1600) for GPT-2 Conv1D.
        # C operates on the d_ff (row) dimension of delta_W.
        # eigenvectors: (d_ff, k), eigenvalues: (k,)
        #
        # For matrix M of shape (d_ff, d_model), applying C along d_ff:
        #   C @ M  approx=  V @ diag(lambda) @ V^T @ M
        # For C^{-1} @ M:
        #   C^{-1} @ M  approx=  V @ diag(1/lambda) @ V^T @ M

        d_ff = eigenvectors.shape[0]

        # Unwhitened: "undo" whitening by applying C to delta_W along d_ff axis
        # delta_unwhitened = C @ delta_W  (approx via low-rank)
        if delta_W.shape[0] == d_ff:
            proj_d = eigenvectors.T @ delta_W  # (k, d_model)
            delta_unwhitened = eigenvectors @ (eigenvalues.unsqueeze(1) * proj_d)  # (d_ff, d_model)
        else:
            delta_unwhitened = delta_W  # fallback

        t_unwhitened = cosine_similarity_flat(delta_unwhitened, g_M)
        tecs_unwhitened.append(t_unwhitened)

        # Whitened: apply C^{-1} to g_M along d_ff axis
        # g_whitened = C^{-1} @ g_M  (approx via low-rank)
        if g_M.shape[0] == d_ff:
            proj_g = eigenvectors.T @ g_M  # (k, d_model)
            g_whitened = eigenvectors @ (proj_g / eigenvalues.unsqueeze(1))  # (d_ff, d_model)
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
