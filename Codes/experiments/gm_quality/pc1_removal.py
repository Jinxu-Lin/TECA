#!/usr/bin/env python3
"""Phase 2b: PC1 Removal Analysis.

Remove the dominant PC1 direction from all attribution gradients and
re-compute TECS + effective dimensionality. If TECS increases after
PC1 removal, the "generic relevance" component was masking a weaker
fact-specific signal.

Usage:
    python -m experiments.gm_quality.pc1_removal --config configs/phase_2_gm_quality.yaml
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
    set_seed, save_results, cosine_similarity_flat, cohens_d,
    bootstrap_ci, paired_test, load_counterfact_facts, get_results_dir,
)
from core.config import load_config


def compute_effective_dimensionality(S):
    """Compute effective dimensionality from singular values via eigenvalue entropy."""
    s2 = S ** 2
    p = s2 / s2.sum()
    p = p[p > 1e-12]  # filter zeros
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def run_pc1_removal(cfg: dict) -> dict:
    """Run PC1 removal analysis on attribution gradients."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    num_facts = data_cfg.get("num_facts", 200)
    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    retrieval_cfg = cfg.get("retrieval", {})
    top_k_candidates = retrieval_cfg.get("top_k_candidates", 100)
    top_k_gradient = retrieval_cfg.get("top_k_gradient", 10)
    index_path = retrieval_cfg.get("index_path")

    print("=" * 60)
    print("Phase 2b: PC1 Removal Analysis")
    print("=" * 60)

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    # Collect ROME deltas and aggregated gradients
    deltas = []
    gradients = []
    case_ids = []

    print(f"\nComputing ROME deltas and TDA gradients for {len(facts)} facts...")
    for i, fact in enumerate(facts):
        try:
            # ROME edit
            edit_result = compute_rome_edit(
                model, tokenizer,
                subject=fact["subject"],
                prompt=fact["prompt"],
                target_new=fact["target_new"],
                target_old=fact["target_old"],
                edit_layer=edit_layer,
                device=device,
            )

            # TDA gradient
            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"] for r in retrieved[:top_k_gradient]]
            weights = [r["score"] for r in retrieved[:top_k_gradient]]

            g_M = compute_aggregated_gradient(
                model, tokenizer, fact["prompt"],
                training_texts, edit_layer, device=device,
                top_k=top_k_gradient, weights=weights,
            )

            deltas.append(edit_result.delta_weight.cpu())
            gradients.append(g_M.cpu())
            case_ids.append(fact["case_id"])

        except Exception as e:
            print(f"  [WARN] Fact {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}] computed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_valid = len(deltas)
    print(f"  {n_valid} valid facts")

    # Stack gradients for SVD
    print("\nComputing PC1 from gradient matrix...")
    G_flat = torch.stack([g.reshape(-1).float() for g in gradients])  # (n, dim)

    # Center
    G_mean = G_flat.mean(dim=0, keepdim=True)
    G_centered = G_flat - G_mean

    # Compute SVD (low-rank since n << dim)
    U, S, Vh = torch.svd_lowrank(G_centered, q=min(n_valid, 100), niter=5)

    eff_dim_before = compute_effective_dimensionality(S)
    pc1_variance_ratio = float((S[0] ** 2) / (S ** 2).sum())

    print(f"  Effective dim (before): {eff_dim_before:.2f}")
    print(f"  PC1 variance ratio: {pc1_variance_ratio:.4f}")
    print(f"  Top-5 singular values: {S[:5].tolist()}")

    # PC1 direction (in original space)
    pc1 = Vh[:, 0]  # (dim,)

    # Remove PC1 from all gradients
    print("\nRemoving PC1 and recomputing TECS...")
    gradients_residual = []
    for g in gradients:
        g_flat = g.reshape(-1).float()
        proj = torch.dot(g_flat, pc1) * pc1
        g_residual = g_flat - proj
        gradients_residual.append(g_residual.reshape(g.shape))

    # Recompute effective dimensionality after PC1 removal
    G_residual_flat = torch.stack([g.reshape(-1).float() for g in gradients_residual])
    G_residual_centered = G_residual_flat - G_residual_flat.mean(dim=0, keepdim=True)
    _, S_residual, _ = torch.svd_lowrank(G_residual_centered, q=min(n_valid, 100), niter=5)
    eff_dim_after = compute_effective_dimensionality(S_residual)

    print(f"  Effective dim (after): {eff_dim_after:.2f}")

    # Compute TECS before and after PC1 removal
    tecs_before = []
    tecs_after = []

    for delta, grad_orig, grad_res in zip(deltas, gradients, gradients_residual):
        t_before = cosine_similarity_flat(delta, grad_orig)
        t_after = cosine_similarity_flat(delta, grad_res)
        tecs_before.append(t_before)
        tecs_after.append(t_after)

    tecs_before_arr = np.array(tecs_before)
    tecs_after_arr = np.array(tecs_after)

    print(f"  TECS before: {tecs_before_arr.mean():.6f} +/- {tecs_before_arr.std():.6f}")
    print(f"  TECS after:  {tecs_after_arr.mean():.6f} +/- {tecs_after_arr.std():.6f}")

    # Statistical test
    test = paired_test(
        np.abs(tecs_after_arr), np.abs(tecs_before_arr),
        "|TECS_after_PC1_removal| vs |TECS_before|",
        seed=seed,
    )

    elapsed = time.time() - start_time

    results = {
        "experiment": "gm_pc1_removal",
        "phase": "2b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": n_valid,
            "seed": seed,
        },
        "pc1_analysis": {
            "pc1_variance_ratio": pc1_variance_ratio,
            "eff_dim_before": eff_dim_before,
            "eff_dim_after": eff_dim_after,
            "eff_dim_increase": eff_dim_after - eff_dim_before,
            "top_5_singular_values_before": S[:5].tolist(),
            "top_5_singular_values_after": S_residual[:5].tolist(),
        },
        "tecs_comparison": {
            "tecs_before_mean": float(tecs_before_arr.mean()),
            "tecs_before_std": float(tecs_before_arr.std()),
            "tecs_after_mean": float(tecs_after_arr.mean()),
            "tecs_after_std": float(tecs_after_arr.std()),
            "abs_tecs_before_mean": float(np.abs(tecs_before_arr).mean()),
            "abs_tecs_after_mean": float(np.abs(tecs_after_arr).mean()),
        },
        "statistical_test": test,
        "decision": {
            "eff_dim_increased": eff_dim_after > eff_dim_before,
            "tecs_increased": float(np.abs(tecs_after_arr).mean()) > float(np.abs(tecs_before_arr).mean()),
            "interpretation": (
                "PC1 removal INCREASED TECS: the generic component was masking a "
                "fact-specific alignment signal"
                if np.abs(tecs_after_arr).mean() > np.abs(tecs_before_arr).mean() and test["p_value"] < 0.05
                else
                "PC1 removal did NOT increase TECS: the low alignment is not due to "
                "a dominant generic gradient component"
            ),
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  PC1 Removal Result:")
    print(f"  Eff-dim: {eff_dim_before:.2f} -> {eff_dim_after:.2f}")
    print(f"  |TECS|: {np.abs(tecs_before_arr).mean():.6f} -> {np.abs(tecs_after_arr).mean():.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "gm_pc1_removal.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2b: PC1 Removal Analysis")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pc1_removal(cfg)


if __name__ == "__main__":
    main()
