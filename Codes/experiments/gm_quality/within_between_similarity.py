#!/usr/bin/env python3
"""Phase 2a: Within-Fact vs Between-Fact Gradient Similarity.

For each of N facts, compute pairwise cosine similarity among its top-k
training gradients (within-fact) and compare to cross-fact gradient
similarity (between-fact). If within >> between, gradients contain
fact-specific information despite the low effective dimensionality.

Usage:
    python -m experiments.gm_quality.within_between_similarity --config configs/phase_2_gm_quality.yaml
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
    set_seed, save_results, paired_test, bootstrap_ci,
    load_counterfact_facts, get_results_dir,
)
from core.config import load_config


def compute_pairwise_cosine(grads):
    """Compute mean pairwise cosine similarity among a list of gradient tensors."""
    n = len(grads)
    if n < 2:
        return 0.0, []

    flat = torch.stack([g.reshape(-1).float() for g in grads])
    flat_norm = F.normalize(flat, dim=1)
    cos_matrix = flat_norm @ flat_norm.T

    # Upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise = cos_matrix[mask].tolist()
    return float(np.mean(pairwise)), pairwise


def run_within_between_similarity(cfg: dict) -> dict:
    """Run within-fact vs between-fact gradient similarity analysis."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    gm_cfg = cfg.get("gm_quality", {})
    top_k = gm_cfg.get("within_between_top_k", 20)
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
    index_path = retrieval_cfg.get("index_path")

    print("=" * 60)
    print("Phase 2a: Within-Fact vs Between-Fact Gradient Similarity")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Num facts: {num_facts}")
    print(f"  Top-k gradients per fact: {top_k}")
    print(f"  Seed: {seed}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.gradient_utils import compute_per_sample_gradients
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    # For each fact: compute top-k gradients, measure within-fact similarity
    within_means = []
    all_grad_representatives = []  # One representative gradient per fact (for between-fact)
    per_fact_results = []

    print(f"\nComputing per-sample gradients for {len(facts)} facts...")

    for i, fact in enumerate(facts):
        try:
            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"] for r in retrieved[:top_k]]

            grads = compute_per_sample_gradients(
                model, tokenizer, training_texts, edit_layer, device=device,
            )

            if len(grads) < 2:
                continue

            within_mean, within_pairwise = compute_pairwise_cosine(grads)
            within_means.append(within_mean)

            # Store mean gradient as representative
            g_mean = torch.stack(grads).mean(dim=0)
            all_grad_representatives.append(g_mean)

            per_fact_results.append({
                "case_id": fact["case_id"],
                "n_grads": len(grads),
                "within_mean_cosine": within_mean,
            })

        except Exception as e:
            print(f"  [WARN] Fact {i} ({fact['subject']}): {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}] computed, within_mean={within_mean:.6f}")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  {len(within_means)} facts with valid gradients")

    # Compute between-fact similarity using representative gradients
    print("\nComputing between-fact similarity...")
    between_mean, between_pairwise = compute_pairwise_cosine(all_grad_representatives)

    # Also compute random between-fact pairs for matched comparison
    rng = np.random.RandomState(seed)
    n_between_samples = len(within_means)
    between_samples = []
    n_reps = len(all_grad_representatives)

    for _ in range(n_between_samples):
        i, j = rng.choice(n_reps, size=2, replace=False)
        g_i = all_grad_representatives[i].reshape(-1).float()
        g_j = all_grad_representatives[j].reshape(-1).float()
        cos = F.cosine_similarity(g_i.unsqueeze(0), g_j.unsqueeze(0)).item()
        between_samples.append(cos)

    within_arr = np.array(within_means)
    between_arr = np.array(between_samples)

    print(f"  Within-fact mean cosine: {within_arr.mean():.6f} +/- {within_arr.std():.6f}")
    print(f"  Between-fact mean cosine: {between_arr.mean():.6f} +/- {between_arr.std():.6f}")

    # Statistical test
    test = paired_test(
        within_arr, between_arr[:len(within_arr)],
        "Within-fact vs Between-fact gradient cosine similarity",
        seed=seed,
    )

    elapsed = time.time() - start_time

    results = {
        "experiment": "gm_within_between",
        "phase": "2a",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": len(within_means),
            "top_k": top_k,
            "seed": seed,
        },
        "within_fact": {
            "mean": float(within_arr.mean()),
            "std": float(within_arr.std()),
            "median": float(np.median(within_arr)),
        },
        "between_fact": {
            "mean": float(between_arr.mean()),
            "std": float(between_arr.std()),
            "median": float(np.median(between_arr)),
            "n_pairs": len(between_samples),
        },
        "statistical_test": test,
        "decision": {
            "within_gt_between": float(within_arr.mean()) > float(between_arr.mean()),
            "significant": test["p_value"] < 0.05,
            "interpretation": (
                "Gradients contain FACT-SPECIFIC information: within-fact similarity "
                f"({within_arr.mean():.6f}) significantly > between-fact ({between_arr.mean():.6f})"
                if test["p_value"] < 0.05 and within_arr.mean() > between_arr.mean()
                else
                "Gradients may be dominated by GENERIC component: within-fact and "
                "between-fact similarity are not significantly different"
            ),
        },
        "per_fact_results": per_fact_results,
    }

    print(f"\n{'=' * 60}")
    print(f"  Within vs Between Result:")
    print(f"  Within: {within_arr.mean():.6f} | Between: {between_arr.mean():.6f}")
    print(f"  Cohen's d: {test['cohens_d']:.4f}")
    print(f"  p-value: {test['p_value']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "gm_within_between.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2a: Within vs Between Gradient Similarity")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_within_between_similarity(cfg)


if __name__ == "__main__":
    main()
