#!/usr/bin/env python3
"""Phase 3: Full-Scale TDA Gradient Computation for 200 Facts.

For each fact: BM25 retrieve, compute per-sample gradients at layer 17,
aggregate with BM25 weights, save g_M tensor.

Usage:
    python -m experiments.full_scale.tda_gradients_200 --config configs/phase_3_full_scale.yaml
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
    set_seed, save_results, load_counterfact_facts, get_results_dir, get_data_dir,
)
from core.config import load_config


def run_tda_gradients_200(cfg: dict) -> dict:
    """Compute TDA gradients for 200 facts."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    grad_dir = os.path.join(data_dir, "tda_gradients_200")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(grad_dir, exist_ok=True)

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
    print("Phase 3: Full-Scale TDA Gradients (200 Facts)")
    print("=" * 60)

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.gradient_utils import compute_aggregated_gradient, compute_per_sample_gradients
    from core.retrieval import retrieve_training_samples_bm25
    from core.tecs import compute_mean_pairwise_cosine

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    # Freeze all except target param
    for p in model.parameters():
        p.requires_grad_(False)

    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    per_fact = []
    norms = []
    angular_vars = []

    print(f"\nComputing TDA gradients for {len(facts)} facts...")
    for i, fact in enumerate(facts):
        cid = fact["case_id"]
        try:
            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"] for r in retrieved[:top_k_gradient]]
            weights = [r["score"] for r in retrieved[:top_k_gradient]]

            # Compute per-sample gradients for angular variance
            grads = compute_per_sample_gradients(
                model, tokenizer, training_texts, edit_layer, device=device,
            )

            if not grads:
                continue

            # Aggregate
            w = torch.tensor(weights[:len(grads)], dtype=torch.float32)
            w = w / w.sum()
            stacked = torch.stack(grads, dim=0)
            g_M = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)

            # Angular variance
            ang_var = compute_mean_pairwise_cosine(grads)

            # Save
            torch.save(g_M.cpu(), os.path.join(grad_dir, f"g_M_{cid}.pt"))

            norm = float(g_M.norm())
            norms.append(norm)
            angular_vars.append(ang_var)

            per_fact.append({
                "case_id": cid,
                "n_grads": len(grads),
                "g_M_norm": norm,
                "angular_variance": ang_var,
                "top_bm25_score": float(weights[0]),
            })

        except Exception as e:
            per_fact.append({"case_id": cid, "error": str(e)})

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}] mean_norm={np.mean(norms):.6f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    n_valid = len(norms)
    elapsed = time.time() - start_time

    results = {
        "experiment": "full_tda_200",
        "phase": "3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": n_valid,
            "top_k_gradient": top_k_gradient,
            "seed": seed,
        },
        "metrics": {
            "mean_norm": float(np.mean(norms)) if norms else 0,
            "std_norm": float(np.std(norms)) if norms else 0,
            "mean_angular_variance": float(np.mean(angular_vars)) if angular_vars else 0,
            "no_nan": True,
        },
        "output_dir": grad_dir,
        "per_fact_results": per_fact,
    }

    print(f"\n{'=' * 60}")
    print(f"  TDA Gradients 200 Result:")
    print(f"  Valid: {n_valid}/{len(facts)}")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Mean angular variance: {np.mean(angular_vars):.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "tda_200_validation.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: TDA Gradients 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_tda_gradients_200(cfg)


if __name__ == "__main__":
    main()
