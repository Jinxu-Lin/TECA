#!/usr/bin/env python3
"""Phase 4: Top-k Ablation.

Vary top-k in {5, 10, 20, 50} for TDA gradient aggregation and measure TECS.
Expected: TECS variation < 20% across k values.

Usage:
    python -m experiments.ablation.topk_ablation --config configs/phase_4_ablation.yaml
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
    set_seed, save_results, cosine_similarity_flat,
    load_counterfact_facts, get_results_dir,
)
from core.config import load_config


def run_topk_ablation(cfg: dict) -> dict:
    """Run top-k ablation: vary k and measure TECS."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    abl_cfg = cfg.get("ablation", {})
    k_values = abl_cfg.get("top_k_values", [5, 10, 20, 50])

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
    print("Phase 4: Top-k Ablation")
    print("=" * 60)
    print(f"  k values: {k_values}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_per_sample_gradients
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    max_k = max(k_values)
    k_results = {k: [] for k in k_values}

    print(f"\nProcessing {len(facts)} facts...")
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
            delta = edit_result.delta_weight.cpu()

            # Retrieve and compute gradients for max_k
            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"] for r in retrieved[:max_k]]
            weights = [r["score"] for r in retrieved[:max_k]]

            grads = compute_per_sample_gradients(
                model, tokenizer, training_texts, edit_layer, device=device,
            )

            if not grads:
                continue

            # For each k, aggregate and compute TECS
            for k in k_values:
                k_grads = grads[:k]
                k_weights = weights[:len(k_grads)]
                w = torch.tensor(k_weights, dtype=torch.float32)
                w = w / w.sum()
                stacked = torch.stack(k_grads)
                g_M = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
                tecs = cosine_similarity_flat(delta, g_M)
                k_results[k].append(tecs)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    # Analysis
    summary = {}
    means = []
    for k in k_values:
        arr = np.array(k_results[k])
        summary[f"k{k}"] = {
            "k": k,
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }
        means.append(float(arr.mean()))
        print(f"  k={k}: TECS={arr.mean():.6f} +/- {arr.std():.6f} (n={len(arr)})")

    # Robustness: variation < 20%
    mean_tecs = np.mean(means)
    max_variation = max(abs(m - mean_tecs) / abs(mean_tecs) for m in means) if abs(mean_tecs) > 1e-12 else 0.0

    elapsed = time.time() - start_time

    results = {
        "experiment": "ablation_topk",
        "phase": "4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "k_values": k_values,
            "num_facts": num_facts,
            "seed": seed,
        },
        "per_k_results": summary,
        "robustness": {
            "mean_tecs_across_k": float(mean_tecs),
            "max_relative_variation": float(max_variation),
            "robust": max_variation < 0.20,
        },
    }

    save_results(results, os.path.join(results_dir, "ablation_topk.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Top-k Ablation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_topk_ablation(cfg)


if __name__ == "__main__":
    main()
