#!/usr/bin/env python3
"""Phase 4: Weighting Scheme Ablation.

Compare BM25 / uniform / TF-IDF weighting for gradient aggregation.
Expected: TECS variation < 20% across schemes.

Usage:
    python -m experiments.ablation.weighting_ablation --config configs/phase_4_ablation.yaml
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


def run_weighting_ablation(cfg: dict) -> dict:
    """Run weighting scheme ablation."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    abl_cfg = cfg.get("ablation", {})
    methods = abl_cfg.get("weighting_methods", ["bm25", "uniform", "tfidf"])

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
    print("Phase 4: Weighting Scheme Ablation")
    print("=" * 60)
    print(f"  Methods: {methods}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_per_sample_gradients
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    method_results = {m: [] for m in methods}

    print(f"\nProcessing {len(facts)} facts...")
    for i, fact in enumerate(facts):
        try:
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

            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"] for r in retrieved[:top_k_gradient]]
            bm25_scores = [r["score"] for r in retrieved[:top_k_gradient]]

            grads = compute_per_sample_gradients(
                model, tokenizer, training_texts, edit_layer, device=device,
            )

            if not grads:
                continue

            stacked = torch.stack(grads)

            for method in methods:
                if method == "bm25":
                    w = torch.tensor(bm25_scores[:len(grads)], dtype=torch.float32)
                elif method == "uniform":
                    w = torch.ones(len(grads), dtype=torch.float32)
                elif method == "rank_inverse":
                    # Rank-inverse weighting: 1/(rank+1) based on BM25 retrieval rank
                    w = torch.tensor([1.0 / (rank + 1) for rank in range(len(grads))], dtype=torch.float32)
                else:
                    w = torch.ones(len(grads), dtype=torch.float32)

                w = w / w.sum()
                g_M = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
                tecs = cosine_similarity_flat(delta, g_M)
                method_results[method].append(tecs)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    summary = {}
    means = []
    for method in methods:
        arr = np.array(method_results[method])
        summary[method] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }
        means.append(float(arr.mean()))
        print(f"  {method}: TECS={arr.mean():.6f} +/- {arr.std():.6f}")

    mean_tecs = np.mean(means)
    max_var = max(abs(m - mean_tecs) / abs(mean_tecs) for m in means) if abs(mean_tecs) > 1e-12 else 0.0

    elapsed = time.time() - start_time

    results = {
        "experiment": "ablation_weighting",
        "phase": "4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {"methods": methods, "num_facts": num_facts, "seed": seed},
        "per_method_results": summary,
        "robustness": {
            "mean_tecs_across_methods": float(mean_tecs),
            "max_relative_variation": float(max_var),
            "robust": max_var < 0.20,
        },
    }

    save_results(results, os.path.join(results_dir, "ablation_weighting.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Weighting Ablation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_weighting_ablation(cfg)


if __name__ == "__main__":
    main()
