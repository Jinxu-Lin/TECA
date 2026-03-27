#!/usr/bin/env python3
"""Phase 4: Gradient Scope Ablation.

Compare single-layer (l* only) vs multi-layer (l* +/- 2) gradient computation.
Expected: Multi-layer may increase alignment (cf. MEMIT result).

Usage:
    python -m experiments.ablation.scope_ablation --config configs/phase_4_ablation.yaml
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


def run_scope_ablation(cfg: dict) -> dict:
    """Run gradient scope ablation: single-layer vs multi-layer."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    abl_cfg = cfg.get("ablation", {})
    multi_layer_range = abl_cfg.get("multi_layer_range", 2)

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

    # Multi-layer: l* +/- range
    multi_layers = list(range(
        max(0, edit_layer - multi_layer_range),
        edit_layer + multi_layer_range + 1,
    ))

    print("=" * 60)
    print("Phase 4: Gradient Scope Ablation")
    print("=" * 60)
    print(f"  Single layer: {edit_layer}")
    print(f"  Multi layers: {multi_layers}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    single_layer_tecs = []
    multi_layer_tecs = []

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
            weights = [r["score"] for r in retrieved[:top_k_gradient]]

            # Single layer
            g_single = compute_aggregated_gradient(
                model, tokenizer, fact["prompt"],
                training_texts, edit_layer, device=device,
                top_k=top_k_gradient, weights=weights,
            )
            tecs_s = cosine_similarity_flat(delta, g_single)
            single_layer_tecs.append(tecs_s)

            # Multi-layer: concatenate gradients from all layers
            multi_grads = []
            for layer in multi_layers:
                g = compute_aggregated_gradient(
                    model, tokenizer, fact["prompt"],
                    training_texts, layer, device=device,
                    top_k=top_k_gradient, weights=weights,
                )
                multi_grads.append(g.cpu().flatten())

            g_multi = torch.cat(multi_grads)

            # For multi-layer TECS, need to also expand delta to match
            # Use zero-padding for non-edit layers
            delta_expanded_parts = []
            for layer in multi_layers:
                if layer == edit_layer:
                    delta_expanded_parts.append(delta.flatten())
                else:
                    delta_expanded_parts.append(torch.zeros_like(delta.flatten()))

            delta_expanded = torch.cat(delta_expanded_parts)
            tecs_m = cosine_similarity_flat(delta_expanded, g_multi)
            multi_layer_tecs.append(tecs_m)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    single_arr = np.array(single_layer_tecs)
    multi_arr = np.array(multi_layer_tecs)

    print(f"  Single-layer: {single_arr.mean():.6f} +/- {single_arr.std():.6f}")
    print(f"  Multi-layer: {multi_arr.mean():.6f} +/- {multi_arr.std():.6f}")

    from experiments.common import paired_test
    test = paired_test(
        np.abs(multi_arr), np.abs(single_arr),
        "|TECS_multi| vs |TECS_single|", seed=seed,
    )

    mean_tecs = np.mean([single_arr.mean(), multi_arr.mean()])
    max_var = abs(single_arr.mean() - multi_arr.mean()) / abs(mean_tecs) if abs(mean_tecs) > 1e-12 else 0

    elapsed = time.time() - start_time

    results = {
        "experiment": "ablation_scope",
        "phase": "4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "edit_layer": edit_layer,
            "multi_layers": multi_layers,
            "num_facts": len(single_layer_tecs),
            "seed": seed,
        },
        "single_layer": {
            "mean": float(single_arr.mean()),
            "std": float(single_arr.std()),
        },
        "multi_layer": {
            "mean": float(multi_arr.mean()),
            "std": float(multi_arr.std()),
        },
        "statistical_test": test,
        "robustness": {
            "relative_variation": float(max_var),
            "robust": max_var < 0.20,
        },
    }

    save_results(results, os.path.join(results_dir, "ablation_scope.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Scope Ablation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_scope_ablation(cfg)


if __name__ == "__main__":
    main()
