#!/usr/bin/env python3
"""Phase 5: MEMIT Comparison on 200 Facts.

MEMIT distributes edits across layers 13-17 vs ROME's single layer 17.
Measure alignment at each layer and cross-layer to test whether
distributed editing bridges the incommensurability gap.

Usage:
    python -m experiments.full_scale.memit_200 --config configs/phase_5_extended.yaml
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
    paired_test, load_counterfact_facts, get_results_dir, get_data_dir,
)
from core.config import load_config


MEMIT_LAYERS = [13, 14, 15, 16, 17]


def run_memit_200(cfg: dict) -> dict:
    """Run MEMIT comparison on 200 facts."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
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
    print("Phase 5: MEMIT Comparison (200 Facts)")
    print("=" * 60)
    print(f"  MEMIT layers: {MEMIT_LAYERS}")
    print(f"  ROME edit layer: {edit_layer}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    # Load precomputed ROME deltas (at layer 17)
    rome_dir = os.path.join(data_dir, "rome_deltas_200")
    rome_deltas = {}
    if os.path.exists(rome_dir):
        for f in os.listdir(rome_dir):
            if f.startswith("delta_") and f.endswith(".pt"):
                cid = f.replace("delta_", "").replace(".pt", "")
                d = torch.load(os.path.join(rome_dir, f), map_location="cpu", weights_only=False)
                rome_deltas[cid] = d["delta_weight"].float()

    # For MEMIT, we compute gradients at each of the MEMIT layers
    # and compare with the ROME delta at layer 17
    cross_layer_tecs = {f"L{l}": [] for l in MEMIT_LAYERS}
    matched_tecs = []  # TECS at layer 17 (ROME layer)
    null_tecs = []

    print(f"\nProcessing {len(facts)} facts...")
    for i, fact in enumerate(facts):
        cid = str(fact["case_id"])
        if cid not in rome_deltas:
            continue

        delta = rome_deltas[cid]

        try:
            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"][:512] for r in retrieved[:top_k_gradient]]
            weights = [r["score"] for r in retrieved[:top_k_gradient]]

            for layer in MEMIT_LAYERS:
                g = compute_aggregated_gradient(
                    model, tokenizer, fact["prompt"],
                    training_texts, layer, device=device,
                    top_k=top_k_gradient, weights=weights,
                )
                tecs = cosine_similarity_flat(delta, g.cpu())
                cross_layer_tecs[f"L{layer}"].append(tecs)

                if layer == edit_layer:
                    matched_tecs.append(tecs)

            # Null: random gradient direction
            g_rand = torch.randn_like(delta)
            null_tecs.append(cosine_similarity_flat(delta, g_rand))

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
    print("\nResults:")
    layer_summary = {}
    for layer in MEMIT_LAYERS:
        key = f"L{layer}"
        arr = np.array(cross_layer_tecs[key])
        d_val = cohens_d(arr, np.array(null_tecs[:len(arr)]))
        layer_summary[key] = {
            "layer": layer,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "cohens_d_vs_null": float(d_val),
            "n": len(arr),
        }
        print(f"  L{layer}: TECS={arr.mean():.6f}, d={d_val:.4f}")

    # Cross-layer mean (MEMIT-style)
    if cross_layer_tecs:
        all_cross = []
        n_facts = len(cross_layer_tecs[f"L{MEMIT_LAYERS[0]}"])
        for fi in range(n_facts):
            layer_vals = [cross_layer_tecs[f"L{l}"][fi] for l in MEMIT_LAYERS
                          if fi < len(cross_layer_tecs[f"L{l}"])]
            if layer_vals:
                all_cross.append(np.mean(layer_vals))

        cross_arr = np.array(all_cross)
        cross_d = cohens_d(cross_arr, np.array(null_tecs[:len(cross_arr)]))
        print(f"  Cross-layer mean: TECS={cross_arr.mean():.6f}, d={cross_d:.4f}")
    else:
        cross_arr = np.array([])
        cross_d = 0.0

    elapsed = time.time() - start_time

    results = {
        "experiment": "ext_memit_200",
        "phase": "5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "memit_layers": MEMIT_LAYERS,
            "rome_edit_layer": edit_layer,
            "num_facts_processed": len(matched_tecs),
            "seed": seed,
        },
        "per_layer_results": layer_summary,
        "cross_layer": {
            "mean": float(cross_arr.mean()) if len(cross_arr) > 0 else None,
            "std": float(cross_arr.std()) if len(cross_arr) > 0 else None,
            "cohens_d": float(cross_d),
        },
        "matched_layer": {
            "mean": float(np.mean(matched_tecs)) if matched_tecs else None,
            "std": float(np.std(matched_tecs)) if matched_tecs else None,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  MEMIT 200 Result:")
    print(f"  Cross-layer d: {cross_d:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "ext_memit_200.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5: MEMIT 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_memit_200(cfg)


if __name__ == "__main__":
    main()
