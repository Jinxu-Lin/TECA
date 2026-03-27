#!/usr/bin/env python3
"""Phase 6: GPT-J TDA Gradient Computation on 100 Facts.

Usage:
    python -m experiments.cross_model.gptj_tda --config configs/phase_6_cross_model.yaml
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


def run_gptj_tda(cfg: dict) -> dict:
    """Compute TDA gradients for GPT-J."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    grad_dir = os.path.join(data_dir, "gptj_tda_gradients")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(grad_dir, exist_ok=True)

    cross_cfg = cfg.get("cross_model", {})
    model_name = cross_cfg.get("model_name", "EleutherAI/gpt-j-6b")
    dtype = cross_cfg.get("dtype", "float16")
    num_facts = cross_cfg.get("num_facts", 100)
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = 5  # GPT-J optimal

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")

    retrieval_cfg = cfg.get("retrieval", {})
    top_k_candidates = retrieval_cfg.get("top_k_candidates", 100)
    top_k_gradient = retrieval_cfg.get("top_k_gradient", 10)
    index_path = retrieval_cfg.get("index_path")

    print("=" * 60)
    print("Phase 6: GPT-J TDA Gradients")
    print("=" * 60)

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=dtype)

    for p in model.parameters():
        p.requires_grad_(False)

    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    per_fact = []
    norms = []

    for i, fact in enumerate(facts):
        cid = fact["case_id"]
        try:
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

            torch.save(g_M.cpu(), os.path.join(grad_dir, f"g_M_{cid}.pt"))
            norm = float(g_M.norm())
            norms.append(norm)
            per_fact.append({"case_id": cid, "g_M_norm": norm})

        except Exception as e:
            per_fact.append({"case_id": cid, "error": str(e)})

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    elapsed = time.time() - start_time

    results = {
        "experiment": "cross_gptj_tda",
        "phase": "6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": len(norms),
            "seed": seed,
        },
        "metrics": {
            "mean_norm": float(np.mean(norms)) if norms else 0,
        },
        "output_dir": grad_dir,
        "per_fact_results": per_fact,
    }

    save_results(results, os.path.join(results_dir, "cross_gptj_tda.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: GPT-J TDA")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_gptj_tda(cfg)


if __name__ == "__main__":
    main()
