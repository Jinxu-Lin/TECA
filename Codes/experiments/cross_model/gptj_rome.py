#!/usr/bin/env python3
"""Phase 6: GPT-J ROME Editing on 100 Facts.

Usage:
    python -m experiments.cross_model.gptj_rome --config configs/phase_6_cross_model.yaml
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


def run_gptj_rome(cfg: dict) -> dict:
    """Run ROME on GPT-J for cross-model validation."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "gptj_rome_deltas")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(rome_dir, exist_ok=True)

    cross_cfg = cfg.get("cross_model", {})
    model_name = cross_cfg.get("model_name", "EleutherAI/gpt-j-6b")
    dtype = cross_cfg.get("dtype", "float16")
    num_facts = cross_cfg.get("num_facts", 100)
    device = cfg.get("model", {}).get("device", "cuda")
    # GPT-J optimal edit layer (from ROME paper)
    edit_layer = 5

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")

    print("=" * 60)
    print("Phase 6: GPT-J ROME Editing")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Edit layer: {edit_layer}")
    print(f"  Dtype: {dtype}")
    print(f"  Num facts: {num_facts}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit

    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=dtype)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    per_fact = []
    success_count = 0

    for i, fact in enumerate(facts):
        cid = fact["case_id"]
        try:
            result = compute_rome_edit(
                model, tokenizer,
                subject=fact["subject"],
                prompt=fact["prompt"],
                target_new=fact["target_new"],
                target_old=fact["target_old"],
                edit_layer=edit_layer,
                device=device,
                backend="builtin",  # Use builtin for GPT-J (no EasyEdit hparams)
            )

            torch.save({
                "case_id": cid,
                "delta_weight": result.delta_weight.cpu(),
                "edit_layer": edit_layer,
                "edit_success": result.edit_success,
            }, os.path.join(rome_dir, f"delta_{cid}.pt"))

            if result.edit_success:
                success_count += 1

            per_fact.append({
                "case_id": cid,
                "edit_success": result.edit_success,
                "pre_prob": result.pre_prob,
                "post_prob": result.post_prob,
            })

        except Exception as e:
            per_fact.append({"case_id": cid, "error": str(e)})

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(facts)}] efficacy={success_count / (i + 1):.1%}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    total = len([p for p in per_fact if "error" not in p])
    efficacy = success_count / total if total > 0 else 0

    elapsed = time.time() - start_time

    results = {
        "experiment": "cross_gptj_rome",
        "phase": "6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "dtype": dtype,
            "num_facts": num_facts,
            "seed": seed,
        },
        "summary": {
            "total_valid": total,
            "success_count": success_count,
            "efficacy_rate": efficacy,
        },
        "output_dir": rome_dir,
        "per_fact_results": per_fact,
    }

    save_results(results, os.path.join(results_dir, "cross_gptj_rome.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: GPT-J ROME")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_gptj_rome(cfg)


if __name__ == "__main__":
    main()
