#!/usr/bin/env python3
"""Phase 3: Full-Scale ROME Editing on 200 Facts.

Run ROME editing on 200 CounterFact facts at layer 17.
Save rank-1 delta components (u, v) for each fact.

Usage:
    python -m experiments.full_scale.rome_200 --config configs/phase_3_full_scale.yaml
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


def run_rome_200(cfg: dict) -> dict:
    """Run ROME editing on 200 facts and save delta tensors."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "rome_deltas_200")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(rome_dir, exist_ok=True)

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    num_facts = data_cfg.get("num_facts", 200)
    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    print("=" * 60)
    print("Phase 3: Full-Scale ROME Editing (200 Facts)")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Edit layer: {edit_layer}")
    print(f"  Num facts: {num_facts}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    per_fact = []
    success_count = 0

    print(f"\nEditing {len(facts)} facts...")
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
            )

            # Save delta tensor
            torch.save({
                "case_id": cid,
                "delta_weight": result.delta_weight.cpu(),
                "edit_layer": edit_layer,
                "edit_success": result.edit_success,
                "pre_prob": result.pre_prob,
                "post_prob": result.post_prob,
            }, os.path.join(rome_dir, f"delta_{cid}.pt"))

            if result.edit_success:
                success_count += 1

            per_fact.append({
                "case_id": cid,
                "subject": fact["subject"],
                "edit_success": result.edit_success,
                "pre_prob": result.pre_prob,
                "post_prob": result.post_prob,
                "delta_norm": float(result.delta_weight.norm()),
            })

        except Exception as e:
            per_fact.append({
                "case_id": cid,
                "subject": fact["subject"],
                "error": str(e),
            })

        if (i + 1) % 20 == 0:
            efficacy = success_count / (i + 1)
            print(f"  [{i + 1}/{len(facts)}] efficacy={efficacy:.1%}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len([p for p in per_fact if "error" not in p])
    efficacy = success_count / total if total > 0 else 0.0
    elapsed = time.time() - start_time

    results = {
        "experiment": "full_rome_200",
        "phase": "3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": num_facts,
            "seed": seed,
        },
        "summary": {
            "total_attempted": len(facts),
            "total_valid": total,
            "success_count": success_count,
            "efficacy_rate": efficacy,
            "gate_pass": efficacy > 0.75,
        },
        "output_dir": rome_dir,
        "per_fact_results": per_fact,
    }

    print(f"\n{'=' * 60}")
    print(f"  ROME 200 Result:")
    print(f"  Efficacy: {efficacy:.1%} ({success_count}/{total})")
    print(f"  Gate (>75%): {'PASS' if efficacy > 0.75 else 'FAIL'}")
    print(f"  Deltas saved to: {rome_dir}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "rome_200_validation.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: Full-Scale ROME 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_rome_200(cfg)


if __name__ == "__main__":
    main()
