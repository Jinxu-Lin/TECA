"""Step 2: Sanity check with CounterFact paraphrases (Plan B pilot).

Validates the end-to-end pipeline on 5 facts using CounterFact paraphrases
before committing to OpenWebText retrieval. Decouples engineering failures
from conceptual failures.

Usage:
    python -m experiments.probe_sanity_check [--num_facts 5] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_utils import load_model_and_tokenizer, get_mlp_weight
from core.rome_utils import compute_rome_edit, flatten_delta
from core.gradient_utils import compute_gradient_at_layer, flatten_gradient
from core.tecs import compute_tecs, compute_angular_variance
from core.retrieval import load_counterfact


def run_sanity_check(
    num_facts: int = 5,
    device: str = "cuda",
    output_dir: str = "results/probe/sanity_check",
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("TECA Probe Sanity Check (Plan B Pilot)")
    print("=" * 50)

    # 1. Load model
    print("\n[1/4] Loading GPT-2-XL...")
    model, tokenizer = load_model_and_tokenizer("gpt2-xl", device=device)

    # 2. Load CounterFact sample
    print(f"\n[2/4] Loading {num_facts} facts from CounterFact...")
    facts = load_counterfact(num_facts=num_facts)

    # 3. For each fact: ROME edit -> gradient -> TECS
    print("\n[3/4] Running ROME edits and computing TECS...")
    results = []
    for i, fact in enumerate(facts):
        print(f"\n  Fact {i+1}/{num_facts}: {fact['subject']} -> {fact['target_new']}")

        # ROME edit
        edit_result = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"],
            prompt=fact["prompt"],
            target_new=fact["target_new"],
            target_old=fact["target_old"],
            device=device,
        )
        print(f"    Edit layer: {edit_result.edit_layer}, "
              f"success: {edit_result.edit_success}, "
              f"prob: {edit_result.pre_prob:.4f} -> {edit_result.post_prob:.4f}")

        # Gradient of test prompt
        grad = compute_gradient_at_layer(
            model, tokenizer, fact["prompt"], edit_result.edit_layer, device,
        )

        # TECS
        tecs_val = compute_tecs(edit_result.delta_weight, grad)
        print(f"    TECS: {tecs_val:.6f}")

        # Check for NaN/zero (the kill signals)
        if tecs_val != tecs_val:  # NaN check
            print("    WARNING: TECS is NaN! Pipeline bug detected.")
        if abs(tecs_val) < 1e-12:
            print("    WARNING: TECS is essentially zero.")

        # Paraphrase gradients for angular variance
        if fact["paraphrases"]:
            para_grads = []
            for para in fact["paraphrases"][:5]:
                pg = compute_gradient_at_layer(
                    model, tokenizer, para, edit_result.edit_layer, device,
                )
                para_grads.append(pg)
            ang_var = compute_angular_variance(para_grads)
            print(f"    Angular variance (paraphrases): {ang_var:.6f}")
        else:
            ang_var = None

        results.append({
            "fact_id": fact["case_id"],
            "subject": fact["subject"],
            "tecs": tecs_val,
            "edit_success": edit_result.edit_success,
            "edit_layer": edit_result.edit_layer,
            "angular_variance": ang_var,
        })

    # 4. Summary
    print("\n[4/4] Summary")
    print("-" * 40)
    tecs_values = [r["tecs"] for r in results]
    has_nan = any(t != t for t in tecs_values)
    all_zero = all(abs(t) < 1e-12 for t in tecs_values)

    print(f"  TECS values: {[f'{t:.6f}' for t in tecs_values]}")
    print(f"  Mean TECS: {sum(tecs_values) / len(tecs_values):.6f}")
    print(f"  Any NaN: {has_nan}")
    print(f"  All zero: {all_zero}")

    if has_nan or all_zero:
        print("\n  VERDICT: FAIL - Pipeline has bugs. Fix before proceeding.")
    else:
        print("\n  VERDICT: PASS - Pipeline produces non-trivial TECS values.")
        print("  Proceed to SVD diagnostic (probe_svd_diagnostic.py) or main probe.")

    # Save results
    out_path = os.path.join(output_dir, "sanity_check_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TECA Probe Sanity Check")
    parser.add_argument("--num_facts", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/probe/sanity_check")
    args = parser.parse_args()

    run_sanity_check(
        num_facts=args.num_facts,
        device=args.device,
        output_dir=args.output_dir,
    )
