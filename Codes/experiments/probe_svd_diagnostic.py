"""Step 1b: SVD projection pre-diagnostic.

Quantifies spectral confound risk before the main probe experiment.
Runs on 5 facts (~30 min) to determine how much of Δθ and g live in the
dominant singular subspace of the weight matrix.

Usage:
    python -m experiments.probe_svd_diagnostic [--num_facts 5] [--top_k 10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_utils import load_model_and_tokenizer, get_mlp_weight
from core.rome_utils import compute_rome_edit
from core.gradient_utils import compute_gradient_at_layer
from core.svd_diagnostics import svd_projection_diagnostic
from core.retrieval import load_counterfact


def run_svd_diagnostic(
    num_facts: int = 5,
    top_k_sv: int = 10,
    device: str = "cuda",
    output_dir: str = "results/probe/svd_diagnostic",
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("TECA SVD Projection Pre-Diagnostic")
    print("=" * 50)

    # 1. Load model
    print("\n[1/3] Loading GPT-2-XL...")
    model, tokenizer = load_model_and_tokenizer("gpt2-xl", device=device)

    # 2. Load facts
    print(f"\n[2/3] Loading {num_facts} facts...")
    facts = load_counterfact(num_facts=num_facts, seed=123)

    # 3. For each fact: ROME edit, gradient, SVD diagnostic
    print(f"\n[3/3] Running SVD diagnostic (top-{top_k_sv} singular vectors)...")
    results = []

    for i, fact in enumerate(facts):
        print(f"\n  Fact {i+1}/{num_facts}: {fact['subject']}")

        # ROME edit to get Δθ and edit_layer
        edit_result = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"],
            prompt=fact["prompt"],
            target_new=fact["target_new"],
            device=device,
        )
        layer = edit_result.edit_layer

        # Gradient
        grad = compute_gradient_at_layer(
            model, tokenizer, fact["prompt"], layer, device,
        )

        # Weight matrix at edit layer
        W = get_mlp_weight(model, layer).detach().cpu()

        # SVD diagnostic
        diag = svd_projection_diagnostic(
            weight_matrix=W,
            delta_weight=edit_result.delta_weight,
            gradient=grad,
            top_k=top_k_sv,
        )
        diag.layer_idx = layer

        print(f"    Layer {layer}:")
        print(f"      Δθ projection onto top-{top_k_sv} SVs: "
              f"{diag.delta_projection_ratio:.4f}")
        print(f"      g  projection onto top-{top_k_sv} SVs: "
              f"{diag.gradient_projection_ratio:.4f}")
        print(f"      Spectral risk: {diag.spectral_risk}")

        results.append({
            "fact_id": fact["case_id"],
            "layer": layer,
            "delta_proj": diag.delta_projection_ratio,
            "grad_proj": diag.gradient_projection_ratio,
            "risk": diag.spectral_risk,
            "top_singular_values": diag.singular_values[:5],
        })

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("-" * 50)

    delta_projs = [r["delta_proj"] for r in results]
    grad_projs = [r["grad_proj"] for r in results]
    risks = [r["risk"] for r in results]

    print(f"  Mean Δθ projection: {sum(delta_projs)/len(delta_projs):.4f}")
    print(f"  Mean g  projection: {sum(grad_projs)/len(grad_projs):.4f}")
    print(f"  Risk levels: {risks}")

    high_risk = sum(1 for r in risks if r == "high")
    if high_risk >= len(results) // 2:
        print("\n  WARNING: Majority of facts show HIGH spectral confound risk.")
        print("  The Null-B placebo test becomes CRITICAL for interpreting results.")
        print("  Consider adding spectral residual TECS (project out top-k SVs).")
    else:
        print("\n  Spectral confound risk is manageable.")
        print("  Proceed to main probe experiment.")

    # Save
    out_path = os.path.join(output_dir, "svd_diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TECA SVD Diagnostic")
    parser.add_argument("--num_facts", type=int, default=5)
    parser.add_argument("--top_k_sv", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/probe/svd_diagnostic")
    args = parser.parse_args()

    run_svd_diagnostic(
        num_facts=args.num_facts,
        top_k_sv=args.top_k_sv,
        device=args.device,
        output_dir=args.output_dir,
    )
