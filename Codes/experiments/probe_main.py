"""Main probe experiment: full TECS computation with all null baselines.

Executes Steps 1, 3-7 from the probe plan:
  1. ROME edits on 50 facts
  3. OpenWebText BM25 retrieval + gradient ranking
  4. Aggregated gradient computation at edit layer
  5. TECS computation
  6. Three-tier null baselines (Null-A, Null-B, Null-C)
  7. Statistical tests and pass/fail evaluation

Usage:
    python -m experiments.probe_main [--num_facts 50] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_utils import load_model_and_tokenizer, get_mlp_weight, num_layers
from core.rome_utils import compute_rome_edit, flatten_delta, EditResult
from core.gradient_utils import (
    compute_gradient_at_layer,
    compute_aggregated_gradient,
    compute_per_sample_gradients,
    flatten_gradient,
)
from core.tecs import compute_tecs, compute_null_a, compute_angular_variance, TECSResult
from core.retrieval import (
    load_counterfact,
    retrieve_training_samples_bm25,
    rank_by_gradient_dot_product,
)
from core.statistics import paired_t_test, check_pass_criteria, format_report


def run_main_probe(
    num_facts: int = 50,
    top_k_retrieval: int = 100,
    top_k_gradient: int = 10,
    null_a_num: int = 10,
    placebo_offsets: list = (-5, 5),
    corpus: str = "openwebtext",
    device: str = "cuda",
    output_dir: str = "results/probe/main",
    bm25_index_path: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    print("=" * 60)
    print("TECA Main Probe Experiment")
    print("=" * 60)
    t0 = time.time()

    # ---------------------------------------------------------------
    # Phase 1: Load model and data
    # ---------------------------------------------------------------
    print("\n[Phase 1] Loading model and data...")
    model, tokenizer = load_model_and_tokenizer("gpt2-xl", device=device)
    facts = load_counterfact(num_facts=num_facts)
    n_layers = num_layers(model)
    print(f"  Model loaded. {n_layers} layers.")
    print(f"  {len(facts)} facts loaded.")

    # ---------------------------------------------------------------
    # Phase 2: ROME edits (all facts)
    # ---------------------------------------------------------------
    print("\n[Phase 2] Running ROME edits...")
    edit_results: Dict[int, EditResult] = {}
    for i, fact in enumerate(facts):
        print(f"  [{i+1}/{num_facts}] {fact['subject']} -> {fact['target_new']}",
              end="", flush=True)
        er = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"],
            prompt=fact["prompt"],
            target_new=fact["target_new"],
            target_old=fact["target_old"],
            device=device,
        )
        edit_results[i] = er
        status = "OK" if er.edit_success else "FAIL"
        print(f"  [{status}] layer={er.edit_layer}")

    edit_layer = edit_results[0].edit_layer  # assume same for all facts
    n_success = sum(1 for er in edit_results.values() if er.edit_success)
    print(f"\n  Edit success: {n_success}/{num_facts} at layer {edit_layer}")

    # ---------------------------------------------------------------
    # Phase 3: Retrieve training samples + compute gradients
    # ---------------------------------------------------------------
    print(f"\n[Phase 3] Retrieving training samples from {corpus}...")
    tecs_results: List[dict] = []

    # Pre-select unrelated fact indices for Null-B (prompt-level placebo)
    import random

    for i, fact in enumerate(facts):
        print(f"\n  Fact [{i+1}/{num_facts}]: {fact['subject']}")

        # 3a. BM25 retrieval
        query = f"{fact['subject']} {fact['target_old']}"
        candidates = retrieve_training_samples_bm25(
            query, top_k=top_k_retrieval,
            corpus_name=corpus, index_path=bm25_index_path,
        )
        print(f"    Retrieved {len(candidates)} candidates")

        if len(candidates) < top_k_gradient:
            print(f"    WARNING: Only {len(candidates)} candidates, need {top_k_gradient}")

        # 3b. Compute per-training-sample gradients (used for both TDA and diagnostics)
        # Each g_i = grad of L(theta; x_i) w.r.t. edit layer weight
        sample_texts = [c["text"][:512] for c in candidates[:top_k_gradient]]
        per_sample_grads = compute_per_sample_gradients(
            model, tokenizer, sample_texts, edit_layer, device,
        )

        # 3c. Aggregate: g_M = mean of training sample gradients (proper TDA)
        if per_sample_grads:
            aggregated_grad = torch.stack(per_sample_grads, dim=0).mean(dim=0)
        else:
            # Fallback: use test prompt gradient if no training samples
            aggregated_grad = compute_gradient_at_layer(
                model, tokenizer, fact["prompt"], edit_layer, device,
            )

        # 3d. Angular variance diagnostic
        ang_var = compute_angular_variance(per_sample_grads)

        # ---------------------------------------------------------------
        # Phase 4-5: TECS computation
        # ---------------------------------------------------------------
        er = edit_results[i]
        tecs_real = compute_tecs(er.delta_weight, aggregated_grad)

        # Null-A: unrelated edit directions (same gradient, different deltas)
        unrelated_indices = [j for j in range(num_facts) if j != i]
        rng = random.Random(42 + i)
        null_a_indices = rng.sample(unrelated_indices, min(null_a_num, len(unrelated_indices)))
        null_a_deltas = [edit_results[j].delta_weight for j in null_a_indices]
        tecs_null_a = compute_null_a(aggregated_grad, null_a_deltas)

        # Null-B: Prompt-level placebo at the EDIT layer.
        # Tests fact-specificity: unrelated facts' prompts should produce lower
        # TECS at the edit layer. If TECS is knowledge-specific, gradients from
        # prompts about *different* facts should NOT align with this fact's
        # ROME delta.
        # This avoids the cross-layer comparison problem (comparing vectors from
        # different parameter subspaces).
        tecs_placebo = {}
        placebo_rng = random.Random(123 + i)
        placebo_fact_indices = placebo_rng.sample(
            unrelated_indices, min(len(placebo_offsets), len(unrelated_indices))
        )
        for pi, pf_idx in enumerate(placebo_fact_indices):
            pf = facts[pf_idx]
            # Compute gradient of an UNRELATED fact's prompt at the edit layer
            grad_unrelated = compute_gradient_at_layer(
                model, tokenizer, pf["prompt"], edit_layer, device,
            )
            # TECS: alignment of this fact's ROME delta with an unrelated prompt's gradient
            tecs_p = compute_tecs(er.delta_weight, grad_unrelated)
            tecs_placebo[f"unrelated_fact_{pf_idx}"] = tecs_p

        # Null-C: edit-failed facts (computed post-hoc)

        placebo_mean = float(np.mean(list(tecs_placebo.values()))) if tecs_placebo else 0.0
        print(f"    TECS(real): {tecs_real:.6f}")
        print(f"    TECS(null-A mean): {np.mean(tecs_null_a):.6f}")
        print(f"    TECS(placebo mean): {placebo_mean:.6f}")
        print(f"    Angular variance: {ang_var:.6f}")

        tecs_results.append({
            "fact_idx": i,
            "fact_id": fact["case_id"],
            "subject": fact["subject"],
            "tecs_real": tecs_real,
            "tecs_null_a": tecs_null_a,
            "tecs_null_a_mean": float(np.mean(tecs_null_a)),
            "tecs_placebo": tecs_placebo,
            "edit_success": er.edit_success,
            "angular_variance": ang_var,
        })

    # ---------------------------------------------------------------
    # Phase 6: Statistical tests
    # ---------------------------------------------------------------
    print("\n[Phase 6] Statistical analysis...")

    real_vals = [r["tecs_real"] for r in tecs_results]
    null_a_means = [r["tecs_null_a_mean"] for r in tecs_results]

    # Test 1: TECS(real) vs TECS(null-A)
    test_real_null = paired_t_test(
        real_vals, null_a_means,
        test_name="TECS(real) vs TECS(null-A)",
    )

    # Test 2: TECS(edit layer) vs TECS(placebo layers)
    edit_layer_vals = real_vals
    placebo_vals = []
    for r in tecs_results:
        placebo_mean = np.mean(list(r["tecs_placebo"].values())) if r["tecs_placebo"] else 0.0
        placebo_vals.append(placebo_mean)

    test_placebo = paired_t_test(
        edit_layer_vals, placebo_vals,
        test_name="TECS(edit layer) vs TECS(placebo layers)",
    )

    # Angular variance check
    mean_ang_var = np.mean([r["angular_variance"] for r in tecs_results])

    # Pass criteria
    pass_criteria = check_pass_criteria(
        test_real_vs_null=test_real_null,
        test_edit_vs_placebo=test_placebo,
        mean_tecs=np.mean(real_vals),
        angular_variance=mean_ang_var,
    )

    # ---------------------------------------------------------------
    # Phase 7: Report
    # ---------------------------------------------------------------
    report = format_report(tecs_results, test_real_null, test_placebo, pass_criteria)
    print("\n" + report)

    # Additional analysis: edit success vs failure
    success_tecs = [r["tecs_real"] for r in tecs_results if r["edit_success"]]
    failure_tecs = [r["tecs_real"] for r in tecs_results if not r["edit_success"]]
    if success_tecs and failure_tecs:
        print(f"\n  TECS(edit success): mean={np.mean(success_tecs):.6f} (n={len(success_tecs)})")
        print(f"  TECS(edit failure): mean={np.mean(failure_tecs):.6f} (n={len(failure_tecs)})")

    # Save
    elapsed = time.time() - t0
    output = {
        "config": {
            "num_facts": num_facts,
            "top_k_retrieval": top_k_retrieval,
            "top_k_gradient": top_k_gradient,
            "corpus": corpus,
            "edit_layer": edit_layer,
        },
        "per_fact_results": tecs_results,
        "statistics": {
            "test_real_vs_null_a": {
                "mean_real": test_real_null.mean_real,
                "mean_null": test_real_null.mean_null,
                "t_stat": test_real_null.t_statistic,
                "p_value": test_real_null.p_value,
                "cohens_d": test_real_null.cohens_d,
                "ci": [test_real_null.ci_low, test_real_null.ci_high],
            },
            "test_placebo": {
                "mean_edit": test_placebo.mean_real,
                "mean_placebo": test_placebo.mean_null,
                "t_stat": test_placebo.t_statistic,
                "p_value": test_placebo.p_value,
                "cohens_d": test_placebo.cohens_d,
            },
            "mean_angular_variance": float(mean_ang_var),
        },
        "pass_criteria": pass_criteria,
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(output_dir, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved to {out_path}")
    print(f"  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TECA Main Probe Experiment")
    parser.add_argument("--num_facts", type=int, default=50)
    parser.add_argument("--top_k_retrieval", type=int, default=100)
    parser.add_argument("--top_k_gradient", type=int, default=10)
    parser.add_argument("--null_a_num", type=int, default=10)
    parser.add_argument("--corpus", type=str, default="openwebtext")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/probe/main")
    parser.add_argument("--bm25_index", type=str, default=None)
    args = parser.parse_args()

    run_main_probe(
        num_facts=args.num_facts,
        top_k_retrieval=args.top_k_retrieval,
        top_k_gradient=args.top_k_gradient,
        null_a_num=args.null_a_num,
        corpus=args.corpus,
        device=args.device,
        output_dir=args.output_dir,
        bm25_index_path=args.bm25_index,
    )
