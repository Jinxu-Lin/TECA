#!/usr/bin/env python3
"""Phase 1c: Semantically Related Facts TECS.

Compute TECS between fact i's ROME edit direction and fact j's attribution
gradient for same-relation vs cross-relation pairs. Tests partial alignment
without requiring exact correspondence.

Expected: Same-relation TECS slightly > cross-relation TECS.
Null result is acceptable -- any signal is informative.

Usage:
    python -m experiments.positive_control.related_facts_tecs --config configs/phase_1_positive_control.yaml
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, tecs_rank1, cohens_d,
    bootstrap_ci, paired_test, load_counterfact_facts, get_results_dir,
)
from core.config import load_config


def run_related_facts_tecs(cfg: dict) -> dict:
    """Run the related facts positive control experiment."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    pc_cfg = cfg.get("positive_control", {})
    num_facts = pc_cfg.get("related_facts_num", 200)
    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    print("=" * 60)
    print("Phase 1c: Semantically Related Facts TECS")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Num facts: {num_facts}")
    print(f"  Seed: {seed}")

    start_time = time.time()

    # Load facts with relation_id grouping
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    # Group by relation
    relation_groups = defaultdict(list)
    for fact in facts:
        rid = fact.get("relation_id", "unknown")
        if rid:
            relation_groups[rid].append(fact)

    # Filter to relations with at least 2 facts
    valid_relations = {k: v for k, v in relation_groups.items() if len(v) >= 2}
    print(f"  Relations with >= 2 facts: {len(valid_relations)}")
    for rid, group in sorted(valid_relations.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {rid}: {len(group)} facts")

    if len(valid_relations) < 2:
        print("  WARNING: Not enough relations for cross-relation comparison.")
        print("  Need at least 2 relations with >= 2 facts each.")
        results = {
            "experiment": "pc_related_facts",
            "phase": "1c",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": "insufficient_relations",
            "n_valid_relations": len(valid_relations),
        }
        save_results(results, os.path.join(results_dir, "pc_related_facts.json"))
        return results

    # Load model and compute ROME + TDA for all facts
    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    retrieval_cfg = cfg.get("retrieval", {})
    top_k_candidates = retrieval_cfg.get("top_k_candidates", 100)
    top_k_gradient = retrieval_cfg.get("top_k_gradient", 10)
    index_path = retrieval_cfg.get("index_path")

    # Compute deltas and gradients for all facts
    fact_data = {}  # case_id -> {delta_weight, gradient}

    print(f"\nComputing ROME deltas and TDA gradients...")
    for i, fact in enumerate(facts):
        cid = fact["case_id"]
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

            # BM25 retrieval + gradient
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

            fact_data[cid] = {
                "delta_weight": edit_result.delta_weight.cpu(),
                "gradient": g_M.cpu(),
                "relation_id": fact.get("relation_id", "unknown"),
            }

        except Exception as e:
            print(f"  [WARN] Fact {cid}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}] computed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  {len(fact_data)} facts with both delta and gradient")

    # Compute cross-fact TECS: same-relation vs cross-relation
    print("\nComputing within-relation and cross-relation TECS...")

    rng = np.random.RandomState(seed)
    within_tecs = []
    cross_tecs = []

    # Get valid case_ids grouped by relation
    relation_cids = defaultdict(list)
    for cid, data in fact_data.items():
        relation_cids[data["relation_id"]].append(cid)

    # Filter to relations with >= 2 valid facts
    valid_rel_cids = {k: v for k, v in relation_cids.items() if len(v) >= 2}
    all_cids = list(fact_data.keys())

    for rid, cids in valid_rel_cids.items():
        # Within-relation pairs
        for i in range(len(cids)):
            for j in range(len(cids)):
                if i == j:
                    continue
                cid_i, cid_j = cids[i], cids[j]
                delta = fact_data[cid_i]["delta_weight"]
                grad = fact_data[cid_j]["gradient"]
                tecs = cosine_similarity_flat(delta, grad)
                within_tecs.append(tecs)

        # Cross-relation pairs (sample same number)
        n_within = len(cids) * (len(cids) - 1)
        other_cids = [c for c in all_cids if c not in cids]
        for _ in range(min(n_within, len(other_cids))):
            cid_i = rng.choice(cids)
            cid_j = rng.choice(other_cids)
            delta = fact_data[cid_i]["delta_weight"]
            grad = fact_data[cid_j]["gradient"]
            tecs = cosine_similarity_flat(delta, grad)
            cross_tecs.append(tecs)

    from experiments.common import cosine_similarity_flat

    within_arr = np.array(within_tecs)
    cross_arr = np.array(cross_tecs)

    print(f"  Within-relation pairs: {len(within_tecs)}")
    print(f"  Cross-relation pairs: {len(cross_tecs)}")
    print(f"  Within mean: {within_arr.mean():.6f} +/- {within_arr.std():.6f}")
    print(f"  Cross mean: {cross_arr.mean():.6f} +/- {cross_arr.std():.6f}")

    # Statistical test (two-sample t-test since pairs are different)
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(within_tecs, cross_tecs)
    d_val = (within_arr.mean() - cross_arr.mean()) / np.sqrt(
        (within_arr.std(ddof=1) ** 2 + cross_arr.std(ddof=1) ** 2) / 2
    ) if min(within_arr.std(), cross_arr.std()) > 1e-12 else 0.0

    elapsed = time.time() - start_time

    results = {
        "experiment": "pc_related_facts",
        "phase": "1c",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts_total": len(facts),
            "num_facts_valid": len(fact_data),
            "num_valid_relations": len(valid_rel_cids),
            "seed": seed,
        },
        "within_relation": {
            "n_pairs": len(within_tecs),
            "mean": float(within_arr.mean()),
            "std": float(within_arr.std()),
            "median": float(np.median(within_arr)),
        },
        "cross_relation": {
            "n_pairs": len(cross_tecs),
            "mean": float(cross_arr.mean()),
            "std": float(cross_arr.std()),
            "median": float(np.median(cross_arr)),
        },
        "statistical_test": {
            "test": "independent_t_test",
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(d_val),
        },
        "decision": {
            "within_gt_cross": float(within_arr.mean()) > float(cross_arr.mean()),
            "significant": p_value < 0.05,
            "interpretation": (
                f"Within-relation TECS ({within_arr.mean():.6f}) "
                f"{'>' if within_arr.mean() > cross_arr.mean() else '<='} "
                f"cross-relation TECS ({cross_arr.mean():.6f}), "
                f"d={d_val:.4f}, p={p_value:.4f}"
            ),
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  Related Facts Result:")
    print(f"  Within-relation: {within_arr.mean():.6f}")
    print(f"  Cross-relation: {cross_arr.mean():.6f}")
    print(f"  Cohen's d: {d_val:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    output_path = os.path.join(results_dir, "pc_related_facts.json")
    save_results(results, output_path)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1c: Related Facts TECS")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_related_facts_tecs(cfg)


if __name__ == "__main__":
    main()
