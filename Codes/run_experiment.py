#!/usr/bin/env python3
"""Unified TECA experiment runner.

Config-driven entry point for all experiment phases (0-7).
Replaces the scattered probe_*.py scripts with a single interface.

Usage:
    # Run all phases from a config
    python run_experiment.py --config configs/base.yaml

    # Run specific phase(s)
    python run_experiment.py --config configs/base.yaml --phase 0
    python run_experiment.py --config configs/base.yaml --phase 0 1 3

    # Override config values from CLI
    python run_experiment.py --config configs/base.yaml --phase 3 data.num_facts=100

    # Dry run: validate config and pipeline without GPU
    python run_experiment.py --config configs/pilot.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.config import load_config, validate_config, config_summary, dump_config


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------

PHASE_NAMES = {
    0: "Sanity Checks",
    1: "Positive Control Experiments",
    2: "g_M Quality Analysis",
    3: "Full-Scale Core Experiments",
    4: "Ablation Experiments",
    5: "Extended Analyses",
    6: "Cross-Model Validation",
    7: "Visualization & Paper Figures",
}


# ---------------------------------------------------------------------------
# Phase 0: Sanity Checks
# ---------------------------------------------------------------------------

def run_phase_0(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 0: ROME validation, gradient check, TECS pipeline check."""
    results = {"phase": 0, "name": PHASE_NAMES[0], "checks": {}}

    num_facts = min(cfg["data"]["num_facts"], 10)  # sanity uses at most 10
    device = cfg["model"]["device"]
    model_name = cfg["model"]["name"]

    if dry_run:
        results["checks"]["rome_validation"] = {
            "status": "dry_run", "num_facts": num_facts, "model": model_name
        }
        results["checks"]["gradient_check"] = {
            "status": "dry_run", "num_facts": num_facts
        }
        results["checks"]["tecs_pipeline"] = {
            "status": "dry_run", "num_facts": num_facts
        }
        results["status"] = "dry_run_ok"
        return results

    import torch
    from core.model_utils import load_model_and_tokenizer, get_mlp_weight
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_gradient_at_layer
    from core.tecs import compute_tecs
    from core.retrieval import load_counterfact

    print("  Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=cfg["model"]["dtype"])
    facts = load_counterfact(
        path=cfg["data"].get("counterfact_path"),
        num_facts=num_facts,
        seed=cfg["data"]["seed"],
    )

    # 0a. ROME validation
    print("  [0a] ROME validation...")
    edit_successes = 0
    for i, fact in enumerate(facts):
        er = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"], prompt=fact["prompt"],
            target_new=fact["target_new"], target_old=fact["target_old"],
            edit_layer=cfg["rome"].get("edit_layer") or cfg["model"].get("edit_layer"),
            device=device,
            v_lr=cfg["rome"]["v_lr"],
            v_num_grad_steps=cfg["rome"]["v_num_grad_steps"],
            clamp_norm_factor=cfg["rome"]["clamp_norm_factor"],
            kl_factor=cfg["rome"]["kl_factor"],
        )
        if er.edit_success:
            edit_successes += 1
        print(f"    [{i+1}/{num_facts}] {fact['subject']}: {'OK' if er.edit_success else 'FAIL'}")

    efficacy = edit_successes / num_facts
    results["checks"]["rome_validation"] = {
        "num_facts": num_facts,
        "successes": edit_successes,
        "efficacy": efficacy,
        "gate_passed": efficacy >= 0.75,
    }

    # 0b. Gradient check
    print("  [0b] Gradient check...")
    grad_ok = True
    edit_layer = er.edit_layer
    for i, fact in enumerate(facts[:5]):
        grad = compute_gradient_at_layer(model, tokenizer, fact["prompt"], edit_layer, device)
        is_nan = bool(torch.isnan(grad).any())
        is_zero = bool((grad.abs() < 1e-12).all())
        shape_ok = list(grad.shape) == [6400, 1600] if model_name == "gpt2-xl" else True
        if is_nan or is_zero or not shape_ok:
            grad_ok = False
            print(f"    [{i+1}] FAIL: nan={is_nan}, zero={is_zero}, shape={list(grad.shape)}")
        else:
            print(f"    [{i+1}] OK: shape={list(grad.shape)}, norm={grad.norm().item():.4f}")

    results["checks"]["gradient_check"] = {"all_ok": grad_ok}

    # 0c. TECS pipeline check
    print("  [0c] TECS pipeline check...")
    tecs_values = []
    for i, fact in enumerate(facts[:5]):
        er_i = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"], prompt=fact["prompt"],
            target_new=fact["target_new"], target_old=fact["target_old"],
            device=device,
        )
        grad_i = compute_gradient_at_layer(model, tokenizer, fact["prompt"], er_i.edit_layer, device)
        tecs_val = compute_tecs(er_i.delta_weight, grad_i)
        tecs_values.append(tecs_val)
        print(f"    [{i+1}] TECS={tecs_val:.6f}")

    has_nan = any(t != t for t in tecs_values)
    all_zero = all(abs(t) < 1e-12 for t in tecs_values)
    results["checks"]["tecs_pipeline"] = {
        "values": tecs_values,
        "has_nan": has_nan,
        "all_zero": all_zero,
        "pipeline_ok": not has_nan and not all_zero,
    }

    gate_passed = (
        results["checks"]["rome_validation"]["gate_passed"]
        and results["checks"]["gradient_check"]["all_ok"]
        and results["checks"]["tecs_pipeline"]["pipeline_ok"]
    )
    results["status"] = "passed" if gate_passed else "failed"
    return results


# ---------------------------------------------------------------------------
# Phase 1: Positive Control Experiments
# ---------------------------------------------------------------------------

def run_phase_1(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 1: ROME vs self, toy model, related facts."""
    results = {"phase": 1, "name": PHASE_NAMES[1], "experiments": {}}

    pc_cfg = cfg.get("positive_control", {})

    if dry_run:
        results["experiments"]["rome_self"] = {
            "status": "dry_run",
            "sigmas": pc_cfg.get("rome_self_sigmas", [0, 0.01, 0.1, 0.5, 1.0]),
            "num_facts": pc_cfg.get("rome_self_num_facts", 100),
        }
        results["experiments"]["toy_model"] = {
            "status": "dry_run",
            "d": pc_cfg.get("toy_model_d", 64),
            "n_pairs": pc_cfg.get("toy_model_n_pairs", 200),
        }
        results["experiments"]["related_facts"] = {
            "status": "dry_run",
            "num_facts": pc_cfg.get("related_facts_num", 200),
        }
        results["status"] = "dry_run_ok"
        return results

    import torch
    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.tecs import compute_tecs, cosine_similarity_flat
    from core.retrieval import load_counterfact
    from core.statistics import paired_t_test

    device = cfg["model"]["device"]
    model_name = cfg["model"]["name"]
    print("  Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=cfg["model"]["dtype"])

    # 1a. ROME vs Self
    print("  [1a] ROME vs Self positive control...")
    sigmas = pc_cfg.get("rome_self_sigmas", [0.0, 0.01, 0.1, 0.5, 1.0])
    n_self = min(pc_cfg.get("rome_self_num_facts", 100), cfg["data"]["num_facts"])
    facts = load_counterfact(
        path=cfg["data"].get("counterfact_path"), num_facts=n_self, seed=cfg["data"]["seed"],
    )

    # Compute ROME deltas
    deltas = []
    for i, fact in enumerate(facts[:20]):  # use first 20 for speed
        er = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"], prompt=fact["prompt"],
            target_new=fact["target_new"], target_old=fact["target_old"],
            device=device,
        )
        deltas.append(er.delta_weight)

    rome_self_results = {}
    for sigma in sigmas:
        tecs_vals = []
        for delta in deltas:
            if sigma == 0.0:
                noisy = delta.clone()
            else:
                noise = torch.randn_like(delta) * sigma * delta.norm()
                noisy = delta + noise
            t = cosine_similarity_flat(delta, noisy)
            tecs_vals.append(t)
        mean_t = float(np.mean(tecs_vals))
        rome_self_results[str(sigma)] = {"mean_tecs": mean_t, "values": tecs_vals}
        print(f"    sigma={sigma}: mean TECS={mean_t:.6f}")

    results["experiments"]["rome_self"] = {
        "status": "completed",
        "results": rome_self_results,
        "monotonic_decrease": all(
            rome_self_results[str(sigmas[i])]["mean_tecs"]
            >= rome_self_results[str(sigmas[i+1])]["mean_tecs"]
            for i in range(len(sigmas) - 1)
        ),
    }

    # 1b. Toy model
    print("  [1b] Toy model positive control...")
    toy_d = pc_cfg.get("toy_model_d", 64)
    toy_n = pc_cfg.get("toy_model_n_pairs", 200)
    toy_result = _run_toy_model_tecs(toy_d, toy_n)
    results["experiments"]["toy_model"] = toy_result

    # 1c. Related facts
    print("  [1c] Related facts positive control...")
    n_related = min(pc_cfg.get("related_facts_num", 200), cfg["data"]["num_facts"])
    facts_full = load_counterfact(
        path=cfg["data"].get("counterfact_path"), num_facts=n_related, seed=cfg["data"]["seed"],
    )
    # Group by relation_id
    by_relation = {}
    for fact in facts_full:
        rid = fact.get("relation_id", "unknown")
        by_relation.setdefault(rid, []).append(fact)

    results["experiments"]["related_facts"] = {
        "status": "completed",
        "num_relations": len(by_relation),
        "relation_sizes": {k: len(v) for k, v in by_relation.items()},
        "note": "Full TECS computation deferred to Phase 3 (requires gradients)",
    }

    results["status"] = "completed"
    return results


def _run_toy_model_tecs(d: int = 64, n_pairs: int = 200) -> Dict[str, Any]:
    """Toy linear associative memory: train, ROME-edit, compute TECS."""
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Build toy model: 3-layer MLP
    model_toy = nn.Sequential(
        nn.Linear(d, d * 4),
        nn.ReLU(),
        nn.Linear(d * 4, d),  # associative layer (index 2)
    )

    # Synthetic data
    keys = torch.randn(n_pairs, d)
    values = torch.randn(n_pairs, d)

    # Train
    optimizer = torch.optim.Adam(model_toy.parameters(), lr=1e-3)
    for epoch in range(500):
        pred = model_toy(keys)
        loss = nn.functional.mse_loss(pred, values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"      Toy model epoch {epoch+1}: loss={loss.item():.6f}")

    # ROME-style edit on the associative layer (layer index 2)
    assoc_layer = model_toy[2]  # nn.Linear(d*4, d)
    W = assoc_layer.weight.detach().clone()  # [d, d*4]

    # Pick a random fact to edit
    edit_idx = 0
    k_star = model_toy[1](model_toy[0](keys[edit_idx:edit_idx+1])).detach().squeeze()  # [d*4]
    v_star = values[edit_idx]  # [d]
    current_v = W @ k_star  # [d]
    v_target = v_star - current_v  # [d]

    # Rank-1 delta: [d, d*4]
    delta = v_target.unsqueeze(1) @ k_star.unsqueeze(0) / (k_star @ k_star)

    # Compute per-sample gradients at the associative layer
    tecs_real_vals = []
    tecs_null_vals = []
    for i in range(min(n_pairs, 50)):
        # Per-sample gradient
        model_toy.zero_grad()
        assoc_layer.weight.requires_grad_(True)
        pred = model_toy(keys[i:i+1])
        loss = nn.functional.mse_loss(pred, values[i:i+1])
        loss.backward()
        grad_i = assoc_layer.weight.grad.detach().clone()
        assoc_layer.weight.requires_grad_(False)

        from core.tecs import cosine_similarity_flat
        t_real = cosine_similarity_flat(delta, grad_i)
        tecs_real_vals.append(t_real)

        # Null: random direction
        rand_delta = torch.randn_like(delta)
        t_null = cosine_similarity_flat(rand_delta, grad_i)
        tecs_null_vals.append(t_null)

    from core.statistics import paired_t_test
    test = paired_t_test(tecs_real_vals, tecs_null_vals, test_name="Toy TECS(real) vs random")

    result = {
        "status": "completed",
        "cohens_d": test.cohens_d,
        "p_value": test.p_value,
        "mean_tecs_real": test.mean_real,
        "mean_tecs_null": test.mean_null,
        "ci": [test.ci_low, test.ci_high],
        "gate_passed": abs(test.cohens_d) > 0.3,
    }
    print(f"      Toy model: d={test.cohens_d:.4f}, p={test.p_value:.6f}, "
          f"gate={'PASS' if result['gate_passed'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Phase 2: g_M Quality Analysis
# ---------------------------------------------------------------------------

def run_phase_2(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 2: Within/between gradient similarity, PC1 removal, retrieval ablation."""
    results = {"phase": 2, "name": PHASE_NAMES[2], "analyses": {}}

    gm_cfg = cfg.get("gm_quality", {})

    if dry_run:
        results["analyses"]["within_between"] = {
            "status": "dry_run",
            "top_k": gm_cfg.get("within_between_top_k", 20),
        }
        results["analyses"]["pc1_removal"] = {
            "status": "dry_run",
            "enabled": gm_cfg.get("pc1_removal", True),
        }
        results["analyses"]["retrieval_ablation"] = {
            "status": "dry_run",
            "methods": gm_cfg.get("retrieval_methods", ["bm25", "tfidf", "contriever", "uniform"]),
        }
        results["status"] = "dry_run_ok"
        return results

    # Full execution requires GPU + data; placeholder for structure
    results["analyses"]["within_between"] = {"status": "pending_gpu"}
    results["analyses"]["pc1_removal"] = {"status": "pending_gpu"}
    results["analyses"]["retrieval_ablation"] = {"status": "pending_gpu"}
    results["status"] = "pending_gpu"
    return results


# ---------------------------------------------------------------------------
# Phase 3: Full-Scale Core Experiments
# ---------------------------------------------------------------------------

def run_phase_3(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 3: 200 facts ROME + TDA + TECS + subspace geometry."""
    results = {"phase": 3, "name": PHASE_NAMES[3], "experiments": {}}

    num_facts = cfg["data"]["num_facts"]
    device = cfg["model"]["device"]

    if dry_run:
        results["experiments"]["rome_editing"] = {
            "status": "dry_run", "num_facts": num_facts,
        }
        results["experiments"]["tda_gradients"] = {
            "status": "dry_run", "num_facts": num_facts,
            "top_k": cfg["retrieval"]["top_k_gradient"],
        }
        results["experiments"]["tecs_core"] = {
            "status": "dry_run", "num_facts": num_facts,
            "null_baselines": ["null_a", "null_b", "null_c", "null_d", "null_e"],
            "bootstrap_n": cfg["statistics"]["bootstrap_n"],
        }
        results["experiments"]["subspace_geometry"] = {
            "status": "dry_run", "num_facts": num_facts,
            "analyses": ["eff_dim", "principal_angles", "cross_projection"],
        }
        results["status"] = "dry_run_ok"
        return results

    # Full execution: delegate to the existing probe_main logic but config-driven
    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_aggregated_gradient, compute_per_sample_gradients
    from core.tecs import compute_tecs, compute_null_a, compute_angular_variance
    from core.retrieval import load_counterfact, retrieve_training_samples_bm25
    from core.statistics import paired_t_test

    print("  Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"], device=device, dtype=cfg["model"]["dtype"],
    )
    facts = load_counterfact(
        path=cfg["data"].get("counterfact_path"),
        num_facts=num_facts, seed=cfg["data"]["seed"],
    )

    # ROME edits
    print(f"  Running ROME edits on {num_facts} facts...")
    edit_results = {}
    for i, fact in enumerate(facts):
        er = compute_rome_edit(
            model, tokenizer,
            subject=fact["subject"], prompt=fact["prompt"],
            target_new=fact["target_new"], target_old=fact["target_old"],
            device=device,
            v_lr=cfg["rome"]["v_lr"],
            v_num_grad_steps=cfg["rome"]["v_num_grad_steps"],
        )
        edit_results[i] = er
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{num_facts}] edits complete")

    edit_layer = edit_results[0].edit_layer
    n_success = sum(1 for er in edit_results.values() if er.edit_success)
    print(f"  Edit success: {n_success}/{num_facts} at layer {edit_layer}")

    results["experiments"]["rome_editing"] = {
        "status": "completed",
        "num_facts": num_facts,
        "successes": n_success,
        "efficacy": n_success / num_facts,
        "edit_layer": edit_layer,
    }

    # TDA gradients + TECS
    print(f"  Computing TDA gradients and TECS...")
    tecs_data = []
    top_k_grad = cfg["retrieval"]["top_k_gradient"]
    null_a_num = cfg["null_baselines"]["null_a_num"]

    for i, fact in enumerate(facts):
        query = f"{fact['subject']} {fact['target_old']}"
        candidates = retrieve_training_samples_bm25(
            query, top_k=cfg["retrieval"]["top_k_candidates"],
            corpus_name=cfg["retrieval"]["corpus"],
            index_path=cfg["retrieval"].get("index_path"),
        )
        sample_texts = [c["text"][:512] for c in candidates[:top_k_grad]]
        agg_grad = compute_aggregated_gradient(
            model, tokenizer, fact["prompt"], sample_texts, edit_layer, device, top_k=top_k_grad,
        )
        per_grads = compute_per_sample_gradients(
            model, tokenizer, sample_texts, edit_layer, device,
        )
        ang_var = compute_angular_variance(per_grads)

        er = edit_results[i]
        tecs_real = compute_tecs(er.delta_weight, agg_grad)

        # Null-A
        unrelated = [j for j in range(num_facts) if j != i]
        rng = random.Random(42 + i)
        null_indices = rng.sample(unrelated, min(null_a_num, len(unrelated)))
        null_a_vals = compute_null_a(agg_grad, [edit_results[j].delta_weight for j in null_indices])

        tecs_data.append({
            "fact_idx": i,
            "tecs_real": tecs_real,
            "tecs_null_a_mean": float(np.mean(null_a_vals)),
            "angular_variance": ang_var,
            "edit_success": er.edit_success,
        })

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{num_facts}] TECS computed")

    # Statistical test
    real_vals = [d["tecs_real"] for d in tecs_data]
    null_vals = [d["tecs_null_a_mean"] for d in tecs_data]
    test = paired_t_test(real_vals, null_vals, test_name="TECS(real) vs Null-A")

    results["experiments"]["tecs_core"] = {
        "status": "completed",
        "per_fact": tecs_data,
        "cohens_d": test.cohens_d,
        "p_value": test.p_value,
        "ci": [test.ci_low, test.ci_high],
        "mean_tecs_real": test.mean_real,
        "mean_tecs_null": test.mean_null,
    }

    results["status"] = "completed"
    return results


# ---------------------------------------------------------------------------
# Phase 4: Ablation Experiments
# ---------------------------------------------------------------------------

def run_phase_4(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 4: Top-k, weighting, loss function, scope ablations."""
    results = {"phase": 4, "name": PHASE_NAMES[4], "ablations": {}}
    abl_cfg = cfg.get("ablation", {})

    if dry_run:
        results["ablations"]["top_k"] = {
            "status": "dry_run",
            "values": abl_cfg.get("top_k_values", [5, 10, 20, 50]),
        }
        results["ablations"]["weighting"] = {
            "status": "dry_run",
            "methods": abl_cfg.get("weighting_methods", ["bm25", "uniform", "tfidf"]),
        }
        results["ablations"]["loss_function"] = {
            "status": "dry_run",
            "functions": abl_cfg.get("loss_functions", ["object_token_ce", "full_sequence_ce", "margin"]),
        }
        results["ablations"]["scope"] = {
            "status": "dry_run",
            "configs": abl_cfg.get("scope_configs", ["single_layer", "multi_layer"]),
        }
        results["status"] = "dry_run_ok"
        return results

    results["status"] = "pending_gpu"
    return results


# ---------------------------------------------------------------------------
# Phase 5: Extended Analyses
# ---------------------------------------------------------------------------

def run_phase_5(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 5: Whitening decomposition, MEMIT comparison."""
    results = {"phase": 5, "name": PHASE_NAMES[5], "analyses": {}}

    if dry_run:
        results["analyses"]["whitening"] = {
            "status": "dry_run",
            "num_facts": cfg["data"]["num_facts"],
        }
        results["analyses"]["memit"] = {
            "status": "dry_run",
            "num_facts": cfg["data"]["num_facts"],
            "layers": "13-17",
        }
        results["status"] = "dry_run_ok"
        return results

    results["status"] = "pending_gpu"
    return results


# ---------------------------------------------------------------------------
# Phase 6: Cross-Model Validation
# ---------------------------------------------------------------------------

def run_phase_6(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 6: GPT-J-6B replication."""
    results = {"phase": 6, "name": PHASE_NAMES[6]}
    cm_cfg = cfg.get("cross_model", {})

    if not cm_cfg.get("enabled", False):
        results["status"] = "skipped (cross_model.enabled = false)"
        return results

    if dry_run:
        results["model"] = cm_cfg.get("model_name", "EleutherAI/gpt-j-6b")
        results["num_facts"] = cm_cfg.get("num_facts", 100)
        results["dtype"] = cm_cfg.get("dtype", "float16")
        results["status"] = "dry_run_ok"
        return results

    results["status"] = "pending_gpu"
    return results


# ---------------------------------------------------------------------------
# Phase 7: Visualization
# ---------------------------------------------------------------------------

def run_phase_7(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Phase 7: Generate paper figures from saved results."""
    results = {"phase": 7, "name": PHASE_NAMES[7]}

    figures = [
        "tecs_distribution",
        "eigenvalue_spectra",
        "principal_angles",
        "cross_projection",
        "memit_heatmap",
        "positive_control",
        "gm_quality",
    ]

    if dry_run:
        results["figures"] = {fig: "dry_run" for fig in figures}
        results["status"] = "dry_run_ok"
        return results

    results["figures"] = {fig: "pending" for fig in figures}
    results["status"] = "pending"
    return results


# ---------------------------------------------------------------------------
# Phase dispatcher
# ---------------------------------------------------------------------------

PHASE_RUNNERS = {
    0: run_phase_0,
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
    7: run_phase_7,
}


def run_phases(
    cfg: Dict[str, Any],
    phases: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run specified experiment phases."""
    if phases is None:
        phases = cfg.get("phases", list(range(8)))

    all_results = {
        "config_summary": config_summary(cfg),
        "dry_run": dry_run,
        "phases": {},
        "timing": {},
    }

    for phase_id in sorted(phases):
        if phase_id not in PHASE_RUNNERS:
            print(f"  WARNING: Unknown phase {phase_id}, skipping.")
            continue

        phase_name = PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
        print(f"\n{'='*60}")
        print(f"Phase {phase_id}: {phase_name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = PHASE_RUNNERS[phase_id](cfg, dry_run=dry_run)
        except Exception as e:
            result = {"phase": phase_id, "status": f"error: {e}"}
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        all_results["phases"][phase_id] = result
        all_results["timing"][phase_id] = round(elapsed, 2)
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Time: {elapsed:.1f}s")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TECA Unified Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/pilot.yaml --dry-run
  python run_experiment.py --config configs/base.yaml --phase 0
  python run_experiment.py --config configs/base.yaml --phase 0 1 3
  python run_experiment.py --config configs/base.yaml data.num_facts=100
        """,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--phase", type=int, nargs="*", default=None, help="Phase(s) to run")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without GPU")
    parser.add_argument("overrides", nargs="*", help="Config overrides as key=value")

    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for ov in (args.overrides or []):
        if "=" in ov:
            k, v = ov.split("=", 1)
            overrides[k] = v

    # Load config
    cfg = load_config(args.config, overrides if overrides else None)

    # Validate
    issues = validate_config(cfg)
    if issues:
        print("Config validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.dry_run:
            print("Fix config issues before running experiments.")
            sys.exit(1)

    # Determine phases
    phases = args.phase if args.phase is not None else cfg.get("phases", list(range(8)))

    # Print header
    print("=" * 60)
    print("TECA Experiment Runner")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Summary: {config_summary(cfg)}")
    print(f"Phases: {phases}")
    print(f"Dry run: {args.dry_run}")

    # Set seed
    set_global_seed(cfg.get("seed", 42))

    # Run
    t_start = time.time()
    all_results = run_phases(cfg, phases=phases, dry_run=args.dry_run)
    total_time = time.time() - t_start
    all_results["total_time_seconds"] = round(total_time, 2)

    # Save results
    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Save config snapshot
    config_dump_path = os.path.join(results_dir, "config_snapshot.yaml")
    dump_config(cfg, config_dump_path)

    # Save results
    tag = "dry_run" if args.dry_run else "run"
    phase_tag = "_".join(str(p) for p in phases)
    results_path = os.path.join(results_dir, f"experiment_{tag}_phase_{phase_tag}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All phases complete. Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Config snapshot: {config_dump_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
