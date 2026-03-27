#!/usr/bin/env python3
"""Phase 1b: Toy Linear Associative Memory TECS.

Constructs a 3-layer MLP where W*k = v holds by construction,
applies ROME-style rank-one edit, computes exact per-sample gradients,
and measures TECS. This tests whether TECS can detect alignment when
the theoretical conditions (linear associative memory) are satisfied.

Expected: TECS Cohen's d > 0.5, rank-one decomposition correlation rho > 0.7.
Gate: If d < 0.3, TECS metric is flawed -- escalate.

Usage:
    python -m experiments.positive_control.toy_model_tecs --config configs/phase_1_positive_control.yaml
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, cohens_d,
    bootstrap_ci, paired_test, get_results_dir,
)
from core.config import load_config


class LinearAssociativeMemory(nn.Module):
    """3-layer MLP with a linear associative memory layer.

    Architecture:
        Layer 1: Linear(d_k, d_hidden) + ReLU
        Layer 2: Linear(d_hidden, d_v)  -- the "associative" layer (ROME target)
        Layer 3: Linear(d_v, d_k)       -- readout (maps back for next-token prediction style loss)
    """

    def __init__(self, d_k: int = 64, d_v: int = 64, d_hidden: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(d_k, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_v)  # associative layer
        self.layer3 = nn.Linear(d_v, d_k)
        self.d_k = d_k
        self.d_v = d_v
        self.d_hidden = d_hidden

    def forward(self, x):
        h = F.relu(self.layer1(x))
        v = self.layer2(h)
        out = self.layer3(v)
        return out, h, v


def generate_associations(n_pairs: int, d_k: int, d_v: int, seed: int = 42):
    """Generate synthetic (key, value) training pairs."""
    rng = torch.Generator().manual_seed(seed)
    keys = torch.randn(n_pairs, d_k, generator=rng)
    values = torch.randn(n_pairs, d_v, generator=rng)
    return keys, values


def train_model(model, keys, values, n_epochs: int = 200, lr: float = 1e-3):
    """Train the model to memorize key-value associations."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    keys_d = keys.to(device)
    values_d = values.to(device)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        out, h, v_pred = model(keys_d)
        # Loss: predict the value correctly AND readout back to key
        loss_value = F.mse_loss(v_pred, values_d)
        loss_readout = F.mse_loss(out, keys_d)
        loss = loss_value + 0.1 * loss_readout

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={loss.item():.6f} "
                  f"(value={loss_value.item():.6f}, readout={loss_readout.item():.6f})")

    model.eval()
    return model


def rome_style_edit(model, key_idx, keys, values_new, device="cpu"):
    """Apply ROME-style rank-one edit to the associative layer.

    For the associative layer W (layer2), compute:
        delta_W = (v_new - W h*) h*^T / (h*^T h*)
    where h* is the hidden representation of the target key.

    Returns the delta_W as a tensor.
    """
    model.eval()
    k = keys[key_idx:key_idx + 1].to(device)
    v_new = values_new.to(device)

    with torch.no_grad():
        _, h_star, v_old = model(k)
        h_star = h_star.squeeze(0)  # (d_hidden,)
        v_old = v_old.squeeze(0)    # (d_v,)

    W = model.layer2.weight.data  # (d_v, d_hidden)

    # Target: v_new - v_old = delta_v
    v_target = v_new.squeeze(0) - v_old  # (d_v,)

    # Rank-1 update: delta_W = v_target @ h_star^T / (h_star^T @ h_star)
    h_norm_sq = torch.dot(h_star, h_star)
    delta_W = v_target.unsqueeze(1) @ h_star.unsqueeze(0) / h_norm_sq  # (d_v, d_hidden)

    return delta_W.cpu(), h_star.cpu(), v_target.cpu()


def compute_per_sample_gradient(model, keys, values, sample_idx, device="cpu"):
    """Compute gradient of loss for a single training sample w.r.t. the associative layer."""
    model.zero_grad()
    model.layer2.weight.requires_grad_(True)

    k = keys[sample_idx:sample_idx + 1].to(device)
    v_true = values[sample_idx:sample_idx + 1].to(device)

    out, h, v_pred = model(k)
    loss = F.mse_loss(v_pred, v_true)
    loss.backward()

    grad = model.layer2.weight.grad.detach().clone().cpu()
    model.zero_grad()
    model.layer2.weight.requires_grad_(False)

    return grad


def run_toy_model_tecs(cfg: dict) -> dict:
    """Run the toy model positive control experiment."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    pc_cfg = cfg.get("positive_control", {})
    d_k = pc_cfg.get("toy_model_d", 64)
    d_v = d_k  # symmetric
    n_pairs = pc_cfg.get("toy_model_n_pairs", 200)
    n_layers = pc_cfg.get("toy_model_n_layers", 3)
    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    device = "cpu"  # Toy model is small enough for CPU

    print("=" * 60)
    print("Phase 1b: Toy Linear Associative Memory TECS")
    print("=" * 60)
    print(f"  d_k = d_v = {d_k}")
    print(f"  n_pairs = {n_pairs}")
    print(f"  Seed: {seed}")

    start_time = time.time()

    # 1. Generate synthetic data
    print("\n[1/5] Generating synthetic associations...")
    keys, values = generate_associations(n_pairs, d_k, d_v, seed=seed)

    # 2. Train the model
    print("\n[2/5] Training toy model...")
    model = LinearAssociativeMemory(d_k, d_v, d_hidden=128).to(device)
    model = train_model(model, keys, values, n_epochs=300, lr=1e-3)

    # Verify training quality
    with torch.no_grad():
        _, _, v_pred = model(keys)
        train_mse = F.mse_loss(v_pred, values).item()
    print(f"  Final training MSE: {train_mse:.6f}")

    # 3. For each of N_EDIT facts, compute ROME edit + gradients + TECS
    n_edit = min(50, n_pairs)  # Edit 50 facts
    print(f"\n[3/5] Computing ROME edits and per-sample gradients for {n_edit} facts...")

    tecs_real = []
    tecs_null = []
    per_fact_results = []

    rng = np.random.RandomState(seed)

    for edit_idx in range(n_edit):
        # Generate a new target value (different from original)
        v_new = torch.randn(1, d_v)

        # Compute ROME-style edit
        delta_W, h_star, v_target = rome_style_edit(model, edit_idx, keys, v_new, device)

        # Compute aggregated gradient from all training samples
        # (In the toy model, we use all samples -- no retrieval needed)
        all_grads = []
        for s in range(n_pairs):
            g = compute_per_sample_gradient(model, keys, values, s, device)
            all_grads.append(g)

        g_aggregated = torch.stack(all_grads).mean(dim=0)

        # TECS: cosine similarity between delta_W and g_aggregated
        tecs = cosine_similarity_flat(delta_W, g_aggregated)
        tecs_real.append(tecs)

        # Null baseline: random permutation of gradient
        g_flat = g_aggregated.reshape(-1)
        perm = torch.randperm(g_flat.shape[0])
        g_null = g_flat[perm].reshape(g_aggregated.shape)
        tecs_n = cosine_similarity_flat(delta_W, g_null)
        tecs_null.append(tecs_n)

        per_fact_results.append({
            "edit_idx": edit_idx,
            "tecs_real": tecs,
            "tecs_null": tecs_n,
            "delta_norm": float(delta_W.norm()),
            "grad_norm": float(g_aggregated.norm()),
        })

        if (edit_idx + 1) % 10 == 0:
            print(f"  [{edit_idx + 1}/{n_edit}] TECS={tecs:.6f}")

    tecs_real_arr = np.array(tecs_real)
    tecs_null_arr = np.array(tecs_null)

    # 4. Statistical analysis
    print("\n[4/5] Statistical analysis...")
    d = cohens_d(tecs_real_arr, tecs_null_arr)
    test = paired_test(tecs_real_arr, tecs_null_arr, "TECS_real vs TECS_null (toy model)", seed=seed)
    ci_real = bootstrap_ci(tecs_real_arr, seed=seed)

    print(f"  TECS_real: mean={tecs_real_arr.mean():.6f} +/- {tecs_real_arr.std():.6f}")
    print(f"  TECS_null: mean={tecs_null_arr.mean():.6f} +/- {tecs_null_arr.std():.6f}")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  p-value: {test['p_value']:.2e}")

    # 5. Rank-one decomposition correlation (if applicable)
    # Check correlation between full TECS and decomposition prediction
    print("\n[5/5] Rank-one decomposition check...")
    decomp_tecs = []
    full_tecs = []

    for edit_idx in range(min(20, n_edit)):
        v_new = torch.randn(1, d_v)
        delta_W, h_star, v_target = rome_style_edit(model, edit_idx, keys, v_new, device)

        for s_idx in range(min(20, n_pairs)):
            g = compute_per_sample_gradient(model, keys, values, s_idx, device)
            tecs_full = cosine_similarity_flat(delta_W, g)
            full_tecs.append(tecs_full)

            # Decomposition: delta_W = v_target outer h_star / ||h||^2
            # TECS ~ (v_target . g . h_star) / norms  (rank-1 identity)
            g_float = g.float()
            v_t = v_target.float()
            h_s = h_star.float()
            dot = v_t @ g_float @ h_s
            norm_prod = v_t.norm() * h_s.norm() * g_float.norm()
            decomp = (dot / norm_prod).item() if norm_prod > 1e-12 else 0.0
            decomp_tecs.append(decomp)

    if full_tecs and decomp_tecs:
        corr = float(np.corrcoef(full_tecs, decomp_tecs)[0, 1])
        print(f"  Rank-1 decomposition correlation: rho = {corr:.4f}")
    else:
        corr = None

    elapsed = time.time() - start_time

    # Decision
    pass_criteria = d > 0.3
    gate_fail = d < 0.3

    results = {
        "experiment": "pc_toy_model",
        "phase": "1b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "d_k": d_k,
            "d_v": d_v,
            "n_pairs": n_pairs,
            "n_edits": n_edit,
            "seed": seed,
            "train_mse": train_mse,
        },
        "tecs_distribution": {
            "real_mean": float(tecs_real_arr.mean()),
            "real_std": float(tecs_real_arr.std()),
            "null_mean": float(tecs_null_arr.mean()),
            "null_std": float(tecs_null_arr.std()),
            "ci_95_real": list(ci_real),
        },
        "statistical_test": test,
        "rank_one_decomposition": {
            "correlation": corr,
            "pass_rho_07": corr is not None and corr > 0.7,
        },
        "decision": {
            "cohens_d": d,
            "pass_d_03": pass_criteria,
            "gate_fail": gate_fail,
            "interpretation": (
                f"PASS: Toy model TECS d={d:.4f} > 0.3 — metric detects alignment "
                f"when theoretical conditions hold"
                if pass_criteria else
                f"FAIL: Toy model TECS d={d:.4f} < 0.3 — ESCALATE: metric may be flawed"
            ),
        },
        "per_fact_results": per_fact_results,
    }

    print(f"\n{'=' * 60}")
    print(f"  Toy Model TECS Result:")
    print(f"  Cohen's d: {d:.4f} (threshold: 0.3)")
    print(f"  Decomposition rho: {corr}")
    print(f"  Decision: {'PASS' if pass_criteria else '*** GATE FAIL ***'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    output_path = os.path.join(results_dir, "pc_toy_model.json")
    save_results(results, output_path)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1b: Toy Model TECS")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_toy_model_tecs(cfg)


if __name__ == "__main__":
    main()
