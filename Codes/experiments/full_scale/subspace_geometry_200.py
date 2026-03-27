#!/usr/bin/env python3
"""Phase 3: Full-Scale Subspace Geometry Analysis (200 facts).

SVD of stacked editing directions and attribution gradients.
Metrics: effective dimensionality, principal angles, cross-projection,
1000 random subspace trials for null distribution.

Depends: full_rome_200, full_tda_200

Usage:
    python -m experiments.full_scale.subspace_geometry_200 --config configs/phase_3_full_scale.yaml
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
    set_seed, save_results, get_results_dir, get_data_dir,
)
from core.config import load_config


def compute_effective_dimensionality(S):
    s2 = S ** 2
    p = s2 / s2.sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def compute_principal_angles(V1, V2):
    M = V1.T @ V2
    _, sigmas, _ = torch.linalg.svd(M)
    sigmas = torch.clamp(sigmas, 0.0, 1.0)
    return torch.acos(sigmas)


def compute_grassmann_distance(angles):
    return float(torch.sqrt(torch.sum(angles ** 2)))


def run_subspace_geometry_200(cfg: dict) -> dict:
    """Run subspace geometry analysis on 200 facts."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    data_dir = get_data_dir(cfg)
    rome_dir = os.path.join(data_dir, "rome_deltas_200")
    grad_dir = os.path.join(data_dir, "tda_gradients_200")
    os.makedirs(results_dir, exist_ok=True)

    K_VALUES = [10, 20, 50]
    N_RANDOM_TRIALS = 1000

    print("=" * 60)
    print("Phase 3: Full-Scale Subspace Geometry (200 Facts)")
    print("=" * 60)

    start_time = time.time()

    # Find common case IDs
    rome_files = {f.replace("delta_", "").replace(".pt", "")
                  for f in os.listdir(rome_dir) if f.startswith("delta_") and f.endswith(".pt")}
    grad_files = {f.replace("g_M_", "").replace(".pt", "")
                  for f in os.listdir(grad_dir) if f.startswith("g_M_") and f.endswith(".pt")}
    common_ids = sorted(rome_files & grad_files, key=lambda x: int(x))
    n_facts = len(common_ids)
    print(f"  Common case IDs: {n_facts}")

    # Load and stack
    print("\nLoading tensors...")
    D_rows = []
    G_rows = []

    for i, cid in enumerate(common_ids):
        d = torch.load(os.path.join(rome_dir, f"delta_{cid}.pt"), map_location="cpu", weights_only=False)
        D_rows.append(d["delta_weight"].float().flatten())
        G_rows.append(
            torch.load(os.path.join(grad_dir, f"g_M_{cid}.pt"), map_location="cpu", weights_only=False)
            .float().flatten()
        )

    D = torch.stack(D_rows, dim=0)  # (n, dim)
    G = torch.stack(G_rows, dim=0)
    dim = D.shape[1]
    del D_rows, G_rows
    gc.collect()

    print(f"  D shape: {D.shape}, G shape: {G.shape}")

    # Project to joint subspace
    print("\nProjecting to joint subspace...")
    DG = torch.cat([D, G], dim=0)
    mean_vec = DG.mean(dim=0, keepdim=True)
    DG_centered = DG - mean_vec

    max_rank = min(n_facts * 2, 200)
    U_joint, S_joint, V_joint = torch.svd_lowrank(DG_centered, q=max_rank, niter=5)

    threshold = S_joint[0] * 1e-10
    effective_rank = int((S_joint > threshold).sum())
    V_basis = V_joint[:, :effective_rank]

    D_proj = (D - mean_vec) @ V_basis
    G_proj = (G - mean_vec) @ V_basis
    del D, G, DG, DG_centered
    gc.collect()

    print(f"  Joint subspace rank: {effective_rank}")

    # SVD of D_proj and G_proj
    max_k = max(K_VALUES)
    U_D, S_D, V_D_t = torch.linalg.svd(D_proj, full_matrices=False)
    U_G, S_G, V_G_t = torch.linalg.svd(G_proj, full_matrices=False)
    V_D_full = V_D_t.T[:, :max_k]
    V_G_full = V_G_t.T[:, :max_k]

    eff_dim_D = compute_effective_dimensionality(S_D[:max_k])
    eff_dim_G = compute_effective_dimensionality(S_G[:max_k])

    print(f"  D eff-dim: {eff_dim_D:.1f}")
    print(f"  G eff-dim: {eff_dim_G:.1f}")

    # Principal angles for each k
    print("\nComputing principal angles...")
    pa_results = {}
    for k in K_VALUES:
        V_D_k = V_D_full[:, :k]
        V_G_k = V_G_full[:, :k]
        angles = compute_principal_angles(V_D_k, V_G_k)
        angles_deg = torch.rad2deg(angles).tolist()
        grassmann = compute_grassmann_distance(angles)

        pa_results[f"k{k}"] = {
            "k": k,
            "angles_degrees": angles_deg,
            "min_angle": min(angles_deg),
            "max_angle": max(angles_deg),
            "mean_angle": float(np.mean(angles_deg)),
            "grassmann_distance": grassmann,
        }
        print(f"  k={k}: min={min(angles_deg):.1f}, mean={np.mean(angles_deg):.1f}")

    # Null distribution
    print(f"\nNull distribution ({N_RANDOM_TRIALS} trials)...")
    rng = np.random.RandomState(seed)
    null_results = {}

    for k in K_VALUES:
        V_D_k = V_D_full[:, :k]
        random_min_angles = []
        for _ in range(N_RANDOM_TRIALS):
            R = rng.standard_normal((effective_rank, k)).astype(np.float32)
            Q, _ = np.linalg.qr(R)
            V_rand = torch.from_numpy(Q[:, :k])
            angles = compute_principal_angles(V_D_k, V_rand)
            random_min_angles.append(torch.rad2deg(angles).min().item())

        real_min = pa_results[f"k{k}"]["min_angle"]
        p_value = sum(1 for a in random_min_angles if a <= real_min) / N_RANDOM_TRIALS

        null_results[f"k{k}"] = {
            "k": k,
            "null_mean_min": float(np.mean(random_min_angles)),
            "null_std_min": float(np.std(random_min_angles)),
            "p_value_min": p_value,
            "structured": p_value < 0.01,
        }
        print(f"  k={k}: real_min={real_min:.1f}, null_mean={np.mean(random_min_angles):.1f}, p={p_value:.4f}")

    # Cross-projection
    print("\nCross-projection...")
    cross_proj = {}
    for k in K_VALUES:
        V_D_k = V_D_full[:, :k]
        V_G_k = V_G_full[:, :k]

        var_G_in_D = float((G_proj @ V_D_k).pow(2).sum() / G_proj.pow(2).sum())
        var_D_in_G = float((D_proj @ V_G_k).pow(2).sum() / D_proj.pow(2).sum())

        cross_proj[f"k{k}"] = {
            "G_in_D": var_G_in_D,
            "D_in_G": var_D_in_G,
        }
        print(f"  k={k}: G-in-D={var_G_in_D:.4f}, D-in-G={var_D_in_G:.4f}")

    elapsed = time.time() - start_time

    results = {
        "experiment": "full_subspace_200",
        "phase": "3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "n_facts": n_facts,
            "k_values": K_VALUES,
            "n_random_trials": N_RANDOM_TRIALS,
            "reduced_dim": effective_rank,
            "seed": seed,
        },
        "subspace_properties": {
            "D_eff_dim": eff_dim_D,
            "G_eff_dim": eff_dim_G,
            "D_top5_sv": S_D[:5].tolist(),
            "G_top5_sv": S_G[:5].tolist(),
        },
        "principal_angles": pa_results,
        "null_distribution": null_results,
        "cross_projection": cross_proj,
    }

    print(f"\n{'=' * 60}")
    print(f"  Subspace Geometry 200 Result:")
    print(f"  D eff-dim: {eff_dim_D:.1f}, G eff-dim: {eff_dim_G:.1f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, "full_subspace_200.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: Subspace Geometry 200")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_subspace_geometry_200(cfg)


if __name__ == "__main__":
    main()
