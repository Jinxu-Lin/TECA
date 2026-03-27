"""Shared utilities for TECA experiment scripts.

Provides common patterns: seed management, result saving, progress reporting,
data loading, and TECS computation helpers.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def set_seed(seed: int = 42) -> None:
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


def _get_git_commit_hash() -> str:
    """Get the short git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).strip().decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results dict to JSON with proper formatting.

    Automatically injects git_commit hash for reproducibility tracking.
    """
    results.setdefault("git_commit", _get_git_commit_hash())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Results saved to {output_path}")


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_counterfact_facts(
    counterfact_path: str,
    num_facts: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """Load and sample CounterFact facts with standardized keys.

    Returns list of dicts with keys:
        case_id, subject, prompt, target_old, target_new, relation_id, paraphrases
    """
    with open(counterfact_path) as f:
        raw = json.load(f)

    rng = random.Random(seed)
    sampled = rng.sample(raw, min(num_facts, len(raw)))

    facts = []
    for entry in sampled:
        rw = entry.get("requested_rewrite", entry)
        prompt_template = rw["prompt"]
        subject = rw["subject"]
        prompt = prompt_template.format(subject) if "{}" in prompt_template else prompt_template

        facts.append({
            "case_id": entry.get("case_id", entry.get("id", 0)),
            "subject": subject,
            "prompt": prompt,
            "prompt_template": prompt_template,
            "target_old": rw.get("target_true", {}).get("str", ""),
            "target_new": rw.get("target_new", {}).get("str", ""),
            "relation_id": rw.get("relation_id", ""),
            "paraphrases": entry.get("paraphrase_prompts", []),
        })
    return facts


def tecs_rank1(u, v, G, needs_t=False):
    """Efficient TECS via rank-1 identity: cos(vec(u v^T), vec(G)) = (u^T G v) / (||u||*||v||*||G||).

    This avoids materializing the full outer product.
    """
    import torch
    u_f, v_f, G_f = u.float(), v.float(), G.float()
    if needs_t:
        G_f = G_f.T
    dot = u_f @ G_f @ v_f
    norm_uv = u_f.norm() * v_f.norm()
    norm_G = G_f.norm()
    if norm_uv < 1e-12 or norm_G < 1e-12:
        return 0.0
    return (dot / (norm_uv * norm_G)).item()


def cosine_similarity_flat(a, b) -> float:
    """Compute cosine similarity between two tensors after flattening."""
    import torch
    import torch.nn.functional as F
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.asarray(x) - np.asarray(y)
    s = diff.std(ddof=1)
    if s > 1e-12:
        return float(diff.mean() / s)
    # std ~ 0: if mean is also ~ 0, no effect; if mean is non-zero, effect is infinite
    m = diff.mean()
    if abs(m) < 1e-12:
        return 0.0
    return float(np.sign(m) * 1e6)  # large sentinel for perfect separation


def bootstrap_ci(data, n_boot: int = 10000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for the mean."""
    data = np.asarray(data)
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(data, len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def paired_test(real, null, name: str, bootstrap_n: int = 10000, seed: int = 42):
    """Run paired t-test with Cohen's d and bootstrap CI."""
    from scipy import stats as scipy_stats
    real = np.asarray(real)
    null = np.asarray(null)
    n = min(len(real), len(null))
    r, nm = real[:n], null[:n]
    d = cohens_d(r, nm)
    t, p = scipy_stats.ttest_rel(r, nm)
    diff = r - nm
    ci = bootstrap_ci(diff, n_boot=bootstrap_n, seed=seed)
    return {
        "name": name,
        "mean_real": float(r.mean()),
        "mean_null": float(nm.mean()),
        "mean_diff": float(diff.mean()),
        "cohens_d": float(d),
        "t_stat": float(t),
        "p_value": float(p),
        "ci_95_low": ci[0],
        "ci_95_high": ci[1],
        "n": int(n),
    }


def get_results_dir(cfg: Dict) -> str:
    """Get the results directory from config."""
    return cfg.get("output", {}).get("results_dir", "_Results")


def get_data_dir(cfg: Dict) -> str:
    """Get the tensor data directory from config."""
    return cfg.get("output", {}).get("tensor_dir", "_Data")


def parse_args(description: str = "TECA Experiment"):
    """Standard argument parser for experiment scripts."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    return parser.parse_args()


def load_experiment_config(args=None):
    """Load config from args or default."""
    from core.config import load_config
    if args is None:
        args = parse_args()
    cfg = load_config(args.config)
    return cfg, getattr(args, "dry_run", False)
