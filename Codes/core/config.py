# Component: Config Loading
# Source: Codes/CLAUDE.md (Config Structure)
# Purpose: Hierarchical YAML config loading with inheritance and CLI override

"""Config loading utilities: base config inheritance, per-phase override, CLI merge."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Default config (fallback if no YAML provided)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "model": {
        "name": "gpt2-xl",
        "device": "cuda",
        "dtype": "float32",
        "edit_layer": None,  # auto-detect
    },
    "rome": {
        "v_lr": 0.5,
        "v_num_grad_steps": 20,
        "clamp_norm_factor": 4.0,
        "kl_factor": 0.0625,
    },
    "data": {
        "dataset": "counterfact",
        "counterfact_path": "data/counterfact.json",
        "num_facts": 200,
        "seed": 42,
    },
    "retrieval": {
        "method": "bm25",
        "corpus": "openwebtext",
        "max_docs": 500_000,
        "top_k_candidates": 100,
        "top_k_gradient": 10,
        "index_path": None,
    },
    "null_baselines": {
        "null_a_num": 10,
        "placebo_offsets": [-5, 5],
    },
    "statistics": {
        "alpha": 0.05,
        "min_cohens_d": 0.5,
        "min_tecs_mean": 0.05,
        "angular_kill_threshold": 0.001,
        "bootstrap_n": 10_000,
    },
    "output": {
        "results_dir": "_Results",
        "save_tensors": False,
        "tensor_dir": "_Data",
    },
    "phases": [0, 1, 2, 3, 4, 5, 6, 7],
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a YAML config file with optional `_base_` inheritance and overrides.

    Inheritance:
        If the YAML file contains a `_base_` key, the referenced config is loaded
        first, then the current file's values are merged on top (deep merge).
        Relative paths are resolved from the current config file's directory.

    Args:
        config_path: Path to a YAML config file. If None, returns defaults.
        overrides: Dict of dot-separated key overrides, e.g. {"data.num_facts": 50}.

    Returns:
        Merged config dict.
    """
    cfg = copy.deepcopy(_DEFAULTS)

    if config_path is not None:
        config_path = str(Path(config_path).resolve())
        file_cfg = _load_yaml_with_inheritance(config_path)
        cfg = _deep_merge(cfg, file_cfg)

    if overrides:
        for key, value in overrides.items():
            _set_nested(cfg, key, value)

    return cfg


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Build config from argparse namespace (CLI integration).

    Extracts --config and all --override.key=value pairs.
    """
    if args is None:
        return load_config()

    config_path = getattr(args, "config", None)
    overrides = {}

    # Collect any key=value overrides from remaining args
    for key in vars(args):
        if key in ("config", "dry_run", "phase", "phases"):
            continue
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val

    return load_config(config_path, overrides if overrides else None)


def validate_config(cfg: Dict[str, Any]) -> list[str]:
    """Validate a config dict. Returns list of warning/error strings (empty = OK)."""
    issues = []

    # Required sections
    for section in ("model", "data", "retrieval", "statistics", "output"):
        if section not in cfg:
            issues.append(f"Missing required section: {section}")

    # Model checks
    model = cfg.get("model", {})
    if model.get("name") not in ("gpt2-xl", "gpt-j-6b", "EleutherAI/gpt-j-6b"):
        issues.append(f"Unrecognized model: {model.get('name')}. Expected gpt2-xl or gpt-j-6b.")

    if model.get("dtype") not in ("float32", "float16", "bfloat16"):
        issues.append(f"Invalid dtype: {model.get('dtype')}")

    # Data checks
    data = cfg.get("data", {})
    if data.get("num_facts", 0) < 1:
        issues.append("data.num_facts must be >= 1")

    # Retrieval checks
    retrieval = cfg.get("retrieval", {})
    if retrieval.get("method") not in ("bm25", "tfidf", "contriever", "uniform"):
        issues.append(f"Unknown retrieval method: {retrieval.get('method')}")

    return issues


def config_summary(cfg: Dict[str, Any]) -> str:
    """Return a human-readable one-line summary of the config."""
    model = cfg.get("model", {}).get("name", "?")
    n_facts = cfg.get("data", {}).get("num_facts", "?")
    phases = cfg.get("phases", [])
    retrieval = cfg.get("retrieval", {}).get("method", "?")
    return (
        f"model={model}, n_facts={n_facts}, phases={phases}, "
        f"retrieval={retrieval}, seed={cfg.get('seed', '?')}"
    )


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    """Write config dict to a YAML file for reproducibility."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml_with_inheritance(path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a YAML file, resolving `_base_` inheritance recursively."""
    if _seen is None:
        _seen = set()

    path = str(Path(path).resolve())
    if path in _seen:
        raise ValueError(f"Circular config inheritance detected: {path}")
    _seen.add(path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is not None:
        # Resolve relative to current config file's directory
        base_path = str((Path(path).parent / base_ref).resolve())
        base_cfg = _load_yaml_with_inheritance(base_path, _seen)
        return _deep_merge(base_cfg, raw)

    return raw


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep-merge *override* into *base*. Lists and scalars are replaced, dicts are merged."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(d: Dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key path.

    Example: _set_nested(d, "data.num_facts", 50) sets d["data"]["num_facts"] = 50.
    """
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to coerce value to int/float/bool if it looks like one
    d[keys[-1]] = _coerce_value(value)


def _coerce_value(v: Any) -> Any:
    """Try to coerce string values to native Python types."""
    if not isinstance(v, str):
        return v
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v
