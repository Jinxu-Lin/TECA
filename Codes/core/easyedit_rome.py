# Component: EasyEdit ROME Backend
# Provides a clean wrapper around EasyEdit's ROME implementation,
# bypassing the problematic top-level __init__.py imports.

"""EasyEdit ROME wrapper for proper ROME editing with C^{-1} covariance."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Bootstrap EasyEdit's ROME modules without triggering __init__.py chain
# ---------------------------------------------------------------------------

_EASYEDIT_ROOT = str(Path(__file__).resolve().parent.parent.parent / "EasyEdit")


def _bootstrap_easyedit():
    """Register EasyEdit modules selectively to avoid dependency hell."""
    if "easyeditor.models.rome.rome_main" in sys.modules:
        return  # Already bootstrapped

    if _EASYEDIT_ROOT not in sys.path:
        sys.path.insert(0, _EASYEDIT_ROOT)

    # Create stub packages to prevent __init__.py from running
    for pkg in [
        "easyeditor",
        "easyeditor.models",
        "easyeditor.models.rome",
        "easyeditor.util",
    ]:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [os.path.join(_EASYEDIT_ROOT, pkg.replace(".", "/"))]
            mod.__package__ = pkg
            sys.modules[pkg] = mod

    import importlib.util

    def _load(mod_name: str, file_path: str):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    base = _EASYEDIT_ROOT + "/easyeditor"

    # util modules
    _load("easyeditor.util.nethook", f"{base}/util/nethook.py")
    _load("easyeditor.util.globals", f"{base}/util/globals.py")
    _load("easyeditor.util.logit_lens", f"{base}/util/logit_lens.py")
    _load("easyeditor.util.generate", f"{base}/util/generate.py")
    _load("easyeditor.util.hparams", f"{base}/util/hparams.py")
    _load("easyeditor.util.runningstats", f"{base}/util/runningstats.py")

    # ROME modules
    _load("easyeditor.models.rome.rome_hparams", f"{base}/models/rome/rome_hparams.py")
    _load("easyeditor.models.rome.repr_tools", f"{base}/models/rome/repr_tools.py")
    _load("easyeditor.models.rome.tok_dataset", f"{base}/models/rome/tok_dataset.py")
    _load("easyeditor.models.rome.layer_stats", f"{base}/models/rome/layer_stats.py")
    _load("easyeditor.models.rome.compute_u", f"{base}/models/rome/compute_u.py")
    _load("easyeditor.models.rome.compute_v", f"{base}/models/rome/compute_v.py")
    _load("easyeditor.models.rome.rome_main", f"{base}/models/rome/rome_main.py")


def _get_rome_modules():
    """Return (execute_rome, ROMEHyperParams, nethook, upd_matrix_match_shape)."""
    _bootstrap_easyedit()
    from easyeditor.models.rome.rome_main import execute_rome, upd_matrix_match_shape
    from easyeditor.models.rome.rome_hparams import ROMEHyperParams
    from easyeditor.util import nethook
    return execute_rome, ROMEHyperParams, nethook, upd_matrix_match_shape


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class EasyEditRomeResult:
    """Container for EasyEdit ROME edit outcome."""
    subject: str
    target_old: str
    target_new: str
    edit_layer: int
    delta_weight: torch.Tensor  # full rank-1 delta, same shape as W
    delta_u: torch.Tensor  # left vector of rank-1 decomposition
    delta_v: torch.Tensor  # right vector of rank-1 decomposition
    edit_success: bool
    pre_prob: float
    post_prob: float


def build_hparams(
    model_name: str = "gpt2-xl",
    edit_layer: int = 17,
    stats_dir: str = "data/stats",
    device: int = 0,
) -> object:
    """Build ROMEHyperParams from the EasyEdit default YAML + overrides."""
    _, ROMEHyperParams, _, _ = _get_rome_modules()

    yaml_path = os.path.join(_EASYEDIT_ROOT, "hparams", "ROME", "gpt2-xl.yaml")
    hparams = ROMEHyperParams.from_hparams(yaml_path)
    hparams.model_name = model_name
    hparams.stats_dir = stats_dir
    hparams.layers = [edit_layer]
    hparams.device = device
    os.makedirs(stats_dir, exist_ok=True)
    return hparams


def compute_rome_edit_easyedit(
    model,
    tokenizer,
    subject: str,
    prompt: str,
    target_new: str,
    target_old: str = "",
    edit_layer: int = 17,
    hparams=None,
    device: str = "cuda",
    stats_dir: str = "data/stats",
) -> EasyEditRomeResult:
    """Run ROME edit using EasyEdit backend and return the rank-1 delta.

    The model weights are restored after measurement — this function is
    non-destructive.
    """
    execute_rome, ROMEHyperParams, nethook, upd_matrix_match_shape = _get_rome_modules()

    if hparams is None:
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        hparams = build_hparams(
            model_name=model.config._name_or_path,
            edit_layer=edit_layer,
            stats_dir=stats_dir,
            device=device_idx,
        )

    weight_name = f"transformer.h.{edit_layer}.mlp.c_proj.weight"
    orig_weight = nethook.get_parameter(model, weight_name).detach().clone()

    # Pre-edit probability
    pre_prob = _target_probability(model, tokenizer, prompt, target_new, device)

    # Build request dict
    request = {
        "prompt": prompt if "{}" not in prompt else prompt,
        "subject": subject,
        "target_new": target_new,
        "target_true": target_old,
    }

    # Execute ROME (returns deltas, restores original weights internally)
    deltas = execute_rome(model, tokenizer, request, hparams)

    # Extract rank-1 components
    delta_u, delta_v = deltas[weight_name]
    delta_W = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
    delta_W_matched = upd_matrix_match_shape(delta_W, orig_weight.shape)

    # Apply delta to measure post-edit success
    with torch.no_grad():
        w = nethook.get_parameter(model, weight_name)
        w[...] += delta_W_matched

    post_prob = _target_probability(model, tokenizer, prompt, target_new, device)
    edit_success = post_prob > pre_prob and post_prob > 0.1

    # Restore original weights
    with torch.no_grad():
        w = nethook.get_parameter(model, weight_name)
        w[...] = orig_weight

    return EasyEditRomeResult(
        subject=subject,
        target_old=target_old,
        target_new=target_new,
        edit_layer=edit_layer,
        delta_weight=delta_W_matched.cpu(),
        delta_u=delta_u.cpu(),
        delta_v=delta_v.cpu(),
        edit_success=edit_success,
        pre_prob=pre_prob,
        post_prob=post_prob,
    )


def _target_probability(model, tokenizer, prompt, target, device):
    """Compute P(target | prompt) — first token probability."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_with_space = " " + target.lstrip()
    target_ids_space = tokenizer.encode(target_with_space, add_special_tokens=False)
    target_ids_nospace = tokenizer.encode(target, add_special_tokens=False)

    with torch.no_grad():
        logits = model(**inputs).logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    prob_space = probs[target_ids_space[0]].item() if target_ids_space else 0.0
    prob_nospace = probs[target_ids_nospace[0]].item() if target_ids_nospace else 0.0
    return max(prob_space, prob_nospace)
