# Component: ROME Editing
# Source: research/method-design.md §2, §3.6
# Ablation config key: N/A (always on)

"""ROME editing utilities: self-contained rank-1 model editing.

Implements the ROME algorithm (Meng et al., 2022) directly, without external
dependencies on EasyEdit or other editing libraries. The algorithm:

1. Compute key vector k* = representation of the subject's last token at the
   target MLP layer (forward pass).
2. Compute target value v* via constrained optimization: find v such that
   the model outputs target_new when the MLP output at the edit layer is
   shifted by (v - W k*) projected onto k*.
3. Apply rank-1 update: W_new = W + (v* - W k*) k*^T / (k*^T C^{-1} k*)
   where C is an empirical covariance estimate (simplified to identity for
   the probe, since we only need the delta direction, not perfect editing).

For the TECA probe, editing fidelity matters less than getting a meaningful
delta direction. The key property we need: delta_weight is a rank-1 matrix
that encodes "what ROME would change to insert this fact".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_utils import get_mlp_proj_param, get_layer_module


@dataclass
class EditResult:
    """Container for a single ROME edit outcome."""

    subject: str
    target_old: str
    target_new: str
    edit_layer: int
    delta_weight: torch.Tensor  # rank-1 delta at edit layer, same shape as W
    edit_success: bool  # whether post-edit model outputs target_new
    pre_prob: float  # P(target_new) before edit
    post_prob: float  # P(target_new) after edit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rome_edit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    subject: str,
    prompt: str,
    target_new: str,
    target_old: str = "",
    edit_layer: Optional[int] = None,
    device: str = "cuda",
    v_lr: float = 5e-1,
    v_num_grad_steps: int = 20,
    clamp_norm_factor: float = 4.0,
    kl_factor: float = 0.0625,
    backend: str = "easyedit",
    easyedit_hparams=None,
    stats_dir: str = "data/stats",
) -> EditResult:
    """Run a single ROME edit and return the rank-1 weight delta at the edit layer.

    backend="easyedit" (default) uses EasyEdit's proper ROME with C^{-1}
    covariance. backend="builtin" uses the simplified self-contained version.
    """
    if edit_layer is None:
        edit_layer = _default_edit_layer(model)

    if backend == "easyedit":
        from .easyedit_rome import compute_rome_edit_easyedit
        ee_result = compute_rome_edit_easyedit(
            model, tokenizer,
            subject=subject, prompt=prompt,
            target_new=target_new, target_old=target_old,
            edit_layer=edit_layer, hparams=easyedit_hparams,
            device=device, stats_dir=stats_dir,
        )
        return EditResult(
            subject=ee_result.subject,
            target_old=ee_result.target_old,
            target_new=ee_result.target_new,
            edit_layer=ee_result.edit_layer,
            delta_weight=ee_result.delta_weight,
            edit_success=ee_result.edit_success,
            pre_prob=ee_result.pre_prob,
            post_prob=ee_result.post_prob,
        )

    # Snapshot pre-edit weight
    param = get_mlp_proj_param(model, edit_layer)
    param_pre = param.detach().clone()

    # Compute pre-edit probability
    pre_prob = _target_probability(model, tokenizer, prompt, target_new, device)

    # Step 1: Compute key vector k* (subject's last token representation at edit layer)
    k_star = _compute_key_vector(model, tokenizer, prompt, subject, edit_layer, device)

    # Step 2: Compute target value v* via optimization
    v_star = _compute_target_value(
        model, tokenizer, prompt, target_new, edit_layer, k_star, device,
        lr=v_lr, num_steps=v_num_grad_steps, clamp_norm_factor=clamp_norm_factor,
        kl_factor=kl_factor,
    )

    # Step 3: Compute rank-1 delta
    # W is the MLP c_proj weight. For GPT-2 Conv1D, weight shape is [in_features, out_features].
    # k_star: [in_features], v_star: [out_features]
    # Current output: W^T @ hidden (Conv1D forward is x @ weight)
    # We want: (W + delta)^T @ k_star to produce v_star
    # delta^T @ k_star = v_star - W^T @ k_star
    # For rank-1: delta = (v_target) @ k_star^T / (k_star^T @ k_star)
    # where v_target = v_star - W^T @ k_star

    W = param_pre.float()  # [in_features, out_features] for Conv1D
    k = k_star.float()

    # W^T @ k gives the current output for key k
    current_v = W.T @ k  # [out_features]
    v_target = v_star.float() - current_v  # [out_features]

    # Rank-1 update: delta = k @ v_target^T / (k^T @ k)
    # Shape: [in_features, out_features] — same as W
    k_norm_sq = torch.dot(k, k)
    if k_norm_sq < 1e-10:
        # Fallback: key vector is near-zero, use unnormalized
        delta = k.unsqueeze(1) @ v_target.unsqueeze(0)
    else:
        delta = (k.unsqueeze(1) @ v_target.unsqueeze(0)) / k_norm_sq

    # Step 4: Apply edit to measure success
    with torch.no_grad():
        param.add_(delta.to(param.dtype).to(param.device))

    post_prob = _target_probability(model, tokenizer, prompt, target_new, device)
    edit_success = post_prob > pre_prob and post_prob > 0.1

    # Restore original weights
    with torch.no_grad():
        param.copy_(param_pre)

    return EditResult(
        subject=subject,
        target_old=target_old,
        target_new=target_new,
        edit_layer=edit_layer,
        delta_weight=delta.cpu(),
        edit_success=edit_success,
        pre_prob=pre_prob,
        post_prob=post_prob,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_edit_layer(model) -> int:
    """Default ROME edit layer for known architectures."""
    name = getattr(model.config, "_name_or_path", "")
    if "gpt2-xl" in name:
        return 17  # ROME paper default for GPT-2-XL
    if "gpt-j" in name:
        return 5
    # Conservative default: middle layer
    from .model_utils import num_layers
    return num_layers(model) // 2


def _target_probability(
    model, tokenizer, prompt: str, target: str, device: str,
) -> float:
    """Compute P(target | prompt) using greedy next-token probability.

    Tries both with and without leading space, since GPT-2 tokenizes
    ' Antarctica' differently from 'Antarctica'.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Try with leading space first (GPT-2 convention for word-initial tokens)
    target_with_space = " " + target.lstrip()
    target_ids_space = tokenizer.encode(target_with_space, add_special_tokens=False)
    target_ids_nospace = tokenizer.encode(target, add_special_tokens=False)

    with torch.no_grad():
        logits = model(**inputs).logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    # Return the max of both variants
    prob_space = probs[target_ids_space[0]].item() if target_ids_space else 0.0
    prob_nospace = probs[target_ids_nospace[0]].item() if target_ids_nospace else 0.0
    return max(prob_space, prob_nospace)


def _compute_key_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    subject: str,
    edit_layer: int,
    device: str,
) -> torch.Tensor:
    """Compute the key vector k*: the MLP input representation at the subject's
    last token position in the edit layer.

    This is the "what key does the model use to look up this subject" vector.
    """
    # Find subject token positions
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    subject_ids = tokenizer.encode(subject, add_special_tokens=False)

    # Find last occurrence of subject tokens in prompt
    subject_end_pos = _find_subject_last_token_pos(prompt_ids, subject_ids)

    # Hook to capture the c_proj input at the edit layer (intermediate MLP activations)
    # For ROME, the key vector must be in the input space of the edited weight (c_proj).
    # c_proj input has dimension d_ff (6400 for GPT-2-XL), not d_model (1600).
    captured = {}

    block = get_layer_module(model, edit_layer)
    mlp = block.mlp

    # Hook into c_proj (the layer being edited) to capture its input
    c_proj = mlp.c_proj if hasattr(mlp, "c_proj") else mlp.fc_out

    def hook_fn(module, input, output):
        # input[0] is the intermediate activation going into c_proj (d_ff dimensional)
        captured["mlp_input"] = input[0].detach()

    handle = c_proj.register_forward_hook(hook_fn)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Extract the representation at the subject's last token position
    # captured["mlp_input"] shape: [1, seq_len, d_model]
    mlp_input = captured["mlp_input"]
    k_star = mlp_input[0, subject_end_pos, :].clone()

    return k_star


def _find_subject_last_token_pos(prompt_ids: List[int], subject_ids: List[int]) -> int:
    """Find the position of the subject's last token in the prompt."""
    if not subject_ids:
        return len(prompt_ids) - 1

    # Search for the subject token sequence in the prompt
    for start in range(len(prompt_ids) - len(subject_ids), -1, -1):
        if prompt_ids[start:start + len(subject_ids)] == subject_ids:
            return start + len(subject_ids) - 1

    # Fallback: if exact match not found (tokenization differences),
    # return a position near the beginning (subjects are usually early)
    return min(len(subject_ids) - 1, len(prompt_ids) - 1)


def _compute_target_value(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target_new: str,
    edit_layer: int,
    k_star: torch.Tensor,
    device: str,
    lr: float = 5e-1,
    num_steps: int = 20,
    clamp_norm_factor: float = 4.0,
    kl_factor: float = 0.0625,
) -> torch.Tensor:
    """Optimize for the target value v* that makes the model output target_new.

    This solves: v* = argmin_v L_edit(v) + kl_factor * L_KL(v)
    where L_edit ensures the model outputs target_new,
    and L_KL keeps the model's other predictions stable.

    We optimize by hooking into the MLP at edit_layer and replacing its output
    at the subject's last token position with a trainable vector.
    """
    # Encode the full target sequence: prompt + target_new
    # Use the full text encoding to find the correct token boundary,
    # since encoding prompt alone vs. as part of full_text can differ.
    full_text = prompt + " " + target_new
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    # Target tokens = full tokens after the prompt portion
    # Find the boundary by encoding prompt separately and computing offset
    target_ids = full_token_ids[len(prompt_token_ids):]
    if not target_ids:
        # Fallback: return the current W @ k_star as v_star (no-op edit)
        W = get_mlp_proj_param(model, edit_layer).detach().float()
        return (W.T @ k_star.float()).to(k_star.device)

    # Get the current MLP output at the subject position (initialization for v*)
    block = get_layer_module(model, edit_layer)
    mlp = block.mlp

    captured_output = {}

    prompt_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
    edit_site_pos = len(prompt_ids_list) - 1  # Last prompt token as edit site

    def capture_hook(module, input, output):
        captured_output["value"] = output.detach().clone()

    handle = mlp.register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    # Initialize v as the current MLP output at the edit site
    current_v = captured_output["value"][0, edit_site_pos, :].clone()
    v_opt = current_v.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([v_opt], lr=lr)

    # Get original logits for KL divergence
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        orig_logits = model(**full_inputs).logits

    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()

        # Hook that replaces the MLP output at the edit position with v_opt
        def edit_hook(module, input, output):
            out = output.clone()
            out[0, edit_site_pos, :] = v_opt
            return out

        handle = mlp.register_forward_hook(edit_hook)
        outputs = model(**full_inputs)
        handle.remove()

        logits = outputs.logits

        # L_edit: cross-entropy loss for generating target_new tokens
        # The target tokens start after the prompt portion in the full encoding
        prompt_boundary = len(prompt_token_ids)
        target_logits = logits[0, prompt_boundary - 1:prompt_boundary - 1 + len(target_ids), :]
        target_tensor = torch.tensor(target_ids, device=device)
        loss_edit = F.cross_entropy(target_logits, target_tensor)

        # L_KL: keep other predictions stable
        loss_kl = kl_factor * F.kl_div(
            F.log_softmax(logits[0], dim=-1),
            F.softmax(orig_logits[0], dim=-1),
            reduction="batchmean",
        )

        loss = loss_edit + loss_kl
        loss.backward()
        optimizer.step()

        # Clamp norm to prevent v* from diverging
        with torch.no_grad():
            v_norm = v_opt.norm()
            max_norm = clamp_norm_factor * current_v.norm()
            if v_norm > max_norm:
                v_opt.mul_(max_norm / v_norm)

    return v_opt.detach()


def flatten_delta(delta: torch.Tensor) -> torch.Tensor:
    """Flatten a weight delta matrix into a 1-D vector for cosine similarity."""
    return delta.reshape(-1).float()
