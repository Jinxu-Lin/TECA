# Component: Gradient Computation
# Source: research/method-design.md §2.1
# Ablation config key: N/A (always on)

"""Gradient computation for TDA: compute per-sample gradients at a target layer."""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_utils import get_mlp_proj_param


def compute_gradient_at_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute ∇_{W_layer} L(θ; test_prompt) — gradient of the next-token
    prediction loss on *test_prompt* w.r.t. the MLP output projection weight
    at *layer_idx*.

    Returns:
        Gradient tensor with the same shape as the weight matrix, on CPU.
    """
    param = get_mlp_proj_param(model, layer_idx)

    # Freeze all parameters to save memory, then enable only target param
    requires_grad_backup = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    model.zero_grad()
    param.requires_grad_(True)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    grad = param.grad.detach().clone().cpu()

    # Restore all requires_grad states
    for p, rg in requires_grad_backup.items():
        p.requires_grad_(rg)
    model.zero_grad()

    return grad


def compute_aggregated_gradient(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_prompt: str,
    training_texts: List[str],
    layer_idx: int,
    device: str = "cuda",
    top_k: int = 10,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute the aggregated TDA gradient g_M = sum_i w_i * g_i / sum(w_i).

    Each g_i = gradient of L(theta; x_i) w.r.t. the edit layer weight, where x_i is a
    training sample. Weights w_i are typically BM25 retrieval scores.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        test_prompt: The test prompt (used for fallback if no training texts).
        training_texts: Retrieved training samples to compute gradients over.
        layer_idx: The layer to compute gradients at.
        device: Device string.
        top_k: Maximum number of training samples to use.
        weights: Per-sample weights (e.g. BM25 scores). If None, uses uniform.

    Returns:
        Aggregated gradient tensor (same shape as weight matrix), on CPU.
    """
    if not training_texts:
        return compute_gradient_at_layer(model, tokenizer, test_prompt, layer_idx, device)

    texts_to_use = training_texts[:top_k]
    grads = compute_per_sample_gradients(model, tokenizer, texts_to_use, layer_idx, device)

    if not grads:
        return compute_gradient_at_layer(model, tokenizer, test_prompt, layer_idx, device)

    # Weighted aggregation (BM25 scores or uniform)
    if weights is not None:
        w = torch.tensor(weights[:len(grads)], dtype=torch.float32)
        w = w / w.sum()  # normalize to sum to 1
        stacked = torch.stack(grads, dim=0)  # [k, *shape]
        aggregated = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    else:
        aggregated = torch.stack(grads, dim=0).mean(dim=0)

    return aggregated


def compute_per_sample_gradients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    layer_idx: int,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Compute individual gradients for each text w.r.t. the target layer.

    Each g_i = ∇_{W_layer} L(θ; text_i).

    Used for:
    - Angular variance diagnostic (pairwise cosine among gradients)
    - Per-sample gradient dot-product ranking (when filtering BM25 candidates)

    Returns:
        List of gradient tensors, each with the same shape as the weight matrix.
    """
    grads = []
    for text in texts:
        g = compute_gradient_at_layer(model, tokenizer, text, layer_idx, device)
        grads.append(g)
    return grads


def flatten_gradient(grad: torch.Tensor) -> torch.Tensor:
    """Flatten gradient to 1-D vector for cosine similarity computation."""
    return grad.reshape(-1).float()
