# Component: Gradient Computation
# Source: research/method-design.md §2.1
# Ablation config key: N/A (always on)

"""Gradient computation for TDA: compute per-sample gradients at a target layer."""

from __future__ import annotations

from typing import List

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

    # Zero all grads, enable grad only for target param
    model.zero_grad()
    requires_grad_backup = param.requires_grad
    param.requires_grad_(True)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    grad = param.grad.detach().clone().cpu()

    # Restore
    param.requires_grad_(requires_grad_backup)
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
) -> torch.Tensor:
    """Compute the aggregated TDA gradient g_M = (1/k) sum_i g_i for top-k training samples.

    Each g_i = gradient of L(theta; x_i) w.r.t. the edit layer weight, where x_i is a
    training sample. This represents "which direction in parameter space did these
    training samples push the model during training" -- the raw gradient TDA
    approximation.

    The aggregated gradient g_M is then compared with the ROME edit direction
    delta_theta to compute TECS. If the training samples that mention the fact
    pushed the model in the same direction as the ROME edit, TECS will be positive.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        test_prompt: The test prompt (used for fallback if no training texts).
        training_texts: Retrieved training samples to compute gradients over.
        layer_idx: The layer to compute gradients at.
        device: Device string.
        top_k: Maximum number of training samples to use.

    Returns:
        Aggregated gradient tensor (same shape as weight matrix), on CPU.
    """
    if not training_texts:
        # Fallback: if no training samples provided, use test prompt gradient
        return compute_gradient_at_layer(model, tokenizer, test_prompt, layer_idx, device)

    # Compute per-training-sample gradients
    texts_to_use = training_texts[:top_k]
    grads = compute_per_sample_gradients(model, tokenizer, texts_to_use, layer_idx, device)

    if not grads:
        return compute_gradient_at_layer(model, tokenizer, test_prompt, layer_idx, device)

    # Aggregate: mean of training sample gradients
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
