"""Model loading and layer access utilities for GPT-2 family models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "gpt2-xl",
    device: str = "cuda",
    dtype: str = "float32",
):
    """Load a HuggingFace causal LM and its tokenizer.

    Returns:
        model: The loaded model on the specified device.
        tokenizer: The corresponding tokenizer with left-padding.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    return model, tokenizer


def get_layer_module(model, layer_idx: int):
    """Return the transformer block at *layer_idx*.

    Works for GPT-2 / GPT-J / GPT-NeoX naming conventions.
    """
    if hasattr(model, "transformer"):
        # GPT-2 / GPT-J
        return model.transformer.h[layer_idx]
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers[layer_idx]
    raise ValueError(f"Unknown model architecture: {type(model)}")


def get_mlp_weight(model, layer_idx: int) -> torch.Tensor:
    """Return the MLP output projection weight matrix at *layer_idx*.

    For GPT-2: transformer.h[l].mlp.c_proj.weight is a Conv1D parameter.
        Conv1D stores weight as [in_features, out_features] = [d_ff, d_model].
        The actual tensor shape is [nx, nf] where nx=in_features, nf=out_features.
        For GPT-2-XL c_proj: nx=6400 (d_ff), nf=1600 (d_model).
        Forward pass: output = input @ weight + bias (i.e., x @ W, no transpose).

    For GPT-J: transformer.h[l].mlp.fc_out.weight is a Linear parameter.
        Linear stores weight as [out_features, in_features] = [d_model, d_ff].

    Returns the raw parameter tensor as-is (no transpose). All consumers
    (gradient_utils, svd_diagnostics, tecs) must use the same convention.
    """
    block = get_layer_module(model, layer_idx)
    mlp = block.mlp

    if hasattr(mlp, "c_proj"):
        # GPT-2 Conv1D: weight shape [in_features, out_features] = [d_ff, d_model]
        # Return as-is. This is the same shape that get_mlp_proj_param returns,
        # so gradients and deltas will have matching shapes.
        return mlp.c_proj.weight
    if hasattr(mlp, "fc_out"):
        # GPT-J Linear: weight shape [out_features, in_features] = [d_model, d_ff]
        return mlp.fc_out.weight
    raise ValueError(f"Cannot find MLP output projection in {type(mlp)}")


def get_mlp_proj_param(model, layer_idx: int) -> torch.nn.Parameter:
    """Return the actual nn.Parameter for the MLP output projection at *layer_idx*.

    This is needed for gradient computation (requires_grad targeting).
    """
    block = get_layer_module(model, layer_idx)
    mlp = block.mlp

    if hasattr(mlp, "c_proj"):
        return mlp.c_proj.weight
    if hasattr(mlp, "fc_out"):
        return mlp.fc_out.weight
    raise ValueError(f"Cannot find MLP output projection in {type(mlp)}")


def num_layers(model) -> int:
    """Return the total number of transformer layers."""
    if hasattr(model, "transformer"):
        return len(model.transformer.h)
    if hasattr(model, "gpt_neox"):
        return len(model.gpt_neox.layers)
    raise ValueError(f"Unknown model architecture: {type(model)}")
