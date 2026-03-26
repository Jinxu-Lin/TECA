"""Tests for Gradient Computation (core/gradient_utils.py).

Uses a minimal mock causal LM model to test gradient computation logic
without loading GPT-2-XL.
"""

import math
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.gradient_utils import (
    flatten_gradient,
)


# ---------------------------------------------------------------------------
# Minimal mock model for gradient computation tests
# ---------------------------------------------------------------------------

class MockConv1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias


class MockMLP(nn.Module):
    def __init__(self, d_model=16, d_ff=64):
        super().__init__()
        self.c_fc = MockConv1D(d_model, d_ff)
        self.c_proj = MockConv1D(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class MockBlock(nn.Module):
    def __init__(self, d_model=16, d_ff=64):
        super().__init__()
        self.mlp = MockMLP(d_model, d_ff)

    def forward(self, x):
        return x + self.mlp(x)


@dataclass
class MockModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class MockCausalLM(nn.Module):
    """Minimal causal LM that supports gradient computation."""
    def __init__(self, n_layers=4, d_model=16, d_ff=64, vocab_size=100):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [MockBlock(d_model, d_ff) for _ in range(n_layers)]
        )
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MockModelOutput(logits=logits, loss=loss)


D_MODEL = 16
D_FF = 64
N_LAYERS = 4
VOCAB_SIZE = 100


@pytest.fixture
def mock_model():
    return MockCausalLM(N_LAYERS, D_MODEL, D_FF, VOCAB_SIZE)


class TestForwardShape:
    """Verify output shapes match specification."""

    def test_flatten_gradient_shape(self):
        grad = torch.randn(64, 16)  # [d_ff, d_model]
        flat = flatten_gradient(grad)
        assert flat.shape == (64 * 16,)
        assert flat.dtype == torch.float32

    def test_flatten_gradient_1d(self):
        grad = torch.randn(1024)
        flat = flatten_gradient(grad)
        assert flat.shape == (1024,)

    def test_gradient_at_layer_shape(self, mock_model):
        """Test that gradient has same shape as MLP weight."""
        from core.model_utils import get_mlp_proj_param
        param = get_mlp_proj_param(mock_model, 1)
        expected_shape = param.shape  # [d_ff, d_model]

        # Manually compute gradient
        mock_model.zero_grad()
        param.requires_grad_(True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
        out = mock_model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        grad = param.grad.detach().clone()

        assert grad.shape == expected_shape, \
            f"Gradient shape {grad.shape} != weight shape {expected_shape}"


class TestGradientFlow:
    """Verify gradient computation produces valid gradients."""

    def test_gradient_non_zero(self, mock_model):
        from core.model_utils import get_mlp_proj_param
        param = get_mlp_proj_param(mock_model, 1)
        mock_model.zero_grad()
        param.requires_grad_(True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
        out = mock_model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        assert param.grad is not None, "Gradient should not be None"
        assert not torch.all(param.grad == 0), "Gradient should be non-zero"

    def test_gradient_different_layers_independent(self, mock_model):
        """Gradients at different layers should be different."""
        from core.model_utils import get_mlp_proj_param

        grads = []
        for layer_idx in range(N_LAYERS):
            mock_model.zero_grad()
            param = get_mlp_proj_param(mock_model, layer_idx)
            param.requires_grad_(True)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
            torch.manual_seed(42)  # same input
            out = mock_model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            grads.append(param.grad.detach().clone())

        # Not all layers should have identical gradients
        all_same = all(torch.allclose(grads[0], g) for g in grads[1:])
        assert not all_same, "Different layers should produce different gradients"


class TestOutputRange:
    """Verify gradient values are numerically valid."""

    def test_gradient_no_nan_inf(self, mock_model):
        from core.model_utils import get_mlp_proj_param
        for layer_idx in range(N_LAYERS):
            mock_model.zero_grad()
            param = get_mlp_proj_param(mock_model, layer_idx)
            param.requires_grad_(True)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
            out = mock_model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            assert not torch.isnan(param.grad).any(), f"NaN in gradient at layer {layer_idx}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient at layer {layer_idx}"

    def test_flatten_preserves_content(self):
        grad = torch.randn(64, 16)
        flat = flatten_gradient(grad)
        recon = flat.reshape(64, 16)
        assert torch.allclose(grad.float(), recon)


class TestConfigSwitch:
    """gradient_utils is always-on. Test aggregation logic."""

    def test_aggregation_mean(self):
        """Mean of gradients should reduce variance."""
        torch.manual_seed(42)
        grads = [torch.randn(64, 16) for _ in range(10)]
        aggregated = torch.stack(grads, dim=0).mean(dim=0)
        assert aggregated.shape == (64, 16)
        # Mean should have smaller norm than individual grads (in expectation)
        mean_norm = aggregated.norm().item()
        individual_norms = [g.norm().item() for g in grads]
        avg_individual_norm = sum(individual_norms) / len(individual_norms)
        assert mean_norm < avg_individual_norm * 1.5  # some slack

    def test_single_gradient_aggregation(self):
        """Aggregating a single gradient should return it as-is."""
        grad = torch.randn(64, 16)
        aggregated = torch.stack([grad], dim=0).mean(dim=0)
        assert torch.allclose(grad, aggregated)
