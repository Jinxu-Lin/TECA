"""Tests for Model Utilities (core/model_utils.py).

Uses a mock GPT-2-like model structure to avoid loading real GPT-2-XL.
"""

import torch
import torch.nn as nn
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model_utils import (
    get_layer_module,
    get_mlp_weight,
    get_mlp_proj_param,
    num_layers,
)


# ---------------------------------------------------------------------------
# Mock GPT-2 model structure
# ---------------------------------------------------------------------------

class MockConv1D(nn.Module):
    """Mimics GPT-2's Conv1D layer (weight shape: [in_features, out_features])."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Conv1D stores weight as [in_features, out_features]
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


class MockTransformer(nn.Module):
    def __init__(self, n_layers=4, d_model=16, d_ff=64):
        super().__init__()
        self.h = nn.ModuleList([MockBlock(d_model, d_ff) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return x


class MockGPT2Model(nn.Module):
    def __init__(self, n_layers=4, d_model=16, d_ff=64):
        super().__init__()
        self.transformer = MockTransformer(n_layers, d_model, d_ff)

    def forward(self, x):
        return self.transformer(x)


D_MODEL = 16
D_FF = 64
N_LAYERS = 4


@pytest.fixture
def mock_model():
    return MockGPT2Model(N_LAYERS, D_MODEL, D_FF)


class TestForwardShape:
    """Verify returned shapes match GPT-2 Conv1D convention."""

    def test_get_layer_module_returns_block(self, mock_model):
        block = get_layer_module(mock_model, 0)
        assert hasattr(block, "mlp"), "Block should have an MLP sub-module"

    def test_get_mlp_weight_shape(self, mock_model):
        # Conv1D weight: [d_ff, d_model] = [64, 16]
        w = get_mlp_weight(mock_model, 0)
        assert w.shape == (D_FF, D_MODEL), f"Expected ({D_FF}, {D_MODEL}), got {w.shape}"

    def test_get_mlp_proj_param_is_parameter(self, mock_model):
        p = get_mlp_proj_param(mock_model, 0)
        assert isinstance(p, nn.Parameter), "Should return an nn.Parameter"
        assert p.shape == (D_FF, D_MODEL)

    def test_num_layers(self, mock_model):
        assert num_layers(mock_model) == N_LAYERS


class TestGradientFlow:
    """Verify gradient flows through the returned weight parameter."""

    def test_gradient_through_mlp_weight(self, mock_model):
        p = get_mlp_proj_param(mock_model, 1)
        p.requires_grad_(True)
        x = torch.randn(1, 5, D_MODEL)
        out = mock_model(x)
        loss = out.sum()
        loss.backward()
        assert p.grad is not None, "Gradient should reach MLP weight"
        assert not torch.all(p.grad == 0), "Gradient should be non-zero"


class TestOutputRange:
    """Verify no NaN/Inf in weight values."""

    def test_weight_no_nan_inf(self, mock_model):
        for layer_idx in range(N_LAYERS):
            w = get_mlp_weight(mock_model, layer_idx)
            assert not torch.isnan(w).any(), f"Layer {layer_idx} weight has NaN"
            assert not torch.isinf(w).any(), f"Layer {layer_idx} weight has Inf"


class TestConfigSwitch:
    """model_utils is infrastructure (always on). Test layer index edge cases."""

    def test_first_layer(self, mock_model):
        block = get_layer_module(mock_model, 0)
        assert block is not None

    def test_last_layer(self, mock_model):
        block = get_layer_module(mock_model, N_LAYERS - 1)
        assert block is not None

    def test_invalid_layer_raises(self, mock_model):
        with pytest.raises(IndexError):
            get_layer_module(mock_model, N_LAYERS)

    def test_unknown_architecture_raises(self):
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Unknown model architecture"):
            get_layer_module(model, 0)

    def test_unknown_architecture_num_layers_raises(self):
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Unknown model architecture"):
            num_layers(model)
