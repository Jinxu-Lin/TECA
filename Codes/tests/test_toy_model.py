"""Tests for the toy linear associative memory model (Phase 1b).

These tests verify the core logic of the toy model experiment
without requiring GPU or large models.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.positive_control.toy_model_tecs import (
    LinearAssociativeMemory,
    generate_associations,
    train_model,
    rome_style_edit,
    compute_per_sample_gradient,
)
from experiments.common import cosine_similarity_flat


class TestLinearAssociativeMemory:
    def test_forward_shapes(self):
        """Forward pass should produce correct output shapes."""
        model = LinearAssociativeMemory(d_k=32, d_v=32, d_hidden=64)
        x = torch.randn(5, 32)
        out, h, v = model(x)
        assert out.shape == (5, 32)
        assert h.shape == (5, 64)
        assert v.shape == (5, 32)

    def test_training_converges(self):
        """Model should learn associations with small loss."""
        keys, values = generate_associations(50, 16, 16, seed=42)
        model = LinearAssociativeMemory(d_k=16, d_v=16, d_hidden=32)
        model = train_model(model, keys, values, n_epochs=200, lr=1e-3)

        with torch.no_grad():
            _, _, v_pred = model(keys)
            mse = F.mse_loss(v_pred, values).item()
        assert mse < 0.1, f"Training MSE too high: {mse}"


class TestRomeStyleEdit:
    def test_edit_produces_rank1(self):
        """ROME-style edit should produce a non-zero delta."""
        keys, values = generate_associations(50, 16, 16, seed=42)
        model = LinearAssociativeMemory(d_k=16, d_v=16, d_hidden=32)
        model = train_model(model, keys, values, n_epochs=100, lr=1e-3)

        v_new = torch.randn(1, 16)
        delta_W, h_star, v_target = rome_style_edit(model, 0, keys, v_new)

        assert delta_W.shape == (16, 32)  # (d_v, d_hidden)
        assert delta_W.norm() > 0


class TestPerSampleGradient:
    def test_gradient_shape(self):
        """Per-sample gradient should match associative layer weight shape."""
        keys, values = generate_associations(50, 16, 16, seed=42)
        model = LinearAssociativeMemory(d_k=16, d_v=16, d_hidden=32)
        model = train_model(model, keys, values, n_epochs=50, lr=1e-3)

        grad = compute_per_sample_gradient(model, keys, values, 0)
        assert grad.shape == (16, 32)  # Same as layer2.weight

    def test_gradient_nonzero(self):
        """Gradient should be non-zero for a non-trivial sample."""
        keys, values = generate_associations(50, 16, 16, seed=42)
        model = LinearAssociativeMemory(d_k=16, d_v=16, d_hidden=32)
        model = train_model(model, keys, values, n_epochs=50, lr=1e-3)

        grad = compute_per_sample_gradient(model, keys, values, 0)
        assert grad.norm() > 0


class TestToyTECS:
    def test_self_alignment(self):
        """TECS of delta with itself should be 1.0."""
        delta = torch.randn(16, 32)
        tecs = cosine_similarity_flat(delta, delta)
        assert abs(tecs - 1.0) < 1e-6

    def test_orthogonal_near_zero(self):
        """TECS between orthogonal vectors should be near 0 in high dim."""
        # In high dimensions, random vectors are approximately orthogonal
        torch.manual_seed(42)
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        tecs = cosine_similarity_flat(a, b)
        assert abs(tecs) < 0.1  # Should be close to 0
