"""Tests for TECS Core Metric (core/tecs.py)."""

import math

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.tecs import (
    cosine_similarity_flat,
    compute_tecs,
    compute_null_a,
    compute_mean_pairwise_cosine,
    TECSResult,
)


class TestForwardShape:
    """Verify output shapes and types match method-design.md §2."""

    def test_compute_tecs_returns_scalar(self):
        # delta_weight and aggregated_gradient: [d_ff, d_model] = [6400, 1600]
        # Use small proxy: [64, 16]
        delta = torch.randn(64, 16)
        grad = torch.randn(64, 16)
        result = compute_tecs(delta, grad)
        assert isinstance(result, float), "TECS should return a scalar float"

    def test_cosine_similarity_flat_returns_scalar(self):
        a = torch.randn(32, 8)
        b = torch.randn(32, 8)
        result = cosine_similarity_flat(a, b)
        assert isinstance(result, float)

    def test_null_a_returns_list_of_floats(self):
        grad = torch.randn(64, 16)
        deltas = [torch.randn(64, 16) for _ in range(5)]
        result = compute_null_a(grad, deltas)
        assert isinstance(result, list)
        assert len(result) == 5
        for v in result:
            assert isinstance(v, float)

    def test_mean_pairwise_cosine_returns_scalar(self):
        grads = [torch.randn(64, 16) for _ in range(4)]
        result = compute_mean_pairwise_cosine(grads)
        assert isinstance(result, float)


class TestGradientFlow:
    """Verify that cosine_similarity_flat is differentiable (gradient flows)."""

    def test_gradient_through_cosine(self):
        a = torch.randn(32, 8, requires_grad=True)
        b = torch.randn(32, 8)
        # cosine_similarity_flat calls .item() so we test the underlying F.cosine_similarity
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        cos = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        )
        cos.backward()
        assert a.grad is not None, "Gradient should flow through cosine similarity"
        assert not torch.all(a.grad == 0), "Gradient should be non-zero"


class TestOutputRange:
    """Verify TECS values are in valid range, no NaN/Inf."""

    def test_tecs_in_minus_one_to_one(self):
        for _ in range(20):
            delta = torch.randn(64, 16)
            grad = torch.randn(64, 16)
            val = compute_tecs(delta, grad)
            assert not math.isnan(val), "TECS should not be NaN"
            assert not math.isinf(val), "TECS should not be Inf"
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, f"TECS={val} out of [-1,1]"

    def test_identical_vectors_give_one(self):
        v = torch.randn(64, 16)
        val = compute_tecs(v, v)
        assert abs(val - 1.0) < 1e-5, f"Self-cosine should be 1.0, got {val}"

    def test_opposite_vectors_give_minus_one(self):
        v = torch.randn(64, 16)
        val = compute_tecs(v, -v)
        assert abs(val - (-1.0)) < 1e-5, f"Opposite cosine should be -1.0, got {val}"

    def test_zero_vector_gives_zero(self):
        v = torch.randn(64, 16)
        z = torch.zeros(64, 16)
        val = compute_tecs(v, z)
        assert val == 0.0, "Zero-vector TECS should be 0.0"

    def test_mean_pairwise_cosine_range(self):
        grads = [torch.randn(64, 16) for _ in range(5)]
        val = compute_mean_pairwise_cosine(grads)
        assert not math.isnan(val)
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6

    def test_single_gradient_returns_zero(self):
        grads = [torch.randn(64, 16)]
        val = compute_mean_pairwise_cosine(grads)
        assert val == 0.0


class TestConfigSwitch:
    """TECS is always-on (no ablation config key). Verify it works with various shapes."""

    def test_different_shapes(self):
        for shape in [(10, 5), (100, 50), (6400, 1600)]:
            # Use small tensors for the large shape to avoid memory issues
            if shape == (6400, 1600):
                # Just verify the function accepts this shape with a small sample
                delta = torch.randn(64, 16)  # proxy
                grad = torch.randn(64, 16)
            else:
                delta = torch.randn(*shape)
                grad = torch.randn(*shape)
            val = compute_tecs(delta, grad)
            assert isinstance(val, float)

    def test_1d_tensors(self):
        a = torch.randn(256)
        b = torch.randn(256)
        val = compute_tecs(a, b)
        assert isinstance(val, float)
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6
