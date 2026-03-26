"""Tests for SVD Diagnostics (core/svd_diagnostics.py)."""

import math
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.svd_diagnostics import svd_projection_diagnostic, SVDDiagnosticResult


class TestForwardShape:
    """Verify output types and structure."""

    def test_returns_svd_diagnostic_result(self):
        W = torch.randn(64, 16)
        delta = torch.randn(64, 16)
        grad = torch.randn(64, 16)
        result = svd_projection_diagnostic(W, delta, grad, top_k=5)
        assert isinstance(result, SVDDiagnosticResult)

    def test_result_fields(self):
        W = torch.randn(64, 16)
        delta = torch.randn(64, 16)
        grad = torch.randn(64, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=5)
        assert isinstance(r.delta_projection_ratio, float)
        assert isinstance(r.gradient_projection_ratio, float)
        assert isinstance(r.spectral_risk, str)
        assert isinstance(r.singular_values, list)
        assert len(r.singular_values) == 5
        assert r.top_k == 5

    def test_singular_values_are_floats(self):
        W = torch.randn(32, 16)
        delta = torch.randn(32, 16)
        grad = torch.randn(32, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=3)
        for sv in r.singular_values:
            assert isinstance(sv, float)


class TestGradientFlow:
    """SVD diagnostics is analysis-only, no gradient needed.
    Test that it handles various tensor configurations."""

    def test_different_dtypes(self):
        for dtype in [torch.float32, torch.float64]:
            W = torch.randn(32, 16, dtype=dtype)
            delta = torch.randn(32, 16, dtype=dtype)
            grad = torch.randn(32, 16, dtype=dtype)
            r = svd_projection_diagnostic(W, delta, grad, top_k=3)
            assert not math.isnan(r.delta_projection_ratio)


class TestOutputRange:
    """Verify projection ratios are in valid range."""

    def test_projection_ratios_in_zero_one(self):
        torch.manual_seed(42)
        W = torch.randn(64, 16)
        delta = torch.randn(64, 16)
        grad = torch.randn(64, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=5)
        assert 0.0 <= r.delta_projection_ratio <= 1.0 + 1e-6, \
            f"delta ratio={r.delta_projection_ratio}"
        assert 0.0 <= r.gradient_projection_ratio <= 1.0 + 1e-6, \
            f"grad ratio={r.gradient_projection_ratio}"

    def test_self_projection_is_one(self):
        # Projecting W onto its own full SVD subspace should give ratio ~1.0
        W = torch.randn(16, 8)
        r = svd_projection_diagnostic(W, W, W, top_k=8)
        assert abs(r.delta_projection_ratio - 1.0) < 0.01, \
            f"Self-projection should be ~1.0, got {r.delta_projection_ratio}"

    def test_zero_delta_gives_zero_ratio(self):
        W = torch.randn(32, 16)
        delta = torch.zeros(32, 16)
        grad = torch.randn(32, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=3)
        assert r.delta_projection_ratio == 0.0

    def test_risk_levels_valid(self):
        W = torch.randn(32, 16)
        delta = torch.randn(32, 16)
        grad = torch.randn(32, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=3)
        assert r.spectral_risk in ("low", "medium", "high")

    def test_no_nan_inf(self):
        torch.manual_seed(42)
        for _ in range(10):
            W = torch.randn(32, 16)
            delta = torch.randn(32, 16)
            grad = torch.randn(32, 16)
            r = svd_projection_diagnostic(W, delta, grad, top_k=5)
            assert not math.isnan(r.delta_projection_ratio)
            assert not math.isnan(r.gradient_projection_ratio)
            assert not math.isinf(r.delta_projection_ratio)
            assert not math.isinf(r.gradient_projection_ratio)


class TestConfigSwitch:
    """SVD diagnostics is analysis-only, always available. Test different top_k values."""

    def test_top_k_1(self):
        W = torch.randn(32, 16)
        delta = torch.randn(32, 16)
        grad = torch.randn(32, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=1)
        assert r.top_k == 1
        assert len(r.singular_values) == 1

    def test_top_k_equals_min_dim(self):
        W = torch.randn(32, 16)
        delta = torch.randn(32, 16)
        grad = torch.randn(32, 16)
        r = svd_projection_diagnostic(W, delta, grad, top_k=16)
        assert r.top_k == 16
        assert len(r.singular_values) == 16
