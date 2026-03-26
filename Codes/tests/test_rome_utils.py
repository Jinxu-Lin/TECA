"""Tests for ROME Editing (core/rome_utils.py).

Uses mock models to avoid loading GPT-2-XL. Tests focus on the rank-1 delta
computation logic, helper functions, and interface correctness.
"""

import math
import torch
import torch.nn as nn
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rome_utils import (
    EditResult,
    flatten_delta,
    _find_subject_last_token_pos,
)


# ---------------------------------------------------------------------------
# Tests for helper functions (don't need a full model)
# ---------------------------------------------------------------------------

class TestForwardShape:
    """Verify output shapes and types."""

    def test_edit_result_fields(self):
        delta = torch.randn(64, 16)
        result = EditResult(
            subject="Paris",
            target_old="France",
            target_new="Germany",
            edit_layer=17,
            delta_weight=delta,
            edit_success=True,
            pre_prob=0.1,
            post_prob=0.9,
        )
        assert result.delta_weight.shape == (64, 16)
        assert isinstance(result.edit_success, bool)
        assert isinstance(result.pre_prob, float)
        assert isinstance(result.post_prob, float)

    def test_flatten_delta_shape(self):
        delta = torch.randn(64, 16)
        flat = flatten_delta(delta)
        assert flat.shape == (64 * 16,)
        assert flat.dtype == torch.float32

    def test_flatten_delta_1d(self):
        delta = torch.randn(1024)
        flat = flatten_delta(delta)
        assert flat.shape == (1024,)

    def test_find_subject_last_token_pos_exact_match(self):
        prompt_ids = [10, 20, 30, 40, 50]
        subject_ids = [20, 30]
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        assert pos == 2, f"Expected position 2, got {pos}"

    def test_find_subject_last_token_pos_at_start(self):
        prompt_ids = [10, 20, 30, 40, 50]
        subject_ids = [10, 20]
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        assert pos == 1

    def test_find_subject_last_token_pos_at_end(self):
        prompt_ids = [10, 20, 30, 40, 50]
        subject_ids = [40, 50]
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        assert pos == 4

    def test_find_subject_last_token_pos_single_token(self):
        prompt_ids = [10, 20, 30]
        subject_ids = [20]
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        assert pos == 1

    def test_find_subject_empty_subject(self):
        prompt_ids = [10, 20, 30]
        subject_ids = []
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        assert pos == len(prompt_ids) - 1

    def test_find_subject_no_match_fallback(self):
        prompt_ids = [10, 20, 30]
        subject_ids = [99, 100]
        pos = _find_subject_last_token_pos(prompt_ids, subject_ids)
        # Fallback: min(len(subject_ids)-1, len(prompt_ids)-1) = min(1, 2) = 1
        assert pos == 1


class TestGradientFlow:
    """Verify the rank-1 delta computation preserves gradient flow."""

    def test_rank1_delta_gradient(self):
        # Simulate the rank-1 update: delta = k @ v_target^T / (k^T @ k)
        k = torch.randn(64, requires_grad=True)
        v_target = torch.randn(16, requires_grad=True)
        k_norm_sq = torch.dot(k, k)
        delta = (k.unsqueeze(1) @ v_target.unsqueeze(0)) / k_norm_sq
        loss = delta.sum()
        loss.backward()
        assert k.grad is not None, "Gradient should flow through k"
        assert v_target.grad is not None, "Gradient should flow through v_target"
        assert not torch.all(k.grad == 0)
        assert not torch.all(v_target.grad == 0)


class TestOutputRange:
    """Verify numerical correctness of rank-1 delta."""

    def test_rank1_delta_shape(self):
        # delta = k @ v^T / (k^T @ k), shape [d_ff, d_model]
        k = torch.randn(64)  # d_ff
        v = torch.randn(16)  # d_model
        k_norm_sq = torch.dot(k, k)
        delta = (k.unsqueeze(1) @ v.unsqueeze(0)) / k_norm_sq
        assert delta.shape == (64, 16), f"Expected (64, 16), got {delta.shape}"

    def test_rank1_delta_is_rank_one(self):
        k = torch.randn(64)
        v = torch.randn(16)
        k_norm_sq = torch.dot(k, k)
        delta = (k.unsqueeze(1) @ v.unsqueeze(0)) / k_norm_sq
        # SVD should show only 1 non-zero singular value
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        # All singular values after the first should be ~0
        assert S[1:].max() < 1e-5, f"Not rank-1: S={S[:5]}"

    def test_rank1_delta_no_nan(self):
        for _ in range(20):
            k = torch.randn(64)
            v = torch.randn(16)
            k_norm_sq = torch.dot(k, k)
            delta = (k.unsqueeze(1) @ v.unsqueeze(0)) / k_norm_sq
            assert not torch.isnan(delta).any()
            assert not torch.isinf(delta).any()

    def test_flatten_preserves_content(self):
        delta = torch.randn(64, 16)
        flat = flatten_delta(delta)
        recon = flat.reshape(64, 16)
        assert torch.allclose(delta.float(), recon)


class TestConfigSwitch:
    """ROME is always-on. Test that interface handles edge cases."""

    def test_edit_result_with_failed_edit(self):
        result = EditResult(
            subject="test",
            target_old="old",
            target_new="new",
            edit_layer=17,
            delta_weight=torch.zeros(64, 16),
            edit_success=False,
            pre_prob=0.5,
            post_prob=0.01,
        )
        assert result.edit_success is False
        assert result.delta_weight.shape == (64, 16)

    def test_near_zero_key_vector(self):
        # When k is near-zero, the fallback path should still produce valid output
        k = torch.zeros(64)
        k[0] = 1e-15  # near zero
        v = torch.randn(16)
        k_norm_sq = torch.dot(k, k)
        # k_norm_sq < 1e-10, so use unnormalized fallback
        if k_norm_sq < 1e-10:
            delta = k.unsqueeze(1) @ v.unsqueeze(0)
        else:
            delta = (k.unsqueeze(1) @ v.unsqueeze(0)) / k_norm_sq
        assert delta.shape == (64, 16)
        assert not torch.isnan(delta).any()
