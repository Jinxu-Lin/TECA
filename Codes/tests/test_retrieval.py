"""Tests for BM25 Retrieval (core/retrieval.py).

Tests focus on the interface logic, ranking, and edge cases.
Does NOT download datasets or build real BM25 indices.
"""

import math
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.retrieval import (
    rank_by_gradient_dot_product,
)


class TestForwardShape:
    """Verify output types and structure."""

    def test_rank_by_gradient_returns_list_of_tuples(self):
        candidates = [{"text": f"doc {i}", "score": float(i), "doc_id": i} for i in range(5)]
        gradients = [torch.randn(64, 16) for _ in range(5)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=3)
        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], dict)
            assert isinstance(item[1], float)

    def test_rank_by_gradient_top_k_respected(self):
        candidates = [{"text": f"doc {i}", "score": float(i), "doc_id": i} for i in range(10)]
        gradients = [torch.randn(64, 16) for _ in range(10)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=5)
        assert len(result) == 5

    def test_rank_by_gradient_top_k_larger_than_candidates(self):
        candidates = [{"text": f"doc {i}", "score": float(i), "doc_id": i} for i in range(3)]
        gradients = [torch.randn(64, 16) for _ in range(3)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=10)
        assert len(result) == 3  # only 3 candidates available


class TestGradientFlow:
    """Ranking uses dot product which preserves gradient flow (if needed)."""

    def test_dot_product_gradient(self):
        test_grad = torch.randn(64, 16, requires_grad=True)
        g = torch.randn(64, 16)
        score = torch.dot(test_grad.reshape(-1).float(), g.reshape(-1).float())
        score.backward()
        assert test_grad.grad is not None
        assert not torch.all(test_grad.grad == 0)


class TestOutputRange:
    """Verify ranking correctness."""

    def test_ranking_is_descending(self):
        candidates = [{"text": f"doc {i}", "doc_id": i} for i in range(5)]
        # Create gradients with known dot products
        test_grad = torch.ones(64, 16)
        gradients = [torch.ones(64, 16) * (i + 1) for i in range(5)]
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=5)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True), \
            f"Scores should be in descending order: {scores}"

    def test_ranking_preserves_candidate_info(self):
        candidates = [{"text": f"doc {i}", "doc_id": i, "score": float(i)} for i in range(5)]
        gradients = [torch.randn(64, 16) for _ in range(5)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=5)
        for cand, score in result:
            assert "text" in cand
            assert "doc_id" in cand

    def test_no_nan_inf_in_scores(self):
        candidates = [{"text": f"doc {i}", "doc_id": i} for i in range(10)]
        gradients = [torch.randn(64, 16) for _ in range(10)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=10)
        for _, score in result:
            assert not math.isnan(score)
            assert not math.isinf(score)


class TestConfigSwitch:
    """retrieval.method is the ablation config key.
    Test that the ranking function works independently of retrieval method."""

    def test_empty_candidates(self):
        result = rank_by_gradient_dot_product([], [], torch.randn(64, 16), top_k=5)
        assert result == []

    def test_single_candidate(self):
        candidates = [{"text": "only doc", "doc_id": 0}]
        gradients = [torch.randn(64, 16)]
        test_grad = torch.randn(64, 16)
        result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=1)
        assert len(result) == 1

    def test_different_tensor_shapes(self):
        """Ranking should work with different weight matrix shapes."""
        for shape in [(32, 8), (64, 16), (128, 32)]:
            candidates = [{"text": f"doc {i}", "doc_id": i} for i in range(3)]
            gradients = [torch.randn(*shape) for _ in range(3)]
            test_grad = torch.randn(*shape)
            result = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=3)
            assert len(result) == 3
