"""Integration tests: verify all core modules import correctly and
interfaces are compatible across components.

All tests are pure CPU, no GPT-2-XL loading required.
"""

import torch
import torch.nn as nn
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModuleImports:
    """3a. Verify all core modules can be imported without errors."""

    def test_import_tecs(self):
        from core.tecs import compute_tecs, compute_null_a, compute_mean_pairwise_cosine, TECSResult
        assert callable(compute_tecs)
        assert callable(compute_null_a)
        assert callable(compute_mean_pairwise_cosine)

    def test_import_model_utils(self):
        from core.model_utils import (
            load_model_and_tokenizer, get_layer_module,
            get_mlp_weight, get_mlp_proj_param, num_layers,
        )
        assert callable(load_model_and_tokenizer)

    def test_import_rome_utils(self):
        from core.rome_utils import compute_rome_edit, EditResult, flatten_delta
        assert callable(compute_rome_edit)

    def test_import_gradient_utils(self):
        from core.gradient_utils import (
            compute_gradient_at_layer, compute_aggregated_gradient,
            compute_per_sample_gradients, flatten_gradient,
        )
        assert callable(compute_gradient_at_layer)

    def test_import_retrieval(self):
        from core.retrieval import (
            load_counterfact, retrieve_training_samples_bm25,
            rank_by_gradient_dot_product,
        )
        assert callable(load_counterfact)

    def test_import_statistics(self):
        from core.statistics import (
            paired_t_test, check_pass_criteria, format_report, TestResult,
        )
        assert callable(paired_t_test)

    def test_import_svd_diagnostics(self):
        from core.svd_diagnostics import svd_projection_diagnostic, SVDDiagnosticResult
        assert callable(svd_projection_diagnostic)

    def test_import_init(self):
        import core
        assert hasattr(core, "__doc__") or True  # just check it imports


class TestInterfaceCompatibility:
    """3b. Verify component interfaces match across the pipeline."""

    def test_rome_delta_shape_matches_tecs_input(self):
        """ROME produces delta_weight [d_ff, d_model].
        TECS consumes delta_weight with same shape."""
        from core.tecs import compute_tecs
        from core.rome_utils import flatten_delta

        d_ff, d_model = 64, 16
        delta = torch.randn(d_ff, d_model)
        grad = torch.randn(d_ff, d_model)

        # TECS should accept the same shape as ROME delta
        tecs = compute_tecs(delta, grad)
        assert isinstance(tecs, float)
        assert -1.0 - 1e-6 <= tecs <= 1.0 + 1e-6

        # flatten_delta should produce a 1D vector
        flat = flatten_delta(delta)
        assert flat.shape == (d_ff * d_model,)

    def test_gradient_shape_matches_tecs_input(self):
        """gradient_utils produces gradients [d_ff, d_model].
        TECS consumes gradients with same shape."""
        from core.tecs import compute_tecs
        from core.gradient_utils import flatten_gradient

        d_ff, d_model = 64, 16
        grad = torch.randn(d_ff, d_model)
        delta = torch.randn(d_ff, d_model)

        tecs = compute_tecs(delta, grad)
        assert isinstance(tecs, float)

        flat = flatten_gradient(grad)
        assert flat.shape == (d_ff * d_model,)

    def test_tecs_results_feed_into_statistics(self):
        """TECS compute_tecs returns floats that statistics.paired_t_test consumes."""
        from core.tecs import compute_tecs, compute_null_a
        from core.statistics import paired_t_test

        d_ff, d_model = 64, 16
        n_facts = 10

        real_tecs = []
        null_tecs = []
        for _ in range(n_facts):
            delta = torch.randn(d_ff, d_model)
            grad = torch.randn(d_ff, d_model)
            real_tecs.append(compute_tecs(delta, grad))
            # Null: use unrelated delta
            null_delta = torch.randn(d_ff, d_model)
            null_tecs.append(compute_tecs(null_delta, grad))

        result = paired_t_test(real_tecs, null_tecs)
        assert result.n == n_facts
        assert isinstance(result.cohens_d, float)
        assert isinstance(result.p_value, float)

    def test_svd_diagnostics_accepts_weight_delta_gradient(self):
        """SVD diagnostics takes weight, delta, and gradient -- all same shape."""
        from core.svd_diagnostics import svd_projection_diagnostic

        d_ff, d_model = 64, 16
        W = torch.randn(d_ff, d_model)
        delta = torch.randn(d_ff, d_model)
        grad = torch.randn(d_ff, d_model)

        result = svd_projection_diagnostic(W, delta, grad, top_k=5)
        assert result.spectral_risk in ("low", "medium", "high")
        assert 0.0 <= result.delta_projection_ratio <= 1.0 + 1e-6
        assert 0.0 <= result.gradient_projection_ratio <= 1.0 + 1e-6

    def test_gradient_ranking_with_retrieval(self):
        """retrieval.rank_by_gradient_dot_product takes candidates + gradients."""
        from core.retrieval import rank_by_gradient_dot_product

        d_ff, d_model = 64, 16
        candidates = [{"text": f"doc {i}", "doc_id": i, "score": float(i)} for i in range(5)]
        gradients = [torch.randn(d_ff, d_model) for _ in range(5)]
        test_grad = torch.randn(d_ff, d_model)

        ranked = rank_by_gradient_dot_product(candidates, gradients, test_grad, top_k=3)
        assert len(ranked) == 3
        # Scores should be in descending order
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_null_a_workflow(self):
        """Full Null-A workflow: compute TECS for real and unrelated deltas."""
        from core.tecs import compute_tecs, compute_null_a

        d_ff, d_model = 64, 16
        grad = torch.randn(d_ff, d_model)
        real_delta = torch.randn(d_ff, d_model)
        unrelated_deltas = [torch.randn(d_ff, d_model) for _ in range(5)]

        real_tecs = compute_tecs(real_delta, grad)
        null_tecs = compute_null_a(grad, unrelated_deltas)

        assert isinstance(real_tecs, float)
        assert len(null_tecs) == 5
        assert all(isinstance(v, float) for v in null_tecs)

    def test_angular_variance_workflow(self):
        """Mean pairwise cosine from per-sample gradients."""
        from core.tecs import compute_mean_pairwise_cosine

        d_ff, d_model = 64, 16
        per_sample_grads = [torch.randn(d_ff, d_model) for _ in range(5)]
        angular_var = compute_mean_pairwise_cosine(per_sample_grads)
        assert isinstance(angular_var, float)
        assert -1.0 - 1e-6 <= angular_var <= 1.0 + 1e-6

    def test_end_to_end_pipeline_mock(self):
        """Simulate the full TECA pipeline with mock data."""
        from core.tecs import compute_tecs, compute_null_a, compute_mean_pairwise_cosine
        from core.statistics import paired_t_test, check_pass_criteria
        from core.svd_diagnostics import svd_projection_diagnostic

        d_ff, d_model = 64, 16
        n_facts = 20

        real_tecs_list = []
        null_tecs_list = []

        for i in range(n_facts):
            torch.manual_seed(i)
            delta = torch.randn(d_ff, d_model)
            grad = torch.randn(d_ff, d_model)
            real_tecs_list.append(compute_tecs(delta, grad))

            null_delta = torch.randn(d_ff, d_model)
            null_tecs_list.append(compute_tecs(null_delta, grad))

        # Statistical testing
        test_result = paired_t_test(real_tecs_list, null_tecs_list)
        assert test_result.n == n_facts

        # Check pass criteria (with mock placebo test)
        placebo_result = paired_t_test(real_tecs_list, null_tecs_list)
        criteria = check_pass_criteria(
            test_result, placebo_result,
            mean_tecs=sum(real_tecs_list) / len(real_tecs_list),
            angular_variance=0.05,
        )
        assert "overall_pass" in criteria

        # SVD diagnostic
        W = torch.randn(d_ff, d_model)
        delta = torch.randn(d_ff, d_model)
        grad = torch.randn(d_ff, d_model)
        svd_result = svd_projection_diagnostic(W, delta, grad, top_k=5)
        assert svd_result.spectral_risk in ("low", "medium", "high")
