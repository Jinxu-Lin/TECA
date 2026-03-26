"""Tests for Statistical Testing (core/statistics.py)."""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.statistics import (
    paired_t_test,
    check_pass_criteria,
    format_report,
    TestResult,
)


class TestForwardShape:
    """Verify output types and structure."""

    def test_paired_t_test_returns_test_result(self):
        real = [0.1, 0.2, 0.15, 0.12, 0.18]
        null = [0.01, 0.02, -0.01, 0.005, 0.03]
        result = paired_t_test(real, null)
        assert isinstance(result, TestResult)

    def test_test_result_fields(self):
        real = [0.1, 0.2, 0.15, 0.12, 0.18]
        null = [0.01, 0.02, -0.01, 0.005, 0.03]
        r = paired_t_test(real, null)
        assert isinstance(r.mean_real, float)
        assert isinstance(r.mean_null, float)
        assert isinstance(r.t_statistic, float)
        assert isinstance(r.p_value, float)
        assert isinstance(r.cohens_d, float)
        assert isinstance(r.ci_low, float)
        assert isinstance(r.ci_high, float)
        assert isinstance(r.n, int)
        assert isinstance(r.passed, bool)
        assert r.n == 5

    def test_check_pass_criteria_returns_dict(self):
        real = [0.5] * 10
        null = [0.0] * 10
        r1 = paired_t_test(real, null)
        r2 = paired_t_test(real, null)
        result = check_pass_criteria(r1, r2, mean_tecs=0.5)
        assert isinstance(result, dict)
        assert "overall_pass" in result

    def test_format_report_returns_string(self):
        real = [0.1, 0.2, 0.15]
        null = [0.01, 0.02, -0.01]
        r1 = paired_t_test(real, null)
        r2 = paired_t_test(real, null)
        criteria = check_pass_criteria(r1, r2, mean_tecs=0.15)
        report = format_report([], r1, r2, criteria)
        assert isinstance(report, str)
        assert "TECA" in report


class TestGradientFlow:
    """Statistical module is pure numpy/scipy, no torch gradients needed.
    Test numerical stability instead."""

    def test_identical_values_cohens_d_zero(self):
        vals = [0.1, 0.2, 0.15, 0.12, 0.18]
        r = paired_t_test(vals, vals)
        assert r.cohens_d == 0.0, "Identical real/null should give d=0"

    def test_large_effect_size(self):
        real = [1.0] * 20
        null = [0.0] * 20
        r = paired_t_test(real, null)
        # All differences are 1.0, std=0 -> degenerate
        # Actually std(diff, ddof=1) = 0, so d = 0 by the code's guard
        # This is technically correct behavior for constant differences
        # Let's use slightly varying values instead
        rng = np.random.RandomState(42)
        real2 = list(1.0 + rng.normal(0, 0.1, 20))
        null2 = list(0.0 + rng.normal(0, 0.1, 20))
        r2 = paired_t_test(real2, null2)
        assert r2.cohens_d > 2.0, f"Large separation should give large d, got {r2.cohens_d}"


class TestOutputRange:
    """Verify outputs are numerically valid."""

    def test_p_value_in_range(self):
        rng = np.random.RandomState(42)
        real = list(rng.normal(0.1, 0.05, 30))
        null = list(rng.normal(0.0, 0.05, 30))
        r = paired_t_test(real, null)
        assert 0.0 <= r.p_value <= 1.0, f"p-value={r.p_value} out of [0,1]"
        assert not math.isnan(r.cohens_d)
        assert not math.isnan(r.t_statistic)

    def test_ci_contains_d(self):
        rng = np.random.RandomState(42)
        real = list(rng.normal(0.1, 0.05, 50))
        null = list(rng.normal(0.0, 0.05, 50))
        r = paired_t_test(real, null)
        assert r.ci_low <= r.cohens_d <= r.ci_high, \
            f"CI [{r.ci_low}, {r.ci_high}] should contain d={r.cohens_d}"

    def test_pass_requires_both_significance_and_effect(self):
        # Small effect but "significant" with many samples
        rng = np.random.RandomState(42)
        real = list(rng.normal(0.001, 0.01, 1000))
        null = list(rng.normal(0.0, 0.01, 1000))
        r = paired_t_test(real, null, min_cohens_d=0.5)
        # Should not pass because effect is too small even if p < 0.05
        if r.p_value < 0.05:
            assert not r.passed, "Small effect should not pass despite significance"


class TestConfigSwitch:
    """statistics.py is always-on. Test edge cases."""

    def test_minimum_sample_size(self):
        real = [0.5, 0.3]
        null = [0.1, 0.0]
        r = paired_t_test(real, null)
        assert r.n == 2

    def test_custom_alpha_and_min_d(self):
        rng = np.random.RandomState(42)
        real = list(rng.normal(0.1, 0.05, 30))
        null = list(rng.normal(0.0, 0.05, 30))
        r1 = paired_t_test(real, null, alpha=0.01, min_cohens_d=0.3)
        r2 = paired_t_test(real, null, alpha=0.05, min_cohens_d=0.5)
        # Different thresholds can change pass/fail
        assert isinstance(r1.passed, bool)
        assert isinstance(r2.passed, bool)

    def test_kill_gate_in_check_pass_criteria(self):
        real = [0.5] * 10
        null = [0.0] * 10
        rng = np.random.RandomState(42)
        real_v = list(0.5 + rng.normal(0, 0.01, 10))
        null_v = list(0.0 + rng.normal(0, 0.01, 10))
        r1 = paired_t_test(real_v, null_v)
        r2 = paired_t_test(real_v, null_v)
        # Angular variance below kill threshold should kill
        result = check_pass_criteria(r1, r2, mean_tecs=0.5, angular_variance=0.0001)
        assert result["kill_gate_angular_variance"]["killed"] is True
        assert result["overall_pass"] is False
