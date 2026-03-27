"""Tests for experiments/common.py utility functions."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.common import (
    set_seed, save_results, cohens_d, bootstrap_ci, paired_test, _json_default,
)


class TestSetSeed:
    def test_reproducibility(self):
        """set_seed should produce reproducible random numbers."""
        set_seed(42)
        a1 = np.random.rand(10)
        set_seed(42)
        a2 = np.random.rand(10)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds(self):
        """Different seeds should produce different sequences."""
        set_seed(42)
        a1 = np.random.rand(10)
        set_seed(99)
        a2 = np.random.rand(10)
        assert not np.array_equal(a1, a2)


class TestSaveResults:
    def test_save_and_load(self):
        """save_results should create a valid JSON file."""
        results = {"key": "value", "number": 42, "array": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_results(results, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == results
        finally:
            os.unlink(path)

    def test_numpy_serialization(self):
        """save_results should handle numpy types."""
        results = {
            "int": np.int64(42),
            "float": np.float32(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_results(results, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["int"] == 42
            assert abs(loaded["float"] - 3.14) < 0.01
            assert loaded["array"] == [1, 2, 3]
            assert loaded["bool"] is True
        finally:
            os.unlink(path)


class TestCohensD:
    def test_perfect_separation(self):
        """Large d for completely separated distributions."""
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([0.0, 1.0, 2.0])
        d = cohens_d(x, y)
        assert d > 5.0  # Very large effect

    def test_identical(self):
        """d = 0 for identical distributions."""
        x = np.array([1.0, 2.0, 3.0])
        d = cohens_d(x, x)
        assert d == 0.0

    def test_small_effect(self):
        """Small d for slightly different distributions."""
        rng = np.random.RandomState(42)
        x = rng.normal(0.1, 1.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        d = cohens_d(x, y)
        assert abs(d) < 0.5  # Should be small


class TestBootstrapCI:
    def test_covers_mean(self):
        """CI should cover the sample mean."""
        data = np.random.RandomState(42).normal(5.0, 1.0, 100)
        low, high = bootstrap_ci(data, n_boot=1000, seed=42)
        assert low < np.mean(data) < high

    def test_narrow_for_tight_data(self):
        """CI should be narrow for low-variance data."""
        data = np.array([1.0] * 50 + [1.01] * 50)
        low, high = bootstrap_ci(data, seed=42)
        assert (high - low) < 0.05


class TestPairedTest:
    def test_significant_difference(self):
        """Should detect significant difference."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        null = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = paired_test(real, null, "test", bootstrap_n=100, seed=42)
        assert result["p_value"] < 0.05
        assert result["cohens_d"] > 0.5
        assert result["n"] == 5
        assert result["name"] == "test"

    def test_no_difference(self):
        """Should not detect difference for identical arrays."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_test(x, x, "same", bootstrap_n=100, seed=42)
        assert result["cohens_d"] == 0.0
