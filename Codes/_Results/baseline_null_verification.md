---
version: "1.0"
created: "2026-03-26"
---

# Null Baseline Verification Report

## Summary

All 5 null baselines are implemented and verified against pilot data (100 facts). Infrastructure is reliable for full-scale experiments.

## Implementation Status

| Null Baseline | Description | Core Function | Pilot Implementation | Pilot Data | Status |
|---------------|-------------|---------------|---------------------|------------|--------|
| Null-A | Random-fact TECS | `core/tecs.py::compute_null_a()` | `pilot_tecs_core.py` L230-235 | pilot_tecs_results.json | VERIFIED |
| Null-B | Wrong-layer TECS (l* +/- 5) | Inline (experiment-specific) | `pilot_tecs_core.py` L287-359 | pilot_tecs_results.json | VERIFIED |
| Null-C | Shuffled-gradient TECS | Inline (experiment-specific) | `pilot_tecs_core.py` L238-247 | pilot_tecs_results.json | VERIFIED |
| Null-D | Random-direction TECS | Inline (experiment-specific) | `pilot_tecs_core.py` L249-256 | pilot_tecs_results.json | VERIFIED |
| Null-E | Test-gradient TECS | Inline (experiment-specific) | `pilot_tecs_core.py` L258-268 | pilot_tecs_results.json | VERIFIED |

## Pilot Null Distribution Statistics

From `_Results/raw/pilot_tecs_results.json` (100 facts, seed=42):

| Baseline | Mean | Std | Cohen's d vs Real | p-value |
|----------|------|-----|-------------------|---------|
| Real TECS | 0.000157 | 0.00676 | -- | -- |
| Null-A (random fact) | -0.000170 | 0.00570 | 0.050 | 0.617 |
| Null-B (wrong layer) | -0.000342 | 0.00339 | 0.078 | 0.731 |
| Null-C (shuffled grad) | -0.0000037 | 0.000310 | 0.024 | 0.813 |
| Null-D (random dir) | 0.0000053 | 0.000307 | 0.022 | 0.824 |
| Null-E (test grad) | -0.00595 | 0.0288 | 0.211 | 0.037 |

## Theoretical Expectations vs Observations

### Null-A (Random-fact TECS)
- **Expected**: Mean near 0, since unrelated facts' edit directions should be orthogonal to the attribution gradient.
- **Observed**: Mean = -0.000170, consistent with expectation. Near-zero in 10M-dimensional space.
- **Verdict**: Distribution is as expected.

### Null-B (Wrong-layer TECS)
- **Expected**: Mean near 0. TECS at non-edit layers should lack the fact-specific signal.
- **Observed**: Mean = -0.000342 across layers [12, 22, 27, 32, 37]. Layer 37 shows slight positive mean (0.0015), but all non-significant.
- **Verdict**: Distribution is as expected.

### Null-C (Shuffled-gradient TECS)
- **Expected**: Mean near 0 with very small std. Permuting gradient elements destroys directional structure.
- **Observed**: Mean = -0.0000037, std = 0.000310 (much smaller than real TECS std). This is consistent with random alignment in ~10M dimensions.
- **Verdict**: Distribution is as expected.

### Null-D (Random-direction TECS)
- **Expected**: Mean near 0 with very small std. Random vectors in ~10M dimensions have near-zero expected cosine.
- **Observed**: Mean = 0.0000053, std = 0.000307. Nearly identical to Null-C.
- **Verdict**: Distribution is as expected.

### Null-E (Test-gradient TECS)
- **Expected**: Potentially different from zero if the test prompt gradient partially captures the editing direction.
- **Observed**: Mean = -0.00595, std = 0.0288 (much higher variance). This is the only baseline with marginal significance (p=0.037), but becomes non-significant after Bonferroni correction (threshold = 0.01).
- **Verdict**: Higher variance is expected since test gradients have variable norms. Distribution is consistent with theory.

## Overall Assessment

All null distributions behave consistently with theoretical predictions:
1. Null-C and Null-D have the tightest distributions (~0.0003 std), as expected for structure-destroying baselines in high-dimensional space.
2. Null-A and Null-B have moderate distributions (~0.003-0.006 std), preserving some gradient/weight structure.
3. Null-E has the widest distribution (0.029 std), expected due to test gradient variability.
4. Real TECS (d=0.050 vs Null-A) is indistinguishable from all baselines, confirming the negative finding is robust.

**Verdict**: Null baseline infrastructure is verified and ready for full-scale (200-fact) experiments.
