---
version: "1.0"
created: "2026-03-17"
last_modified: "2026-03-25"
---

# Probe Results: TECA

> [ASSIMILATED: generated from iter_001/exp/results/ + iter_001/supervisor/experiment_analysis.md]

## Executive Summary

**Verdict: NEGATIVE PATH (d = 0.050 << 0.2 threshold)**

TECS (TDA-Editing Consistency Score) at ROME's editing layer l*=17 is indistinguishable from all five null baselines. The editing and attribution directions in parameter space show no detectable geometric alignment. Subsequent negative path analyses reveal:
- H6 (whitening) REJECTED: C^{-1} is NOT the source of incommensurability
- H7 (structured incommensurability) CONFIRMED: the misalignment is structured, not random
- MEMIT provides partial bridge: cross-layer d ~ 0.63

## Phase 1: ROME Validation — GO

| Metric | Value |
|--------|-------|
| Efficacy | 100% (100/100 facts) |
| Mean P(new) post-edit | 0.978 |
| Mean P(old) post-edit | 0.00019 |
| Avg time/fact | 19.0s |
| Weight modified | transformer.h.17.mlp.c_proj.weight |
| Delta shape | (6400, 1600) |

**Lesson**: FP16 essential for shared GPU; EasyEdit proper ROME achieves 100% efficacy vs 14% for simplified implementation.

## Phase 2: TDA Gradient Computation — GO

| Metric | Value |
|--------|-------|
| Valid gradients | 100/100 facts |
| Mean g_M norm | 0.1896 |
| Angular variance | 0.048 (moderate consistency) |
| NaN count | 0 |

## Phase 3: TECS Core Measurement — NEGATIVE

| Metric | Value |
|--------|-------|
| TECS mean | 0.000157 |
| TECS std | 0.00676 |
| 95% Bootstrap CI | [-0.00117, 0.00146] |
| Positive/Negative ratio | 56/44 (near random) |
| Cohen's d vs Null-A | 0.050 |
| Cohen's d vs Null-B | 0.078 |
| Cohen's d vs Null-C | 0.024 |
| Cohen's d vs Null-D | 0.022 |
| Cohen's d vs Null-E | 0.211 (Null-E itself has high variance) |
| Bonferroni-corrected | ALL non-significant |

**Decision gate**: d(vs Null-A) = 0.050 ≤ 0.2 → NEGATIVE PATH

## Negative Path: Subspace Geometry (H7)

**Editing subspace (D)**: effective dimensionality = 40.8. Flat spectrum — editing directions are distributed across ~40 dimensions. Condition number = 2.0 (well-conditioned).

**Attribution subspace (G)**: effective dimensionality = 1.2. PC1 captures 91% of variance. Condition number = 33.2. Attribution is dominated by a single common direction.

**Principal angles**: All minimum angles >> 0 at k=10 (63.7°), k=20 (59.7°), k=50 (56.8°). No angles below 45°.

**Null comparison**: Minimum angles are NOT significantly smaller than random subspace baselines (p=0.084 at k=10, p=0.989 at k=20, p=1.0 at k=50). The misalignment is at least as severe as random.

**Cross-projection asymmetry**:
- G-in-D = 17.3% (attribution partially projects onto editing subspace)
- D-in-G = 1.0% (editing does NOT project onto attribution subspace)
- This asymmetry reveals that the narrow attribution subspace has marginal overlap with the broad editing subspace, but not vice versa.

## Negative Path: Whitening Decomposition (H6)

| Variant | Mean | Std |
|---------|------|-----|
| TECS raw | 0.000157 | 0.00676 |
| TECS unwhitened | -0.00103 | 0.00631 |
| TECS whitened | -0.0000221 | 0.00518 |

Cohen's d (unwhitened vs raw) = -0.198, p = 0.051 (non-significant)

**H6 REJECTED**: Removing ROME's C^{-1} whitening does NOT increase TECS. The geometric incommensurability is fundamental, not an artifact of ROME's covariance rotation.

## Negative Path: MEMIT Comparison

MEMIT editing across layers 13-17 (30 facts):
- Cross-layer TECS d ~ 0.63 (detectable alignment across layers)
- Matched-layer TECS d >> 6.0 (trivially high — same layer, same method)
- Delta norms increase monotonically from layer 13 (0.41) to layer 17 (0.62)

**Interpretation**: Multi-layer distributed editing partially bridges the incommensurability gap that single-layer ROME cannot cross. This suggests the editing-attribution geometric relationship is partially recoverable with richer editing methods.

## Key Conclusions

1. **ROME editing and TDA attribution operate in geometrically incommensurable parameter subspaces** — TECS is indistinguishable from noise
2. **The incommensurability is NOT due to ROME's whitening** — it persists even with raw (unwhitened) directions
3. **The subspace structure is radically asymmetric**: editing spans ~40 dimensions (flat), attribution collapses to ~1 dimension (dominated by PC1)
4. **Cross-projection is one-directional**: attribution partially overlaps editing (17%) but editing does not overlap attribution (1%)
5. **MEMIT partially bridges the gap** (d ~ 0.63), suggesting multi-layer approaches access more of the parameter space
6. **This provides a parameter-level explanation for Hase et al. (2023)**: localization (causal tracing) and editing access fundamentally different geometric structures in parameter space

## Data Artifacts

All raw data in `iter_001/exp/results/`:
- `pilot_rome_results.json` — Phase 1
- `pilot_tda_results.json` — Phase 2
- `pilot_tecs_results.json` — Phase 3
- `negative_subspace_results.json` — H7
- `negative_whitening_results.json` — H6
- `negative_memit_results.json` — MEMIT comparison
- Delta tensors: `iter_001/exp/results/rome_deltas/` (100 .pt files)
