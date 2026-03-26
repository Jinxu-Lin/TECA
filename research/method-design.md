---
version: "1.1"
entry_mode: "dr_revise"
iteration_major: 1
iteration_minor: 1
---

# Method Design: TECA

> [ASSIMILATED: generated from iter_001/idea/proposal.md + iter_001/plan/methodology.md + pilot results]

## 1. Overview

We measure the geometric relationship between two independent probes of factual knowledge in transformer MLPs: rank-one model editing (ROME/MEMIT) and training data attribution (TDA) gradients. The study follows a gated, dual-outcome design with a novel metric (TECS) and a six-component geometric analysis framework.

**Post-pilot framing**: Since TECS ~ 0, the paper becomes a **geometric incommensurability characterization** study, not a validation tool paper.

## 2. Core Method: TECS

### 2.1 Definition

For a factual association (subject s, relation r, object o) edited at layer l* of an MLP with weight matrix W ∈ R^{d_v × d_k}:

```
TECS = cos(vec(Δ_W_E), vec(g_M))
```

where:
- Δ_W_E = ROME's rank-one update at l* (outer product u·v^T)
- g_M = Σ_{i ∈ top-k} w_i · ∇_W L(z_i; θ) / ||...|| — BM25-weighted, normalized aggregation of per-sample training gradients at l*

### 2.2 Theoretical Foundation (Rank-One Decomposition)

Under the linear associative memory model (W·k = v), TECS admits a rank-one decomposition:

```
TECS(z_i) = sign(α) · cos(C^{-1}k*, k_i) · cos(v* - Wk*, d_v_i)
```

This decomposition:
- Predicts the null distribution: E[TECS_random²] ~ 1/d_k (Proposition 2)
- Predicts signal amplification: SNR ~ ρ_k · ρ_v · √d_k (Proposition 3)
- For GPT-2-XL (d_k=1600), even weak correlations (ρ_k · ρ_v > 0.08) should produce Cohen's d > 0.3

→ experiment-design.md §3 (Five Null Baselines)

## 3. Six-Component Geometric Analysis Framework

### 3.1 TECS Core Measurement
Direct cosine similarity between editing direction and attribution gradient.
→ experiment-design.md §3 (Phase 3)

### 3.2 Subspace Dimensionality Analysis
Effective dimensionality via eigenvalue entropy: d_eff = exp(-Σ p_i log p_i) where p_i are normalized eigenvalues from SVD of the stacked direction matrices.
→ experiment-design.md §5.1 (Subspace Geometry)

### 3.3 Principal Angle Analysis
scipy.linalg.subspace_angles between S_E = span(Δ_W_1, ..., Δ_W_N) and S_A = span(g_1, ..., g_N). Compare against random subspace baseline (1000 trials).
→ experiment-design.md §5.1

### 3.4 Cross-Projection Analysis
- G-in-D: fraction of attribution variance captured by editing subspace
- D-in-G: fraction of editing variance captured by attribution subspace
- Asymmetry reveals directional overlap structure.
→ experiment-design.md §5.1

### 3.5 Whitening Decomposition (H6)
Compare TECS_whitened (standard, with C^{-1}) vs TECS_unwhitened (raw, without C^{-1}).
Tests whether ROME's statistical whitening is the primary source of geometric incommensurability.
→ experiment-design.md §5.2

### 3.6 MEMIT Comparison
MEMIT distributes edits across layers 13-17 vs ROME's single layer 17. Measure alignment at each layer and cross-layer to test whether distributed editing bridges the incommensurability gap.
→ experiment-design.md §5.3

## 4. Hypotheses

| ID | Statement | Path | Status |
|----|-----------|------|--------|
| H1 | TECS at l* > null baselines (d > 0.3) | Core | REJECTED (d=0.050) |
| H2 | Rank-one decomposition correlates with full TECS (ρ > 0.7) | Core | UNTESTED (requires positive signal) |
| H3 | TECS predicts editing success (ρ > 0.2) | Positive | N/A (H1 rejected) |
| H4 | TECS layer-specific (peak at l*) | Positive | N/A (H1 rejected) |
| H5 | Mid-range spectral band selectivity | Positive | N/A (H1 rejected) |
| H6 | Whitening explains geometric gap (d > 0.3) | Negative | REJECTED |
| H7 | Structured incommensurability (θ_min < random) | Negative | CONFIRMED |

## 5. Dual-Outcome Design Principle

The paper is designed to produce a publishable contribution regardless of TECS direction:
- **TECS > 0 (d > 0.2)**: "Editing-attribution geometric consistency" paper
- **TECS ~ 0 (d ≤ 0.2)**: "Structured geometric incommensurability" paper (THIS OUTCOME)

Both outcomes advance understanding of how knowledge is geometrically organized in parameter space.

## 6. Addressing Known Concerns

### Circular reasoning (Contrarian)
We do not claim TECS "validates" TDA. We measure geometric relationships and characterize the structure of any misalignment. The null result itself is the finding.

### Dimensional concentration (Theorist)
Five null baselines (including Null-E random direction) calibrate against high-dimensional noise floor. Theoretical framework predicts E[TECS²] ~ 1/d_k.

### ROME artifacts (Contrarian)
Whitening decomposition (H6) directly tests whether ROME's C^{-1} rotation is the source. H6 rejected — the gap is fundamental.

### TDA gradient reliability (Empiricist)
Phase 2 gradient sanity checks passed. Angular variance = 0.048 (moderate consistency, not noise).

## 7. Positive Control Experiment (Design Review Addition)

To establish that TECS *can* detect alignment when it exists (addressing the "measurement failure" alternative), we design a three-tier positive control:

### 7.1 Tier 1: Trivial Positive (ROME vs. Self)
Compute TECS between a ROME edit direction and itself (or a noisy copy). Expected TECS ~ 1.0 (or proportional to noise level). This is a sanity check that the metric pipeline works.

### 7.2 Tier 2: Constructive Positive (Toy Linear Associative Memory)
Construct a multi-layer toy model where W·k = v holds by construction:
- Architecture: 3-layer MLP with ReLU, d_k = d_v = 64
- Training data: 200 synthetic (key, value) associations
- Editing: apply ROME-style rank-one update at the associative layer
- Attribution: compute exact per-sample gradients (no BM25 retrieval needed — full training set is small)
- Prediction: TECS should be significantly positive because (a) the linear memory assumption holds, (b) attribution gradients are exact, and (c) the rank-one decomposition predictions should hold

This tests whether TECS detects alignment when the theoretical conditions are satisfied. If it does, the GPT-2-XL null result is informative about real knowledge geometry.

### 7.3 Tier 3: Non-Trivial Positive (Semantically Related Facts)
For a fact (s, r, o), compute TECS between its ROME edit direction and attribution gradients for semantically related facts (same relation, different subject). If related facts share geometric structure, TECS should be detectably higher than cross-category comparisons. This tests partial alignment without requiring exact correspondence.

## 8. g_M Quality Analysis (Design Review Addition)

The attribution eff-dim = 1.2 (PC1 captures 91%) raises the question: is this a genuine property of how LLMs store knowledge influence, or an artifact of BM25 retrieval?

### 8.1 Within-Fact vs. Between-Fact Gradient Similarity
For each fact, compute pairwise cosine similarity among its top-k training gradients (within-fact) and compare to cross-fact gradient similarity (between-fact). If within-fact >> between-fact, the gradients contain fact-specific information despite the low eff-dim. If within-fact ~ between-fact, the gradients are dominated by a shared "generic relevance" component.

### 8.2 Retrieval Method Ablation
Compare attribution quality across retrieval methods:
| Method | Description | Expected Effect on eff-dim |
|--------|-------------|---------------------------|
| BM25 (baseline) | Lexical matching | eff-dim ~ 1.2 (current) |
| TF-IDF | Alternative lexical | Similar to BM25 |
| Contriever | Dense retrieval | Potentially higher eff-dim |
| Oracle top-k | If available, use known training data | Upper bound on eff-dim |

### 8.3 PC1 Removal Analysis
Remove the dominant PC1 direction from all attribution gradients and re-compute TECS on the residual. If TECS increases after PC1 removal, the "generic relevance" component was masking a weaker fact-specific signal.

## 9. "Trivially Expected" Defense (Theoretical Decomposition)

The theoretical rank-one decomposition (§2.2) provides the key defense against the "trivially expected" objection:

- Under the linear associative memory model, TECS admits an analytic form with predictable SNR ~ rho_k · rho_v · sqrt(d_k)
- For GPT-2-XL (d_k = 1600), even weak correlations (rho_k · rho_v > 0.08) should produce Cohen's d > 0.3
- The fact that TECS ~ 0 despite these predictions means either: (a) the linear memory assumptions fail badly in practice, or (b) BM25 attribution is too crude to capture the signal
- The positive control (§7.2) distinguishes these: if the toy model shows TECS > 0, then (a) is confirmed for real transformers
- This argument transforms a "trivially expected" null result into a quantitative falsification of specific theoretical predictions
