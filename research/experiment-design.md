---
version: "1.1"
entry_mode: "dr_revise"
iteration_major: 1
iteration_minor: 1
---

# Experiment Design: TECA

> [ASSIMILATED: generated from iter_001/plan/methodology.md + iter_001/idea/hypotheses.md + pilot results]

## 1. Model and Dataset
← method-design.md §2 (TECS Definition)

- **Model**: GPT-2-XL (1.5B parameters, 48 layers, d_k = d_v = 1600)
- **Dataset**: CounterFact (Meng et al., 2022) — 200 facts for full experiments, 100 used in pilot
- **Editing framework**: EasyEdit (pinned commit hash) for ROME/MEMIT
- **TDA retrieval**: BM25 over WikiText/Wikipedia, top-20 documents per fact
- **Hardware**: Single NVIDIA RTX 4090 (24GB VRAM), GPT-2-XL in FP16

## 2. Phase 1: ROME Editing Validation
← method-design.md §2.1

- N facts from CounterFact at layer l*=17
- Metric: efficacy (P(new) > P(old) post-edit)
- **Gate**: efficacy > 75%
- **Pilot result**: 100% efficacy (100/100), PASSED

## 3. Phase 3: Core TECS Measurement
← method-design.md §3.1

### Five Null Baselines
← method-design.md §2.2 (Theoretical Foundation)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Null-A | Random-fact TECS | Fact-specificity control |
| Null-B | Wrong-layer TECS (l* ± 5) | Layer-specificity control |
| Null-C | Shuffled-gradient TECS | Gradient-structure control |
| Null-D | Random-direction TECS | Dimensional-concentration control |
| Null-E | Test-gradient TECS | Gradient-source control |

### Primary Metric
Cohen's d (TECS_real vs Null-A) with 10,000 bootstrap 95% CI. Bonferroni correction for 5 comparisons.

### Decision Gate
- d > 0.2 → Positive path
- d ≤ 0.2 → Negative path (TRIGGERED: d = 0.050)

## 4. Phase 4: Ablation Study (PENDING for full paper)

| Axis | Settings | Purpose |
|------|----------|---------|
| Top-k cutoff | k ∈ {5, 10, 20, 50} | TDA aggregation sensitivity |
| Weighting | BM25 / uniform / TF-IDF | Weighting scheme robustness |
| Loss definition | Object token CE / full-sequence / margin | Loss definition sensitivity |
| Gradient scope | Layer l* only / layers [l*-2, l*+2] | Scope sensitivity |

Robustness criterion: TECS Cohen's d variation < 20% across settings.

## 5. Negative Path Extensions (PILOT COMPLETE)
← method-design.md §3.2-3.6

### 5.1 Subspace Geometry (H7)
← method-design.md §3.2, §3.3, §3.4

- SVD of stacked editing directions D (100 × 10.24M) and attribution gradients G (100 × 10.24M)
- Effective dimensionality via eigenvalue entropy
- Principal angle analysis at k ∈ {10, 20, 50}
- Null distribution: 1000 random subspace trials in reduced (dim=199) space
- Cross-projection: G-in-D and D-in-G variance ratios

**Pilot results**:
- D_eff_dim = 40.8, G_eff_dim = 1.2
- Min angles: 63.7° (k=10), 59.7° (k=20), 56.8° (k=50)
- p-values vs random: 0.084 (k=10), 0.989 (k=20), 1.0 (k=50)
- G-in-D = 17.3%, D-in-G = 1.0%

### 5.2 Whitening Decomposition (H6)
← method-design.md §3.5

- Compute covariance matrix C from 100 WikiText samples at layer 17
- TECS_unwhitened: use raw k* instead of C^{-1}k*
- Compare TECS_whitened vs TECS_unwhitened via paired t-test + Cohen's d

**Pilot result**: d = -0.198, p = 0.051 — H6 REJECTED (whitening not the cause)

### 5.3 MEMIT Comparison
← method-design.md §3.6

- MEMIT editing across layers 13-17 (simplified, identity covariance)
- Cross-layer TECS: alignment between MEMIT delta at layer L and TDA gradient at layer 17
- Matched-layer TECS: alignment between MEMIT delta and TDA gradient at the same layer

**Pilot result**: Cross-layer d ~ 0.63, matched-layer d >> 6.0

## 6. Positive Control Experiments (Design Review Addition)
← method-design.md §7

### 6.1 Tier 1: ROME vs. Self (Sanity Check)
- Compute TECS(delta_W, delta_W + epsilon) for epsilon ~ N(0, sigma^2 I) at sigma ∈ {0, 0.01, 0.1, 0.5, 1.0}
- Expected: TECS decreases monotonically from 1.0
- Purpose: Confirm metric pipeline correctness
- GPU time: <5 min

### 6.2 Tier 2: Toy Linear Associative Memory
- Construct 3-layer MLP (d=64, ReLU) trained on 200 synthetic (key, value) pairs
- Apply ROME-style rank-one edit at the associative layer
- Compute exact per-sample gradients (full training set, no retrieval)
- Measure TECS and compare to rank-one decomposition predictions
- Expected: TECS significantly > 0 (d > 0.5), decomposition correlation rho > 0.7
- **Gate**: If toy model TECS ~ 0, the metric itself is flawed — escalate
- GPU time: ~10 min (CPU sufficient)

### 6.3 Tier 3: Semantically Related Facts
- Group CounterFact facts by relation type (e.g., "born in", "capital of")
- Compute TECS between fact i's ROME edit and fact j's attribution gradient for same-relation vs cross-relation pairs
- Expected: Same-relation TECS slightly > cross-relation TECS
- This is exploratory — any signal is informative, null result is acceptable
- GPU time: ~20 min (reuses existing tensors)

## 6.5 g_M Quality Analysis (Design Review Addition)
← method-design.md §8

### 6.5.1 Within-Fact vs Between-Fact Gradient Similarity
- For each fact, compute pairwise cosine sim among its top-k training gradients
- Compute cross-fact gradient similarity (random pairs from different facts)
- Statistical test: paired t-test on within-fact vs between-fact means
- Expected: If within > between, gradients contain fact-specific signal

### 6.5.2 Retrieval Method Ablation
| Method | Implementation | Priority |
|--------|---------------|----------|
| BM25 (baseline) | rank_bm25 | Done |
| TF-IDF | sklearn TfidfVectorizer | P1 |
| Contriever | facebook/contriever | P1 |
| Uniform (no retrieval) | Random top-k from corpus | P1 |
- For each method: compute g_M, measure eff-dim, re-compute TECS
- GPU time: ~1 hour total (gradient computation dominates)

### 6.5.3 PC1 Removal Analysis
- Remove PC1 direction from all attribution gradients: g_M' = g_M - (g_M · pc1) * pc1
- Re-compute TECS on residual g_M'
- Re-compute eff-dim on residual
- Expected: eff-dim increases substantially; TECS may increase if PC1 masked signal
- GPU time: ~5 min (reuses existing gradients)

## 7. Full-Scale Experiment Plan (UPDATED)

| Experiment | N facts | Status | Priority |
|------------|---------|--------|----------|
| Core TECS (200 facts) | 200 | PENDING | P0 |
| Subspace geometry (200 facts) | 200 | PENDING | P0 |
| **Positive control: Toy model** | N/A | PENDING | **P0** |
| **Positive control: ROME vs self** | 200 | PENDING | **P0** |
| **g_M quality: within vs between** | 200 | PENDING | **P0** |
| **g_M quality: PC1 removal** | 200 | PENDING | **P0** |
| Ablation (4 axes) | 200 | PENDING | P1 |
| Whitening decomposition (200 facts) | 200 | PENDING | P1 |
| MEMIT full (200 facts) | 200 | PENDING | P1 |
| **g_M quality: retrieval ablation** | 200 | PENDING | **P1** |
| **Positive control: related facts** | 200 | PENDING | **P1** |
| Cross-model (GPT-J 6B) | 100 | PENDING | P2 |

## 7. Statistical Analysis Plan

- Primary: Cohen's d with 10,000 bootstrap 95% CI
- Multiple comparisons: Bonferroni correction
- Effect size thresholds: d > 0.2 (small), d > 0.5 (medium), d > 0.8 (large)
- Subspace tests: permutation tests (1000 trials) for principal angle comparisons
- All seeds fixed at 42
- All intermediate tensors saved for reproducibility

## 8. Cross-Model Validation (Design Review Addition)

### 8.1 GPT-J-6B (Primary Cross-Model)
- Model: EleutherAI/gpt-j-6b (6B parameters, 28 layers, d_model=4096)
- Hardware: Single RTX 4090 (24GB) — requires FP16 + gradient checkpointing or offloading
- Editing: ROME at GPT-J's optimal layer (determined by causal tracing or ROME paper recommendation)
- Dataset: 100 CounterFact facts (subset of 200)
- Experiments: Core TECS + subspace geometry + positive control
- Expected: If incommensurability replicates, the structural claim is strengthened

### 8.2 Analysis
- Compare TECS effect sizes across models
- Compare dimensionality asymmetry (eff-dim ratio)
- Compare cross-projection profiles
- If results diverge: characterize which structural properties are model-dependent

## 9. Expected Visualizations (for paper)

| Figure | Content | Status |
|--------|---------|--------|
| Fig 1 | Architecture diagram: ROME direction vs TDA gradient in parameter space | PENDING |
| Fig 2 | TECS distribution: real vs Null-A overlay with effect size | Data ready |
| Fig 3 | Subspace dimensionality: eigenvalue spectra for D and G | Data ready |
| Fig 4 | Principal angle distribution vs random baseline | Data ready |
| Fig 5 | Cross-projection asymmetry diagram | Data ready |
| Fig 6 | MEMIT cross-layer alignment heatmap | Data ready |
| Table 1 | Main TECS results (all 5 null comparisons) | Data ready |
| Table 2 | Subspace geometry summary | Data ready |
| Table 3 | Ablation study | PENDING |

## 10. Computational Budget (UPDATED)

| Task | GPU Time | Status |
|------|----------|--------|
| Phase 1 (ROME) | 32 min | DONE |
| Phase 2 (TDA gradients) | 4 min | DONE |
| Phase 3 (TECS core) | ~50 min | DONE |
| Negative path extensions | ~25 min | DONE |
| Full-scale (200 facts) | ~2 hours | PENDING |
| **Positive control: ROME vs self** | ~5 min | PENDING |
| **Positive control: Toy model** | ~10 min | PENDING |
| **Positive control: Related facts** | ~20 min | PENDING |
| **g_M quality analysis** | ~30 min | PENDING |
| **Retrieval method ablation** | ~1 hour | PENDING |
| Ablation (4 axes) | ~1 hour | PENDING |
| Cross-model (GPT-J 6B) | ~3 hours | PENDING |
| **Total remaining** | **~8 hours** | |
