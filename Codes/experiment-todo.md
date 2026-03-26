# Experiment TODO: TECA

> Blueprint v1.0 — Generated from method-design.md v1.1 + experiment-design.md v1.1
> Remote execution on xuchang3 via SSH MCP. Local code at `~/Research/TECA/Codes/`.

## Environment Preparation

- [ ] **env-setup**: Clone/update repo on xuchang3, install dependencies
  ```bash
  ssh xuchang3
  cd /home/jinxulin/sibyl_system  # or new path
  pip install torch transformers easyedit scipy rank_bm25 scikit-learn
  pip freeze > requirements.txt
  ```
- [ ] **data-check**: Verify CounterFact dataset (200 facts), WikiText corpus for BM25
- [ ] **gpu-verify**: Confirm RTX 4090, CUDA version, FP16 support
- [ ] **pilot-data-check**: Verify all pilot tensors are accessible
  - `iter_001/exp/results/rome_deltas/` (100 .pt files)
  - `iter_001/exp/results/pilot_tecs_results.json`
  - `iter_001/exp/results/negative_subspace_results.json`
- [ ] **seed-config**: Set global seed = 42 (torch, numpy, random, CUDA deterministic)

## Phase 0: Sanity Checks (~10 min)

- [ ] **sanity-rome-200**: Run ROME on 200 CounterFact facts at layer 17
  - Config: `configs/rome_200.yaml`
  - Expected: efficacy > 95% (pilot was 100%)
  - Output: `_Results/rome_200_validation.json`
  - **Gate**: efficacy < 75% → debug ROME setup

- [ ] **sanity-gradient-check**: Verify TDA gradient computation on 10 facts
  - Config: `configs/sanity_gradient.yaml`
  - Expected: all gradients non-zero, no NaN, shapes match (6400, 1600)
  - Output: `_Results/sanity_gradient.json`

- [ ] **sanity-tecs-pipeline**: Compute TECS for 10 facts, compare to pilot values
  - Expected: values within floating-point tolerance of pilot
  - Output: `_Results/sanity_tecs_pipeline.json`

## Phase 1: Positive Control Experiments (~35 min) — P0 PRIORITY

> Must complete BEFORE interpreting the main null result. Establishes metric validity.

### 1a. ROME vs Self (Trivial Positive) — ~5 min

- [ ] **pc-rome-self**: Compute TECS(delta_W, delta_W + epsilon) for sigma ∈ {0, 0.01, 0.1, 0.5, 1.0}
  - Script: `experiments/positive_control/rome_self_check.py`
  - Config: `configs/pc_rome_self.yaml`
  - Data: 100 pilot ROME deltas (reuse `iter_001/exp/results/rome_deltas/`)
  - Expected: TECS decreases monotonically from ~1.0
  - Pass: TECS(sigma=0) = 1.0, TECS(sigma=1.0) > 0.3
  - Output: `_Results/pc_rome_self.json`
  - **If fail**: metric pipeline is broken — stop all experiments, debug

### 1b. Toy Linear Associative Memory — ~10 min (CPU sufficient)

- [ ] **pc-toy-model**: Build and train toy model, compute TECS with ground-truth attribution
  - Script: `experiments/positive_control/toy_model_tecs.py`
  - Config: `configs/pc_toy_model.yaml`
  - Architecture: 3-layer MLP, d_k = d_v = 64, ReLU activation, 200 synthetic (key, value) pairs
  - Steps: Train → ROME-style rank-one edit → exact per-sample gradients → TECS
  - Expected: TECS Cohen's d > 0.5, rank-one decomposition correlation rho > 0.7
  - Pass: d > 0.3 (TECS detects alignment in controlled setting)
  - Output: `_Results/pc_toy_model.json`
  - **If fail (d < 0.3)**: TECS metric itself is flawed — ESCALATE to design phase

### 1c. Semantically Related Facts — ~20 min

- [ ] **pc-related-facts**: Within-relation vs cross-relation TECS comparison
  - Script: `experiments/positive_control/related_facts_tecs.py`
  - Config: `configs/pc_related_facts.yaml`
  - Data: 200 CounterFact facts grouped by relation type
  - Metric: Mean TECS(same-relation pairs) vs Mean TECS(cross-relation pairs)
  - Expected: Slight signal (same > cross), but null result is acceptable
  - Output: `_Results/pc_related_facts.json`

## Phase 2: g_M Quality Analysis (~1.5 hours) — P0 PRIORITY

> Resolves the eff-dim = 1.2 concern before full-scale experiments.

### 2a. Within-Fact vs Between-Fact Gradient Similarity — ~15 min

- [ ] **gm-within-between**: Compute gradient similarity structure
  - Script: `experiments/gm_quality/within_between_similarity.py`
  - Config: `configs/gm_within_between.yaml`
  - For each of 200 facts: pairwise cosine sim among top-20 training gradients
  - Compare: mean within-fact sim vs mean between-fact sim (paired t-test)
  - Expected: within > between → gradients have fact-specific content
  - Output: `_Results/gm_within_between.json`

### 2b. PC1 Removal Analysis — ~5 min

- [ ] **gm-pc1-removal**: Remove dominant direction and re-analyze
  - Script: `experiments/gm_quality/pc1_removal.py`
  - Config: `configs/gm_pc1_removal.yaml`
  - Steps: Compute PC1 from all g_M → project out → re-compute TECS + eff-dim
  - Expected: eff-dim increases; TECS may increase if PC1 masked signal
  - Output: `_Results/gm_pc1_removal.json`

### 2c. Retrieval Method Ablation — ~1 hour

- [ ] **gm-retrieval-tfidf**: TF-IDF retrieval → g_M → TECS + eff-dim
  - Script: `experiments/gm_quality/retrieval_ablation.py --method tfidf`
  - Expected: Similar to BM25
  - Output: `_Results/gm_retrieval_tfidf.json`

- [ ] **gm-retrieval-contriever**: Dense retrieval (Contriever) → g_M → TECS + eff-dim
  - Script: `experiments/gm_quality/retrieval_ablation.py --method contriever`
  - Expected: Potentially higher eff-dim (semantic vs lexical matching)
  - Output: `_Results/gm_retrieval_contriever.json`

- [ ] **gm-retrieval-uniform**: Uniform random retrieval → g_M → TECS + eff-dim
  - Script: `experiments/gm_quality/retrieval_ablation.py --method uniform`
  - Expected: Baseline — if eff-dim similar, retrieval method doesn't matter
  - Output: `_Results/gm_retrieval_uniform.json`

## Phase 3: Full-Scale Core Experiments (~2 hours) — P0 PRIORITY

> Scale pilot results from 100 → 200 facts.

- [ ] **full-rome-200**: ROME editing on 200 facts (if not done in Phase 0)
  - Script: `experiments/full_scale/rome_200.py`
  - Reuse Phase 0 results if available
  - Output: `_Data/rome_deltas_200/` (200 .pt files)

- [ ] **full-tda-200**: TDA gradient computation for 200 facts
  - Script: `experiments/full_scale/tda_gradients_200.py`
  - Config: `configs/full_tda.yaml`
  - Output: `_Data/tda_gradients_200/` (200 .pt files)

- [ ] **full-tecs-200**: Core TECS measurement + 5 null baselines
  - Script: `experiments/full_scale/tecs_core_200.py`
  - Depends: full-rome-200, full-tda-200
  - Metric: Cohen's d with 10,000 bootstrap 95% CI, Bonferroni correction
  - Expected: d ~ 0.05 (consistent with pilot), CI excludes 0.2
  - Output: `_Results/full_tecs_200.json`

- [ ] **full-subspace-200**: Subspace geometry analysis (200 facts)
  - Script: `experiments/full_scale/subspace_geometry_200.py`
  - Depends: full-rome-200, full-tda-200
  - Metrics: eff-dim, principal angles, cross-projection, 1000 random trials
  - Expected: Consistent with pilot (D eff-dim ~40, G eff-dim ~1-2)
  - Output: `_Results/full_subspace_200.json`

## Phase 4: Ablation Experiments (~1 hour) — P1 PRIORITY

- [ ] **ablation-topk**: Top-k ∈ {5, 10, 20, 50} → TECS
  - Script: `experiments/ablation/topk_ablation.py`
  - Expected: TECS variation < 20% across k
  - Output: `_Results/ablation_topk.json`

- [ ] **ablation-weighting**: BM25 / uniform / TF-IDF weighting → TECS
  - Script: `experiments/ablation/weighting_ablation.py`
  - Expected: TECS variation < 20%
  - Output: `_Results/ablation_weighting.json`

- [ ] **ablation-loss**: Object-token CE / full-sequence CE / margin loss → TECS
  - Script: `experiments/ablation/loss_ablation.py`
  - Expected: TECS variation < 20%
  - Output: `_Results/ablation_loss.json`

- [ ] **ablation-scope**: Layer l* only / layers [l*-2, l*+2] → TECS
  - Script: `experiments/ablation/scope_ablation.py`
  - Expected: Multi-layer may increase alignment (cf. MEMIT result)
  - Output: `_Results/ablation_scope.json`

## Phase 5: Extended Analyses (~1.5 hours) — P1 PRIORITY

- [ ] **ext-whitening-200**: Whitening decomposition on 200 facts
  - Script: `experiments/full_scale/whitening_200.py`
  - Expected: Confirm H6 rejection (d ~ -0.2, non-significant)
  - Output: `_Results/ext_whitening_200.json`

- [ ] **ext-memit-200**: MEMIT comparison on 200 facts (proper covariance)
  - Script: `experiments/full_scale/memit_200.py`
  - Expected: Cross-layer d ~ 0.6 (consistent with pilot 0.63)
  - Output: `_Results/ext_memit_200.json`

## Phase 6: Cross-Model Validation (~3 hours) — P2 PRIORITY

> Run ONLY after Phases 1-5 succeed.

- [ ] **cross-gptj-setup**: Load GPT-J-6B in FP16 with gradient checkpointing
  - Verify: fits in 24GB VRAM for forward + single-layer gradient
  - If OOM: use CPU offloading or reduce batch size

- [ ] **cross-gptj-rome**: ROME editing on 100 facts (GPT-J, optimal layer)
  - Script: `experiments/cross_model/gptj_rome.py`
  - Output: `_Data/gptj_rome_deltas/`

- [ ] **cross-gptj-tda**: TDA gradients for 100 facts (GPT-J)
  - Script: `experiments/cross_model/gptj_tda.py`
  - Output: `_Data/gptj_tda_gradients/`

- [ ] **cross-gptj-tecs**: Core TECS + subspace geometry (GPT-J)
  - Script: `experiments/cross_model/gptj_tecs.py`
  - Depends: cross-gptj-rome, cross-gptj-tda
  - Expected: TECS ~ 0 (replicates GPT-2-XL finding)
  - Output: `_Results/cross_gptj_tecs.json`

- [ ] **cross-gptj-positive**: Positive control on GPT-J (ROME vs self + related facts)
  - Script: `experiments/cross_model/gptj_positive_control.py`
  - Output: `_Results/cross_gptj_positive.json`

## Phase 7: Visualization & Paper Figures

- [ ] **viz-tecs-distribution**: TECS real vs Null-A overlay with effect size annotation
- [ ] **viz-eigenvalue-spectra**: Editing vs attribution eigenvalue spectra (side by side)
- [ ] **viz-principal-angles**: Angle distribution vs random baseline (violin plot)
- [ ] **viz-cross-projection**: Asymmetry diagram (G-in-D vs D-in-G)
- [ ] **viz-memit-heatmap**: Cross-layer alignment heatmap
- [ ] **viz-positive-control**: Toy model TECS vs noise level; GPT-2-XL positive control results
- [ ] **viz-gm-quality**: eff-dim comparison across retrieval methods; within vs between similarity

## Execution Order & Dependencies

```
Phase 0 (sanity) ────────────────────────────────────┐
                                                      │
Phase 1 (positive control) ──── [Gate: toy model d>0.3?] ──→ Phase 3 (full scale)
                                                      │
Phase 2 (g_M quality) ───────────────────────────────┘
                                                      │
                                              Phase 3 complete
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                             Phase 4 (ablation) Phase 5 (extended) Phase 6 (cross-model)
                                    │                 │                 │
                                    └─────────────────┴─────────────────┘
                                                      │
                                              Phase 7 (visualization)
```

**Critical gate**: If Phase 1b (toy model) fails with d < 0.3, STOP and escalate — the TECS metric itself may be flawed.

## Time Budget Summary

| Phase | Estimated Time | Priority | Depends On |
|-------|---------------|----------|------------|
| Phase 0: Sanity | ~10 min | P0 | — |
| Phase 1: Positive Control | ~35 min | P0 | Phase 0 |
| Phase 2: g_M Quality | ~1.5 hours | P0 | Phase 0 |
| Phase 3: Full Scale | ~2 hours | P0 | Phase 1 gate |
| Phase 4: Ablation | ~1 hour | P1 | Phase 3 |
| Phase 5: Extended | ~1.5 hours | P1 | Phase 3 |
| Phase 6: Cross-Model | ~3 hours | P2 | Phase 3 |
| Phase 7: Visualization | ~30 min | P1 | All |
| **Total** | **~10 hours** | | |

All experiments fit within 1x RTX 4090 (24GB). Phases 1-3 are the critical path (~4 hours). Phases 4-6 can run in any order after Phase 3.
