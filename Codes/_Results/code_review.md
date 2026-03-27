# TECA Code Review Report

**Date**: 2026-03-26
**Reviewer**: Automated DL Research Code Review (6-dimension)
**Codebase**: `/home/jinxulin/TECA/Codes/`

---

## Overall Assessment

| Dimension | Verdict | Summary |
|-----------|---------|---------|
| 1. Architecture Faithfulness | **Pass** | File structure matches CLAUDE.md mapping; deep/shallow separation enforced |
| 2. Component Faithfulness | **Concern** | 3 issues: gradient aggregation missing BM25 weighting, ROME rank-1 formula deviation, missing Null-C/D/E baselines |
| 3. Ablation Engineering | **Concern** | Config keys exist for retrieval method but ablation axes (loss function, scope, weighting) are not wired to working code |
| 4. DL Common Bugs | **Concern** | 2 issues: gradient accumulation risk across facts, missing `torch.use_deterministic_algorithms` |
| 5. Reproducibility | **Concern** | Missing `__init__.py` for `core/` package discovery (only has docstring), no `requirements.txt` version pinning, no config dump in probe_main.py |
| 6. Computational Efficiency | **Pass** | Reasonable memory management; large tensors moved to CPU; BM25 caching in place |

**Overall Verdict**: **Concern** -- The codebase is architecturally sound and the core TECS metric is correctly implemented. However, several deviations from method-design.md (gradient weighting, null baselines, ROME formula) and incomplete ablation wiring need attention before full-scale experiments.

---

## Dimension 1: Architecture Faithfulness

**Verdict: Pass**

### 1.1 File Structure vs CLAUDE.md Mapping

| Component (CLAUDE.md) | Expected File | Actual File | Match |
|------------------------|---------------|-------------|-------|
| TECS Core Metric | `core/tecs.py` | `core/tecs.py` | YES |
| Model Utilities | `core/model_utils.py` | `core/model_utils.py` | YES |
| ROME Editing | `core/rome_utils.py` | `core/rome_utils.py` | YES |
| Gradient Computation | `core/gradient_utils.py` | `core/gradient_utils.py` | YES |
| BM25 Retrieval | `core/retrieval.py` | `core/retrieval.py` | YES |
| Statistical Testing | `core/statistics.py` | `core/statistics.py` | YES |
| SVD Diagnostics | `core/svd_diagnostics.py` | `core/svd_diagnostics.py` | YES |
| Config Loading | `core/config.py` | `core/config.py` | YES (not in CLAUDE.md mapping table but referenced in Config Structure) |

### 1.2 Deep/Shallow Separation

- `core/` contains only reusable, model-agnostic computation modules -- **correct**.
- `experiments/` contains pipeline scripts and pilot experiments -- **correct**.
- `run_experiment.py` and `evaluate.py` are top-level entry points -- **correct**.
- No experiment-specific logic leaks into `core/` -- **verified**.

### 1.3 Dependency Direction

CLAUDE.md specifies:
```
Level 0 (no deps):  tecs.py, model_utils.py, statistics.py, svd_diagnostics.py
Level 1:            rome_utils.py (← model_utils), gradient_utils.py (← model_utils)
Level 2:            retrieval.py (← model_utils)
```

Verified imports:
- `tecs.py`: imports only torch, no core deps -- **correct**
- `model_utils.py`: imports only torch + transformers -- **correct**
- `statistics.py`: imports only numpy + scipy -- **correct**
- `svd_diagnostics.py`: imports only torch -- **correct**
- `rome_utils.py`: imports `from .model_utils import ...` -- **correct**
- `gradient_utils.py`: imports `from .model_utils import ...` -- **correct**
- `retrieval.py`: no import from model_utils -- **note**: CLAUDE.md says retrieval depends on model_utils, but actual code does not. This is acceptable (retrieval handles BM25 only, model interaction happens in experiment scripts).

---

## Dimension 2: Component Faithfulness

**Verdict: Concern**

### 2.1 TECS Core Metric (`core/tecs.py`)

**Method-design.md Section 2.1**: TECS = cos(vec(Delta_W_E), vec(g_M))

- `compute_tecs()` calls `cosine_similarity_flat()` which flattens both tensors and computes cosine similarity -- **correct**.
- `cosine_similarity_flat()` handles zero-norm vectors (returns 0.0) -- **correct**.
- `compute_null_a()` correctly computes TECS with unrelated edit directions -- **correct**.
- `compute_mean_pairwise_cosine()` correctly computes mean pairwise cosine among gradients -- **correct**.

**Status: Pass**

### 2.2 ROME Editing (`core/rome_utils.py`)

**Method-design.md Section 2**: Delta_W_E = (v* - W^T k*) k*^T / (k*^T C^{-1} k*) where C is empirical covariance.

**[CONCERN-2.2a]** `core/rome_utils.py` line 120: The rank-1 delta uses `k*^T @ k*` (identity covariance) instead of `k*^T @ C^{-1} @ k*`. The docstring at line 17 acknowledges this simplification ("simplified to identity for the probe"). While the code comment explains the rationale, method-design.md Section 3.5 (H6 Whitening Decomposition) explicitly tests whether C^{-1} matters. This simplification is **intentional** but means the "ROME delta" computed here differs from the true ROME algorithm's delta direction. The TECS values computed against this simplified delta are not directly comparable to what would be computed with the full ROME covariance-weighted delta.

- Impact: Medium. The pilot already uses this simplified version successfully. However, the paper's claims about "ROME direction" strictly refer to the covariance-weighted version. This should be clearly documented as "simplified ROME (identity covariance)" in the experiment results.

**[CONCERN-2.2b]** `core/rome_utils.py` line 295: In `_compute_target_value()`, `subject_end_pos` is set to `len(prompt_ids_list) - 1` (last prompt token), but in `_compute_key_vector()` (line 204), `subject_end_pos` is found by searching for the actual subject token position. These two functions use different positions for the same concept. The target value optimization hooks into the MLP at the last prompt token, while the key vector is extracted at the subject's last token. This is actually correct behavior for ROME (key is at subject position, value insertion is at last token), but the variable naming `subject_end_pos` in `_compute_target_value` is misleading -- it is not the subject position, it is the edit site position.

- Impact: Low (code behavior is correct, naming is confusing).

### 2.3 Gradient Computation (`core/gradient_utils.py`)

**Method-design.md Section 2.1**: g_M = Sum_{i in top-k} w_i * grad_W L(z_i; theta) / ||...|| -- BM25-weighted, normalized aggregation.

**[CONCERN-2.3a]** `core/gradient_utils.py` line 96: `compute_aggregated_gradient()` uses **uniform mean** (`torch.stack(grads, dim=0).mean(dim=0)`) instead of BM25-weighted aggregation. Method-design.md explicitly specifies "BM25-weighted, normalized aggregation of per-sample training gradients". The BM25 scores are available in the retrieval results but are never passed to this function.

- File: `core/gradient_utils.py`, line 96
- Impact: **High**. This is a direct deviation from the method specification. The aggregation weighting could affect the gradient direction and thus the TECS values.

**[CONCERN-2.3b]** `core/gradient_utils.py` line 96: Method-design.md specifies normalization (`/ ||...||`) in the g_M formula, but the code uses un-normalized mean aggregation. The aggregated gradient is not L2-normalized before being passed to TECS. Since TECS uses cosine similarity (which is scale-invariant), the normalization of g_M does not affect the TECS value. However, the BM25 weighting (2.3a) **does** affect direction and therefore matters.

- Impact: None on TECS (cosine is scale-invariant), but the weighting issue (2.3a) remains.

### 2.4 BM25 Retrieval (`core/retrieval.py`)

- `retrieve_training_samples_bm25()` returns BM25 scores with candidates -- **correct**.
- Three-level cache (pickle, in-memory, build fresh) -- **correct**.
- `rank_by_gradient_dot_product()` re-ranks by gradient dot product -- **correct**.
- `load_counterfact()` with proper seeded sampling -- **correct**.

**Status: Pass**

### 2.5 Statistical Testing (`core/statistics.py`)

- `paired_t_test()` computes paired t-test, Cohen's d, 95% CI -- matches method-design.md Section 4.
- Cohen's d formula: `mean(diff) / std(diff, ddof=1)` -- correct for paired samples.
- CI uses normal approximation (`d +/- 1.96 * se_d`) -- **note**: experiment-design.md Section 7 specifies "10,000 bootstrap 95% CI" for the primary metric. The code uses normal approximation in `statistics.py` and only has bootstrap in `evaluate.py`. The `bootstrap_n` config key exists but is not used in the core statistical module.

**[CONCERN-2.5a]** `core/statistics.py` lines 62-65: The CI for Cohen's d uses a normal approximation rather than the specified bootstrap method. The `evaluate.py` file has `bootstrap_cohens_d()` (line 155) but it is not used in the main experiment pipeline (`run_experiment.py` Phase 3 calls `paired_t_test()` which uses the approximation).

- File: `core/statistics.py`, lines 62-65; `evaluate.py`, line 155
- Impact: Low. The normal approximation is a reasonable alternative, and bootstrap is available in evaluate.py. However, the paper should use the bootstrap CI.

### 2.6 SVD Diagnostics (`core/svd_diagnostics.py`)

- Correctly computes projection ratios using both left and right singular vectors -- **correct**.
- Risk assessment thresholds (>0.8 high, >0.5 medium) are reasonable -- **correct**.

**Status: Pass**

### 2.7 Missing Null Baselines

**[CONCERN-2.7]** Experiment-design.md Section 3 defines **five** null baselines (Null-A through Null-E):

| Baseline | Status in Code |
|----------|----------------|
| Null-A (random-fact TECS) | Implemented in `probe_main.py` and `run_experiment.py` Phase 3 |
| Null-B (wrong-layer TECS) | **Partially implemented** -- `probe_main.py` computes a "placebo" using unrelated facts' gradients at the edit layer, but experiment-design.md specifies wrong-layer (l* +/- 5), not wrong-fact. The `probe_config.yaml` has `placebo_layer_offsets: [-5, 5]` but `probe_main.py` ignores these and uses unrelated-fact placebo instead. |
| Null-C (shuffled-gradient TECS) | **Not implemented** in any experiment script |
| Null-D (random-direction TECS) | **Not implemented** -- only used in the toy model positive control in `run_experiment.py` Phase 1 |
| Null-E (test-gradient TECS) | **Not implemented** in any experiment script |

- Files: `experiments/probe_main.py`, `run_experiment.py`
- Impact: **High**. Three of five null baselines specified in experiment-design.md are missing. The paper's statistical framework relies on all five for comprehensive null calibration.

---

## Dimension 3: Ablation Engineering

**Verdict: Concern**

### 3.1 Config Keys for Ablation Axes

Experiment-design.md Section 4 defines 4 ablation axes:

| Axis | Config Key | Config Present | Code Wired |
|------|-----------|----------------|------------|
| Top-k cutoff | `ablation.top_k_values` | YES (`base.yaml` line 61) | **NO** -- Phase 4 in `run_experiment.py` returns `pending_gpu` |
| Weighting | `ablation.weighting_methods` | YES (`base.yaml` line 62) | **NO** -- no code to switch between BM25/uniform/TF-IDF weighting in gradient aggregation |
| Loss definition | `ablation.loss_functions` | YES (`base.yaml` line 63) | **NO** -- `gradient_utils.py` only supports next-token CE loss (via `model(**inputs, labels=inputs["input_ids"])`) |
| Gradient scope | `ablation.scope_configs` | YES (`base.yaml` line 64) | **NO** -- `gradient_utils.py` computes gradient at a single specified layer; no multi-layer aggregation |

**[CONCERN-3.1]** All four ablation axes have config keys but none are wired to executable code. Phase 4 in `run_experiment.py` (line 591) returns `pending_gpu` without any implementation.

- Files: `run_experiment.py` lines 566-592, `core/gradient_utils.py`
- Impact: Medium. These are P1 priority experiments (not blocking), but the config keys exist without backing implementation. The code infrastructure (config loading, phase dispatch) is ready, but the actual ablation logic is missing.

### 3.2 Retrieval Method Ablation

- `core/config.py` validates `retrieval.method` against `("bm25", "tfidf", "contriever", "uniform")` (line 151).
- `core/retrieval.py` only implements BM25; there is no TF-IDF, Contriever, or Uniform retrieval.
- `base.yaml` `gm_quality.retrieval_methods` lists all four, but Phase 2 in `run_experiment.py` returns `pending_gpu`.

### 3.3 Positive Control Configs

- `positive_control.rome_self_sigmas`, `toy_model_d`, `toy_model_n_pairs` -- all config keys present and wired in Phase 1. **Pass**.

---

## Dimension 4: DL Common Bugs

**Verdict: Concern**

### 4.1 Data Leakage

- No data leakage detected. ROME edits are applied and immediately restored (`rome_utils.py` lines 123-131). Model weights are unchanged between facts.
- Training and test data are properly separated (CounterFact facts vs. OpenWebText retrieval).

**Status: Pass**

### 4.2 Shape/Broadcasting

- CLAUDE.md documents tensor shapes clearly: MLP weight `[6400, 1600]`, ROME delta same shape, gradient same shape.
- `cosine_similarity_flat()` flattens both tensors before computing -- immune to broadcasting issues.
- `rome_utils.py` rank-1 outer product: `k.unsqueeze(1) @ v_target.unsqueeze(0)` produces `[d_ff, 1] @ [1, d_model] = [d_ff, d_model]` -- **correct**.

**Status: Pass**

### 4.3 Loss Reduction Mode

- `gradient_utils.py` line 39: `model(**inputs, labels=inputs["input_ids"])` uses HuggingFace default loss (mean reduction over tokens). This is consistent across all gradient computations.
- `rome_utils.py` line 338: `F.cross_entropy(target_logits, target_tensor)` uses default mean reduction.

**Status: Pass**

### 4.4 Random Seed

**[CONCERN-4.4a]** `run_experiment.py` `set_global_seed()` (lines 47-59) correctly seeds `random`, `numpy`, `torch`, `torch.cuda`, and sets `cudnn.deterministic = True`. However:

- `probe_main.py` and `probe_sanity_check.py` do NOT call `set_global_seed()`. They rely on individual `random.Random(42 + i)` instances for null baseline selection, but do not set global torch/numpy seeds.
- `probe_svd_diagnostic.py` uses `seed=123` for `load_counterfact()` (line 48), inconsistent with the project-wide `seed=42`.

- File: `experiments/probe_main.py` (no global seed call), `experiments/probe_svd_diagnostic.py` line 48
- Impact: Low for probe scripts (which are being superseded by `run_experiment.py`), but the SVD diagnostic uses a different seed.

### 4.5 Train/Eval Mode

- `model_utils.py` line 32: `model.eval()` is called after loading -- **correct**.
- `gradient_utils.py` does NOT call `model.train()` before computing gradients, which means dropout (if any) stays off and batch norm uses running stats. For GPT-2-XL, this is correct (no dropout in eval, no batch norm). The gradient computation does use `torch.enable_grad()` implicitly by setting `param.requires_grad_(True)` before the forward pass.

**Status: Pass**

### 4.6 Gradient Accumulation Risk

**[CONCERN-4.6]** `gradient_utils.py` `compute_gradient_at_layer()` lines 34-47: The function calls `model.zero_grad()` at the start, but only zeros gradients for the model parameters. If this function is called in a loop (as in `compute_per_sample_gradients`), each call does `model.zero_grad()` + forward + backward + clone grad. This is correct for per-sample gradient isolation. However:

- Line 36: `requires_grad_backup = param.requires_grad` saves and restores the requires_grad state, but `model.zero_grad()` only zeros existing `.grad` attributes -- it does not affect other parameters' `requires_grad` flags. If other parameters happen to have `requires_grad=True` (which they do by default for a pretrained model), gradients will accumulate on them even though they are not used. This wastes memory but does not affect correctness since only `param.grad` is read.

- File: `core/gradient_utils.py`, lines 34-47
- Impact: Low. Memory waste from unused gradient accumulation on non-target parameters. Could be mitigated by wrapping the forward-backward in `torch.no_grad()` for all parameters except the target, or by using `param.grad.zero_()` instead of `model.zero_grad()`.

### 4.7 torch.use_deterministic_algorithms

**[CONCERN-4.7]** CLAUDE.md Reproducibility Checklist specifies "Deterministic operations where possible (torch.use_deterministic_algorithms)". `run_experiment.py` sets `cudnn.deterministic = True` and `cudnn.benchmark = False`, but does NOT call `torch.use_deterministic_algorithms(True)`. This means some CUDA operations (e.g., scatter, index_add) may still be non-deterministic.

- File: `run_experiment.py` line 56-57
- Impact: Low. For this project's workload (dense matmul, cosine similarity), non-deterministic scatter operations are unlikely to be invoked. But it violates the stated checklist.

---

## Dimension 5: Reproducibility

**Verdict: Concern**

### 5.1 Seed Full Chain

- `run_experiment.py` `set_global_seed()`: seeds random, numpy, torch, cuda -- **correct**.
- `probe_main.py` uses per-fact RNG (`random.Random(42 + i)`) for null baseline selection -- **correct for within-run determinism**, but no global seed.
- `run_experiment.py` Phase 3 uses `random.Random(42 + i)` for null baseline selection -- **correct**.
- CounterFact sampling uses `random.Random(seed)` -- **correct**.

**Status: Pass** (in the unified runner; probe scripts have the noted gap).

### 5.2 Environment Locking

**[CONCERN-5.2]** `requirements.txt` uses minimum version constraints (`torch>=2.0`, `transformers>=4.35`, etc.) instead of pinned versions. This makes exact reproduction difficult across different environments.

- File: `/home/jinxulin/TECA/Codes/requirements.txt`
- Impact: Medium. For a research project, pinned versions or a `requirements.lock` would ensure exact reproducibility.

### 5.3 Results Recording

- `run_experiment.py` saves results as JSON with config snapshot (`dump_config` at line 801) -- **correct**.
- `probe_main.py` saves results JSON with config summary but NOT a full config dump (line 249-282).
- `evaluate.py` generates markdown reports with gate summaries -- **correct**.

**Status: Pass** (in the unified runner).

### 5.4 Config Completeness

- `base.yaml` covers all experiment parameters comprehensively.
- Config inheritance (`_base_` key) works correctly via `_load_yaml_with_inheritance()`.
- `validate_config()` checks required sections and valid values.
- **Missing**: `rome` section is not validated (e.g., v_lr range, v_num_grad_steps range).

**Status: Pass** with minor gap.

### 5.5 BM25 Index Versioning

- `build_bm25_index.py` saves pickled index -- **correct**.
- Index is loaded via `--bm25_index` CLI argument or `retrieval.index_path` config -- **correct**.
- CLAUDE.md Reproducibility Checklist: "BM25 index cached and versioned (pickle)" -- implemented.

**Status: Pass**

---

## Dimension 6: Computational Efficiency

**Verdict: Pass**

### 6.1 GPU Memory

- `model_utils.py` line 32: `model.eval()` called -- reduces memory for inference.
- `gradient_utils.py` line 43: `grad.detach().clone().cpu()` -- gradients moved to CPU immediately after extraction. Good.
- `rome_utils.py` line 138: `delta.cpu()` -- delta stored on CPU. Good.
- `rome_utils.py` line 106: Weight upcast to float32 for computation: `W = param_pre.float()` -- necessary for numerical precision.
- No `torch.no_grad()` wrapper in `compute_gradient_at_layer()` for the non-target parameters (see CONCERN-4.6), but this is a minor memory issue.

### 6.2 Data Loading

- BM25 index caching (`_BM25_CACHE` global dict) avoids rebuilding across calls -- **efficient**.
- `build_bm25_index.py` pre-builds and pickles the index -- **correct workflow**.
- `load_counterfact()` with priority-ordered loading (local -> download -> HF) -- **robust**.

### 6.3 Redundant Computation

- `probe_main.py` Phase 3 computes `per_sample_grads` and then aggregates them, and also separately calls `compute_angular_variance()` on the same gradients. The `compute_aggregated_gradient()` function in `gradient_utils.py` also calls `compute_per_sample_gradients()` internally. In `probe_main.py`, gradients are computed once and reused for both aggregation and angular variance -- **efficient**.

- However, `run_experiment.py` Phase 3 (line 515-521) calls BOTH `compute_aggregated_gradient()` AND `compute_per_sample_gradients()` separately, resulting in **double gradient computation**:
  - Line 515-517: `compute_aggregated_gradient()` internally calls `compute_per_sample_gradients()` and aggregates
  - Line 518-520: `compute_per_sample_gradients()` called again for angular variance

**[NOTE-6.3]** `run_experiment.py` Phase 3 lines 515-521: Double computation of per-sample gradients. The `compute_aggregated_gradient()` function already computes per-sample gradients internally. Calling `compute_per_sample_gradients()` again wastes ~50% of the gradient computation time.

- Impact: Medium (doubles the gradient computation time, which is the dominant cost).

### 6.4 CPU/GPU Transfers

- Gradients are moved to CPU immediately after computation -- minimal GPU memory pressure.
- ROME deltas stored on CPU -- correct (they are reused across facts for null baselines).
- No unnecessary round-trips detected.

---

## Fix Recommendations

| Priority | Issue | File:Line | Recommendation |
|----------|-------|-----------|----------------|
| **P0** | Gradient aggregation uses uniform mean instead of BM25-weighted | `core/gradient_utils.py:96` | Add `weights` parameter to `compute_aggregated_gradient()`, pass BM25 scores from retrieval results. Compute `weighted_sum = sum(w_i * g_i) / sum(w_i)` instead of `mean`. |
| **P0** | Missing Null-C, Null-D, Null-E baselines | `run_experiment.py` Phase 3 | Implement: Null-C (shuffle gradient indices across facts), Null-D (random Gaussian direction same shape as delta), Null-E (gradient from test prompt instead of training samples). |
| **P0** | Null-B uses wrong-fact instead of wrong-layer | `experiments/probe_main.py:155-175` | Compute TECS at layers l*-5 and l*+5 using the SAME fact's gradient and delta, per experiment-design.md. Note: cross-layer comparison requires careful handling since different layers have different parameter spaces. The current wrong-fact approach may actually be more meaningful -- document the design decision. |
| **P1** | Double gradient computation in Phase 3 | `run_experiment.py:515-521` | Refactor: compute per-sample gradients once, use them for both aggregation and angular variance. |
| **P1** | Bootstrap CI not used for primary metric | `core/statistics.py:62-65` | Add bootstrap CI option to `paired_t_test()` or ensure `evaluate.py`'s `bootstrap_cohens_d()` is used for final reporting. |
| **P1** | Missing ablation code for 4 axes | `run_experiment.py:566-592` | Implement Phase 4: (a) top-k cutoff loop, (b) weighting scheme switch in gradient aggregation, (c) alternative loss functions in gradient computation, (d) multi-layer gradient scope. |
| **P2** | requirements.txt not pinned | `requirements.txt` | Add a `requirements.lock` or pin exact versions for reproducibility. |
| **P2** | torch.use_deterministic_algorithms not called | `run_experiment.py:56-57` | Add `torch.use_deterministic_algorithms(True)` after setting cudnn flags. |
| **P2** | ROME uses identity covariance | `core/rome_utils.py:120` | Document clearly in experiment results. Consider implementing full ROME with empirical covariance for comparison (as a separate experiment, since H6 tests this). |
| **P3** | Misleading variable name in _compute_target_value | `core/rome_utils.py:295` | Rename `subject_end_pos` to `edit_site_pos` or `last_prompt_token_pos` for clarity. |
| **P3** | SVD diagnostic uses different seed | `experiments/probe_svd_diagnostic.py:48` | Change `seed=123` to `seed=42` for consistency, or document the intentional difference. |

---

## Test Coverage Assessment

| Module | Test File | Coverage Level | Notes |
|--------|-----------|----------------|-------|
| `core/tecs.py` | `tests/test_tecs.py` | Good | Shape, range, edge cases, gradient flow |
| `core/model_utils.py` | `tests/test_model_utils.py` | Good | Mock model, shape verification, error handling |
| `core/rome_utils.py` | `tests/test_rome_utils.py` | Partial | Tests helpers and rank-1 math; no end-to-end test with real/mock model |
| `core/gradient_utils.py` | `tests/test_gradient_utils.py` | Good | Mock model gradient computation, shape and flow verification |
| `core/retrieval.py` | `tests/test_retrieval.py` | Partial | Only tests `rank_by_gradient_dot_product`; no test for BM25 retrieval or `load_counterfact` |
| `core/statistics.py` | `tests/test_statistics.py` | Good | All functions tested, edge cases covered |
| `core/svd_diagnostics.py` | `tests/test_svd_diagnostics.py` | Good | Shapes, ranges, edge cases |
| `core/config.py` | None | **Missing** | No test file for config loading, inheritance, validation, or CLI override |
| Integration | `tests/test_integration.py` | Good | Module imports, interface compatibility, end-to-end mock pipeline |

**Missing test**: No `tests/test_config.py` for the config loading system (inheritance resolution, deep merge, coercion, validation).
