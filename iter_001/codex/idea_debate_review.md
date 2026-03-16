# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-17
**模型**: Codex (GPT-5)

## 评审意见

**Overall score: 6/10 (promising but overconfident).**

### 1) Novelty assessment
- **Core idea is moderately novel**: directly comparing a ROME edit direction with a TDA gradient direction at the same layer is a useful new diagnostic.
- **But it is not "from scratch" novel**: it sits at the intersection of already-known disconnects:
  - localization vs editing success mismatch (Hase et al., 2023),
  - factual-attribution vs causal-influence mismatch in TDA (Chang et al., ICLR 2025).
- Likely novelty level: **new metric + synthesis**, not a new paradigm.

### 2) Methodological rigor
- 5 nulls + 4 ablation axes are good.
- Missing/underweighted confounders:
  1. **Parameterization dependence** (cosine in parameter space is not function-invariant).
  2. **Selection bias** if analysis conditions on successful edits.
  3. **Lexical overlap confound** from BM25 retrieval.
  4. **Norm/scale confound** across parameter blocks (layernorm, whitening, precision effects).
  5. **Synthetic benchmark artifact risk** (CounterFact-style evaluation inflation).
  6. **Implementation sensitivity** (ROME vs r-ROME behavior).

### 3) Theoretical soundness
- The rank-one decomposition is **algebraically plausible under strict assumptions** (linear associative memory, clean rank-one structure, stable covariance whitening).
- Practical usefulness is limited if gradients are not close to rank-one (likely in real runs).
- Fisher-TECS/KFAC is conceptually motivated, but in this setup it risks becoming another noisy approximation layer unless you show stability checks and ablations.

### 4) Risk analysis
- **85% publishable is too optimistic.**
- More realistic:
  - ~55-65% publishable (including solid negative-result paper),
  - ~20-30% ambiguous/noisy outcome,
  - ~10-20% dead-end.
- Biggest risks:
  - TECS dominated by measurement artifacts,
  - unreliable TDA signals on proxy corpora,
  - strong result not generalizing beyond GPT-2-XL / CounterFact.

### 5) Overlooked shared blind spots
Across all six perspectives, the biggest common miss is:
- **They treat parameter-space direction as intrinsically meaningful**, but equivalent reparameterizations can rotate/reshape these directions without changing behavior.
Other shared blind spots:
- overreliance on synthetic edit evaluation,
- weak linkage to function-space behavior (actual outputs/reasoning consistency),
- underestimation of benchmark leakage and teacher-forcing artifacts.

### 6) Practical feasibility (4090, ~80 min)
- **Not feasible as written** for full scope (200 facts + layer sweep + 5 nulls + 4 ablations + diagnostics).
- 4090 memory is likely enough for GPT-2-XL editing runs, but runtime is the blocker.
- Roughly: layer sweep alone multiplies cost by ~48; adding nulls/ablations explodes combinations. Expect **hours to days**, not 80 minutes.

### 7) Publication potential
- Best fit: **ACL/EMNLP (likely Findings/Main depending execution)** as a careful diagnostic/evaluation study.
- For NeurIPS/ICLR main-track strength, you'd need:
  - stronger invariance handling,
  - broader model coverage,
  - clearer causal claims beyond correlation diagnostics.

### 8) Concrete improvements
1. Add a **function-space companion metric** (logit/output behavior alignment), not only parameter-space cosine.
2. Use **r-ROME** explicitly and compare against original ROME.
3. Pre-register primary hypotheses and control multiple comparisons.
4. Add **token-overlap-controlled retrieval baselines** (BM25 vs dense vs random lexical match).
5. Validate on at least one **realistic benchmark** (e.g., WILD/QAEdit-style eval), not only CounterFact.
6. Reduce scope for feasibility: start with fixed-layer study, then expand to sweep.
7. Report **seed variance + bootstrap CIs + effect sizes**, not only p-values.
8. Include a small controlled setting where true influential data is known, to calibrate TDA trustworthiness.

---

**Sources**
- ROME (NeurIPS 2022): https://arxiv.org/abs/2202.05262
- MEMIT (ICLR 2023): https://arxiv.org/abs/2210.07229
- Does Localization Inform Editing? (NeurIPS 2023): https://arxiv.org/abs/2301.04213
- Rebuilding ROME / r-ROME (2024): https://arxiv.org/abs/2403.07175
- Ripple effects benchmark (TACL 2024): https://doi.org/10.1162/tacl_a_00644
- Mirage of Model Editing (ACL 2025): https://aclanthology.org/2025.acl-long.745/
- Superficial Editing analysis (ACL 2025): https://aclanthology.org/2025.acl-long.868/
- STEAM (Findings EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.585/
- Scalable Influence & Fact Tracing (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/65798a76cc176c29b6bfefe84b0a03ff-Abstract-Conference.html
- DDA TDA method (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.782/
- Influence functions (ICML 2017): https://proceedings.mlr.press/v70/koh17a.html
- K-FAC (ICML 2015): https://proceedings.mlr.press/v37/martens15.html
- EasyEdit runtime/memory table: https://github.com/zjunlp/EasyEdit
- GPT-2-XL config (48 layers): https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json

## 评分

6/10
