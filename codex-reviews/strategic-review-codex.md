# Codex External Strategic Review — TECA

> External AI review via GPT-5.3-codex (Codex MCP). Non-blocking, for reference only.

**Overall judgment:** The direction is interesting but currently an **exploratory geometry study**, not yet a credible "independent validator" for TDA.

## 1. Gap reality and importance
- **Real and important:** Retraining-based evaluation is costly and unstable in large/non-convex regimes. Evidence: Revisiting the Fragility of Influence Functions shows Spearman can be misleading (their "miss-relation" example) and retraining trajectories can be unstable in non-convex settings.
- **Not fully solved, but partly addressed:** Newer work (e.g., DATE-LM, NeurIPS 2025) already tackles evaluation quality with unified, application-driven protocols and reports that no method dominates.
- **Inference:** TECA's gap should be framed narrowly as: "no accepted *parameter-space* cross-check for TDA," not "no non-retraining evaluation exists."

## 2. Scientific soundness of TECS + fatal flaws
- **Sound as hypothesis-generation**, weak as causal validation.
- **Fatal flaw A (proxy validity):** TECS assumes ROME edit direction reflects factual storage location. Hase et al. (2301.04213) reports localization signals are weakly predictive of edit success and often near-zero correlation.
- **Fatal flaw B (independence):** TDA gradients and ROME updates are both functions of the same model geometry; alignment can come from shared spectral/optimization structure, not true data causality.
- **Fatal flaw C (measurement invariance):** Raw cosine in weight space is parameterization-dependent; flattening rank-1 updates can create artifacts.
- **Fatal flaw D (data semantics):** If "training samples" are from CounterFact splits, that is not GPT-2 pretraining data attribution; interpretation collapses.

## 3. Probe adequacy
- Current probe is **okay for pilot**, **insufficient for core-claim validation**.
- Major upgrades needed:
  1. Replace/augment t-test with permutation + bootstrap CIs.
  2. Add placebo controls: non-edit layers, random rank-1 vectors with matched norms/spectrum.
  3. Stratify by edit success and relation type.
  4. Replicate across at least one additional editor (MEMIT or r-ROME) and one additional model.
  5. Use a model with known training corpus provenance (or clearly limit claim scope).

## 4. Contribution ceiling / venue
- **As currently designed:** Workshop or weak Findings.
- **If strengthened with robust controls + multi-model/editor generalization:** ACL/EMNLP Findings is realistic.
- **Main-track NeurIPS/ICLR:** unlikely unless TECS demonstrates strong predictive utility beyond existing benchmarks (e.g., DATE-LM tasks) and addresses identifiability.

## 5. Biggest underestimated risks
1. **CounterFact/provenance mismatch** (most serious practical risk).
2. **Circularity disguised as triangulation** (shared confounds).
3. **High-dimensional effect illusions** (small cosines become "significant").
4. **Researcher degrees of freedom** (layer choice, k, prompt/loss definitions, null design).
5. **Interpretation asymmetry:** Both positive and negative results are easy to over-interpret without stronger controls.

**Recommended reframe:** Position TECA as a **diagnostic lens on parameter-space compatibility** between attribution and editing, not as a standalone validator of TDA correctness.

## Sources
- https://arxiv.org/abs/2303.12922
- https://arxiv.org/abs/2301.04213
- https://arxiv.org/abs/2202.05262
- https://arxiv.org/abs/2210.07229
- https://arxiv.org/abs/2507.09424
- https://aclanthology.org/2025.findings-emnlp.775/
- https://aclanthology.org/2024.emnlp-main.1210/
- https://aclanthology.org/2024.findings-emnlp.903/
