# P3 Critique Summary

## Reject Risk Assessment

**Preliminary judgment**: Borderline — the paper has a strong conceptual contribution but critical experimental gaps (PENDING results). If all PENDING experiments are completed and support the claims, the paper is likely a poster accept. In current form with PENDING placeholders, it would be desk-rejected or receive weak reject.

**Aggregate scores**: Novelty 7, Soundness 6, Experiment 5, Presentation 7, Reproducibility 7. **Mean: 6.4/10**.

## Critical Issues (must fix before submission)

### Section: Experiments — Positive Control
- **Issue**: All three tiers of positive control experiments are PENDING. These are essential to the paper's central claim that the null TECS result is informative rather than a metric failure.
- **Tag**: needs-additional-experiments
- **Sources**: Experiment Critique [Critical], Soundness Critique [Major]

### Section: Experiments — Sample Size
- **Issue**: All results use N=100 pilot data. Full-scale N=200 experiments are PENDING.
- **Tag**: needs-additional-experiments

### Section: Experiments — Attribution Quality Confound
- **Issue**: The g_M quality analysis (within/between similarity, retrieval ablation, PC1 removal) is entirely PENDING. Without it, the paper cannot distinguish "knowledge geometry property" from "BM25 artifact."
- **Tag**: needs-additional-experiments
- **Sources**: Soundness Critique [Critical], Experiment Critique [Major]

## Major Issues (should fix before submission)

### Section: Method — Rank-One Decomposition Gap
- **Issue**: Eq. 4 decomposes per-sample TECS, but experiments use aggregated TECS. The connection is not established.
- **Tag**: rewrite-fixable
- **Source**: Soundness Critique [Critical]

### Section: Experiments — Cross-Model Validation
- **Issue**: Claims about "structural properties of transformers" based on single model. GPT-J-6B results PENDING.
- **Tag**: needs-additional-experiments
- **Source**: Experiment Critique [Major]

### Section: Experiments — MEMIT Scale
- **Issue**: MEMIT analysis uses only N=30 with simplified implementation.
- **Tag**: needs-additional-experiments
- **Source**: Experiment Critique [Major]

### Section: Throughout — No Figures
- **Issue**: Zero figures in a geometry paper. Need conceptual diagram, TECS distribution, eigenvalue spectra, cross-projection visualization.
- **Tag**: rewrite-fixable
- **Source**: Presentation Critique [Major]

### Section: Introduction — Too Long + Contribution Inflation
- **Issue**: ~2 pages with 5 contributions. Reduce to ~1.5 pages and 3 contributions.
- **Tag**: rewrite-fixable
- **Source**: Presentation Critique [Major]

### Section: Throughout — Conditional Claims
- **Issue**: Claims should be conditional on BM25 attribution until g_M quality analysis resolves the confound.
- **Tag**: rewrite-fixable
- **Source**: Soundness Critique [Critical]

### Section: Experiments — Principal Angle Interpretation
- **Issue**: p-values > 0.05 at all k; paper over-interprets as "structured." Separate "incommensurability exists" from "incommensurability is structured."
- **Tag**: rewrite-fixable
- **Source**: Soundness Critique [Major]

## Minor Issues

| Section | Issue | Tag |
|---|---|---|
| Method 3.2 | Attribution pipeline underspecified (BM25 params, corpus details) | rewrite-fixable |
| Experiments 4.1 | EasyEdit commit hash not included | rewrite-fixable |
| Experiments 4.5 | MEMIT uses simplified implementation — state this explicitly | rewrite-fixable |
| Throughout | "34:1" vs "40:1" inconsistency | rewrite-fixable |
| Abstract | SNR formula unusual in abstract | rewrite-fixable |
| Experiments 4.2 | No individual fact analysis (top/bottom TECS) | rewrite-fixable |
| Experiments 4.3 | Missing confidence intervals for geometric measurements | rewrite-fixable |

## Priority Order for P4 Editing

1. **Complete PENDING experiments** (blocking — cannot submit without positive control + full-scale)
2. **Add figures** (4-5 figures are essential for a geometry paper)
3. **Tone down claims** to conditional pending g_M quality analysis
4. **Trim Introduction** and reduce contribution list
5. **Fix decomposition gap** (per-sample vs. aggregated TECS)
6. **Fix principal angle interpretation**
7. **Add pipeline specification details**
8. **Fix minor inconsistencies**
