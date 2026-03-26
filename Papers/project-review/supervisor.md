# Supervisor Review Report

## Overall
- **Quality Score**: 7 / 10
- **Core Assessment**: This is a well-conceived research project that asks a genuinely novel question (editing-attribution geometric relationship) and provides an honest, rigorous initial answer (structured incommensurability under BM25). The methodology is sound, the statistical rigor exceeds most comparable papers, and the dual-outcome design demonstrates intellectual maturity. The main gap is execution: critical experiments remain PENDING.
- **Submission Readiness**: Needs Improvement (complete PENDING experiments before submission)
- **AC Decision Simulation**: Borderline Accept --- "Novel question, clean methodology, but incomplete experimental validation. Strong poster if PENDING experiments support claims."

## Dimension Scores

| Dimension | Score | Brief |
|-----------|-------|-------|
| Problem Quality | 8/10 | Genuinely novel, well-motivated gap between editing and attribution communities |
| Method-Problem Fit | 8/10 | TECS directly measures the gap; six-component framework provides comprehensive characterization |
| Method Rigor | 7/10 | Sound methodology; rank-one decomposition is useful but algebraically straightforward |
| Experiment Sufficiency | 5/10 | Pilot data convincing for direction; many critical experiments PENDING |
| Presentation | 7/10 | Clear writing, good structure; needs figures |
| Overall Contribution | 7/10 | Advances understanding of knowledge geometry; practical impact limited |

## Contribution-Evidence Audit

| Claim | Argumentation | Validation | Evidence Strength | Assessment |
|-------|-------------|-----------|-------------------|-----------|
| C1: TECS metric + decomposition | Section 3.2-3.3 | Table 1, Appendix B | Medium (decomposition untested on toy model) | Solid methodology; decomposition value depends on positive control |
| C2: Structured incommensurability | Section 3.5 | Tables 2-3, cross-projection | Medium (confounded by BM25) | Core finding needs g_M quality analysis to resolve confound |
| C3: Framework + positive control | Sections 3.5-3.6 | Partial (pilot only) | Weak (positive control PENDING) | Framework is complete but validation incomplete |

## Risk Assessment: Most Likely Reviewer Challenges

1. **Challenge**: "The null result is trivially expected --- of course cosine similarity in 10M-dimensional space is zero."
   **Current Defense**: Adequate --- the rank-one decomposition shows TECS SHOULD be nonzero under linear memory assumptions, and the five null baselines calibrate against dimensional effects.
   **Suggested Supplement**: The positive control (Tier 2) would definitively address this.

2. **Challenge**: "BM25 attribution is too crude; this is measuring retrieval quality, not knowledge geometry."
   **Current Defense**: Insufficient --- the paper acknowledges the confound but cannot resolve it without the PENDING retrieval ablation.
   **Suggested Supplement**: Complete Section 4.7 experiments.

3. **Challenge**: "Single model (GPT-2-XL) is insufficient for structural claims about transformers."
   **Current Defense**: Insufficient --- no cross-model results yet.
   **Suggested Supplement**: At minimum, run Pythia-410M as a lightweight cross-model check.

4. **Challenge**: "The Hase et al. connection is tenuous --- they study localization vs. editing, not attribution vs. editing."
   **Current Defense**: Adequate (after P4 revision) --- the paper now carefully distinguishes the two questions and frames the connection as "both point to different operations accessing different structures."

5. **Challenge**: "What is the practical implication? If editing and attribution don't align, so what?"
   **Current Defense**: Adequate --- implications for cross-paradigm tools, TDA evaluation, and linear memory model validity are discussed.

## Best Practices Checklist
- [x] Reproducible (details complete --- model, layer, seed, hardware, statistical method)
- [x] Statistical significance reported (Cohen's d, bootstrap CIs, Bonferroni)
- [ ] Ablations cover key components (PENDING)
- [x] Compared with recent work (Hase et al. 2023, RepT, MDA, RIF)
- [x] Limitations explicitly stated (Section 5)
- [ ] Code/data release planned (stated as "upon acceptance")
- [x] Ethics considered (N/A --- characterization study)

## Improvement Suggestions (prioritized)
1. Complete positive control experiment (Tier 2: toy model) --- highest priority, resolves "metric doesn't work" objection
2. Complete g_M quality analysis (retrieval ablation) --- resolves BM25 confound
3. Add 4-5 figures (eigenvalue spectra, TECS distribution, cross-projection diagram)
4. Run at least one cross-model check (Pythia-410M is cheapest)
5. Scale MEMIT to 200 facts with proper implementation
