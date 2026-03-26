# Experiment Critique Report

## Overall Assessment
- **Score**: 5 / 10
- **Core Assessment**: The experimental setup is solid for the core TECS measurement, with well-designed null baselines and appropriate statistical methodology. However, the paper currently has more PENDING experiments than completed ones, and several critical experiments (positive control, g_M quality, cross-model) that directly support the main claims are not yet available. The pilot scale (N=100) is small for a NeurIPS paper making broad structural claims.
- **Would this dimension cause reject at a top venue?**: Yes, in current form — too many PENDING results for critical experiments. After completion, likely No.

## Issues (by severity)

### [Critical] Positive control experiments are PENDING but essential
- **Location**: Experiments 4.6
- **Problem**: The positive control experiments (Tiers 1-3) are the paper's most important methodological contribution — they distinguish "metric failure" from "informative null result." All three tiers are PENDING. Without them, a reviewer can legitimately argue that TECS ~ 0 simply means the metric doesn't work.
- **Simulated reviewer phrasing**: "The paper proposes positive controls to validate TECS but does not present any results. The claim that the null result is 'informative about knowledge geometry' rests entirely on these controls. Without them, I cannot evaluate whether TECS is measuring anything meaningful."
- **Suggested fix**: This is a must-complete experiment before submission. Prioritize Tier 2 (toy model) as it provides the strongest evidence.

### [Critical] Sample size (N=100) insufficient for structural claims
- **Location**: All experimental sections
- **Problem**: All reported results use N=100 pilot facts. For claims about "structural properties" of knowledge geometry, N=200 (or more) with proper power analysis would be expected at NeurIPS. The pilot is informative but underpowered for some measurements (e.g., principal angle p=0.084 at k=10 is suggestive but non-significant).
- **Simulated reviewer phrasing**: "The results are based on 100 CounterFact facts. Given the high dimensionality of the parameter space (d_k * d_v = 2.56M), it is unclear whether 100 observations are sufficient to characterize subspace geometry. A power analysis would be helpful."
- **Suggested fix**: Complete full-scale 200-fact experiments. Add power analysis or justify N=200 sufficiency.

### [Major] Ablation study is entirely PENDING
- **Location**: Experiments 4.8
- **Problem**: The four-axis ablation (top-k, weighting, loss, gradient scope) is entirely PENDING. Without it, the paper cannot claim the null result is robust to methodological choices.
- **Suggested fix**: Complete at least top-k and weighting ablations before submission.

### [Major] Cross-model validation is PENDING
- **Location**: Experiments 4.9
- **Problem**: The paper makes claims about "structural properties of transformer knowledge storage" based on a single model (GPT-2-XL). Without GPT-J-6B results, the generalization claim is unsupported.
- **Simulated reviewer phrasing**: "The authors claim the incommensurability is a structural property of transformers, but test only GPT-2-XL. At minimum, one additional model family is needed."
- **Suggested fix**: Complete GPT-J-6B experiments or at minimum Pythia-410M (computationally cheaper).

### [Major] MEMIT analysis has only N=30 facts
- **Location**: Experiments 4.5
- **Problem**: The MEMIT comparison uses only 30 facts with a simplified implementation. N=30 is insufficient for reliable effect size estimation (d ~ 0.63 has very wide CI at N=30).
- **Suggested fix**: Scale MEMIT to 200 facts with proper implementation and report confidence intervals.

### [Minor] Missing table/figure for TECS distribution
- **Location**: Experiments 4.2
- **Problem**: The paper describes the TECS distribution verbally but does not include a histogram or density plot comparing real TECS vs. Null-A. This is a standard visualization for null result papers.
- **Suggested fix**: Add a figure showing overlapping distributions of real TECS and Null-A baselines.

### [Minor] No failure case analysis
- **Location**: Experiments
- **Problem**: The paper does not examine individual facts with the highest/lowest TECS values. Are there ANY facts where TECS is notably positive? What characterizes them?
- **Suggested fix**: Add a brief analysis of the top-10 and bottom-10 TECS facts — do they have distinct properties (e.g., common relations, specific categories)?

## Strengths
- Five null baselines with Bonferroni correction is a strength.
- The whitening decomposition (H6) is a clever experiment that eliminates a natural alternative explanation.
- The cross-projection asymmetry analysis provides genuine insight beyond "they don't align."

## Summary Recommendations
The experimental framework is well-designed, but execution is incomplete. The paper cannot be submitted until positive control, full-scale core TECS, and at least one cross-model experiment are complete. The PENDING placeholders should be treated as blocking tasks.
