# Soundness Critique Report

## Overall Assessment
- **Score**: 6 / 10
- **Core Assessment**: The core methodology is straightforward and correctly applied. However, there are significant logical gaps in the causal reasoning chain: the paper claims "structured incommensurability" as a property of knowledge geometry, but the evidence cannot distinguish this from an artifact of the specific attribution method (BM25) and its known low quality for LLMs. The rank-one decomposition derivation has a gap between the per-sample formula and the aggregated TECS used in practice.
- **Would this dimension cause reject at a top venue?**: Possibly — the "attribution quality confound" is a significant soundness concern that the paper acknowledges but cannot fully resolve without the PENDING experiments.

## Issues (by severity)

### [Critical] Causal claim outstrips evidence — attribution quality confound
- **Location**: Throughout; especially Introduction P3, Conclusion
- **Problem**: The paper claims "structured geometric incommensurability between editing and attribution subspaces" as a finding about knowledge geometry. But with eff-dim = 1.2 for attribution (PC1 = 91%), it is equally plausible that BM25-based attribution is simply too crude to produce meaningful directions. The paper acknowledges this (Section 4.7 g_M quality analysis) but the PENDING experiments that would resolve it are not yet available. Without these results, the claim is under-supported.
- **Simulated reviewer phrasing**: "The authors report that attribution gradients have effective dimensionality 1.2 — this means essentially all attribution directions point the same way. The 'incommensurability' finding may simply reflect that BM25-based TDA produces degenerate gradients, not that editing and attribution access different knowledge structures. The g_M quality analysis (Section 4.7) is critical to the paper's claims but all results are PENDING."
- **Suggested fix**: (1) Tone down claims in current text to conditional: "Under BM25-based attribution, we observe structured incommensurability..." (2) Clearly state that the g_M quality analysis results will determine whether the finding is about knowledge geometry or attribution quality. (3) Consider moving the attribution quality analysis earlier (Section 4.3 or 4.4) to address the confound before building on it.

### [Critical] Rank-one decomposition applies to single-sample TECS, not aggregated TECS
- **Location**: Method 3.3
- **Problem**: Eq. 4 decomposes TECS for a single training sample $z_i$, but the actual TECS measured in experiments uses the aggregated gradient $g_M = \sum_i w_i \nabla L$. The decomposition's predictions (SNR scaling, noise floor) may not carry over to the aggregated version. The paper does not address this gap.
- **Simulated reviewer phrasing**: "The rank-one decomposition in Eq. 4 assumes a single training sample's gradient, but the experiments use an aggregated gradient over top-K samples. The aggregation step is non-trivial — it involves BM25 weighting and normalization — and the theoretical predictions may not hold for the aggregate."
- **Suggested fix**: Add a paragraph explicitly connecting the per-sample decomposition to the aggregated metric. Either: (a) show that the aggregation preserves the decomposition structure (under appropriate assumptions), or (b) acknowledge that the decomposition applies to the per-sample version and the aggregation introduces additional averaging that could weaken or strengthen the signal.

### [Major] Principal angle interpretation is misleading
- **Location**: Experiments 4.3, Table 3
- **Problem**: The p-values for principal angle comparison (0.084, 0.989, 1.000) show that editing-attribution angles are NOT significantly smaller than random. The paper interprets this as "confirms structural separation." But the correct interpretation is weaker: the editing and attribution subspaces are no more aligned than random subspaces. This does NOT mean they are "structured" — it means we cannot distinguish them from random. The "structured" claim comes from the dimensionality asymmetry, not the principal angles.
- **Simulated reviewer phrasing**: "The principal angle analysis shows the subspaces are not MORE aligned than random, but the authors claim 'structured incommensurability.' Random incommensurability would also produce these p-values. The 'structured' aspect comes from the dimensionality analysis, and the paper should be clearer about which evidence supports which claim."
- **Suggested fix**: Separate the "incommensurability" claim (supported by TECS + principal angles) from the "structured" claim (supported by dimensionality asymmetry + cross-projection asymmetry). The principal angles show the gap exists; the dimensionality analysis shows it has structure.

### [Major] MEMIT analysis uses simplified implementation
- **Location**: Experiments 4.5
- **Problem**: The MEMIT comparison uses "identity covariance" (simplified MEMIT), not the actual MEMIT algorithm. The cross-layer d ~ 0.63 could be an artifact of the simplification. The paper notes this limitation only in the research documents, not in the paper text.
- **Suggested fix**: Explicitly state that the MEMIT analysis uses a simplified implementation and that full MEMIT results are pending. Note whether identity covariance would over- or under-estimate alignment.

### [Minor] Missing variance/std for key measurements
- **Location**: Tables 2, 3; Cross-projection results
- **Problem**: Effective dimensionality, principal angles, and cross-projection ratios are reported as point estimates without confidence intervals or variance measures. For N=100 facts, bootstrapping these would be straightforward.
- **Suggested fix**: Add bootstrap 95% CIs for all key geometric measurements.

## Strengths
- Five null baselines with Bonferroni correction is rigorous for the core TECS measurement.
- The dual-outcome design prevents p-hacking (negative path was pre-registered).
- The positive control design (Section 3.6) is methodologically sound and addresses the most important alternative explanation.

## Summary Recommendations
The paper's soundness depends critically on the PENDING experiments (positive control, g_M quality analysis). Without these, the causal chain from "TECS ~ 0" to "structured incommensurability of knowledge geometry" has a significant gap — the attribution quality confound. The paper should be more explicit about which claims require the PENDING data and which are already supported.
