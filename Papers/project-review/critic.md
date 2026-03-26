# Critic Review Report

## Overall
- **Attack Strength**: 6 / 10
- **Core Weakness**: The paper's main claim (structured incommensurability as a property of knowledge geometry) cannot be distinguished from an artifact of degenerate BM25-based attribution (eff-dim=1.2) without the PENDING g_M quality experiments. The positive control --- the paper's most important methodological defense --- is also PENDING.
- **Most Likely Reject Reason**: "The null result is unsurprising and the geometric characterization may reflect BM25 retrieval degeneracy rather than knowledge geometry."

## Issues (by severity)

### [Critical] Attribution degeneracy as confound
- **Location**: Throughout; especially Sections 4.3, 4.7
- **Problem**: With eff-dim = 1.2 for attribution gradients, the "34:1 asymmetry" is largely a statement about BM25 retrieval quality, not knowledge geometry. The paper acknowledges this but cannot resolve it without the PENDING g_M quality experiments.
- **Evidence**: PC1 captures 91% of attribution variance. Cross-projection G-in-D = 17.3% is consistent with a random 1D subspace projected onto a 40D subspace.
- **Simulated Reviewer Attack**: "You show that BM25-retrieved gradients are nearly degenerate (eff-dim=1.2). Of course they don't align with ROME edits --- they barely align with anything. Your 'structured incommensurability' is an artifact of using the crudest possible attribution method. Run your analysis with Contriever or exact IF and I'll reconsider."
- **Suggested Fix**: Complete the retrieval ablation (Section 4.7) before submission. If eff-dim increases but TECS remains zero, the finding is robust.

### [Major] Positive control is PENDING
- **Location**: Sections 3.6, 4.6
- **Problem**: The paper's primary defense against "the metric doesn't work" is the positive control on a toy model. This experiment has not been run.
- **Simulated Reviewer Attack**: "How do I know TECS can detect alignment at all? You propose a positive control but don't report results. For all I know, TECS is insensitive to alignment even when it exists."
- **Suggested Fix**: Run the toy model experiment. This is the single highest-priority missing piece.

### [Major] "Structured" claim overinterpretation
- **Location**: Sections 4.3 (Principal Angles)
- **Problem**: Principal angles at all k are not significantly different from random baselines. The paper revised to acknowledge this, but still uses "structured incommensurability" as the core framing. The "structured" part rests on the dimensionality asymmetry alone, which could be an attribution artifact.
- **Simulated Reviewer Attack**: "You claim 'structured' incommensurability, but your principal angle analysis shows no significant difference from random. The only 'structure' is that one set of directions is 1-dimensional --- which could be a BM25 artifact."
- **Suggested Fix**: More carefully distinguish: "the incommensurability is confirmed (TECS ~ 0); the evidence for structure (dimensionality asymmetry) is suggestive but confounded by attribution quality."

### [Minor] MEMIT analysis underpowered
- **Location**: Section 4.5
- **Problem**: N=30 facts, simplified implementation. Effect size CI at N=30 is very wide.
- **Suggested Fix**: Scale to N=200 with proper MEMIT.

## Proxy Metric Gaming Check
- **Result**: Pass (N/A)
- **Analysis**: This is a characterization study, not a method paper. There is no metric to game --- the paper reports null results honestly with proper statistical methodology. The dual-outcome pre-registration is a positive sign.

## Missing Baselines / Ablations
1. Random subspace cross-projection baseline (for contextualizing 17.3% / 1.0%)
2. Exact influence function gradients (if computationally feasible) as an attribution upper bound
3. Pythia family (controlled scaling) as a lighter cross-model check

## "Kill Shot" Test
**Question**: "Given that BM25 retrieval produces nearly degenerate attribution gradients (eff-dim=1.2, PC1=91%), isn't your entire geometric characterization just measuring BM25's degeneracy? What happens with a non-degenerate attribution method?"

**Defense**: The paper has the right experiment designed (Section 4.7 retrieval ablation) but results are PENDING. Without them, this question is unanswerable.
