# External AI Review --- P7 --- TECA

## Overall Impression
A methodologically rigorous study asking a genuinely novel question. The paper's greatest strength is the dual-outcome pre-registration and five null baselines. The greatest concern remains the same as P3: the BM25 degeneracy confound is unresolved, and the positive control --- the paper's most important methodological contribution --- exists only as a design, not as results.

## Blind Spot Report
- **Blind spot 1: The paper assumes ROME edits reflect "knowledge location" but ROME itself may be an artifact.** ROME uses causal tracing to select layer 17, but causal tracing has been shown to be sensitive to implementation details (e.g., corruptions method, which tokens are restored). If layer 17 is not truly the "knowledge layer," the entire comparison is between a potentially arbitrary editing layer and attribution gradients, which makes the null result less informative.
- **Blind spot 2: The 34:1 ratio is not surprising given the mathematical structure.** ROME computes one rank-one update per fact, and these updates are explicitly designed to be different from each other (minimizing cross-talk). Of COURSE they span many dimensions. Attribution gradients, on the other hand, are aggregations of BM25-weighted samples that share a common retrieval mechanism. The asymmetry may be trivially explained by the mathematical design of each method, not by knowledge geometry.
- **Blind spot 3: The paper treats "cosine similarity ~ 0 in high dimensions" as a finding, but this is the default.** The theoretical decomposition addresses this partially, but the paper could benefit from explicitly computing the expected TECS under a simple null model (e.g., ROME directions are random rank-one matrices, attribution directions are rank-one matrices with shared PC1) and showing the observed TECS is consistent or inconsistent with this null.

## Strengths
- Five null baselines with Bonferroni correction exceeds typical rigor in this area.
- The whitening decomposition experiment is clever and provides genuine mechanistic insight.
- Honest handling of null results with pre-registered dual-outcome design.
- The MEMIT cross-layer finding (d ~ 0.63) adds constructive depth.

## Weaknesses / Concerns
- Positive control is designed but not executed.
- The connection to Hase et al. is indirect: they study localization vs. editing success, this paper studies attribution geometry vs. editing geometry. The link requires assuming localization ~ attribution, which is not established.
- The paper does not adequately explain why the editing subspace is 40-dimensional. Is this a property of CounterFact facts or of ROME's C^{-1} decorrelation? The H6 experiment shows whitening isn't the sole cause of TECS ~ 0, but doesn't tell us about the editing dimensionality.

## Simpler Alternative Challenge
A simpler paper: (1) Compute cosine similarity between ROME edits and IF gradients. (2) It's zero. (3) Report the eigenvalue spectra. (4) Show the toy model positive control works. This 4-page analysis paper (no six-component framework, no theoretical decomposition) would convey 80% of the insight and be more convincing because it would be complete.

## Specific Recommendations
1. Run the toy model positive control before anything else --- it's CPU-only and takes 10 minutes.
2. Add a "mathematical null model" that predicts the 34:1 ratio from ROME's and BM25's mathematical structure alone, without reference to knowledge geometry.
3. Consider moving the six-component framework to the appendix and presenting only the 3 most informative components in the main paper (dimensionality, cross-projection, whitening) to save space for complete experimental results.

## Score
6.5 / 10 --- Borderline at NeurIPS. The question is novel and the methodology is rigorous, but too many PENDING experiments. With completed experiments, I would vote 7-7.5.
