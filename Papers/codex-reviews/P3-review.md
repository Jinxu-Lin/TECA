# External AI Review — P3 — TECA

## Overall Impression
The paper asks a fresh and well-motivated question about the geometric relationship between knowledge editing and attribution in parameter space. The methodology is clean and the statistical rigor (five null baselines, Bonferroni correction, effect sizes) exceeds typical papers in this area. The biggest concern is that the paper may be a beautifully executed measurement of an artifact — BM25-based attribution gradients with eff-dim=1.2 are so degenerate that claiming anything about "knowledge geometry" from them is premature.

## Blind Spot Report
- **Blind spot 1: The attribution is not really "training data attribution"**. BM25 retrieval over WikiText finds lexically similar documents, not actual training data. GPT-2 was trained on WebText, not WikiText. The paper measures alignment between ROME edits and gradients of *proxy* training data retrieved by *lexical matching*. The "attribution" label implies a stronger connection to actual training influence than what is being computed.
- **Blind spot 2: Cosine similarity in d=10.24M dimensions is almost always near-zero**. In very high dimensions, random vectors are nearly orthogonal by concentration of measure. The paper's null baselines control for this, but the framing could acknowledge more explicitly that near-zero cosine similarity is the *default* in this space, not a finding.
- **Blind spot 3: The paper conflates two questions**. (1) "Do editing and attribution directions align?" and (2) "What is the geometric structure of each?" The null TECS answers (1), but the six-component analysis mostly answers (2) independently. The link between them (the dimensionality asymmetry *explains* why TECS ~ 0) is asserted but not formally shown.

## Strengths
- The dual-outcome pre-registration is excellent methodology and should become standard.
- The positive control design (especially Tier 2: toy model) directly addresses the strongest objection.
- The whitening decomposition experiment is clever and eliminates a plausible mechanistic explanation.
- Effect sizes (Cohen's d) rather than p-values as primary metrics is good practice.

## Weaknesses / Concerns
- The paper claims "parameter-level explanation for Hase et al. (2023)" but Hase et al. studied localization-editing disconnect (causal tracing vs. editing success), not attribution-editing disconnect. The paper bridges localization and attribution without justification — these are different operations.
- The theoretical decomposition's value is unclear. It predicts TECS > 0 "under the linear memory model," but the linear memory model is known to be a rough approximation. Showing the model fails is not surprising.
- With 91% of attribution variance in PC1, the cross-projection asymmetry (17.3% G-in-D) may simply reflect: attribution's PC1 has ~17% overlap with the 40-dimensional editing subspace, which is geometrically unremarkable for a 40D subspace in a 100D effective space.

## Simpler Alternative Challenge
A simpler version of this paper: (1) Compute TECS. (2) It's zero. (3) Show eigenvalue spectra of both direction sets. (4) Done. The six-component framework, theoretical decomposition, and positive control add rigor but also complexity. A reviewer might prefer the simpler version with more complete experiments over the comprehensive framework with PENDING results.

## Specific Recommendations
1. Rename "training data attribution gradients" to "proxy attribution gradients" or "retrieval-based gradients" to be honest about what is being measured.
2. Add a formal argument (not just assertion) connecting the dimensionality asymmetry to near-zero TECS. E.g.: if attribution has eff-dim k_A in a d-dimensional space, expected TECS under independent subspaces scales as sqrt(k_A * k_E / d^2).
3. Compute and report the expected cross-projection ratio for random subspaces of matching dimensions (40D and 1D in 100D space) as a baseline for the 17.3% / 1.0% figures.
4. Address the WikiText vs. WebText training data mismatch explicitly.

## Score
6 / 10 — Borderline at NeurIPS. Strong question and methodology, but the attribution quality confound and extensive PENDING results prevent acceptance in current form. With completed experiments and toned-down claims, could reach 7-8.
