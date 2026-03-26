# Reproducibility Critique Report

## Overall Assessment
- **Score**: 7 / 10
- **Core Assessment**: The paper provides good high-level reproducibility information (model, dataset, layer, framework, seed). The core TECS computation is simple and well-specified. However, several implementation details are missing, and the attribution pipeline has unspecified parameters that could significantly affect results.
- **Would this dimension cause reject at a top venue?**: No — the missing details are typical for an initial submission and can be addressed in camera-ready.

## Issues (by severity)

### [Major] Attribution pipeline underspecified
- **Location**: Experiments 4.1 (Attribution paragraph), Method 3.2 (Eq. 3)
- **Problem**: The attribution gradient computation has several unspecified details: (1) Which WikiText/Wikipedia version and split? (2) How are documents segmented for BM25 retrieval? (3) What is the document length distribution? (4) How is the "object token" defined when it spans multiple tokens? (5) What is the exact BM25 configuration (k1, b parameters)? These choices could significantly affect g_M and thus TECS.
- **Suggested fix**: Add appendix with full attribution pipeline specification. At minimum, specify corpus size, document segmentation, and BM25 parameters in main text.

### [Major] EasyEdit version/commit not specified
- **Location**: Experiments 4.1
- **Problem**: "We use EasyEdit with ROME" — EasyEdit has had multiple versions with different ROME implementations. The paper mentions a "pinned commit hash" in the research documents but does not include it in the paper text.
- **Suggested fix**: Include the exact EasyEdit commit hash or version number.

### [Minor] Gradient computation memory management unspecified
- **Location**: Experiments 4.1
- **Problem**: Computing per-sample gradients for GPT-2-XL in FP16 at layer 17's MLP weight (6400 x 1600 = 10.24M parameters) requires ~40MB per gradient. For top-K=20 samples, this is 800MB per fact. The paper doesn't discuss memory management (sequential vs. batch gradient computation, gradient accumulation).
- **Suggested fix**: Add brief note on gradient computation approach (likely sequential per-sample).

### [Minor] SVD computation details missing
- **Location**: Method 3.5
- **Problem**: For the stacked direction matrix (N x 10.24M for N=100), full SVD is impractical. The paper likely uses truncated SVD or PCA but doesn't specify.
- **Suggested fix**: Specify whether full SVD, truncated SVD, or randomized SVD is used, and the number of components retained.

### [Minor] Covariance matrix estimation
- **Location**: Experiments 4.4 (Whitening decomposition)
- **Problem**: "Compute covariance matrix C from 100 WikiText samples at layer 17." Only 100 samples for a 1600x1600 covariance matrix (rank-deficient). This may affect the whitening experiment.
- **Suggested fix**: Report the number of samples used for covariance estimation and note whether regularization was applied.

## Strengths
- Random seed fixed at 42 and consistently reported.
- Statistical methodology (bootstrap, Bonferroni, Cohen's d) is standard and reproducible.
- Hardware clearly specified (single RTX 4090).
- CounterFact is a well-known public benchmark.
- TECS itself is a simple formula (cosine similarity) that is trivially reproducible.

## Summary Recommendations
The core experiment (TECS = cosine similarity between ROME delta and aggregated gradient) is highly reproducible. The main reproducibility risk is in the attribution pipeline, which has several unspecified parameters. An appendix with full pipeline details and code release would resolve all concerns.
