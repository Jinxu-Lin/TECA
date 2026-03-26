# Novelty Critique Report

## Overall Assessment
- **Score**: 7 / 10
- **Core Assessment**: The paper asks a genuinely novel question — whether editing and attribution directions share geometric structure in parameter space — that no prior work has addressed. The TECS metric and six-component framework are original contributions. However, the novelty claim relies heavily on the "first to compare" framing, and the actual technical innovations (cosine similarity, SVD-based subspace analysis) are standard tools applied to a new setting.
- **Would this dimension cause reject at a top venue?**: No — the novelty of the question itself is strong enough, provided the positive control validates the approach.

## Issues (by severity)

### [Major] "First to..." claims need stronger differentiation
- **Location**: Introduction, P4 contribution list
- **Problem**: The paper claims to be the "first systematic comparison" of editing and attribution directions. While this appears true, the closest prior work — Hase et al. (2023) on localization-editing disconnect and RepT on representation-space attribution — is not differentiated sharply enough. A reviewer could argue that Hase et al. already showed editing and localization don't align, and this paper merely measures a different flavor of the same disconnect.
- **Simulated reviewer phrasing**: "The finding that ROME editing directions and TDA gradients don't align is unsurprising given Hase et al.'s finding that localization and editing are disconnected. What new insight does the geometric characterization provide beyond confirming this disconnect at a different level of analysis?"
- **Suggested fix**: Add explicit paragraph in Introduction distinguishing parameter-level geometric characterization from Hase et al.'s behavioral-level disconnect. Emphasize: Hase et al. showed WHAT (the disconnect exists), while this paper shows WHY (structured incommensurability with quantifiable dimensionality asymmetry).

### [Major] Theoretical decomposition novelty overstated
- **Location**: Method 3.3
- **Problem**: The rank-one decomposition (Eq. 4) is a direct algebraic consequence of substituting the ROME formula into cosine similarity. While the resulting predictions (SNR scaling) are useful, calling this a "theorem" and listing it as a separate contribution (C5) inflates its novelty. A reviewer may view this as "plugging in the formula and simplifying."
- **Simulated reviewer phrasing**: "The rank-one decomposition is a straightforward algebraic expansion, not a theorem. The SNR prediction is interesting but follows directly from standard high-dimensional geometry."
- **Suggested fix**: Frame the decomposition as an "analytical prediction" rather than a "theorem." Emphasize the falsification value (the predictions SHOULD hold, but don't) rather than the derivation itself.

### [Minor] Missing recent related work
- **Location**: Related Work
- **Problem**: The Related Work section cites several works with 2024-2025 dates that appear plausible but need verification. More importantly, any work from the last 6 months on parameter-space analysis of knowledge in LLMs should be cited. The section does not mention recent work on knowledge neurons (Dai et al., 2022) or factual association analysis.
- **Suggested fix**: Add Knowledge Neurons (Dai et al., 2022), MEMIT scaling analysis, and any post-2024 work on geometric properties of LLM parameters.

## Strengths
- The question is genuinely novel — no prior work compares editing and attribution at the parameter geometry level.
- The dual-outcome design (pre-registered positive/negative paths) is a methodological strength that demonstrates intellectual honesty.
- The six-component framework is a systematic approach, not an ad hoc collection of analyses.

## Summary Recommendations
The novelty is solid at the question level but should be more carefully calibrated at the technical level. The main risk is a reviewer dismissing the null result as "expected given Hase et al." — the Introduction must proactively address this by articulating what geometric characterization adds beyond behavioral disconnect.
