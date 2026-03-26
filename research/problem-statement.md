---
version: "2.0"
created: "2026-03-17"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 2
iteration_minor: 0
---

# Problem Statement: TECA v2

## 1. Gap Definition

### 1.1 Gap Candidate List

| # | Candidate Gap | Derivation Path | Importance | Novelty | Tractability |
|---|--------------|-----------------|------------|---------|-------------|
| G1 | **Parameter-space geometric incommensurability between knowledge editing and training data attribution** — editing (ROME/MEMIT) and attribution (IF/TRAK) operate on the same weight matrices but occupy geometrically distinct subspaces, with no existing characterization of this disconnect | Hase et al. (2023) localization-editing disconnect + our pilot TECS d=0.050 + Zhang et al. (2501.18887) unified attribution framework reveals the editing-attribution boundary is the least explored | High — resolves a foundational question about knowledge geometry | High — first parameter-space geometric characterization; no prior work connects these at the subspace level | High — pilot data already in hand; 1x4090 sufficient |
| G2 | **Structural dimensionality asymmetry as root cause of cross-paradigm failure** — editing spans ~40D (flat spectrum), attribution collapses to ~1D (PC1=91%), creating a 40:1 dimensional mismatch that may explain why cross-paradigm validation systematically fails across the field | Pilot subspace analysis + RepT (2510.02334) finding that representation-space attribution outperforms parameter-space attribution in LLMs + MDA (2601.21996) subspace IF framework | High — explains a class of failures, not just one | Very High — dimensionality asymmetry as a structural phenomenon is novel | High — quantifiable, testable, verifiable with positive control |
| G3 | **Missing theoretical prediction for when editing-attribution alignment should exist** — the linear associative memory model predicts TECS > 0 under specific conditions (rank-one decomposition), but no work has tested when these conditions hold vs. fail, nor what structural properties break them | TECS rank-one decomposition theorem + pilot falsification (theory predicts signal, reality shows none) + RIF (2506.06656) showing IF accuracy depends on effective dimension | Medium-High — bridges theory-practice gap in knowledge editing | High — first falsification-based analysis of linear associative memory assumptions | Medium — requires positive control construction |
| G4 | **Attribution method quality as confound in cross-paradigm comparison** — attribution eff-dim=1.2 (dominated by single direction) raises the question: is incommensurability a property of knowledge geometry, or an artifact of BM25-based attribution being too crude? | Pilot g_M analysis + MAGIC (2504.16430) showing attribution quality varies dramatically with method + SOURCE (2405.12186) providing superior gradients via unrolled differentiation | Medium — methodological validity concern | Medium — attribution method ablation is known to matter | High — can test with multiple attribution methods |

### 1.2 Selected Gap (Primary: G1+G2 combined)

**One-sentence**: Existing work treats knowledge editing and training data attribution as independent operations on shared parameters, but our pilot reveals they occupy geometrically incommensurable subspaces with a 40:1 dimensional asymmetry — a structural phenomenon that provides the first parameter-level explanation for the persistent localization-editing disconnect (Hase et al., 2023) and challenges the linear associative memory assumptions underlying ROME.

**Detailed analysis**:

The gap is of the "done but has fundamental flaw" type: the knowledge editing community assumes a linear associative memory model (W·k = v) where knowledge is stored as rank-one associations, while the TDA community assumes gradient directions reflect knowledge influence. Both communities implicitly assume their parameter-space directions are meaningful proxies for "knowledge location" — but neither has tested whether these two notions of "knowledge direction" share geometric structure.

Our pilot provides the first empirical test: they do not. TECS = cos(vec(delta_W_E), vec(g_M)) is indistinguishable from noise (d=0.050), and the underlying geometry reveals a radical structural asymmetry: editing directions distribute across ~40 dimensions (flat eigenvalue spectrum, condition number 2.0), while attribution directions collapse to effectively 1 dimension (PC1 captures 91%, condition number 33.2). This 40:1 dimensional mismatch is not merely "different directions in the same space" — it is a fundamental structural incompatibility.

**Why this matters beyond TECA**: The structured incommensurability finding has implications for:
1. Any attempt to use TDA to validate or explain model editing (e.g., "which training data was responsible for the knowledge that ROME edits?")
2. The viability of cross-paradigm knowledge tools (editing + attribution + localization)
3. The validity of the linear associative memory model: if W·k = v held precisely, TECS should be detectable (our rank-one decomposition predicts SNR ~ rho_k · rho_v · sqrt(d_k))

### 1.3 Root Cause Analysis

**Type**: Structural mismatch between two knowledge paradigms' implicit geometric assumptions

**Why chain (4 levels)**:

1. **Why is TECS ~ 0?** — Editing and attribution directions in R^{d_v x d_k} have near-zero cosine similarity.

2. **Why near-zero cosine similarity?** — The two direction sets occupy geometrically distinct subspaces: editing spans a ~40D manifold with flat spectrum; attribution collapses to a ~1D subspace dominated by a single principal component. Their subspace angles (min 56.8° at k=50) exceed random baselines.

3. **Why do they occupy different subspaces?** — Two hypotheses:
   - (a) **Operational asymmetry**: ROME computes a precision intervention (C^{-1}k* × (v* - Wk*)), which is a single rank-one update engineered to change one fact while minimally disturbing others. TDA gradients aggregate influence signals across many training samples, dominated by a common "knowledge-reading" direction that reflects how the model processes factual queries generally (PC1 = 91%).
   - (b) **Optimization vs. influence**: Editing solves an optimization problem (minimize perturbation subject to fact change), producing directions orthogonal to the current weight manifold. Attribution measures sensitivity of loss to weights, producing directions aligned with the steepest descent geometry. These are fundamentally different mathematical operations on the same space.

4. **Why is the operational asymmetry so extreme (40:1)?** — ROME's C^{-1} whitening decorrelates the key space, spreading edits across many principal components. But H6 rejection shows this is NOT the cause — even unwhitened TECS ~ 0. The real root cause is that BM25-based attribution produces gradients dominated by a single "retrieval relevance" direction (the BM25 weighting creates a near-degenerate aggregation), while ROME's rank-one update is computed from the specific factual association's key-value structure. **The attribution method's crude retrieval may artificially collapse the attribution subspace**.

**Oracle verification**: If an oracle gave us perfect attribution gradients (true per-fact influence on the exact parameters ROME modifies), would TECS become nonzero? This is testable via positive control (toy model with known ground truth). If even perfect attribution shows TECS ~ 0, the incommensurability is fundamental. If perfect attribution shows TECS > 0, the incommensurability is partially an attribution quality artifact.

### 1.4 Gap Three-Dimensional Evaluation

**Importance**: HIGH
- Affects all three knowledge paradigm communities (editing, attribution, localization)
- Explains an open puzzle (Hase et al., 2023) that has no existing parameter-level answer
- Becomes more important as knowledge editing scales (MEMIT, mass-editing) and attribution becomes standard practice
- The dimensional asymmetry finding (40:1) unlocks a new structural understanding beyond "they don't align"

**Novelty**: VERY HIGH
- No prior work compares editing and attribution at parameter-space geometry level (confirmed via Zhang et al. 2501.18887 survey — this boundary is unexplored)
- The structured incommensurability characterization (effective dimensionality, principal angles, cross-projection asymmetry) is methodologically novel
- The positive control experiment (testing when TECS CAN work) is a novel falsification-based approach

**Tractability**: HIGH
- Pilot data for 100 facts already exists; scaling to 200 is straightforward (~2h GPU)
- Single RTX 4090 is sufficient for GPT-2-XL and GPT-J-6B
- Positive control (toy model) is computationally cheap
- Attribution method ablation uses existing frameworks (EasyEdit, LogIX)

## 2. Research Questions

### 2.1 Main RQ

**RQ-Main**: Is the geometric incommensurability between knowledge editing directions and training data attribution directions in transformer MLP parameter space a fundamental structural property of how knowledge is organized, or is it contingent on specific methodological choices (editing method, attribution method, model scale)?

**Falsifiability**: If we find a configuration (attribution method, model, or editing method) where TECS is significantly positive (d > 0.3), the "fundamental structural" claim is falsified — incommensurability is contingent. If TECS ~ 0 across all configurations AND the positive control succeeds, the claim is supported.

**Predictive power**: The answer predicts (a) whether cross-paradigm knowledge tools can work in principle, (b) which properties of editing/attribution methods would need to change for alignment, (c) whether the linear associative memory model is a good description of real knowledge storage.

### 2.2 Sub-RQs

**RQ1 (Existence + Positive Control)**: Can TECS detect editing-attribution alignment in a controlled setting where alignment is known to exist (toy linear associative memory), and does it fail to detect alignment in real transformers?

*Rationale*: Addresses the "trivially expected" objection and the "measurement failure" alternative explanation simultaneously. If TECS works on toy model but fails on GPT-2-XL, the null result is informative about real knowledge geometry rather than metric failure.

**RQ2 (Structural Characterization)**: What is the geometric structure of the incommensurability — specifically, what are the effective dimensionalities, principal angles, and cross-projection profiles of editing vs. attribution subspaces, and how do these compare to random subspace baselines?

*Rationale*: Moves beyond "TECS ~ 0" to a rich geometric characterization that reveals WHY the incommensurability exists.

**RQ3 (Attribution Method Sensitivity)**: Does the attribution method quality (BM25 vs. TF-IDF vs. gradient-based retrieval; aggregation top-k; loss definition) significantly affect the incommensurability, and specifically, does the attribution subspace dimensionality (currently 1.2) increase with better retrieval?

*Rationale*: Directly tests whether the 40:1 asymmetry is a property of knowledge geometry or an artifact of crude attribution. The suspiciously low eff-dim=1.2 must be explained.

**RQ4 (Cross-Method Generalization)**: Does the incommensurability pattern hold across (a) editing methods (ROME single-layer vs. MEMIT multi-layer), (b) models (GPT-2-XL vs. GPT-J-6B vs. Pythia family), and (c) fact categories (CounterFact subsets)?

*Rationale*: NeurIPS requires generalization beyond a single model. MEMIT pilot (d~0.63) already suggests method matters. Cross-model is essential for any structural claim.

## 3. Attack Angle

### 3.1 Candidate Attack Angles (Summary)

| # | Angle | Root Cause Match | Feasibility |
|---|-------|-----------------|-------------|
| A1 | **Geometric incommensurability characterization with positive control** — Six-component analysis framework + toy model validation + attribution ablation | Direct — characterizes the structural mismatch from multiple geometric perspectives, with positive control proving the metric works | High — builds on pilot, 1x4090 |
| A2 | Representation-space TECS (inspired by RepT) — compute TECS in hidden state space instead of parameter space | Indirect — sidesteps root cause rather than characterizing it | Medium — novel but changes the question |
| A3 | Theoretical analysis of linear associative memory failure conditions | Addresses theory-practice gap specifically | Medium — may be too narrow for standalone contribution |

### 3.2 Selected Attack Angle: A1 (Geometric Incommensurability Characterization with Positive Control)

**Core idea** (2 paragraphs):

We conduct a systematic geometric characterization of the editing-attribution parameter-space relationship, centered on a novel positive control experiment that establishes TECS's validity before interpreting the null result. First, we construct a toy linear associative memory model where the ground-truth editing-attribution relationship is known by construction — if TECS detects alignment there, the null result in real transformers is informative about knowledge geometry rather than metric failure. This positive control is the critical methodological innovation that elevates the negative result from "our metric didn't work" to "knowledge editing and attribution access fundamentally different geometric structures."

Second, we deploy a six-component geometric analysis framework (TECS core measurement, subspace dimensionality, principal angles, cross-projection, whitening decomposition, multi-method comparison) augmented with an attribution method ablation that tests whether the 40:1 dimensional asymmetry is fundamental or an artifact of BM25 retrieval. The framework is applied across multiple models (GPT-2-XL, GPT-J-6B, Pythia-410M), multiple editing methods (ROME, MEMIT), and multiple attribution configurations, producing a comprehensive geometric atlas of knowledge incommensurability. The theoretical rank-one decomposition of TECS provides analytical predictions for the toy model that can be checked against the real model's failure mode.

**Root cause causal match**: The attack angle directly characterizes the structural mismatch (root cause level 2-3) through geometric analysis, tests whether attribution quality is a confound (root cause level 4), and establishes measurement validity through the positive control (eliminating the "bad metric" alternative).

**Probe result support**: All six components have been piloted on 100 facts with clear results. The positive control is the key addition — it was identified as a critical gap in the design review.

### 3.3 Limitations and Risks

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| "Trivially expected" reviewer objection | HIGH | HIGH | Positive control + theoretical prediction (rank-one decomposition predicts TECS > 0 under linear memory assumptions) establish that alignment SHOULD exist under well-defined conditions |
| Attribution eff-dim=1.2 is an artifact of BM25 | MEDIUM | HIGH | Attribution method ablation (BM25 / TF-IDF / gradient-based / oracle top-k) will quantify this; if eff-dim increases, characterize the residual incommensurability |
| Toy model positive control is too trivial | MEDIUM | MEDIUM | Design the toy model to be non-trivial: multi-layer, nonlinear activation, realistic d_k/d_v ratio; show TECS transitions from > 0 to ~ 0 as model complexity increases |
| Cross-model results are inconsistent | LOW | HIGH | Use Pythia family for controlled scaling analysis (same architecture, different sizes); GPT-J for architecture diversity |
| Geometric framework is descriptive not explanatory | MEDIUM | MEDIUM | The rank-one decomposition provides an analytical explanation (key-alignment vs value-alignment decomposition); the positive control provides causal evidence |
| NeurIPS reviewers reject negative result papers | MEDIUM | HIGH | Frame as "geometric characterization" paper with positive methodological contributions (TECS, framework, positive control design), not as "we tried X and it didn't work" |

## 4. Probe Results Integration

### 4.1 Verified Hypotheses

| Hypothesis | Evidence | Signal Strength |
|-----------|----------|-----------------|
| H7: Structured incommensurability | Min principal angles > random baselines at all k; cross-projection asymmetry (17.3% vs 1.0%) | STRONG — clear geometric structure, not random noise |
| H_MEMIT (partial): Multi-layer editing shows different alignment | Cross-layer d ~ 0.63 (detectable), matched-layer d >> 6.0 (trivial) | MODERATE — 30 facts only, needs scaling |

### 4.2 Unverified Hypotheses

| Hypothesis | Why Unverified | Verification Plan |
|-----------|---------------|-------------------|
| H2: Rank-one decomposition empirical validity | Requires positive TECS signal to test correlation; can be tested on toy model positive control | Toy model experiment: compute both full TECS and decomposed TECS, measure correlation |
| Attribution method sensitivity | Only BM25 tested; eff-dim=1.2 may be an artifact | Attribution ablation (BM25 / TF-IDF / gradient / varying top-k) |
| Cross-model generalization | Only GPT-2-XL | GPT-J-6B + Pythia-410M experiments |
| TECS positive control validity | No ground-truth setting tested | Toy linear associative memory with known editing-attribution relationship |

### 4.3 Unexpected Findings

| Finding | Potential Impact |
|---------|-----------------|
| Attribution eff-dim = 1.2 (PC1 = 91%) | Suggests BM25-based attribution may be collapsing all facts to a single "relevance" direction. This is either a deep finding about how TDA gradients are structured in LLMs (consistent with RepT's finding that parameter-space TDA has low SNR), or an attribution quality artifact. **Must be resolved — it is a potential confound for the entire incommensurability claim.** |
| Cross-projection asymmetry (G-in-D = 17.3%, D-in-G = 1.0%) | The one-directional overlap suggests attribution's narrow subspace marginally intersects the broad editing subspace, but not vice versa. This asymmetry may reflect that attribution captures a "generic knowledge access" direction that happens to weakly overlap with part of the editing manifold. |
| MEMIT cross-layer d ~ 0.63 | Multi-layer editing partially bridges the gap. This is a constructive finding — it suggests the editing-attribution relationship is partially recoverable with richer editing methods, which strengthens the "fundamental geometry" narrative (the gap is real but not absolute). |

### 4.4 Probe Limitations

| Limitation | Impact | Mitigation in Full Experiments |
|-----------|--------|-------------------------------|
| N=100 facts (pilot) | Sufficient for direction, underpowered for subtle effects | Scale to N=200 for full experiments |
| Single model (GPT-2-XL) | Cannot claim generality | Add GPT-J-6B + Pythia-410M |
| Single attribution method (BM25) | Cannot disentangle geometry from attribution quality | Attribution method ablation |
| No positive control | Cannot prove TECS metric works | Toy model positive control |
| MEMIT simplified (identity covariance) | May not reflect true MEMIT behavior | Use proper MEMIT with full covariance |
| No theoretical baseline for expected TECS | Cannot quantify how far from "should be" | Rank-one decomposition predictions on toy model |

## 5. Metadata

- Based on Startup output version: project.md v1.0 (assimilated)
- Probe results source: `Codes/_Results/probe_result.md`
- GPU resource constraint: 1x NVIDIA RTX 4090 (24GB VRAM), remote server (xuchang3). GPT-2-XL fits in FP16; GPT-J-6B needs careful memory management; Pythia-410M is lightweight.
- Episteme sources consulted: RepT (2510.02334), MDA (2601.21996), Unified Attribution (2501.18887), Infusion (2602.09987), RIF (2506.06656), MAGIC (2504.16430)
- Key change from v1.0: Added positive control requirement (RQ1), attribution method ablation (RQ3), cross-model validation (RQ4); reframed from "negative result paper" to "geometric characterization paper with positive methodological contributions"
