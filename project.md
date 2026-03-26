---
version: "1.0"
status: "complete"
decision: "Go with focus"
created: "2026-03-16"
last_modified: "2026-03-25"
---

# Project: TECA — TDA-Editing Consistency Analysis

> [ASSIMILATED: generated from TECA_old/project-startup.md + TECA/iter_001/idea/proposal.md + pilot results]

## 1. Overview

### 1.1 Topic
Investigating the geometric relationship between model editing directions (ROME/MEMIT) and training data attribution (TDA) gradient directions in transformer MLP parameter space, using the novel TECS (TDA-Editing Consistency Score) metric.

### 1.2 Initial Idea
TDA evaluation suffers from a fundamental bootstrapping problem: all existing methods (LDS, LOO, Spearman) rely on retrain-based ground truth, which is systematically unreliable on non-convex large models. TECA proposes an entirely independent validation dimension: comparing TDA attribution directions with model editing directions in parameter space.

The core observation is that TDA and model editing operate on the same object in parameter space but from opposite directions — TDA traces "which training samples influenced this knowledge" while ROME actively modifies "where knowledge is stored in parameters." If both point to the same parameter subspace, they mutually validate. If not, the geometric incommensurability itself is scientifically informative, explaining the persistent localization-editing disconnect (Hase et al., 2023).

TECS = cos(vec(delta_W_E), vec(g_M)) at ROME's editing layer l*, measuring cosine similarity between the rank-one editing update and the aggregated attribution gradient.

### 1.3 Baseline Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | ROME (Meng et al., 2022) | 2202.05262 | Rank-one editing, causal tracing — editing end of TECS |
| 2 | MEMIT (Meng et al., 2022) | 2210.07229 | Multi-layer distributed editing — contrast with ROME |
| 3 | Revisiting IF Fragility | 2303.12922 | Spearman miss-relation — motivation for independent evaluation |
| 4 | IF on LLMs (Wang et al.) | 2409.19998 | IF triple failure on LLMs — risk factor for TDA end |
| 5 | Infusion | 2602.09987 | IF inverse optimization — bridge between IF and parameter space |
| 6 | MDA | 2601.21996 | Subspace IF — supports subspace comparison design |
| 7 | Hase et al. (2023) | 2301.04213 | Localization-editing disconnect — key puzzle TECA addresses |

### 1.4 Available Resources
- **GPU**: 1x NVIDIA RTX 4090 (24GB VRAM), remote server (xuchang3)
- **Timeline / DDL**: NeurIPS 2026 submission
- **Existing Assets**: GPT-2-XL, CounterFact dataset, EasyEdit library, all pilot tensors saved

---

## 2. Problem & Approach

### 2.1 Baseline Analysis

#### What they solved
ROME/MEMIT: deterministic knowledge editing via rank-one updates. TDA methods (IF, TRAK): training data influence estimation. Each has independent evaluation paradigms.

#### What they didn't solve
No work has connected editing and attribution at the level of parameter-space geometry. The editing and TDA communities developed in isolation.

#### Why they didn't solve it
Different evaluation paradigms; no natural metric bridges the two. The question "do editing and attribution directions point the same way?" has never been asked systematically.

### 2.2 Problem Definition
- **Problem**: TDA evaluation relies entirely on retrain-based ground truth, which is systematically unreliable. Meanwhile, knowledge editing produces deterministic parameter updates in the same space as TDA gradients, yet no one has measured their geometric relationship.
- **Authenticity**: The evaluation gap is well-documented (miss-relation in Spearman, IF triple failure on LLMs).
- **Importance**: Understanding whether different knowledge operations access commensurable parameter subspaces is fundamental to knowledge representation theory.
- **Value layer**: "Nobody has done it" — first systematic comparison of editing and attribution directions.

### 2.3 Root Cause Analysis
- Symptom: TDA methods lack independent (non-retrain) validation
- Intermediate: No metric bridges editing and attribution communities
- Root Cause: The geometric relationship between editing update vectors and attribution gradient vectors has never been characterized — they live in the same space but nobody measured their alignment

### 2.4 Proposed Approach
Compute TECS (cosine similarity between ROME's rank-one update and aggregated TDA gradient) across hundreds of facts with five null baselines. Dual-outcome design: positive TECS reveals geometric consistency; near-zero TECS with structured characterization reveals the geometric basis for the localization-editing disconnect. Under the linear associative memory model, TECS admits a rank-one decomposition into key-alignment and value-alignment terms.

### 2.5 Core Assumptions

| # | Assumption | Type | Source | Support | If false |
|---|-----------|------|--------|---------|----------|
| 1 | TDA gradients and ROME edits share geometric structure at l* | Empirical | Researcher | Weak | TECS meaningless (CONFIRMED FALSE — pilot d=0.05) |
| 2 | ROME rank-1 update reflects knowledge storage location | Theoretical | ROME paper | Medium (disputed) | TECS measures ROME bias, not knowledge |
| 3 | IF/TRAK attribution directions are reliable | Empirical | TDA literature | Medium | Both ends unreliable |
| 4 | Cosine similarity is a reasonable alignment measure | Methodological | Researcher | Medium | Need alternative metrics |
| 5 | Editing layer l* comparison suffices | Structural | ROME assumption | Medium (disputed) | Need multi-layer analysis |

---

## 3. Validation Strategy

### 3.1 Idea Type Classification
New perspective / measurement: first parameter-space geometric comparison of editing and attribution.

### 3.2 Core Hypothesis
H1 (CONFIRMED FALSE): TECS at l* significantly higher than null baselines (Cohen's d > 0.3). Actual d = 0.050.
H7 (CONFIRMED): Structured incommensurability — editing and attribution occupy distinct parameter subspaces.

### 3.3 Probe Experiment Design
Phase 1: ROME validation (100 facts, GPT-2-XL, EasyEdit). Phase 2: TDA gradient computation. Phase 3: TECS measurement vs 5 null baselines. Phase 4 (negative path): Subspace geometry + whitening decomposition + MEMIT comparison.

### 3.4 Pass / Fail Criteria

| Result | Condition | Action |
|--------|-----------|--------|
| Positive path | Cohen's d > 0.2 | Dose-response + layer sweep + spectral analysis |
| Negative path | Cohen's d ≤ 0.2 | Subspace geometry + whitening + MEMIT (THIS PATH TAKEN) |
| Dead end | Random misalignment, no structure | Pivot required |

### 3.5 Time Budget & Resources
- Phase 1-3: ~85 min GPU (COMPLETE)
- Negative path extensions: ~40 min GPU (COMPLETE)
- Full-scale experiments: ~2 hours GPU (PENDING)

### 3.6 Failure Diagnosis Plan

| Failure mode | Signature | Meaning | Action |
|-------------|-----------|---------|--------|
| TECS ~ 0, structured | Min angle < random baseline | Editing/attribution in distinct organized subspaces | Characterize → negative result paper |
| TECS ~ 0, whitening gap | Unwhitened >> whitened | ROME's C^{-1} rotates away from natural geometry | Mechanistic explanation paper |
| TECS ~ 0, random | Min angle ~ random baseline | No geometric content | Pivot required |

---

## 4. Review

### 4.1 Review History

| Round | Date | Decision | Key Changes |
|-------|------|----------|-------------|
| TECA_old Debate | 2026-03-16 | Go with focus | 6-perspective startup debate (Noesis V1) |
| Sibyl Pilot | 2026-03-17 | Negative path | TECS d=0.050, entered negative path |
| Negative Path | 2026-03-17 | Proceed | H6 rejected, H7 confirmed, MEMIT d~0.63 |

### 4.2 Latest Assessment Summary
- **Contrarian**: Circular reasoning concern validated — two unreliable tools cannot validate each other. Claim properly scoped.
- **Empiricist**: Kill gates worked as designed. Negative result with rigorous methodology is publishable.
- **Theorist**: Rank-one decomposition theorem still valuable for theoretical framework even with null TECS.
- **Pragmatist**: All experiments within budget. Negative path reuses existing data efficiently.
- **Interdisciplinary**: Structured incommensurability connects to CLS theory — editing as "hippocampal" and attribution as "neocortical" encoding.

### 4.3 Decision
- **Decision**: Pass (Go with focus → Negative result paper)
- **Rationale**: TECS ~ 0 with structured characterization is a publishable finding. Explains Hase et al. disconnect at parameter level.
- **Key Risks**: Reviewer skepticism about negative results; need to expand from 100 to 200 facts for full paper.
- **Unresolved Disputes**: Whether structured incommensurability is interesting enough for NeurIPS oral vs poster.

### 4.4 Conditions for Next Module
- Expand to 200 facts with full ablation study
- Complete MEMIT multi-layer analysis
- Frame as "geometric incommensurability" paper, not "TECS validation" paper
- Key contribution: first empirical evidence that editing and attribution access geometrically incommensurable parameter subspaces

<!-- Complete debate records: legacy/teca-noesis/phase-outcomes/debate/ + Reviews/init/ -->
