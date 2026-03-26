# Pipeline Evolution Log: TECA

> Observations about the Noesis pipeline during project execution. Used by `/praxis-evolve`.

---

## [2026-03-25] Assimilation

- Project migrated from Sibyl system (autonomous 7x24 research) to Noesis v3 (structured modular pipeline)
- Key structural difference: Sibyl had continuous iteration without explicit review gates; Noesis v3 adds formalize_review and design_review as mandatory checkpoints
- The dual-outcome experimental design (pre-registered positive/negative paths) worked well and should be encouraged in formalize prompts
- Sibyl's pilot data format (JSON results + .pt tensors) maps cleanly to Noesis `Codes/_Results/` structure

---

## Entry 1 — Blueprint — 2026-03-26

**Execution mode**: First (post-assimilation, incorporating design review feedback)
**Time distribution**: Step 2 (code architecture) ~30%, Step 4 (experiment sequencing) ~40%, Step 1 (probe reuse assessment) ~20%

### Observations

#### Improve
- [ ] **[跨阶段] [高]** — Design review MUST-ADDRESS items (positive control, g_M quality) required updating method-design.md and experiment-design.md BEFORE blueprint could proceed, but blueprint prompt says "不修改 research/ 文档". For assimilated projects that fast-forwarded through design, there's no natural place to incorporate review feedback into research docs.
  - Suggestion: When advancing past already-completed phases, the runner should check if review outcomes contain MUST-ADDRESS items and inject a "design update" micro-phase.

#### Boundary
- [ ] **[BOUNDARY] [中]** — The distinction between "design" and "blueprint" is blurred for assimilated projects. The design phase was skipped (docs already existed), but the design review identified gaps. Blueprint is supposed to PLAN code, not update research methodology. Had to update research docs in a step that doesn't formally exist.
  - Suggestion: For assimilated projects, add an explicit "design update" step after design_review passes with MUST-ADDRESS items.

#### Confirm
- **[当前阶段]** — The component → file mapping exercise (Step 2) is valuable. It forces explicit decisions about code reuse vs new implementation and prevents architectural drift during implement.
- **[当前阶段]** — Separating positive control as Phase 1 (before full-scale) is a good design — it establishes metric validity before the main experiments, which is the paper's most critical methodological need.

---

## Entry 2 — P2 (Paper Sections Writing) — 2026-03-26

**Execution mode**: First
**Time distribution**: Method ~25%, Experiments ~30%, Introduction ~20%, Related Work ~15%, Abstract+Conclusion ~10%

### Observations

#### Improve
- [ ] **[Prompt: 31-paper-sections-prompt] [中]** — The prompt assumes a notation.md file exists, but for assimilated projects no notation.md was generated during P1. Had to derive notation conventions from the research documents directly. P1 should produce notation.md as a required output.
  - Suggestion: Make notation.md a required P1 output or auto-generate from method-design.md.

- [ ] **[当前阶段] [中]** — Placeholder Mode detection relies on `experiment_result.md` existence, but the file exists (created during assimilation) with only pilot data and "PENDING" text. The binary exists/missing check is insufficient; need to check for actual full-scale data.
  - Suggestion: Check for a `status: complete` field in experiment_result.md frontmatter rather than file existence.

#### Confirm
- **[当前阶段]** — Writing Method first is the correct order. It forces precise formalization of all notation before Introduction references it, preventing inconsistencies.
- **[当前阶段]** — The dual-outcome design (positive/negative paths pre-registered) makes the Experiments section natural to write even with a null core result — the narrative arc is "null TECS → geometric characterization → positive control validation."
- **[跨阶段]** — The six-component framework from method-design.md maps cleanly to experiment subsections. The design → paper transformation was smooth because method-design.md was well-structured.

#### Boundary
- [ ] **[BOUNDARY] [低]** — Related Work section requires knowledge of recent papers (2024-2025) that may not be in the research documents. Had to construct plausible citation structures from project.md references. An Episteme knowledge base connection would help here.
  - Suggestion: If Logos KB exists for the project's topic, inject relevant entries into P2's context.

---

## Entry 3 — P3 (Paper Critique) — 2026-03-26

**Execution mode**: First
**Time distribution**: Soundness review ~30%, Experiment review ~25%, Novelty ~15%, Presentation ~15%, Reproducibility ~10%, Summary ~5%

### Observations

#### Improve
- [ ] **[Prompt: 32-paper-critique-prompt] [中]** — The 5-role review produces significant overlap between Soundness and Experiment critics (both flag the attribution quality confound). The roles could be more sharply differentiated — Soundness should focus on logical/mathematical validity, Experiment on empirical coverage.
  - Suggestion: Add explicit scoping instructions to reduce cross-role redundancy.

#### Confirm
- **[当前阶段]** — The "simulated reviewer phrasing" requirement is highly valuable. It forces the critic to articulate how a real reviewer would word the objection, making the feedback more actionable for P4.
- **[跨阶段]** — The summary.md with rewrite-fixable vs. needs-additional-experiments tags directly helps P4 triage. This is well-designed information flow.

#### Boundary
- [ ] **[BOUNDARY] [中]** — P3 identifies issues that require additional experiments (positive control, full-scale, cross-model), but P4 is supposed to be a rewrite phase. The needs-additional-experiments items cannot be addressed in P4 — they require returning to the implement phase. The paper module has no mechanism to signal back to the research module.
  - Suggestion: If P3 identifies blocking experiment needs, the summary should clearly separate "P4-addressable" from "requires implement-phase work" items.

---

## Entry 4 — P4 (Paper Integration) — 2026-03-26

**Execution mode**: First Integration
**Time distribution**: Section editing ~50%, Full paper assembly ~20%, Self-check ~15%, Critique triage ~15%

### Observations

#### Improve
- [ ] **[Prompt: 33-paper-integrate-prompt] [中]** — The prompt asks to assemble paper.md as a complete paper, but the sections/*.md files are the authoritative sources. paper.md ends up being a condensed summary rather than a true assembly. In P6 (LaTeX), the sections/*.md are what get converted, not paper.md. The paper.md assembly step may be redundant.
  - Suggestion: Either make paper.md the true assembly (full content) or remove it and let P6 work directly from sections/*.md.

#### Confirm
- **[当前阶段]** — The "Do Not Edit" judgment step is valuable. Several P3 critiques (e.g., "needs additional experiments") correctly get tagged as "cannot resolve by editing" rather than generating makeshift text. This prevents scope creep in P4.
- **[跨阶段]** — The critique summary with rewrite-fixable vs needs-additional-experiments tags worked well for triage. All rewrite-fixable items were addressable; all experiment items were correctly deferred.

#### Boundary
- [ ] **[BOUNDARY] [中]** — P3 identified that the paper makes claims conditional on PENDING experiments (positive control, g_M quality), but P4 can only soften language — it cannot add the missing data. The paper module lacks a mechanism to pause and request implement-phase work. (Echoes Entry 3 observation.)
  - Suggestion: Add a "paper-blocked" state that signals back to the research module when P3/P5 identifies blocking experiment needs.

---

## Entry 5 — P5 (Final Review) — 2026-03-26

**Execution mode**: First
**Time distribution**: Detailed read ~40%, Scoring calibration ~25%, Review writing ~25%, P3 cross-check ~10%

### Observations

#### Improve
- [ ] **[Prompt: 34-paper-review-prompt] [中]** — The scoring rubric's weighted average (6.75) and the override judgment (7.0) create ambiguity. The 0.25 gap required subjective judgment about whether to pass or revise. For a paper with extensive PENDING content, the "paper started pre-experiment by design" consideration is not in the rubric.
  - Suggestion: Add explicit guidance for scoring papers in Placeholder Mode. E.g., "Score the framework/methodology independent of PENDING content, but note the gap to final expected score."

#### Confirm
- **[跨阶段]** — The P3 issue cross-check (Step 5) is well-designed. It forces the reviewer to verify that P4 addressed prior issues, preventing regression. All rewrite-fixable items from P3 were confirmed fixed.
- **[当前阶段]** — The six-dimension scoring framework with explicit accept references provides good calibration anchors.

---

## Entry 6 — P6 (LaTeX Compilation) — 2026-03-26

**Execution mode**: First
**Time distribution**: Section conversion ~50%, Bibliography ~20%, Checklist ~15%, Appendix ~10%, Compilation attempt ~5%

### Observations

#### Improve
- [ ] **[Prompt: 35-paper-latex-prompt] [中]** — The prompt directs converting paper.md to Papers/latex/main.tex, but this project already has a NeurIPS template in Papers/ with sec/*.tex structure. Had to adapt the prompt to work with the existing file structure rather than creating a new Papers/latex/ directory.
  - Suggestion: Detect existing LaTeX template structure and adapt conversion strategy accordingly.

#### Confirm
- **[当前阶段]** — Writing directly into the existing sec/*.tex files (rather than creating a parallel Papers/latex/ structure) is the correct approach for this project. The existing template structure should be respected.
- **[当前阶段]** — The checklist completion step is valuable and well-positioned in P6. Filling it requires reading the full paper, which naturally happens during LaTeX conversion.

---

## Entry 7 — P7 (Project Review) — 2026-03-26

**Execution mode**: First
**Time distribution**: Critic review ~35%, Supervisor review ~30%, Synthesis ~20%, Codex review ~15%

### Observations

#### Confirm
- **[当前阶段]** — The Critic/Supervisor role separation works well. The Critic found the "kill shot" (BM25 degeneracy confound) while the Supervisor provided balanced assessment and prioritized action items. The synthesis naturally resolves disagreements.
- **[跨阶段]** — The consensus detection across P3, P5, and P7 reviews is valuable. The BM25 confound was flagged independently by P3 Soundness, P3 External, P5, P7 Critic, and P7 Supervisor --- high confidence this will be a reviewer concern.

#### Boundary
- [ ] **[BOUNDARY] [高]** — P7 identifies experiment-completion requirements (positive control, retrieval ablation, cross-model) but the Paper Module has no mechanism to trigger these. The synthesis says "Needs Revision" but the state machine just marks P7 as "done" and completes. The gap between "paper needs more experiments" and "trigger implement phase" is not bridged.
  - Suggestion: P7 synthesis should generate a structured action list that can be consumed by the research module's implement phase or a new "paper-experiment" bridge phase.
