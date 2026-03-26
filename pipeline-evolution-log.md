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
