# Iteration Log: TECA

> Most recent entries first.

---

## [2026-03-25] Assimilation into Noesis v3

**Action**: Project assimilated from Sibyl system + TECA_old (Noesis V1) into Noesis v3 framework.
**Key decisions**:
- Research Module set to `implement` (pilot complete, full-scale experiments pending)
- Paper Module set to `P2` (LaTeX template with section drafts exists)
- Legacy materials preserved in `legacy/teca-noesis/`
- All pilot data and results carried forward from `iter_001/exp/`

---

## [2026-03-17] Negative Path Complete

**Action**: Executed negative path analyses after TECS core measurement returned d=0.050.
**Results**:
- H6 (whitening) REJECTED — C^{-1} is not the source of incommensurability
- H7 (structured incommensurability) CONFIRMED — min angles > random baselines
- MEMIT cross-layer d ~ 0.63 — multi-layer editing partially bridges gap
**Decision**: PROCEED with negative result paper. No pivot needed.

**Excluded directions**:
- Behavioral probing alternative (Contrarian suggestion) — deferred, parameter-space characterization is the unique contribution
- Knowledge fingerprinting pivot — not triggered (d=0.050 is clearly negative, not ambiguous)

---

## [2026-03-17] Core TECS Pilot

**Action**: Phase 1-3 pilot experiments (100 facts, GPT-2-XL, ROME at layer 17).
**Results**: TECS mean = 0.000157, Cohen's d = 0.050. All 5 null baselines non-significant after Bonferroni.
**Decision**: NEGATIVE PATH triggered. d ≤ 0.2.

---

## [2026-03-16] Project Startup (TECA_old, Noesis V1)

**Action**: 6-perspective startup debate. Go with focus.
**Kill gates set**: angular variance < 0.05 → STOP; TECS d < 0.5 → STOP; gradient norm control kills signal → STOP.
**Key risk**: Circular reasoning (Contrarian). Mitigated by scoping claim as "geometric characterization" not "validation tool".
