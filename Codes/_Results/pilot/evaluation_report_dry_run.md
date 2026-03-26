# TECA Evaluation Report

Generated: 2026-03-26 20:49:20

## Configuration

- **Config summary**: model=gpt2-xl, n_facts=50, phases=[0, 1, 3], retrieval=bm25, seed=42
- **Results directory**: `_Results/pilot`

---

## Phase 0: Sanity Checks

**Status**: dry_run

| Metric | Value |
|--------|-------|
| rome_validation.efficacy | 0.950000 |
| rome_validation.gate | PASS |
| gradient_check.gate | PASS |
| tecs_pipeline.gate | PASS |

---

## Phase 1: Positive Controls

**Status**: dry_run

| Metric | Value |
|--------|-------|
| rome_vs_self.monotonic_decrease | True |
| rome_vs_self.gate | PASS |
| toy_model.cohens_d | 0.800000 |
| toy_model.p_value | 0.001000 |
| toy_model.gate | PASS |

---

## Gate Summary

| Phase | Gate | Status |
|-------|------|--------|
| 0: Sanity Checks | ROME efficacy >= 75% | PASS |
| 0: Sanity Checks | Gradient check | PASS |
| 0: Sanity Checks | TECS pipeline | PASS |
| 1: Positive Controls | ROME vs self monotonic | PASS |
| 1: Positive Controls | Toy model d > 0.3 | PASS |
