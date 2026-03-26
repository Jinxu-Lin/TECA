---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
---

# Experiment Results: TECA

> [ASSIMILATED: pilot results complete, full-scale PENDING]

## Status: PILOT COMPLETE, FULL-SCALE PENDING

Pilot experiments (100 facts) are complete across all phases including negative path analyses. Full-scale experiments (200 facts + ablation) are pending.

## Pilot Results Summary

See `Codes/_Results/probe_result.md` for detailed pilot results.

### Core Finding
TECS ≈ 0 (Cohen's d = 0.050 vs null baselines). Structured geometric incommensurability between editing and attribution subspaces confirmed.

### Key Numbers
- TECS mean = 0.000157, std = 0.00676
- Editing eff-dim = 40.8, attribution eff-dim = 1.2
- H6 (whitening) REJECTED, H7 (structured incommensurability) CONFIRMED
- MEMIT cross-layer d ~ 0.63
- G-in-D = 17.3%, D-in-G = 1.0%

## Pending Full-Scale Experiments

1. Expand TECS core measurement from 100 to 200 facts
2. Four-axis ablation study (top-k, weighting, loss, gradient scope)
3. Subspace geometry at 200-fact scale
4. MEMIT full comparison (200 facts)
5. Optional: cross-model validation on GPT-J 6B
