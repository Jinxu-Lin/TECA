## [Pragmatist] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Engineering Feasibility

**All components proven.** The pilot demonstrated:
- EasyEdit ROME: 100% efficacy, 19s/fact
- TDA gradient computation: 100% valid, ~2.4s/fact
- TECS computation: deterministic, no convergence issues
- Subspace analysis: CPU-only, ~4 min
- Whitening decomposition: ~1.5 min
- MEMIT: 30/30 successful, ~0.8s/fact

### Computational Budget Verification

| Experiment | Estimated | Actual (pilot) | Status |
|------------|-----------|----------------|--------|
| ROME validation | 15 min | 32 min | Done (FP16 optimization applied) |
| TDA gradients | 20 min | 4 min | Done (faster than expected) |
| TECS measurement | 15 min | ~50 min | Done (5 null baselines = more compute) |
| Negative path | 40 min | ~25 min | Done |
| Full scale (200 facts) | ~2 hrs | — | Pending |
| Ablation (4 axes) | ~1 hr | — | Pending |
| Cross-model (GPT-J) | ~3 hrs | — | Optional |

Total remaining: ~3-6 hours GPU. Well within RTX 4090 capability.

### Implementation Complexity

Low. The codebase is functional:
- `pilot_rome_validation.py` — ROME validation
- `pilot_tecs_core.py` — Core TECS measurement
- `negative_subspace_geometry.py` — Subspace analysis
- `negative_whitening.py` — Whitening decomposition
- `negative_memit_experiment.py` — MEMIT comparison

Scaling from 100 → 200 facts requires changing one parameter. No new code needed for core experiments.

### Time Estimate

- Full-scale experiments: 1 day
- Paper writing: 1-2 weeks (LaTeX template exists)
- Revision buffer: 1 week
- Total: 2-3 weeks to submission-ready paper

### Assessment: PASS

No engineering risks. Budget is well within constraints. The code exists and works. This is purely an execution task.
