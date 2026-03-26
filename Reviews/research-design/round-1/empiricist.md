## [Empiricist] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Statistical Power

With N=100 facts, the study has adequate power to detect medium effects (d > 0.5) but may miss small effects (d = 0.2-0.3). The observed d = 0.050 is far below even the detection floor, so the null conclusion is robust. Expanding to 200 facts improves power but won't change the core conclusion.

Bootstrap CI [-0.00117, 0.00146] spanning zero with N=100 is definitive for the null hypothesis.

### Baseline Fairness

All five null baselines are well-constructed:
- Null-A (random fact): Controls for fact-specificity — the most important control
- Null-B (wrong layer): Controls for layer-specificity
- Null-C (shuffled gradient): Controls for gradient structure
- Null-D (random direction): Controls for dimensional concentration
- Null-E (test gradient): Additional control, though its high variance (std = 0.029 vs 0.007 for TECS) makes it less informative

**Missing baseline**: A "random editing method" baseline (e.g., random rank-one update with the same norm as ROME) would strengthen the claim that the null result is about ROME specifically, not about rank-one updates in general. However, this is minor — Null-D effectively serves this role.

### Probe → Full Experiment Bridge

The scaling from 100 → 200 facts is straightforward:
- Same code, same pipeline
- Same model, same layer
- Only difference: more data points

This is not a typical "toy → full" scale gap. The pilot IS the experiment at reduced N. Risk of scaling failure: essentially zero.

### Reproducibility

Excellent:
- Fixed seed (42)
- EasyEdit pinned commit
- All tensors saved
- All results in JSON with timestamps
- Configuration recorded in each result file

### Assessment: PASS

The experimental design meets high standards for a measurement study. The null result is statistically robust. The scaling plan is minimal-risk. Recommend the cross-model validation (GPT-J) as the most impactful addition.
