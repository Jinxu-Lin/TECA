## [Interdisciplinary] Formalize Review — TECA

> Advisory review (assimilation context: pilot experiments complete)

### Cross-Domain Perspective

The editing-attribution incommensurability finding has interesting parallels in other fields:

1. **Complementary Learning Systems (CLS) theory**: ROME can be viewed as "hippocampal" (fast, sparse, targeted) encoding, while TDA gradients reflect "neocortical" (slow, distributed, statistical) encoding. The incommensurability is predicted by CLS: these are fundamentally different learning systems that encode knowledge in different geometric structures. This is not a bug — it's a feature of dual-system architectures.

2. **Multi-messenger astronomy analogy**: Different probes (gravitational waves vs electromagnetic radiation) reveal different aspects of the same event. The fact that editing and attribution "see" different geometric structures in parameter space is analogous to how different physical probes reveal complementary (not redundant) information.

3. **Spectral theory / random matrix theory**: The radical dimensionality asymmetry (editing eff-dim = 40.8 vs attribution eff-dim = 1.2) is reminiscent of the Marchenko-Pastur distribution in random matrix theory. The attribution subspace collapsing to effectively 1 dimension suggests the TDA gradients are dominated by a single principal mode — possibly a systematic bias rather than fact-specific information.

### Alternative Framing

The project could benefit from framing the incommensurability not as a "failure" of TECS but as evidence for **dual-geometry knowledge organization**: knowledge editing requires distributed, multi-dimensional updates (eff-dim 40.8) while attribution signals are concentrated in a narrow subspace (eff-dim 1.2). This asymmetry is itself a finding about knowledge representation.

### Alternative Attack Angles

If the current framing doesn't resonate with reviewers:
- **Angle B (from startup debate)**: Behavioral probing — compare ROME edit effects vs TDA top-k deletion effects in behavior space. This avoids parameter-space assumptions entirely.
- **Angle C**: Focus paper entirely on the dimensionality asymmetry and what it reveals about TDA gradient structure.

### Theoretical Value of RQs

RQ3 (incommensurability characterization) and RQ5 (MEMIT contrast) are the strongest. RQ1 (existence) is resolved (negative). RQ2 (decomposition) requires positive signal. RQ4 (whitening) is resolved (negative).

The MEMIT finding (d ~ 0.63) is particularly interesting from a CLS perspective: distributing edits across layers partially bridges the gap between "fast writing" and "slow learning" geometries.

### Overall Assessment: PASS

The problem framing is sound and has rich cross-domain connections. The dual-geometry interpretation elevates the paper from "null result" to "geometric structure discovery." Recommend incorporating the dimensionality asymmetry and CLS framing prominently in the paper.
