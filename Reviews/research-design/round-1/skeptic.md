## [Skeptic] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Weakest Component

**The TDA gradient construction is the weakest link.** Several compounding approximations:

1. BM25 retrieval from a Wikipedia subset (not GPT-2's actual training data, WebText)
2. Top-20 documents only — sparse sampling of training influence
3. BM25-weighted aggregation — arbitrary weighting scheme
4. Single-layer gradient at l*=17 — ignores gradient flow through other layers

Each approximation is reasonable in isolation, but their composition may produce g_M that is more "BM25 retrieval bias" than "TDA attribution signal." The attribution eff-dim = 1.2 (91% variance in PC1) is consistent with this concern — a systematic BM25 retrieval bias would manifest as a dominant principal component.

### Most Likely Failure Point

**The "trivially expected" reviewer objection is the most dangerous.** If reviewers are not convinced that comparing ROME directions and TDA gradients is informative (because they optimize fundamentally different objectives), no amount of rigorous methodology will save the paper. This is an argumentative challenge, not a technical one.

### Alternative Explanations

1. **TECS ~ 0 because g_M is noise**: Attribution eff-dim = 1.2 suggests g_M is dominated by a common mode (possibly BM25 bias), not fact-specific information. The "incommensurability" may actually be "TDA gradients carry no useful directional information."

2. **TECS ~ 0 because ROME directions encode statistical correction, not knowledge**: ROME's C^{-1} rotates the update to compensate for correlated keys. The editing direction is partly a statistical artifact, not purely a knowledge direction.

3. **MEMIT d ~ 0.63 is an artifact of shared l*=17**: MEMIT edits layers 13-17, including the same layer used for TDA gradients. The "partial bridging" may reflect shared layer effects, not genuine alignment.

### What Would Change My Mind

- g_M eff-dim > 5 with a different retrieval method (confirming fact-specific attribution signal)
- Cross-model replication on GPT-J with similar findings
- A control experiment showing that TECS is high for a known-aligned pair (e.g., two IF gradients for the same fact)

### Assessment: PASS (advisory, with reservations)

The methodology is sound for what it measures. My concern is about what it *means*, not how it's done. The alternative explanation that g_M is noise (not fact-specific) needs to be ruled out or acknowledged. The paper should include a "TDA gradient quality" analysis section.
