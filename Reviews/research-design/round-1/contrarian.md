## [Contrarian] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Assumption Challenges

1. **g_M quality is unverified.** The Phase 2 "sanity check" (gradient norm > 1e-8 and angular variance = 0.048) is necessary but insufficient. Angular variance of 0.048 means top-k gradients are nearly orthogonal — the aggregated g_M is averaging nearly random directions. This is consistent with g_M being noise, not signal. The attribution eff-dim = 1.2 further supports this: the "TDA direction" is not a rich fact-specific signal but a single dominant mode (likely BM25 retrieval bias).

2. **ROME Δ_W encodes more than knowledge.** ROME's constrained least-squares solution minimizes ||C^{-1}(k* ⊗ (v* - v))||, which means Δ_W is shaped by: (a) the target key k*, (b) the value error v* - v, (c) the key covariance C. Component (c) encodes statistical properties of ALL keys, not just the target fact. Comparing this with a single-fact TDA gradient is comparing apples (population statistics) with oranges (single-sample statistics).

### Overfit to Pilot Concern

**Low risk.** The pilot IS the experiment at reduced N. There's no model training, no hyperparameter tuning, and no iterative optimization. The design decisions (choice of baselines, metrics, thresholds) were set before experiments. This is not a typical ML experiment where overfitting to validation is a risk.

### Strongest Counterargument

**The most damaging reviewer comment would be:** "The authors compute cosine similarity between two very different mathematical objects — a constrained optimization solution and an aggregated gradient — and find they don't align. This is expected and uninformative."

**Defense needed:** The paper must clearly articulate WHY the comparison is informative even given different optimization objectives. The theoretical decomposition (TECS = key-alignment × value-alignment) provides part of this defense — it shows that under the linear associative memory model, alignment IS expected if knowledge representations are consistent. The null result then means the model assumptions underlying ROME (linear associative memory) do not match the actual gradient geometry.

### Missing Experiments

- **Positive control**: Show that TECS IS high for a case where alignment is expected (e.g., two ROME edits for similar facts, or ROME edit vs ROME edit for the same fact with different targets). Without a positive control, we can't distinguish "TECS doesn't work" from "the underlying geometry is incommensurable."

### Assessment: PASS (advisory)

The design is sound for its purpose. The positive control experiment is the most important missing piece. The "trivially expected" objection must be addressed head-on in the paper.
