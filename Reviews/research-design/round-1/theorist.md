## [Theorist] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Logical Closure

Gap → Root Cause → Method → Why Solves chain is **complete and sound**:
- Gap: No parameter-space comparison of editing and attribution directions
- Root Cause: These communities developed in isolation; no natural bridge metric
- Method: TECS (cosine similarity) + six-component geometric framework
- Why Solves: Directly measures the geometric relationship and characterizes its structure

The rank-one decomposition theorem (TECS = key-alignment × value-alignment) provides theoretical grounding. Under the linear associative memory model, this is mathematically correct.

### Theoretical Correctness

1. **TECS definition**: Well-defined cosine similarity in R^{d_v × d_k}. Flattening to vec() preserves the angle. Correct.

2. **Rank-one decomposition**: Under W*k = v, the decomposition TECS(z_i) = sign(α) · cos(C^{-1}k*, k_i) · cos(v* - Wk*, d_v_i) is algebraically valid. The linear associative memory assumption is standard (used by ROME itself).

3. **Null distribution prediction**: E[TECS²] ~ 1/d_k from concentration of measure. For d_k = 1600, expected |TECS_null| ~ 0.025. Observed TECS_null std ~ 0.006 — lower than predicted, likely because the actual directions are not uniformly random on the sphere.

4. **Effective dimensionality**: Entropy-based measure is standard and well-defined.

5. **Principal angle analysis**: scipy.linalg.subspace_angles is numerically stable and correct for this application.

### Hidden Assumptions

- ROME's C^{-1} makes the editing direction depend on the covariance of all keys, not just the target fact. This is by design (least-squares solution) but means delta_W encodes statistical information about the entire key distribution, not just the target knowledge.
- BM25 retrieval for TDA is a proxy — the actual training data of GPT-2 is not fully available. The 20 retrieved documents may not include the actual training documents that shaped the knowledge.

### Assessment: PASS

The theoretical framework is sound. The decomposition theorem adds interpretive value even though H1 was rejected. The hidden assumptions are acknowledged and do not invalidate the analysis.
