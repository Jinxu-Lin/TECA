## [Contrarian] Formalize Review — TECA

> Advisory review (assimilation context: pilot experiments complete, negative path analyses done)

### Assumption Challenges

- **Assumption 1 (TDA-ROME alignment at l*)**: This was the project's central hypothesis, and it has been conclusively falsified (d = 0.050). The project correctly pivoted to characterizing the incommensurability rather than claiming alignment. However, the falsification raises a deeper question: if the two methods operate in fundamentally different subspaces, is the *comparison itself* scientifically meaningful, or is it measuring an artifact of method design choices?

- **Assumption 2 (ROME reflects knowledge storage)**: The ROME/causal tracing debate remains unresolved. The finding that TECS ~ 0 could equally mean (a) knowledge geometry is incommensurable between operations, OR (b) ROME's update direction is a constrained optimization artifact unrelated to knowledge geometry. The project acknowledges this but the distinction matters for claim strength.

- **Framing as "structured incommensurability"**: The subspace analysis (H7) actually showed min angles are NOT significantly smaller than random (p = 0.084 at k=10, p = 0.989 at k=20). This contradicts the "structured" claim. The misalignment is indistinguishable from random at most k values. The asymmetric cross-projection (G-in-D = 17.3%, D-in-G = 1.0%) is the only structured signal.

### Counterfactual Scenario

**If the core insight is wrong** (i.e., the incommensurability is trivially expected): ROME's rank-one update is a constrained least-squares solution in a whitened space; TDA gradients are natural gradients in parameter space. These are mathematically different objects by construction. The "finding" that they don't align may be as surprising as finding that apples and oranges have different shapes.

**Most likely reviewer objection**: "Why would anyone expect these directions to align? ROME solves argmin_v ||C^{-1}(k* ⊗ (v - v*))||, while TDA computes ∇_W L. These optimize different objectives in different geometries. Near-zero cosine similarity is the null hypothesis, not a finding."

### Underestimated Competition

- Hase et al. (2023) already established that localization doesn't predict editing. TECA's parameter-level explanation is an extension, not a fundamentally new finding.
- STEAM (Jeong et al., 2025) discusses "isolated residual streams" — related geometric characterization.

### Survival Line Assessment

**If the paper can only show "ROME edits and TDA gradients point in different directions"**: Borderline poster at best. The paper needs the geometric characterization framework + MEMIT contrast + theoretical decomposition to reach poster-level contribution.

### Overall Assessment: PASS (advisory)

Despite the above concerns, the gap is genuine (no systematic parameter-space comparison exists), the methodology is rigorous (5 null baselines, effect sizes, Bonferroni), and the dual-outcome design is well-executed. The framing needs tightening — specifically around what "structured" means given the H7 null results — but the direction is sound.
