## [Comparativist] Formalize Review — TECA

> Advisory review (assimilation context: pilot experiments complete)

### Gap Novelty Assessment

**Gap is genuine and novel.** No prior work systematically compares editing update vectors and TDA gradient vectors in parameter space. The closest related work:

1. **Hase et al. (2023)**: Shows causal tracing localization doesn't predict editing success — but operates at activation level, not parameter-space geometry. TECA provides the parameter-level explanation.

2. **STEAM (Jeong et al., 2025)**: "Isolated residual streams" — related concept but from the perspective of activation-space analysis, not parameter-space directional comparison.

3. **MDA (2601.21996)**: Computes IF in parameter subspaces — methodologically related but doesn't connect to editing directions.

4. **Infusion (2602.09987)**: IF inverse optimization connects IF to parameter interventions — most related conceptually, but works in vision domain and doesn't do the geometric characterization TECA does.

### Competition Window

- No direct competitor identified doing parameter-space editing-attribution geometric analysis
- The topic (knowledge editing + TDA connection) is niche enough that concurrent work risk is low
- NeurIPS 2026 timeline is reasonable — competition window > 6 months

### Positioning Recommendations

The paper should position itself as:
- NOT: "A new TDA evaluation method" (TECS failed at this)
- YES: "First systematic geometric characterization of editing vs attribution parameter subspaces, revealing structured incommensurability"

The Hase et al. puzzle is the strongest motivation hook — TECA provides a concrete, parameter-level explanation for why localization doesn't predict editing.

### Differentiation from Negative Result Papers

The paper is not merely "X doesn't work." It provides:
1. A principled metric (TECS) with theoretical foundation
2. A six-component characterization framework
3. Specific mechanistic findings (asymmetric cross-projection, MEMIT partial bridging)
4. Explanatory value (connects to Hase et al. puzzle)

This differentiates it from "we tried X and it didn't work" papers.

### Overall Assessment: PASS

Gap is novel, positioning is defensible, and competition risk is low. The connection to the Hase et al. puzzle provides strong motivation that elevates the paper above a simple negative result.
