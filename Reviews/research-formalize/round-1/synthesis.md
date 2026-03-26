## Formalize Review Synthesis — TECA (Round 1, Advisory)

> Advisory review context: pilot experiments complete, negative path analyses done, assimilation into Noesis v3.

### Divergence Map

**Consensus (all 4 perspectives):**
1. The gap is genuine — no prior work systematically compares editing and attribution directions in parameter space
2. The methodology is rigorous — 5 null baselines, effect sizes, Bonferroni correction, dual-outcome design
3. The project is feasible and resource-efficient — total compute < 6 hours GPU for full paper
4. The pilot execution was exemplary — kill gates worked as designed, pivot was data-driven

**Main Divergence:**
- **Contrarian** vs **Interdisciplinary/Comparativist**: Is the null TECS result "trivially expected" (different objectives → different directions by construction) or "scientifically informative" (reveals dual-geometry knowledge organization)?
  - **Judgment**: Both are partially right. The paper must proactively address the "trivially expected" objection (Contrarian's strongest point) while framing the characterization as the contribution, not the null result itself.

**Unique Insights:**
- **[Contrarian]** H7 (structured incommensurability) is actually NOT confirmed at most k values (p = 0.989 at k=20, p = 1.0 at k=50). The "structured" framing needs revision — the asymmetric cross-projection is the real signal.
- **[Interdisciplinary]** The radical dimensionality asymmetry (40.8 vs 1.2) may be the most interesting finding, deserving prominence in the paper.
- **[Comparativist]** The Hase et al. (2023) puzzle connection is the strongest motivation hook.

---

### Priority Sorting

**Must Address:**
1. **Framing of "structured" incommensurability** (Contrarian) — H7 p-values contradict the "structured" claim at k=20 and k=50. Must reframe: the asymmetric cross-projection and dimensionality asymmetry are the structured signals, not the principal angles.
2. **"Trivially expected" objection** (Contrarian) — Must argue why comparing these directions is informative even if they optimize different objectives. The theoretical decomposition and CLS framing help.

**Should Address:**
3. Cross-model validation (Pragmatist) — GPT-J 6B would significantly strengthen generalizability claims
4. CLS/dual-geometry framing (Interdisciplinary) — Elevates paper from null result to structure discovery

**Shelved:**
- Behavioral probing alternative (Interdisciplinary) — The parameter-space characterization IS the unique contribution; behavioral probing is a different paper
- Random matrix theory formalization (Interdisciplinary) — Interesting but beyond scope of first paper

---

### Judgment

**PASS (advisory)**

The problem formulation is sound, well-motivated, and backed by complete pilot data. The gap is novel (confirmed by Comparativist), the execution is feasible (confirmed by Pragmatist), and the negative result has genuine scientific value when properly framed (confirmed by Interdisciplinary). The Contrarian's concerns about H7 framing and the "trivially expected" objection are real but addressable through careful paper writing, not through re-doing the formalization.

**Routing**: Pass → design (proceed to full experiment design for 200-fact scale)

---

### Revised Research Direction

No major revision needed. Minor adjustments:
1. De-emphasize "structured incommensurability" framing in favor of "dimensionality asymmetry + asymmetric cross-projection" as the primary characterization
2. Lead with the Hase et al. puzzle as motivation (strongest hook)
3. Frame MEMIT bridging as evidence for the role of distributed vs localized editing
4. Proactively address the "trivially expected" objection in the introduction

---

### Next Phase Focus

1. **Scale experiments to 200 facts** — same pipeline, higher statistical power
2. **Four-axis ablation study** — robustness verification
3. **Tighten H7 claims** — use dimensionality asymmetry + cross-projection as primary evidence, not principal angles alone
4. **Optional: GPT-J 6B cross-model** — generalizability

---

### Unresolved Open Questions

- **"Trivially expected" defense**: How to argue that the comparison is informative despite different optimization objectives — Impact: HIGH — Becomes deal-breaker if reviewers are not convinced
- **Attribution eff-dim = 1.2**: Is this a TDA methodology limitation (BM25 retrieval bias) or a genuine property of attribution geometry? — Impact: MEDIUM — Could be investigated with different retrieval methods
- **NeurIPS vs ICLR**: Negative result papers have mixed reception — venue choice matters — Impact: LOW — NeurIPS Datasets & Benchmarks track may be more receptive
