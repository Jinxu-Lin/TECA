## Design Review Synthesis — TECA (Round 1, Advisory)

> Advisory review context: pilot experiments complete, assimilation into Noesis v3.

### Divergence Map

**Consensus (all 6 perspectives):**
1. The methodology is rigorous — 5 null baselines, Cohen's d with bootstrap CI, Bonferroni correction, pre-registered decision gates
2. The implementation is proven — all pilot code works, scaling is straightforward
3. The null result is statistically robust — d = 0.050 is definitively below any reasonable threshold
4. The remaining work (200 facts + ablation) is low-risk engineering

**Main Divergences:**

- **Skeptic/Contrarian** vs **Theorist/Empiricist**: Is g_M (aggregated TDA gradient) a meaningful "attribution direction" or noise dominated by BM25 retrieval bias?
  - **Judgment**: Legitimate concern. The eff-dim = 1.2 is suspicious. The paper should include a gradient quality analysis section. However, even if g_M is partially noise, the measurement itself is informative — it reveals that TDA gradients as commonly computed do not align with editing directions.

- **Contrarian** vs **Others**: Is a positive control experiment needed?
  - **Judgment**: Yes, this would significantly strengthen the paper. Show that TECS CAN detect alignment when it exists (e.g., ROME edit vs itself, or ROME edits for semantically similar facts).

**Unique Insights:**
- **[Skeptic]** Attribution eff-dim = 1.2 may indicate BM25 bias, not fact-specific information — needs investigation
- **[Contrarian]** ROME Δ_W encodes population statistics (C^{-1}) while g_M is per-fact — fundamentally different mathematical objects
- **[Theorist]** The theoretical decomposition provides the best defense against the "trivially expected" objection
- **[Methodologist]** Retrieval method ablation (BM25 vs dense retrieval) would address g_M quality concerns

---

### Priority Sorting

**Must Address:**
1. **Positive control experiment** (Contrarian) — Show TECS can detect alignment in a known-positive case. This is essential for the paper's credibility.
2. **g_M quality analysis** (Skeptic) — Investigate whether eff-dim = 1.2 reflects BM25 bias or genuine attribution structure. Consider showing within-fact vs between-fact gradient similarity.

**Should Address:**
3. **Cross-model validation** (Empiricist, Pragmatist) — GPT-J 6B would strengthen generalizability
4. **Retrieval method ablation** (Methodologist) — BM25 vs dense retrieval to test sensitivity
5. **Paper argumentation** — Proactively address "trivially expected" objection using theoretical decomposition

**Shelved:**
- Fisher information metric alternative (too complex for first paper)
- Random matrix theory null distribution (interesting but not needed for core claims)

---

### Judgment

**PASS (advisory)**

The experimental design is sound, well-controlled, and proven by pilot execution. The six-component geometric framework provides a comprehensive characterization toolkit. The two critical additions (positive control + g_M quality analysis) are feasible within the compute budget and would significantly strengthen the paper. No fundamental or blocking issues identified.

**Routing**: Pass → blueprint (proceed to implementation planning for full-scale experiments)

---

### Method-Experiment Alignment Check

| Method Component | Experiment | Ablation | Status |
|-----------------|-----------|----------|--------|
| TECS metric | Phase 3 core | 4-axis ablation | Pilot done, full pending |
| 5 null baselines | Phase 3 | — | Pilot done |
| Subspace geometry | H7 analysis | k ∈ {10,20,50} | Pilot done |
| Whitening decomposition | H6 analysis | — | Pilot done |
| MEMIT comparison | MEMIT experiment | — | Pilot done (30 facts) |
| Rank-one decomposition | — | — | Theoretical only (H1 rejected) |
| **Positive control** | **MISSING** | — | **NEEDED** |
| **g_M quality** | **MISSING** | — | **NEEDED** |

---

### Next Phase Focus

1. **Add positive control experiment** — essential for paper credibility
2. **Add g_M quality analysis** — address Skeptic's concern about attribution noise
3. **Scale to 200 facts** — straightforward
4. **Run 4-axis ablation** — robustness verification
5. **Optional: GPT-J cross-model** — generalizability

---

### Unresolved Open Questions

- **"Trivially expected" defense**: Must be addressed in paper introduction and discussion — Impact: HIGH
- **Attribution eff-dim = 1.2**: Is this BM25 bias or genuine? — Impact: MEDIUM — Affects interpretation of findings
- **MEMIT d ~ 0.63 interpretation**: Is this genuine alignment or shared-layer artifact? — Impact: LOW — Interesting but secondary to core finding
