# Project Review Synthesis

## Review Summary

| Role | Status | Core Finding |
|------|--------|-------------|
| Critic | Done | Attack strength 6/10. Kill shot: BM25 degeneracy confound. Positive control PENDING. |
| Supervisor | Done | Quality 7/10. Novel question, sound methodology, incomplete execution. Borderline Accept with completed experiments. |
| External | Skipped | Codex MCP not available. |

## Composite Assessment
- **Supervisor Score**: 7 / 10
- **Critic Attack Strength**: 6 / 10
- **Submission Recommendation**: Needs Revision --- complete PENDING experiments before submission

## Critical Issues (must resolve before submission)

1. **Positive control experiment (Tier 2: toy model)** --- Without this, the paper cannot distinguish "metric failure" from "informative null result." Both Critic and Supervisor flag this as the single highest priority.

2. **Attribution quality analysis (retrieval ablation)** --- The BM25 degeneracy confound (eff-dim=1.2) undermines the core claim. Both reviewers identify this as the most likely rejection reason. Must demonstrate that incommensurability persists with non-degenerate attribution methods, OR honestly report that it doesn't.

3. **Full-scale core TECS (200 facts)** --- Pilot N=100 is underpowered for a NeurIPS structural claim.

## Major Issues (should resolve)

1. **Figures** --- Zero figures in a geometry paper. Need 4-5 visualizations.
2. **Cross-model validation** --- At minimum Pythia-410M as a lightweight check.
3. **MEMIT scale** --- N=30 with simplified implementation is too weak.

## Consensus and Conflicts

### Issues flagged by multiple reviewers
- BM25 degeneracy confound (Critic + Supervisor + P3 Soundness + P3 External) --- **HIGHEST CONSENSUS ISSUE**
- Positive control PENDING (Critic + Supervisor + P3 Experiment)
- Single-model limitation (Supervisor + P3 Experiment)

### Reviewer disagreements
- Critic rates the "structured" framing as overinterpretation (score: 6); Supervisor considers it adequately caveated after P4 revisions (score: 7). **Decision**: Keep current framing but add random subspace cross-projection baseline to contextualize 17.3%/1.0% numbers. This resolves the disagreement with data.

## Proxy Metric Gaming Verdict
- **Status**: Pass (N/A)
- **Details**: This is a characterization study with no optimization target. The dual-outcome pre-registration and five null baselines demonstrate honest methodology. No gaming risk.

## Submission Strategy

- **Current State**: Major revision (PENDING experiments are blocking)
- **Venue Fit**: NeurIPS 2026 --- appropriate for a geometric characterization study with rigorous methodology
- **Predicted Outcome**: Borderline Accept (poster) if experiments support claims; Weak Reject without experiments
- **Action Items** (prioritized):
  1. Run positive control Tier 2 (toy model) --- ~10 min GPU, CPU sufficient
  2. Run full-scale TECS (200 facts) --- ~2 hours GPU
  3. Run g_M retrieval ablation (BM25/TF-IDF/Contriever) --- ~1 hour GPU
  4. Create figures (eigenvalue spectra, TECS distribution, cross-projection, MEMIT heatmap)
  5. Run Pythia-410M cross-model check --- ~30 min GPU
  6. Scale MEMIT to 200 facts with proper covariance --- ~1 hour GPU
  7. Fill all PENDING placeholders via /praxis-paper-fill
- **Most Likely Rejection Reason**: "BM25 attribution is too degenerate; the geometric characterization may be an artifact of the attribution method."
- **Rebuttal Preparation**:
  - If g_M ablation shows eff-dim increases with Contriever but TECS stays zero: "The incommensurability is robust to attribution quality."
  - If g_M ablation shows TECS increases with better retrieval: "We quantify the attribution quality contribution to incommensurability." Either way, the experiment provides an answer.
  - For "trivially expected" objection: cite rank-one decomposition SNR prediction + positive control results.
  - For "single model" objection: cite cross-model results (if available) or Pythia-410M check.
