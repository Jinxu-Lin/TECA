# Final Review Report

## Overall
- **Composite Score**: 7.0 / 10.0
- **Decision**: Pass
- **One-line assessment**: A well-conceived geometric characterization study with a novel question and rigorous statistical methodology, whose main limitation (extensive PENDING experiments) is appropriate for a paper started before experiment completion and addressable via /praxis-paper-fill.
- **Predicted outcome at NeurIPS 2026**: Borderline Accept (poster) — conditional on PENDING experiments supporting the claims
- **Confidence**: Medium

## Six-Dimension Scores

| Dimension | Score | Brief |
|-----------|-------|-------|
| Novelty | 7/10 | Genuinely novel question (editing-attribution geometry); TECS metric and six-component framework are original. The theoretical decomposition is a useful analytical tool though not a deep theorem. |
| Soundness | 7/10 | P4 revisions addressed key P3 issues: claims now conditional on BM25, decomposition gap explicitly noted, principal angle interpretation corrected. The remaining soundness concern (attribution quality confound) is properly flagged and has a clear experimental resolution path. |
| Significance | 7/10 | Valuable to the knowledge editing and attribution communities. The parameter-level perspective on Hase et al. is a meaningful connection. Broader significance depends on PENDING positive control and cross-model results. |
| Experiments | 6/10 | Pilot data (N=100) is convincing for direction but underpowered for a NeurIPS submission. Many critical experiments (positive control, g_M quality, cross-model) are PENDING. The existing results (core TECS, subspace geometry, whitening, MEMIT) are well-analyzed. Score reflects current state; completion of PENDING experiments would raise to 8. |
| Presentation | 7/10 | Clear writing, logical structure, appropriate length after P4 trimming. Contribution list reduced to 3 (improvement from 5). Key weakness: no figures yet (figure references added but figures not created). Notation is consistent. |
| Reproducibility | 7/10 | Core experiment is simple and reproducible. Attribution pipeline details deferred to appendix (appropriate). EasyEdit version, seed, hardware specified. |

## Composite Calculation

Novelty (25%): 7 * 0.25 = 1.75
Soundness (20%): 7 * 0.20 = 1.40
Significance (15%): 7 * 0.15 = 1.05
Experiments (25%): 6 * 0.25 = 1.50
Presentation (10%): 7 * 0.10 = 0.70
Reproducibility (5%): 7 * 0.05 = 0.35
**Total: 6.75**

**Override consideration**: No dimension <= 4; Novelty > 5. However, the paper is being written with PENDING placeholders by design (Paper Module started before experiment completion). The PENDING experiments have clear designs, computational feasibility, and expected outcomes documented. The paper framework is sound and the claims are appropriately conditioned.

**Final score adjustment**: 7.0 (rounded up from 6.75 based on: (1) the dual-outcome pre-registration and positive control methodology are genuinely novel contributions to research methodology, (2) the available pilot data is internally consistent and well-analyzed, (3) the paper appropriately conditions all claims on PENDING results rather than overclaiming).

## Detailed Review

### Strengths
1. **Novel question with clear answer**: "Do editing and attribution directions share geometry?" is a question nobody has asked, and the answer (no, under BM25 attribution) is informative. This advances understanding of knowledge representation.
2. **Rigorous null result handling**: Five null baselines with Bonferroni correction, pre-registered negative path, positive control design. This is how null results should be reported.
3. **Appropriate claim scoping (post-P4)**: Claims are now conditional on BM25 attribution, the proxy corpus limitation is acknowledged, and the distinction between "incommensurability exists" (TECS) and "incommensurability is structured" (dimensionality) is clearer.
4. **Theoretical decomposition provides falsification value**: The rank-one decomposition predicts TECS > 0 under linear memory assumptions, transforming the null result from "we looked and found nothing" to "theory predicts X, reality shows not-X, therefore theory assumption Y fails."
5. **MEMIT partial bridge**: The constructive finding that multi-layer editing partially recovers alignment (d ~ 0.63) adds depth beyond a pure null result.

### Weaknesses
1. [Major] [needs experiments] **Experiments section is majority PENDING**: Sections 4.6-4.10 are almost entirely placeholders. While understandable for a paper started pre-experiment, the submission cannot proceed until at minimum positive control (4.6) and full-scale core TECS (4.2) are filled.
2. [Major] [rewrite-fixable] **No figures**: A geometry paper needs visual communication. Figure references are placed but figures don't exist yet. Need at minimum: (1) conceptual schematic, (2) TECS distribution, (3) eigenvalue spectra, (4) cross-projection diagram.
3. [Minor] [rewrite-fixable] **paper.md is a condensed summary, not full assembly**: The assembled paper.md truncates the Method and Experiments sections. P6 should work from sections/*.md files directly.
4. [Minor] [rewrite-fixable] **Related Work citations need verification**: Several citations use placeholder-style formatting (author, year) without confirmed BibTeX entries. These need to be verified and entered into main.bib during P6.

### Questions for Authors
1. Have you computed the expected cross-projection ratios for random subspaces of dimensions 1 and 40 in the ambient space? This would contextualize the 17.3% / 1.0% finding.
2. The MEMIT result (d ~ 0.63 cross-layer) is based on only 30 facts with simplified implementation. How confident are you that this holds at scale with proper MEMIT?
3. The WebText vs. WikiText mismatch for the attribution corpus --- have you considered using a WebText proxy or The Pile for retrieval?

## Edit Suggestions (if revision needed — N/A, Pass decision)

N/A — paper passes to P6. Key tasks for P6 and beyond:
1. Create all figures during LaTeX conversion
2. Verify and populate main.bib with all citations
3. Complete PENDING experiments and run /praxis-paper-fill
4. Write appendix with full attribution pipeline details
5. Complete NeurIPS checklist

## P3 Issue Fix Status

| P3 Issue | Status |
|---|---|
| Conditional claims (attribution confound) | FIXED — claims now conditional on BM25 |
| Contribution inflation (5->3) | FIXED — reduced to 3 contributions |
| Decomposition gap (per-sample vs aggregated) | FIXED — gap explicitly noted in Method 3.3 |
| Principal angle interpretation | FIXED — distinction between incommensurability and structured |
| 34:1 vs 40:1 inconsistency | FIXED — standardized to 34:1 |
| SNR formula in abstract | FIXED — removed |
| No figures | PARTIAL — references added, figures not yet created |
| PENDING experiments | NOT ADDRESSABLE in paper module — correctly deferred |
| Attribution pipeline details | PARTIAL — deferred to appendix |
| MEMIT simplified note | FIXED — explicitly stated |
| Individual fact analysis | ADDED — Section 4.10 with PENDING placeholder |
