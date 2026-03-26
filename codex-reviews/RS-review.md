# External AI Review — RS — TECA

**Reviewer**: External AI (Codex)
**Date**: 2026-03-16
**Artifact**: `research/problem-statement.md` v1.2 (entry_mode: rs_revise, iteration 1.2)

---

## Overall Impression

This is the most intellectually self-aware problem statement I have reviewed in this domain. The authors have done something unusual: they systematically documented the ways their own approach could fail, downgraded their claim from "independent TDA validator" to "exploratory geometry probe," and designed layered null baselines. The core observation — that RRO is epistemically circular and unreliable at scale — is correct, well-documented, and undervalued by the community.

The danger is that this self-awareness has become a liability. After two rounds of revision, the document now contains more hedging infrastructure than scientific claim. The gap is important enough for a main-track paper; the current attack angle is positioned for a Findings slot. There is a mismatch between the ambition of the problem and the reach of the proposed solution — and the document knows it but has not resolved it. The next step is not more hedging. It is a strategic choice: either strengthen the attack to match the gap's importance, or reframe the problem so it is proportionate to the attack.

**Elevator pitch test**: "Two kinds of parameter operations — training attribution and knowledge editing — should leave similar geometric traces if they operate on the same knowledge structures. We measure whether they do." This passes. Two sentences, novel question, unknown answer. A reviewer would keep reading.

**Competing lab test**: A TDA lab (e.g., the TRAK/MAGIC authors) would view this as a methodological critique of their evaluation framework — they would pay attention. A model editing lab (ROME/MEMIT) would see it as an external use of their tools — interesting but not threatening. No one is currently racing to answer this specific question. Competitive risk is low.

---

## The Blind Spot Report

**Blind Spot 1: The gap and the solution are decoupled — and the paper does not close the loop.**

The stated gap is: RRO ground truth is unreliable. The proposed contribution is: TECS measures parameter-space geometric alignment between TDA attribution directions and ROME editing directions. These are related but distinct. TECS does not make RRO more reliable. It does not provide an alternative to LDS. Even a strongly positive TECS result leaves the practitioner without guidance on which TDA method to use for practical attribution tasks. The paper will face the reviewer question: "You identified a crisis in TDA evaluation. Your proposed metric does not resolve it. What does the paper actually contribute?" The current answer — "we open a new analysis dimension" — is defensible but thin. This gap has been acknowledged in §2.3 point 4 but not addressed architecturally.

**Blind Spot 2: The directionality of TECS is ambiguous and unaddressed.**

TECS = cos(Δθ_E^{l*}, g_M^{l*}). ROME's Δθ_E is the direction that reduces loss for the *new* fact o'. The gradient g_M is computed w.r.t. the original test prompt's loss (the *original* fact o). These two vectors have opposing causal relationships to the same fact slot: one writes a new answer, one reflects the contribution of training data to the existing answer. Should they align (positive cosine) or oppose (negative cosine)? The document never states the expected sign, which means the pass criterion (TECS mean > 0.05) could be looking in the wrong direction. If the correct prediction is negative alignment — ROME's edit direction opposes the training attribution direction — a null result near zero would mask a signal in the negative cosine range. This is a potential fatal flaw that should be resolved theoretically before the pilot begins.

**Blind Spot 3: LDS is used as the validation standard for a metric designed to complement LDS.**

RQ2 proposes to validate TECS by measuring Spearman correlation between TECS method rankings and LDS method rankings. The paper's entire motivation rests on the claim that LDS is an unreliable ground truth. Using LDS to validate TECS is circular: a high correlation means TECS agrees with LDS (which may share LDS's noise), not that TECS is measuring something meaningful. The document briefly notes that TECS and LDS measure "fundamentally different attributes" — but this makes RQ2's validation criterion even less coherent. If they measure different things, low Spearman would not be a failure; it would be expected. RQ2 needs to be redesigned with a validation standard that does not depend on LDS.

**Blind Spot 4: The training data sourcing problem can invalidate the experiment silently.**

The pilot requires retrieving training documents that causally contributed to GPT-2-XL's encoding of specific facts. BM25 retrieval from OpenWebText returns lexically similar documents, not causally influential ones. For common facts, BM25 may return documents that *discuss* the fact without being among the model's actual knowledge sources — Wikipedia mirrors, news aggregators, or synthetic text. For rare facts, BM25 may return zero useful results. In both cases, g_M would be the gradient of non-influential documents, and TECS would measure an undefined quantity. The document acknowledges this but lists it as a "risk" with no pre-validation plan. A 30-minute manual check of 10 facts' retrieved documents should precede the full pilot — not follow a negative result.

**Blind Spot 5: The rank-1 structure of ROME's update creates output-space confound.**

ROME's Δθ_E^{l*} is a rank-1 outer product (value direction × key direction). When flattened to a vector, the cosine similarity with g_M is dominated by the value direction — the specific direction that increases probability of the target token. For factual completions where the target token is a common noun (city names, country names, people), the value direction aligns with the "high-probability token" direction in that layer. Gradients computed on test prompts for next-token prediction also point toward directions that increase probability of high-probability completions. For common factual completions, TECS > 0 could reflect shared output-space pressure (both vectors point toward "make high-probability tokens more probable") rather than shared knowledge localization. This is a different and more fundamental confound than the spectral artifact already identified — it is specific to the rank-1 structure of ROME's formulation.

---

## Strengths

1. **The gap is real, documented, and timely.** The critique of RRO (miss-relation in Spearman + non-convex retraining divergence) rests on published evidence (2303.12922) and is not adequately addressed by the field. The problem statement earns its claim.

2. **Three-tier null baselines are methodologically credible.** Null-A (unrelated fact edit directions), Null-B (placebo layers), and Null-C (editing failure cases) represent genuine rigor. Very few papers in this space pre-register this level of experimental control.

3. **The SVD front-loaded spectral diagnosis is a smart addition.** Running spectral analysis *before* the full pilot provides decision-relevant information cheaply. This is the right order of operations.

4. **Claim hierarchy is graduated and honest.** The document explicitly distinguishes what can be claimed under different experimental outcomes. This would hold up under reviewer scrutiny.

5. **Cross-field novelty is real and not easily scooped.** Bridging TDA evaluation and model editing via parameter-space geometry is genuinely novel. The "two parameter-space operations on the same knowledge object should share geometric structure" hypothesis is the kind of question that should have been asked and wasn't.

6. **The pilot scope is appropriate.** 50 facts, GPT-2-XL, raw gradients only: this is the right minimum footprint to test core feasibility before investing in EK-FAC or multi-model experiments.

---

## Weaknesses / Concerns

**Critical:**

W1 — **Directionality ambiguity in TECS.** As described in Blind Spot 2, the expected sign of cos(Δθ_E, g_M) is not established. This must be resolved before the pilot — not as a post-hoc interpretation.

W2 — **RQ2 validation is self-undermining.** Using LDS to validate a complement to LDS is circular. RQ2 needs redesign. Suggested alternative: use a small set of facts with known memorization status (e.g., facts with measured perplexity drop during training) as a non-LDS anchor for validation.

W3 — **Rank-1 output-space confound is unmitigated.** The confound described in Blind Spot 5 is not in the risk register and has no current control. Mitigation: decompose TECS into the v-direction component and k-direction component of ROME's rank-1 update, and report each separately. If signal comes only from the v-direction, the interpretation narrows significantly.

**Moderate:**

W4 — **Training data retrieval quality is unvalidated.** A silent retrieval failure would produce uninterpretable results without a pre-pilot manual check.

W5 — **Placebo-layer test is weaker than claimed.** Layers l* ± 5 have correlated spectral structure with l* in transformer MLPs. A matched-spectrum null (random rank-1 direction at l* with the same singular value distribution as Δθ_E) would provide stronger control against spectral confound than comparing nearby layers.

W6 — **The gap-to-solution bridge is weak.** The paper's narrative needs a clearer statement of what a practitioner gains from a positive TECS result. Even a single concrete use case (e.g., "TECS could serve as a cheap pre-screening signal to identify TDA methods worth running LDS on") would strengthen the motivation.

---

## The "Simpler Alternative" Challenge

**Alternative 1 — LDS noise quantification (Direct, publishable in 1 week):**
Instead of proposing a new metric, directly characterize LDS variance. Run LDS on a small model (GPT-2-medium) five times with different random seeds; measure rank correlation variance across seeds and across different subset sizes. If LDS rankings are seed-dependent (Spearman < 0.7 across seeds), this is a clean, direct, and impactful paper establishing the problem without requiring a new method. This would land at ACL/EMNLP main track as a methods critique. TECA could then follow as "given that LDS is noisy, here is an alternative signal."

**Alternative 2 — RepSim at edit layer (Simpler, already validated):**
Instead of TECS (parameter-space gradient cosine), compute RepSim (activation cosine between training samples and test facts) specifically at the ROME-identified edit layer l*. If RepSim at l* is higher than at other layers, this provides evidence for layer-specific knowledge localization without the ROME Δθ_E directionality problem. This has existing empirical support (2409.19998 reports RepSim outperforming IF) and would directly answer whether the edit layer is where attribution signals are strongest.

**Alternative 3 — Subspace version of TECS (More principled):**
Replace single-vector cosine with principal angle between the k-dimensional gradient subspace (from top-k training samples' individual gradient vectors) and ROME's 1-dimensional edit direction. This is geometrically more principled, statistically more stable, and avoids the averaging operation that suppresses individual gradient directions. The principal angle formulation also naturally provides a null distribution via random subspace sampling.

None of these alternatives obviously dominates TECS — but each should be compared explicitly in the paper's motivation section.

---

## Specific Recommendations

**Before running the pilot:**

1. Resolve the directionality question analytically. Write out the causal story: if training sample x_i contributed to the model knowing (s, r, o), the model's parameters θ were pushed toward higher p(o | s, r). ROME's Δθ_E also increases p(o' | s, r) for the new fact. Does the original training gradient point toward or away from the edit direction? This is a 1-hour theoretical exercise. If the expected relationship is negative, the pass criterion needs to change.

2. Add the matched-spectrum null as a third null baseline alongside Null-A and Null-B. Generate 100 random rank-1 matrices at l* with the same singular value profile as ROME's actual update; use their TECS distribution as the spectral-corrected null. This adds < 15 minutes to the pilot and substantially strengthens the experimental design.

3. Manually validate BM25 retrieval for 10 facts before committing to the full pilot. Look at the top-5 retrieved documents for each fact and assess whether they plausibly influenced GPT-2-XL's encoding of that fact. This 30-minute check can prevent 6-10 wasted GPU hours.

4. Redesign RQ2. Replace "Spearman correlation with LDS" with an anchor that does not depend on LDS. Options: (a) memorization-based anchor (facts with high vs. low training set frequency, measured via perplexity); (b) known-provenance mini-corpus (20 synthetic facts introduced via fine-tuning with known training samples, where the causal attribution is ground truth).

**During the pilot:**

5. Report TECS decomposed into v-direction and k-direction components of ROME's rank-1 update. If signal comes exclusively from the v-direction, the interpretation changes: TECS is measuring whether TDA gradients point toward the target-token direction, not toward knowledge localization.

6. Stratify the 50 facts by BM25 retrieval coverage (high / low entity frequency in WebText) and report TECS separately. This enables the interpretation: if TECS is positive only for high-coverage facts, the signal is about retrieval quality, not knowledge geometry.

---

## Score

**4.5 / 10** — calibrated to NeurIPS/ICLR main track

Breakdown:
- Gap quality: +2.5 (real, documented, important)
- Attack novelty: +1.5 (genuinely cross-field, unoccupied space)
- Experimental design rigor: +1.0 (pre-specified kill gates, three null baselines)
- Unresolved directionality ambiguity (W1): -1.5
- Circular RQ2 validation (W3): -1.5
- Rank-1 output-space confound (W3, Blind Spot 5): -1.0
- Gap-solution decoupling (Blind Spot 1): -1.0

With the three pre-pilot fixes (directionality resolution, matched-spectrum null, RQ2 redesign), a positive pilot result would be credible at EMNLP/ACL Findings (6.5). Reaching main track requires: at minimum two models, the rank-1 decomposition analysis, and a validation anchor independent of LDS. The ceiling is real but the current design does not reach it.

The core scientific question is worth asking. Run the theoretical directionality check first — it costs one hour and could save the entire pilot from going in the wrong direction.
