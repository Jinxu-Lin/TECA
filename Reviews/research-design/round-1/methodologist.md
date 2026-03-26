## [Methodologist] Design Review — TECA

> Advisory review (assimilation context: experiments complete)

### Evaluation Protocol Assessment

**Strengths:**
1. Five null baselines provide comprehensive controls (fact-specificity, layer-specificity, gradient-structure, dimensional-concentration, gradient-source)
2. Cohen's d with bootstrap CI as primary metric (not p-values alone) — best practice
3. Bonferroni correction for multiple comparisons
4. Pre-registered decision gates (d > 0.2 threshold set before experiments)
5. Seeds fixed, tensors saved, EasyEdit commit pinned

**Concerns:**
1. **BM25 retrieval coverage**: Only 20 documents per fact from a Wikipedia subset. GPT-2's actual training data (WebText) is not fully available. The attribution gradients may not reflect the actual training influence.

2. **Ablation study pending**: The pilot was done with a single configuration. The four-axis ablation (top-k, weighting, loss, gradient scope) is necessary to confirm the null result is robust across settings.

3. **No data leakage risk**: Not applicable — this is a measurement study, not a prediction task. Correct.

### Ablation Coverage

Planned ablation axes:
- Top-k cutoff: k ∈ {5, 10, 20, 50} — **adequate**
- Weighting: BM25 / uniform / TF-IDF — **adequate**
- Loss definition: object token CE / full-sequence / margin — **adequate**
- Gradient scope: single layer / multi-layer — **adequate**

Missing ablation:
- **Retrieval method**: BM25 is one approach; contriever or other dense retrieval could produce different TDA gradients
- **Aggregation method**: Mean vs subspace projection (top-r SVD) — the startup debate (Theorist) suggested this

### Hyperparameter Selection

No hyperparameters to tune — TECS is a direct measurement. This is a strength of the experimental design.

### Assessment: PASS

The evaluation protocol is rigorous for a measurement study. The pending ablation study is important but straightforward to execute. Recommend adding retrieval method ablation if time permits.
