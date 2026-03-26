# Experiments

## 4.1 Experimental Setup

**Models.** We use GPT-2-XL (1.5B parameters, 48 layers, $d_k = d_v = 1600$) as the primary model. {{PENDING: cross\_model | Cross-model validation on GPT-J-6B (6B parameters, 28 layers) | TECS $d$ and eff-dim ratios for GPT-J-6B}}.

**Dataset.** We use CounterFact \citep{meng2022locating}, a benchmark of counterfactual knowledge editing prompts. {{PENDING: full\_scale\_n | Full-scale experiment uses $N$ facts | $N=200$}}. Pilot experiments use 100 facts.

**Editing.** We use EasyEdit with ROME at layer $l^* = 17$ for GPT-2-XL, following the original recommendation. MEMIT edits span layers 13--17. All edits achieve {{PENDING: full\_rome\_efficacy | Full-scale ROME efficacy | $>$95\%}} efficacy (pilot: 100\%).

**Attribution.** Training data attribution uses BM25 retrieval over WikiText/Wikipedia as a proxy corpus to identify the top-$K=20$ most relevant documents per fact. We note that this retrieves documents lexically similar to the factual query, not verified training data; GPT-2 was trained on WebText. Per-sample gradients $\nabla_{W^{(l^*)}} \mathcal{L}(x_i; \theta)$ are computed at the editing layer in FP16. The aggregated gradient $g_M$ is BM25-weighted and $\ell_2$-normalized. Full attribution pipeline details (BM25 parameters, corpus statistics, document segmentation) are provided in Appendix~\ref{app:attribution_details}.

**Statistics.** Effect sizes are Cohen's $d$ with 10,000-iteration bootstrap 95\% CIs. Multiple comparisons use Bonferroni correction ($\alpha = 0.01$ for 5 null baselines). Subspace comparisons use permutation tests (1000 random subspace trials). Seed fixed at 42.

**Hardware.** Single NVIDIA RTX 4090 (24GB) with GPT-2-XL in FP16.

## 4.2 Core TECS Measurement

Table~\ref{tab:tecs_core} presents the core TECS results against all five null baselines (pilot: 100 facts). Figure~\ref{fig:tecs_distribution} shows the overlapping distributions.

TECS is indistinguishable from all null baselines (mean = $1.57 \times 10^{-4}$, std = $6.76 \times 10^{-3}$, 95\% bootstrap CI $[-1.17 \times 10^{-3}, 1.46 \times 10^{-3}]$). The positive/negative ratio is 56/44, consistent with symmetric noise centered at zero. The primary effect size ($d = 0.050$ vs.\ Null-A) falls far below the pre-registered $d > 0.2$ threshold.

{{PENDING: full\_scale\_tecs | Full-scale TECS results with 200 facts | Expected $d < 0.1$, consistent with pilot}}

## 4.3 Geometric Characterization

Having established that TECS detects no scalar alignment, we apply the six-component framework to characterize the structure underlying this null result. We emphasize that the geometric properties reported below are measured under BM25-based attribution; Section~\ref{sec:gm_quality} examines whether these properties change with alternative attribution methods.

### Subspace Dimensionality

The editing and attribution direction sets exhibit a radical dimensionality asymmetry (Table~\ref{tab:subspace}). Editing directions span a $\sim$40-dimensional manifold with a flat eigenvalue spectrum (condition number 2.0, Figure~\ref{fig:eigenvalue_spectra}), indicating that different facts produce geometrically diverse editing updates. Attribution gradients collapse to an effectively one-dimensional subspace where a single principal component captures 91\% of the variance ($d_{\text{eff}} = 1.2$, condition number 33.2). This 34:1 dimensionality ratio reveals that, under BM25-based retrieval, ROME editing explores a rich parameter subspace while attribution is dominated by a single shared direction.

### Principal Angle Analysis

At all subspace dimensions $k \in \{10, 20, 50\}$, the minimum principal angles between editing and attribution subspaces are large ($>$56$^\circ$) and not significantly smaller than random subspace baselines ($p = 0.084$ at $k=10$, $p = 0.989$ at $k=20$, $p = 1.000$ at $k=50$). This establishes that the two subspaces are incommensurable --- their geometric separation is at least as severe as that of random subspaces. The ``structured'' aspect of this incommensurability comes not from the principal angles themselves (which are consistent with random), but from the dimensionality asymmetry and cross-projection pattern described below.

### Cross-Projection Asymmetry

The overlap between subspaces is markedly asymmetric (Figure~\ref{fig:cross_projection}):
- \textbf{G-in-D} = 17.3\%: attribution variance partially captured by the editing subspace.
- \textbf{D-in-G} = 1.0\%: editing variance is essentially invisible from the attribution subspace.

This one-directional overlap indicates that the narrow attribution subspace marginally intersects the broad editing manifold, likely because both involve the same weight matrix. However, the editing manifold's $\sim$40 dimensions are almost entirely invisible from attribution's single dominant direction. This asymmetry distinguishes the observed incommensurability from a purely random relationship: random 1D and 40D subspaces in this ambient space would produce cross-projection ratios of approximately {{PENDING: random\_cross\_projection | Expected cross-projection for random subspaces of matching dimension | $\sim$20\% and $\sim$0.5\%}}, which we use as a baseline comparison.

{{PENDING: full\_scale\_geometry | Full-scale subspace geometry at 200 facts | Expected eff-dim ratio $\sim$30--40:1, cross-projection asymmetry maintained}}

## 4.4 Whitening Decomposition

We test whether ROME's statistical whitening ($C^{-1}$) is the primary source of incommensurability. The unwhitened variant (using raw $k^*$ instead of $C^{-1}k^*$) does not increase TECS ($d = -0.198$, $p = 0.051$). ROME's covariance rotation is not the source of the geometric gap. This eliminates a natural mechanistic explanation --- the incommensurability is more fundamental than a statistical preprocessing artifact.

## 4.5 Multi-Method Comparison: MEMIT

MEMIT distributes edits across layers 13--17. We measure alignment at each layer (pilot: 30 facts). Cross-layer TECS (layers 13--16 delta vs.\ layer 17 gradient) shows $d \approx 0.63$ (medium effect), while matched-layer TECS is trivially high ($d \gg 6.0$). This indicates that MEMIT's distributed editing partially bridges the incommensurability gap, suggesting the editing-attribution geometric relationship is partially recoverable when editing information spans multiple layers. We note that this analysis uses a simplified MEMIT implementation (identity covariance); {{PENDING: full\_memit | Full-scale MEMIT with proper covariance, 200 facts | Expected cross-layer $d \sim 0.5$--$0.8$}}.

## 4.6 Positive Control Experiments

{{PENDING: positive\_control\_self | Tier 1 self-alignment results | TECS decreases monotonically from 1.0 as noise increases}}

{{PENDING: positive\_control\_toy | Tier 2 toy model results | TECS significantly $> 0$ ($d > 0.5$), decomposition correlation $\rho > 0.7$, confirming metric validity}}

{{PENDING: positive\_control\_related | Tier 3 semantically related facts results | Same-relation TECS slightly $>$ cross-relation}}

The positive control experiments are essential for interpreting the null result: they establish that TECS can detect alignment when theoretical conditions are satisfied. If TECS is significantly positive in the toy linear associative memory but indistinguishable from noise in GPT-2-XL, the null result is informative about real knowledge geometry rather than metric failure.

## 4.7 Attribution Quality Analysis ($g_M$)
\label{sec:gm_quality}

The attribution gradient's effective dimensionality of 1.2 (PC1 = 91\%) raises a critical question: does the 34:1 asymmetry reflect a genuine property of knowledge geometry, or is it an artifact of BM25-based retrieval producing degenerate gradients? We address this through three analyses.

{{PENDING: gm\_within\_between | Within-fact vs.\ between-fact gradient similarity | Expected within $>$ between if gradients contain fact-specific signal}}

{{PENDING: gm\_pc1\_removal | PC1 removal analysis: TECS and eff-dim after removing dominant component | Expected eff-dim increases, TECS may increase if PC1 masked signal}}

{{PENDING: gm\_retrieval\_ablation | Retrieval method ablation: BM25 vs.\ TF-IDF vs.\ Contriever vs.\ uniform | Expected eff-dim varies by method, TECS remains near zero}}

If attribution effective dimensionality increases substantially with improved retrieval methods while TECS remains near zero, the incommensurability is robust to attribution quality. If TECS increases with better retrieval, the incommensurability is partially an attribution quality artifact --- an important finding in itself.

## 4.8 Ablation Study

{{PENDING: ablation\_topk | Top-$k$ ablation: $k \in \{5, 10, 20, 50\}$ | Expected TECS Cohen's $d$ variation $< 20\%$}}

{{PENDING: ablation\_weighting | Weighting ablation: BM25 vs.\ uniform vs.\ TF-IDF | Expected minimal effect on TECS}}

{{PENDING: ablation\_loss | Loss ablation: object token CE vs.\ full-sequence vs.\ margin | Expected minimal effect}}

{{PENDING: ablation\_scope | Gradient scope ablation: layer $l^*$ only vs.\ layers $[l^*-2, l^*+2]$ | Expected minimal effect}}

The ablation study tests robustness of the null result across four methodological axes. A robust null result (Cohen's $d$ variation $< 20\%$ across settings) would indicate the incommensurability is not an artifact of specific hyperparameter choices.

## 4.9 Cross-Model Validation

{{PENDING: cross\_model\_gptj | GPT-J-6B results: TECS $d$, eff-dim, principal angles, cross-projection | Expected replication of incommensurability pattern}}

Cross-model validation on GPT-J-6B (6B parameters, different architecture) tests whether the geometric incommensurability generalizes beyond GPT-2-XL.

## 4.10 Individual Fact Analysis

To understand whether the null result masks heterogeneous behavior across facts, we examine the distribution tails. {{PENDING: fact\_analysis | Top-10 and bottom-10 TECS facts: properties, relation types, categories | Expected no systematic pattern distinguishing high/low TECS facts}}.
