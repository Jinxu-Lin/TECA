# Methodology

## 3.1 Problem Formulation

We study the geometric relationship between two independent probes of factual knowledge in transformer MLP parameter space: rank-one model editing directions and training data attribution gradients. Given a transformer with $L$ layers, each containing an MLP with weight matrix $W^{(l)} \in \mathbb{R}^{d_v \times d_k}$, and a factual association $z = (s, r, o)$ (subject $s$, relation $r$, object $o$), we ask: do the parameter-space directions identified by knowledge editing and training data attribution share geometric structure?

For each fact $z$, knowledge editing (ROME) produces a rank-one weight update $\Delta W_E \in \mathbb{R}^{d_v \times d_k}$ at a critical layer $l^*$, while training data attribution produces an aggregated gradient $g_M \in \mathbb{R}^{d_v \times d_k}$ at the same layer. Both are elements of the same parameter space. We characterize their geometric relationship through a novel metric (TECS) and a six-component analysis framework.

## 3.2 TECS: TDA-Editing Consistency Score

**Motivation.** If knowledge editing and attribution access the same underlying knowledge representation, their parameter-space directions should exhibit detectable alignment. We need a metric that directly measures this geometric relationship without relying on retrain-based ground truth.

**Definition.** For a factual association $z = (s, r, o)$ edited at layer $l^*$, we define the TDA-Editing Consistency Score as:
\begin{equation}
\text{TECS}(z) = \cos\!\bigl(\operatorname{vec}(\Delta W_E),\; \operatorname{vec}(g_M)\bigr)
\label{eq:tecs}
\end{equation}
where $\operatorname{vec}(\cdot)$ flattens a matrix into a vector, $\Delta W_E$ is the ROME rank-one editing update, and $g_M$ is the normalized aggregated attribution gradient.

**Editing direction.** ROME computes the rank-one update at layer $l^*$ as:
\begin{equation}
\Delta W_E = \frac{(v^* - W^{(l^*)} k^*)  (C^{-1} k^*)^\top}{(C^{-1} k^*)^\top k^*}
\label{eq:rome_update}
\end{equation}
where $k^* \in \mathbb{R}^{d_k}$ is the subject's key representation, $v^* \in \mathbb{R}^{d_v}$ is the target value encoding the new fact, and $C = \mathbb{E}[k k^\top]$ is the key covariance matrix estimated from a reference corpus. This update modifies the MLP's key-value association at a single critical layer while minimally perturbing other stored associations.

**Attribution direction.** We compute the aggregated attribution gradient as:
\begin{equation}
g_M(z) = \frac{\sum_{i \in \text{top-}K} w_i \cdot \nabla_{W^{(l^*)}} \mathcal{L}(x_i; \theta)}{\bigl\|\sum_{i \in \text{top-}K} w_i \cdot \nabla_{W^{(l^*)}} \mathcal{L}(x_i; \theta)\bigr\|}
\label{eq:attribution_gradient}
\end{equation}
where $\{x_i\}_{i \in \text{top-}K}$ are the $K$ training documents most relevant to fact $z$ (retrieved via BM25 over a proxy corpus), $w_i$ are BM25 relevance weights, and $\mathcal{L}$ is the cross-entropy loss on the object token. We note that BM25 retrieval produces proxy attribution gradients --- the retrieved documents approximate, rather than exactly identify, the training data responsible for the fact.

## 3.3 Theoretical Foundation: Rank-One Decomposition

**Motivation.** Under the linear associative memory model ($W k = v$), TECS admits an analytic decomposition that yields testable predictions about when editing-attribution alignment should exist.

For a single training sample $z_i$, substituting Eq.~\ref{eq:rome_update} into Eq.~\ref{eq:tecs} and expanding yields:
\begin{equation}
\text{TECS}(z_i) = \operatorname{sign}(\alpha) \cdot \cos(C^{-1}k^*,\; k_i) \cdot \cos(v^* - Wk^*,\; \delta_{v,i})
\label{eq:tecs_decomposition}
\end{equation}
where $\alpha$ is a positive scalar, $k_i$ is the key representation of training sample $i$, and $\delta_{v,i}$ is the value-space gradient component. TECS decomposes into a product of key-space alignment and value-space alignment.

This per-sample decomposition does not directly apply to the aggregated metric in Eq.~\ref{eq:tecs}, which involves a weighted sum of gradients. However, it provides two analytical predictions that bound the expected behavior. First, the expected squared TECS under random alignment satisfies $\mathbb{E}[\text{TECS}_{\text{random}}^2] \sim 1/d_k$, establishing the noise floor. Second, the signal-to-noise ratio for the aggregated metric scales approximately as $\text{SNR} \sim \rho_k \cdot \rho_v \cdot \sqrt{d_k}$, where $\rho_k$ and $\rho_v$ measure average key-space and value-space correlations across the top-$K$ samples. For GPT-2-XL ($d_k = 1600$), even weak average correlations ($\rho_k \cdot \rho_v > 0.08$) should produce Cohen's $d > 0.3$. The failure to observe this signal constrains how badly the linear associative memory assumptions fail in practice, or alternatively, how poorly BM25-based retrieval identifies the relevant training samples.

## 3.4 Null Baselines

To calibrate TECS against chance-level alignment, we employ five null baselines:

\begin{enumerate}
\item \textbf{Null-A (Random-fact)}: TECS between fact $z_i$'s editing direction and fact $z_j$'s attribution gradient ($i \neq j$). Controls for fact-specificity.
\item \textbf{Null-B (Wrong-layer)}: TECS computed at layer $l^* \pm 5$ instead of $l^*$. Controls for layer-specificity.
\item \textbf{Null-C (Shuffled-gradient)}: TECS with randomly permuted gradient components. Controls for gradient structure.
\item \textbf{Null-D (Random-direction)}: TECS between the editing direction and a random unit vector. Controls for dimensional concentration.
\item \textbf{Null-E (Test-gradient)}: TECS using test-set gradients instead of training-set gradients. Controls for gradient source.
\end{enumerate}

All comparisons use Cohen's $d$ with 10,000-iteration bootstrap 95\% confidence intervals, corrected for multiple comparisons via Bonferroni adjustment ($\alpha = 0.01$).

## 3.5 Six-Component Geometric Analysis Framework

When TECS fails to detect alignment, we deploy a structured analysis framework to characterize the geometry of the incommensurability.

**Component 1: Subspace dimensionality.** We compute the effective dimensionality of each direction set via eigenvalue entropy:
\begin{equation}
d_{\text{eff}} = \exp\!\Bigl(-\sum_i p_i \log p_i\Bigr), \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}
\label{eq:eff_dim}
\end{equation}
where $\{\sigma_i\}$ are the singular values from the SVD of the stacked direction matrix. This measures how many dimensions capture the variance of the direction set.

**Component 2: Principal angle analysis.** We compute the principal angles between the editing subspace $\mathcal{S}_E = \operatorname{span}(\Delta W_1, \ldots, \Delta W_N)$ and the attribution subspace $\mathcal{S}_A = \operatorname{span}(g_1, \ldots, g_N)$ at subspace dimensions $k \in \{10, 20, 50\}$. We compare minimum principal angles against a random subspace baseline (1000 trials) to test whether the editing-attribution misalignment exceeds chance levels.

**Component 3: Cross-projection analysis.** We measure asymmetric overlap via two variance ratios:
\begin{align}
\text{G-in-D} &= \frac{\|P_{\mathcal{S}_E} G\|_F^2}{\|G\|_F^2}, &
\text{D-in-G} &= \frac{\|P_{\mathcal{S}_A} D\|_F^2}{\|D\|_F^2}
\label{eq:cross_projection}
\end{align}
where $P_{\mathcal{S}}$ denotes projection onto subspace $\mathcal{S}$. Asymmetry between these ratios reveals whether the overlap is directional.

**Component 4: Whitening decomposition.** We test whether ROME's statistical whitening ($C^{-1}$ in Eq.~\ref{eq:rome_update}) is the primary source of geometric incommensurability by comparing standard TECS against an unwhitened variant using raw $k^*$.

**Component 5: Multi-method comparison (MEMIT).** MEMIT distributes edits across multiple layers ($l^* - 4$ through $l^*$). We measure TECS between MEMIT deltas and attribution gradients both within-layer and cross-layer, testing whether distributed editing bridges the incommensurability gap.

**Component 6: Attribution quality analysis.** We assess $g_M$ quality through: (a) within-fact vs.\ between-fact gradient similarity, (b) retrieval method ablation (BM25, TF-IDF, dense retrieval), and (c) PC1 removal to test whether the dominant shared component masks a weaker fact-specific signal. This component is essential for distinguishing properties of knowledge geometry from artifacts of the attribution pipeline.

## 3.6 Positive Control Experiments

A critical methodological concern is whether TECS can detect alignment when it exists. We address this through a three-tier positive control design.

**Tier 1: Self-alignment (sanity check).** We compute TECS between a ROME editing direction and a noisy copy of itself at varying noise levels, confirming the metric pipeline is correct.

**Tier 2: Toy linear associative memory.** We construct a controlled setting where editing-attribution alignment holds by design: a 3-layer MLP ($d = 64$, ReLU) trained on 200 synthetic key-value associations, with ROME-style rank-one edits and exact per-sample gradients (no retrieval approximation). Under these conditions, the rank-one decomposition predicts TECS $> 0$. If this prediction holds for the toy model but fails for real transformers, the null result in GPT-2-XL is informative about knowledge geometry rather than metric failure.

**Tier 3: Semantically related facts.** We compute TECS between one fact's ROME edit direction and attribution gradients for semantically related facts (same relation type, different subject), testing for partial geometric structure.
