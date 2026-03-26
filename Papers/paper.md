# Decomposing the Geometry of Knowledge Operations: Structured Incommensurability Between Editing and Attribution in Parameter Space

## Abstract

Knowledge editing and training data attribution both operate on transformer MLP weight matrices to probe factual knowledge, yet whether their parameter-space directions share geometric structure is unknown. We introduce TECS (TDA-Editing Consistency Score), a cosine similarity metric between ROME's rank-one editing updates and aggregated attribution gradients, and apply it to GPT-2-XL across {{PENDING: abstract_n_facts | number of CounterFact facts | 200}} CounterFact facts with five null baselines and Bonferroni correction. TECS is indistinguishable from noise (Cohen's $d = 0.050$), but a six-component geometric analysis reveals this null result arises from structured incommensurability: editing directions span a ~40-dimensional manifold while attribution gradients collapse to an effectively one-dimensional subspace (34:1 dimensionality ratio), with asymmetric cross-projection (17.3% vs. 1.0%) and principal angles comparable to random baselines. A theoretical rank-one decomposition predicts that even weak correlations should produce detectable signal under the linear associative memory model, constraining how severely these assumptions fail in practice. {{PENDING: abstract_positive_control | Positive control results | Expected: TECS significantly > 0 on toy model, confirming metric validity}}. These findings provide a parameter-level perspective on the localization-editing disconnect and establish that, under current attribution methods, different knowledge operations access geometrically distinct structures in the same parameter space.

## 1. Introduction

Knowledge editing and training data attribution represent two fundamental approaches to understanding factual knowledge in large language models. Knowledge editing methods such as ROME (Meng et al., 2022a) and MEMIT (Meng et al., 2022b) perform targeted rank-one updates to MLP weight matrices, treating knowledge as localized key-value mappings in parameter space. Training data attribution (TDA) methods based on influence functions (Koh & Liang, 2017) and gradient-based techniques (Park et al., 2023) estimate which training examples are responsible for a model's factual predictions by computing parameter-space gradients. Both paradigms operate on the same weight matrices from complementary directions --- editing modifies parameters to change knowledge, while attribution measures parameter sensitivity to identify knowledge sources --- yet no prior work has examined whether these two notions of "knowledge direction" share geometric structure.

This gap has concrete consequences. Hase et al. (2023) demonstrated a persistent disconnect between knowledge localization (via causal tracing) and knowledge editing success, yet offered no parameter-level explanation. The editing and attribution communities have developed in isolation, each with independent evaluation paradigms. A basic geometric question remains unanswered: when ROME identifies a parameter-space direction for editing a fact, and TDA identifies a direction for attributing the same fact to training data, do these directions share any geometric structure?

In this paper, we introduce the TDA-Editing Consistency Score (TECS), a cosine similarity metric between rank-one editing updates and aggregated attribution gradients at the editing layer, and apply it to GPT-2-XL across CounterFact facts with five null baselines and Bonferroni correction. TECS is indistinguishable from noise (Cohen's $d = 0.050$). A six-component geometric analysis framework reveals this null result arises from a structured incommensurability under BM25-based attribution: editing directions span a ~40-dimensional manifold while attribution gradients collapse to an effectively one-dimensional subspace (34:1 dimensionality ratio), with asymmetric cross-projection (17.3% vs. 1.0%) and principal angles comparable to random baselines. A theoretical rank-one decomposition predicts that even weak correlations should produce detectable signal under the linear associative memory model, transforming the null result into a quantitative constraint on how severely these assumptions fail.

Our contributions are:
- We propose TECS, the first metric for comparing knowledge editing directions and attribution gradients in parameter space, with a theoretical rank-one decomposition that yields testable predictions under the linear associative memory model.
- We characterize structured geometric incommensurability between editing and attribution subspaces --- a 34:1 effective dimensionality ratio, large principal angles, and asymmetric cross-projection --- providing a parameter-level perspective on the localization-editing disconnect (Hase et al., 2023).
- We develop a six-component geometric analysis framework and a positive control methodology using a toy linear associative memory to validate metric sensitivity, establishing that the null result in real transformers is informative about knowledge geometry rather than metric failure.

## 2. Related Work

### Knowledge Editing in Language Models

Knowledge editing methods modify specific factual associations in pretrained language models without full retraining. ROME (Meng et al., 2022a) performs rank-one updates to MLP weight matrices at a critical layer identified by causal tracing, treating factual knowledge as key-value associations in a linear associative memory. MEMIT (Meng et al., 2022b) extends this to multi-layer distributed editing for batch fact insertion. More recent approaches include PMET (Li et al., 2024), which operates on both attention and MLP modules, and constrained fine-tuning methods (Zhu et al., 2020). Knowledge neurons (Dai et al., 2022) provide a complementary perspective by identifying individual neurons associated with factual knowledge. Our work examines the editing assumption directly by measuring whether ROME's editing directions share geometric structure with independently computed attribution directions.

### Training Data Attribution

Training data attribution methods estimate the influence of individual training examples on model predictions. Influence functions (Koh & Liang, 2017) approximate leave-one-out effects via the inverse Hessian, but scale poorly and exhibit fragility on deep networks (Basu et al., 2021). TRAK (Park et al., 2023) projects gradients to a lower-dimensional space for computational tractability. Recent work has revealed significant limitations: Wang et al. (2024) document failure modes of influence functions on LLMs. In the subspace approach, MDA (Kwon et al., 2024) decomposes influence into subspace components, and RIF (Tang et al., 2025) shows that influence function accuracy depends critically on effective dimensionality. Our work characterizes the geometric properties of attribution gradients and their relationship to an independent knowledge probe (editing).

### Knowledge Localization and the Localization-Editing Disconnect

Causal tracing (Meng et al., 2022a) identifies MLP layers causally responsible for factual predictions. Hase et al. (2023) demonstrated that knowledge localization does not reliably predict editing success. We note that our work addresses a related but distinct question: Hase et al. study whether localization predicts editing success (behavioral), while we study whether editing and attribution directions share geometric structure (geometric). Both findings point to different knowledge operations accessing different parameter-space structures.

### Geometric Analysis of Neural Network Parameter Space

Geometric perspectives on neural network parameters have yielded insights through loss landscape analysis (Li et al., 2018), mode connectivity (Draxler et al., 2018; Frankle et al., 2020), probing (Hewitt & Manning, 2019), and embedding geometry (Ethayarajh, 2019). However, the geometry of parameter-space directions associated with specific knowledge operations has not been studied.

### Unified Perspectives on Knowledge Operations

Recent surveys (Zhang et al., 2025) have attempted to unify knowledge editing and attribution, but focus on methodological comparisons rather than geometric relationships. MAGIC (Zhao et al., 2025) demonstrates that attribution quality varies dramatically with retrieval methods, motivating our attribution quality analysis. None examines whether parameter-space directions from different knowledge operations are geometrically commensurable.

## 3. Methodology

### 3.1 Problem Formulation

We study the geometric relationship between two independent probes of factual knowledge in transformer MLP parameter space. For each fact $z = (s, r, o)$, knowledge editing (ROME) produces a rank-one weight update $\Delta W_E \in \mathbb{R}^{d_v \times d_k}$ at layer $l^*$, while training data attribution produces an aggregated gradient $g_M \in \mathbb{R}^{d_v \times d_k}$ at the same layer. We characterize their geometric relationship through TECS and a six-component analysis framework.

### 3.2 TECS: TDA-Editing Consistency Score

TECS(z) = cos(vec(Delta_W_E), vec(g_M))

where ROME computes Delta_W_E = (v* - W k*)(C^{-1} k*)^T / (C^{-1} k*)^T k*, and g_M is the normalized BM25-weighted aggregation of per-sample training gradients at layer l*.

### 3.3 Theoretical Foundation: Rank-One Decomposition

Under the linear associative memory model, TECS decomposes into key-space alignment times value-space alignment. This yields predictions: E[TECS_random^2] ~ 1/d_k (noise floor) and SNR ~ rho_k * rho_v * sqrt(d_k). For GPT-2-XL (d_k=1600), even weak correlations (rho_k * rho_v > 0.08) should produce Cohen's d > 0.3.

### 3.4 Null Baselines

Five null baselines: Null-A (random-fact), Null-B (wrong-layer), Null-C (shuffled-gradient), Null-D (random-direction), Null-E (test-gradient). Bonferroni-corrected at alpha = 0.01.

### 3.5 Six-Component Framework

1. Subspace dimensionality (eigenvalue entropy)
2. Principal angle analysis
3. Cross-projection analysis
4. Whitening decomposition
5. Multi-method comparison (MEMIT)
6. Attribution quality analysis

### 3.6 Positive Control Experiments

Tier 1 (self-alignment), Tier 2 (toy linear associative memory), Tier 3 (semantically related facts).

## 4. Experiments

### 4.1 Setup

GPT-2-XL (1.5B, 48 layers, d_k=d_v=1600), CounterFact, ROME at l*=17, BM25 top-20, single RTX 4090.

### 4.2 Core TECS: d = 0.050, all null baselines non-significant.

### 4.3 Geometric Characterization: eff-dim 40.8 vs 1.2 (34:1), principal angles > 56 deg, G-in-D = 17.3%, D-in-G = 1.0%.

### 4.4 Whitening: H6 rejected (d = -0.198, p = 0.051).

### 4.5 MEMIT: cross-layer d ~ 0.63.

### 4.6-4.10: Positive control, g_M quality, ablation, cross-model, individual fact analysis — {{PENDING}}.

## 5. Conclusion

First geometric characterization of editing-attribution relationship. 34:1 dimensionality asymmetry under BM25 attribution. Parameter-level perspective on localization-editing disconnect. Limitations: single layer, BM25 proxy corpus, single model.

## References

[See main.bib]
