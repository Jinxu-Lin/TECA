# Notation Reference

| Symbol | Definition | First appears |
|---|---|---|
| $W^{(l)}$ | MLP weight matrix at layer $l$, $\in \mathbb{R}^{d_v \times d_k}$ | Method 3.1 |
| $d_k, d_v$ | Key and value dimensions (both 1600 for GPT-2-XL) | Method 3.1 |
| $l^*$ | Critical editing layer (layer 17 for GPT-2-XL) | Method 3.1 |
| $z = (s, r, o)$ | Factual association (subject, relation, object) | Method 3.1 |
| $\Delta W_E$ | ROME rank-one editing update | Method 3.2 |
| $g_M$ | Aggregated attribution gradient (normalized) | Method 3.2 |
| TECS | TDA-Editing Consistency Score, $\cos(\operatorname{vec}(\Delta W_E), \operatorname{vec}(g_M))$ | Method 3.2 |
| $k^*$ | Subject's key representation | Method 3.2 |
| $v^*$ | Target value encoding new fact | Method 3.2 |
| $C$ | Key covariance matrix $\mathbb{E}[kk^\top]$ | Method 3.2 |
| $K$ | Number of top retrieved documents (default 20) | Method 3.2 |
| $w_i$ | BM25 relevance weight for document $i$ | Method 3.2 |
| $\mathcal{L}$ | Cross-entropy loss on object token | Method 3.2 |
| $d_{\text{eff}}$ | Effective dimensionality (eigenvalue entropy) | Method 3.5 |
| $\mathcal{S}_E, \mathcal{S}_A$ | Editing and attribution subspaces | Method 3.5 |
| G-in-D, D-in-G | Cross-projection variance ratios | Method 3.5 |
| $\rho_k, \rho_v$ | Key-space and value-space correlations | Method 3.3 |
| $d$ (Cohen's) | Effect size metric | Method 3.4 |
