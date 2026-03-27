# Code Review Report

> 审查时间: 2026-03-27
> 审查基线: method-design.md v1.1 + experiment-design.md v1.1 + experiment-todo.md v1.0
> 代码快照: 6ecca1f

## 总体评估

| 维度 | 判定 | 关键发现 |
|------|------|---------|
| 架构忠实度 | Pass | 所有 method-design 组件和 experiment-todo 脚本均有对应实现，core/experiments 分层清晰 |
| 组件忠实度 | Concern | Null-E 实现为 sign-flip 而非 test-gradient；whitening_200.py covariance hook 在错误维度空间；weighting ablation 的 "tfidf" 实际是 rank-inverse |
| Ablation 工程 | Pass | 4 个 ablation 维度均可通过 config YAML 驱动，无需修改代码 |
| DL 常见 Bug | Concern | Null-C/D/E 随机操作缺少显式 seed generator；gradient_utils.py 未冻结非目标参数（部分调用者遗漏外部冻结） |
| 可复现性 | Concern | 结果 JSON 不含 git commit hash；requirements.txt 缺少 scikit-learn；Null baselines 随机操作未使用 seeded torch Generator |
| 计算效率 | Pass | 逐 fact 处理 + gc.collect() + empty_cache() 模式正确，显存需求在 24GB 内 |

**总体判定**: Needs Fix（2 项必须修复涉及实验忠实度，5 项建议修复涉及可复现性和效率）

---

## 维度 1: 架构忠实度

### 1.1 method-design 组件 -> core/ 映射

| method-design 组件 | core/ 文件 | 状态 |
|---------------------|-----------|------|
| TECS Core (S2) | `core/tecs.py` | OK |
| TDA Gradient (S2.1) | `core/gradient_utils.py` | OK |
| ROME Editing (S2) | `core/rome_utils.py` + `core/easyedit_rome.py` | OK |
| BM25 Retrieval (S2.1, S8.2) | `core/retrieval.py` | OK |
| Statistics (S4) | `core/statistics.py` | OK |
| SVD Diagnostics (S3.2) | `core/svd_diagnostics.py` | OK |
| Config | `core/config.py` | OK |
| Model Utils | `core/model_utils.py` | OK |

### 1.2 experiment-todo 脚本 -> experiments/ 映射

| experiment-todo 实验 | experiments/ 脚本 | 状态 |
|----------------------|-------------------|------|
| Phase 0: sanity checks | `run_experiment.py` | OK |
| Phase 1a: ROME vs Self | `positive_control/rome_self_check.py` | OK |
| Phase 1b: Toy Model | `positive_control/toy_model_tecs.py` | OK |
| Phase 1c: Related Facts | `positive_control/related_facts_tecs.py` | OK |
| Phase 2a: Within/Between | `gm_quality/within_between_similarity.py` | OK |
| Phase 2b: PC1 Removal | `gm_quality/pc1_removal.py` | OK |
| Phase 2c: Retrieval Ablation | `gm_quality/retrieval_ablation.py` | OK |
| Phase 3: ROME 200 | `full_scale/rome_200.py` | OK |
| Phase 3: TDA 200 | `full_scale/tda_gradients_200.py` | OK |
| Phase 3: TECS Core 200 | `full_scale/tecs_core_200.py` | OK |
| Phase 3: Subspace Geometry | `full_scale/subspace_geometry_200.py` | OK |
| Phase 4: Top-k Ablation | `ablation/topk_ablation.py` | OK |
| Phase 4: Weighting Ablation | `ablation/weighting_ablation.py` | OK |
| Phase 4: Loss Ablation | `ablation/loss_ablation.py` | OK |
| Phase 4: Scope Ablation | `ablation/scope_ablation.py` | OK |
| Phase 5: Whitening | `full_scale/whitening_200.py` | OK |
| Phase 5: MEMIT | `full_scale/memit_200.py` | OK |
| Phase 6: Cross-Model | `cross_model/gptj_*.py` (4 files) | OK |

### 1.3 core/experiments 分层

- `core/` 只包含可复用核心组件，无实验特定逻辑。
- `experiments/` 脚本都通过 `from core.xxx import ...` 调用核心逻辑。
- `experiments/common.py` 提供实验间共享的统计和 IO 工具。
- 模块依赖方向：experiments -> core -> model_utils。

**判定: Pass**

---

## 维度 2: 组件忠实度

### 2.1 core/tecs.py

- `compute_tecs()` 调用 `cosine_similarity_flat()`，即 `cos(vec(delta_W), vec(g_M))`，与 method-design S2.1 一致。
- Flatten 使用 `.reshape(-1).float()`，正确处理任意 shape。
- 零除保护：`norm < 1e-12` 返回 0.0。
- `compute_null_a` 正确（用 unrelated facts 的 delta）。
- `compute_mean_pairwise_cosine` 的 O(n^2) 对 top-k=20 可接受。

**状态: Pass**

### 2.2 core/gradient_utils.py

- `compute_gradient_at_layer`: zero_grad -> enable grad on target param -> forward -> backward -> clone grad -> restore。正确。
- `compute_aggregated_gradient`: 支持 BM25 权重参数 `weights`（line 90-94），normalize 后加权求和。正确。
- 当 `weights=None` 时使用均匀 mean（line 96）。正确。

**Concern C-2.2a**: `compute_gradient_at_layer` 仅对目标参数 enable grad 但未冻结其他参数。虽然 `loss.backward()` 后只读取 `param.grad`，backward pass 仍会为所有 `requires_grad=True` 的参数计算梯度，浪费显存。`tda_gradients_200.py`（line 69-70）和 `loss_ablation.py`（line 129-131）在外部冻结了所有参数，但 `retrieval_ablation.py`、`pc1_removal.py`、`within_between_similarity.py` 没有。

- 文件: `core/gradient_utils.py` line 31-48
- 影响: 显存浪费（可能多用 ~4-6GB），在 24GB GPU 上余量减小但仍可行。不影响正确性。
- 严重度: Low-Medium

### 2.3 core/easyedit_rome.py

- 正确 bootstrap EasyEdit 模块，避免 `__init__.py` 依赖链。
- Delta 提取: `deltas[weight_name]` 返回 `(delta_u, delta_v)`，通过 `delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)` 重建 rank-1 delta，然后 `upd_matrix_match_shape` 匹配原始权重形状。正确。
- 非破坏性: 先备份权重，应用 delta 测量 post_prob，然后恢复。正确。

**状态: Pass**

### 2.4 core/rome_utils.py (builtin backend)

- Key vector: `_compute_key_vector` 使用 forward hook 在 `c_proj` 输入端捕获中间激活（d_ff 维度）。正确。
- Target value optimization: 通过 hook 替换 MLP 输出实现约束优化。正确。
- Rank-1 update: `delta = (k @ v_target^T) / (k^T @ k)`，使用 identity covariance（C=I）。Default backend 是 easyedit（使用真正的 C^{-1}），builtin 是 simplified backup。文档已说明。
- edit success 判定: `post_prob > pre_prob and post_prob > 0.1`。合理。

**状态: Pass**

### 2.5 core/retrieval.py

- BM25: 使用 `rank_bm25.BM25Okapi`，三级缓存（pickle / in-memory / build-fresh）。正确。
- CounterFact 加载: 支持本地文件 / 自动下载 / HuggingFace，seed 控制 sampling。正确。
- `rank_by_gradient_dot_product`: 存在但未被实验脚本使用。无害。

**状态: Pass**

### 2.6 core/statistics.py

- Cohen's d: `np.mean(diff) / np.std(diff, ddof=1)`。标准 paired-samples Cohen's d。正确。
- Bootstrap CI: `np.random.RandomState(seed)` 控制 seed，`np.percentile(2.5, 97.5)`。正确。
- Paired t-test: `scipy.stats.ttest_rel`。正确。
- Normal approximation CI for d: `se_d = sqrt(1/n + d^2/(2n))`。标准公式。正确。
- `experiments/common.py` 中的 `paired_test` 支持 bootstrap CI (`bootstrap_n` 参数)。正确。

**状态: Pass**

### 2.7 core/svd_diagnostics.py

- Projection ratio: `P_k(M) = Uk @ (Uk^T @ M @ Vk^T) @ Vk`，双侧投影到 top-k 奇异向量子空间。Frobenius 范数比。正确。
- Risk assessment: 阈值硬编码（0.8 high, 0.5 medium），合理。

**状态: Pass**

### 2.8 core/config.py

- YAML 继承: `_base_` key 递归加载 + deep merge。正确。
- CLI override: dot-separated key path。正确。
- Validation: 检查 model name, dtype, num_facts, retrieval method。合理。
- Defaults 与 `base.yaml` 一致。

**状态: Pass**

### 2.9 experiments/positive_control/toy_model_tecs.py

- 3-layer MLP (d_k=64, d_v=64, d_hidden=128)，与 method-design S7.2 一致。
- Training: MSE loss on values + 0.1 * readout loss，合理。
- ROME-style edit: `delta_W = v_target outer h_star / (h_star^T h_star)`，正确的 rank-1 update。
- per-sample gradient: 对所有 200 个 training sample 计算梯度，均匀聚合。因为 toy model 不需要 retrieval，均匀权重合理。
- TECS 使用 cosine similarity，所以 g_M 的 L2 归一化不影响结果。正确。
- Null baseline: `torch.randperm` 打乱 gradient 元素（类似 Null-C）。可接受。

**Concern C-2.9a**: Rank-1 decomposition correlation check（line 259-279）重新生成 `v_new = torch.randn(1, d_v)`（line 261），使用的编辑与主循环不同。统计上无问题，但 decomposition 验证不是针对主实验的同一组编辑。

- 文件: `experiments/positive_control/toy_model_tecs.py` line 259-261
- 严重度: Low

### 2.10 experiments/full_scale/tecs_core_200.py (5 Null Baselines)

**[FAIL] Null-E 实现与 experiment-design 不匹配。**

experiment-design S3 Table:
- Null-E: **Test-gradient TECS** -- "Gradient-source control"

method-design S2 的含义是 Null-E 应使用 test prompt（即 CounterFact prompt 本身）的梯度代替 training sample 的梯度 g_M。但代码实现（line 149-156）是对 g_M 的每个元素做 **random sign flip**:
```python
signs = torch.sign(torch.randn_like(gm))
g_sign_flipped = gm * signs
```
注释也承认: `# Null-E: test-gradient (use delta as its own "test gradient" direction proxy, rotated by random orthogonal transform...)`。

实际实现是 Null-C 的变体（sign flip vs shuffle），不是 test-gradient baseline。

- 文件: `experiments/full_scale/tecs_core_200.py` line 148-156
- 影响: Null-E 不测试 "gradient source" 控制变量。
- 严重度: Medium（实验解释力降低，但 Null-A/C/D 已覆盖大部分控制需求）

**Concern C-2.10b**: Null-C 和 Null-D 的 `torch.randperm` / `torch.randn_like` 未使用显式 seed generator（line 133-145）。依赖全局 torch RNG。虽然 `set_seed(seed)` 在开头调用，不同 PyTorch 版本 / 设备可能产生不同序列。

- 文件: `experiments/full_scale/tecs_core_200.py` line 133-156
- 严重度: Low-Medium

### 2.11 experiments/gm_quality/retrieval_ablation.py

- TF-IDF: `sklearn.TfidfVectorizer` + cosine similarity。正确。
- Contriever: `facebook/contriever` CLS token embedding + cosine similarity。正确。
- Uniform: `np.random.RandomState(seed)` 随机采样。正确。
- Effective dimensionality: `torch.svd_lowrank` + eigenvalue entropy。正确。

**状态: Pass**

### 2.12 experiments/ablation/loss_ablation.py

- object_token_ce: 取最后 3 个 token 的 CE。正确。
- full_sequence_ce: 标准 `model(labels=input_ids).loss`。正确。
- margin: `log P(correct) - log P(second_best)`，对 masked logits 重新 log_softmax 后取 max。正确。

**状态: Pass**

### 2.13 experiments/full_scale/whitening_200.py

**[FAIL] Covariance matrix hook 在错误的维度空间。**

`compute_covariance_matrix` (line 52) hooks into `block.mlp.c_fc`（即 c_fc 的 **输入**，d_model=1600 维度），产生 C 形状为 (1600, 1600)。

但 ROME 的 covariance C 应该在 key space（c_proj 的 **输入** = d_ff = 6400 维度）。EasyEdit 的 ROME 实现中，C = E[k k^T]，k 是 c_proj 输入激活（6400 维）。

代码在 line 155 用 `delta_W.shape[1] == C.shape[0]`（即 1600 == 1600）匹配并执行 `delta_W @ C`（(6400,1600) @ (1600,1600)），这在数学上是把 C 应用到 delta_W 的 value/output space（d_model=1600），而非 key space（d_ff=6400）。

- 文件: `experiments/full_scale/whitening_200.py` line 52
- 影响: whitening decomposition（H6 分析）在错误空间中进行，结论可能无效。
- 严重度: Medium-High

### 2.14 experiments/full_scale/memit_200.py

代码使用 ROME delta at layer 17 与 MEMIT 各层的 TDA gradient 做 cosine similarity，但未实际运行 MEMIT 编辑获取各层的 MEMIT delta。这是有意简化（pilot 中已使用此方法），与 method-design S3.6 "Measure alignment at each layer" 的措辞有偏差但可接受。

- 严重度: Low

### 2.15 experiments/ablation/weighting_ablation.py

**Concern C-2.15a**: "tfidf" weighting 方法（line 109-110）实际使用 `1.0 / (rank + 1)` rank-inverse weighting，不是真正的 TF-IDF 分数。注释写 `# Simple TF-IDF-like: inverse document frequency proxy using BM25 rank`。这与 experiment-design S4 中 "TF-IDF" 条目的含义不符。

- 文件: `experiments/ablation/weighting_ablation.py` line 109-110
- 影响: ablation 结果标注为 "TF-IDF" 但实际是 rank-inverse weighting，可能误导论文读者。
- 严重度: Low-Medium

### 2.16 experiments/full_scale/subspace_geometry_200.py

- 使用 `torch.svd_lowrank` 投影到 joint subspace（避免对 200 x 10.24M 矩阵做 full SVD）。正确。
- `compute_principal_angles` 使用 `torch.linalg.svd(V1^T @ V2)` + `arccos(clamp(sigmas, 0, 1))`。标准方法，正确。
- 1000 random subspace trials for p-value。与 experiment-design S5.1 一致。
- Cross-projection: `(G @ V_D)^2.sum() / G^2.sum()`。正确。

**状态: Pass**

### 2.17 experiments/positive_control/rome_self_check.py

- Noise: `torch.randn_like(delta) * sigma * delta.norm()`。与 experiment-design S6.1 一致（epsilon ~ N(0, sigma^2 ||delta||^2 I)）。
- Monotonicity check: `means[i] >= means[i+1] - 0.01` 允许 0.01 tolerance。合理。
- Pass criteria: TECS(sigma=0)=1.0, TECS(sigma=1.0)>0.3。与 experiment-todo 一致。

**状态: Pass**

### 2.18 experiments/gm_quality/pc1_removal.py

- PC1 提取: `svd_lowrank` -> `Vh[:, 0]` 作为 PC1 方向。正确。
- PC1 removal: `g_flat - dot(g_flat, pc1) * pc1`。标准投影移除，正确。
- 对比 |TECS| before vs after。正确。

**状态: Pass**

**维度 2 判定: Concern**

---

## 维度 3: Ablation 工程

### 3.1 Config 驱动性

| Ablation 维度 | Config Key | 默认值 | 脚本 | 可切换 |
|---------------|-----------|--------|------|--------|
| Top-k | `ablation.top_k_values` | [5, 10, 20, 50] | `topk_ablation.py` | Yes |
| Weighting | `ablation.weighting_methods` | ["bm25", "uniform", "tfidf"] | `weighting_ablation.py` | Yes |
| Loss | `ablation.loss_functions` | ["object_token_ce", "full_sequence_ce", "margin"] | `loss_ablation.py` | Yes |
| Scope | `ablation.multi_layer_range` | 2 | `scope_ablation.py` | Yes |

### 3.2 Config 文件完整性

- `configs/base.yaml`: 完整覆盖所有 ablation 参数。
- `configs/phase_4_ablation.yaml`: 继承 base，override ablation 参数。
- 所有 ablation 脚本通过 `cfg.get("ablation", {})` 读取参数。

### 3.3 Robustness 判定

所有 ablation 脚本计算 `max_relative_variation` 并与 20% 阈值比较（与 experiment-design S4 一致）。

**判定: Pass**

---

## 维度 4: DL 常见 Bug

### 4.1 Shape/Broadcasting

- TECS cosine similarity: `a.reshape(-1).float()` + `F.cosine_similarity` -- 正确。
- gradient aggregation: `stacked * w.view(-1, *([1] * (stacked.dim() - 1)))` -- 正确 broadcast 权重。
- SVD projection 切片: `U[:, :top_k]`, `Vh[:top_k, :]` -- 正确。

**状态: Pass**

### 4.2 数值稳定性

- Cosine similarity 零除: `norm < 1e-12` 返回 0.0。正确。
- SVD: `torch.linalg.svd` (full) 和 `torch.svd_lowrank` (low-rank, niter=5)。正确。
- Eigendecomposition: `whitening_200.py` 使用 `torch.linalg.eigh` + threshold `max * 1e-6`。正确。
- Entropy 中 log(0) 防护: `p = p[p > 1e-12]`。正确。

**状态: Pass**

### 4.3 Seed 管理

**Concern C-4.3a**: Null-C/D/E 在 `tecs_core_200.py` 中使用 `torch.randperm` 和 `torch.randn_like` 依赖全局 torch RNG，未使用 `torch.Generator`。虽然 `set_seed(seed)` 已调用，不同 PyTorch 版本/设备可能产生不同序列。

- 文件: `experiments/full_scale/tecs_core_200.py` line 133-156
- 影响: 跨环境可复现性受损。

**Concern C-4.3b**: `toy_model_tecs.py` line 203 `v_new = torch.randn(1, d_v)` 在循环中消耗全局 RNG，影响后续 `torch.randperm` 确定性。若 try/except 跳过某些 fact，RNG 状态会漂移。

- 文件: `experiments/positive_control/toy_model_tecs.py` line 203, 222
- 严重度: Low

**Concern C-4.3c**: `rome_self_check.py` line 117 `torch.randn_like(delta)` 使用全局 RNG 生成 noise，虽然 torch seed 已设置但跨版本不保证。

- 严重度: Low

### 4.4 梯度计算

- per-sample gradient: 每次 `model.zero_grad()` 后 forward+backward。正确。
- detach/clone: `param.grad.detach().clone().cpu()`。正确。
- requires_grad restore: 备份后恢复。正确。
- no_grad: forward hooks 和 key vector 计算正确使用 `torch.no_grad()`。

**Concern C-4.4a**: `gradient_utils.py` 中 `compute_gradient_at_layer` 只对目标参数 enable grad，未 freeze 其他参数。`loss.backward()` 会对所有 `requires_grad=True` 参数计算梯度。部分调用者未在外部冻结:
- `experiments/gm_quality/retrieval_ablation.py`
- `experiments/gm_quality/pc1_removal.py`
- `experiments/gm_quality/within_between_similarity.py`

- 影响: 显存浪费 ~4-6GB，不影响正确性。
- 严重度: Medium（可能导致 OOM 如果显存紧张）

### 4.5 Memory 管理

- 所有 full-scale 脚本在 fact 循环中调用 `gc.collect()` + `torch.cuda.empty_cache()`。
- 处理完后 `del model; gc.collect()`。
- 逐 fact 处理（避免 200 facts x 大 tensor OOM）。
- `subspace_geometry_200.py` 使用 `svd_lowrank` 而非 full SVD。

**状态: Pass**

**维度 4 判定: Concern**

---

## 维度 5: 可复现性

### 5.1 Seed 全链路

`experiments/common.py::set_seed()` 设置 `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`。所有实验脚本在入口处调用。完整。

但 Null baselines 的 torch 随机操作未使用显式 Generator（见 C-4.3a）。

### 5.2 结果 JSON 元数据

**Concern C-5.2a**: 结果 JSON 包含 `timestamp`, `config`, `seed`，但不包含 **git commit hash**。experiment-todo S Environment Preparation 要求结果包含 git commit 信息。

- 文件: `experiments/common.py` line 41-45
- 严重度: Low

### 5.3 requirements.txt

**Concern C-5.3a**: `requirements.txt` 缺少 `scikit-learn`，但 `retrieval_ablation.py`（line 50）和 `weighting_ablation.py` 使用 `sklearn`。

- 文件: `Codes/requirements.txt`
- 影响: 干净环境安装后无法运行 Phase 2c TF-IDF retrieval ablation。
- 严重度: Low

当前 `requirements.txt` 已有版本 pin（如 `torch==2.5.1`），这比之前的 `>=` 约束改善很多。

### 5.4 中间结果保存

- ROME deltas: `_Data/rome_deltas_200/delta_{cid}.pt`。正确。
- TDA gradients: `_Data/tda_gradients_200/g_M_{cid}.pt`。正确。
- Config: `base.yaml` 中 `save_tensors: true`。正确。

### 5.5 Config 快照

Phase-specific YAML configs 使用 `_base_: "base.yaml"` 继承，参数可追溯。但结果 JSON 中只包含部分 config 信息，不是完整的 resolved config dump。

**维度 5 判定: Concern**

---

## 维度 6: 计算效率

### 6.1 逐 fact 处理

所有 full-scale 实验逐 fact 处理，每次只在 GPU 上加载一个 fact 的 tensor。避免 200 facts 同时在 GPU 上的 OOM。

### 6.2 不必要的重复计算

Ablation 实验（topk, weighting, loss, scope）各自独立加载模型并重新计算 ROME edits，未复用 Phase 3 预计算结果。这是效率优化机会而非正确性问题。

**优化点**: `topk_ablation.py` 和 `weighting_ablation.py` 先计算 max_k 个 gradients 再 subset，正确避免了重复梯度计算。

### 6.3 GPU/CPU 转换

- Gradient 在 GPU 上计算，`.cpu()` 后存储。无不必要 ping-pong。
- SVD 在 CPU 上进行。正确。

### 6.4 显存需求估算

| 组件 | 估算 VRAM |
|------|----------|
| GPT-2-XL (FP32) | ~6GB |
| 单个 fact gradient 计算 (forward + backward) | ~2-4GB |
| ROME edit (forward + hooks) | ~2GB |
| BM25 index (CPU) | 0 GPU |
| **总计 (正常, 已 freeze 非目标参数)** | **~10-12GB** |
| **总计 (未 freeze, 见 C-4.4a)** | **~14-18GB** |

24GB RTX 4090 充足，但未 freeze 时余量较小。

**维度 6 判定: Pass**

---

## 修复建议

### 必须修复（影响实验忠实度）

| # | 维度 | 问题 | 修复方案 | 文件 |
|---|------|------|---------|------|
| F1 | 组件忠实度 | Null-E 实现为 random sign flip 而非 experiment-design 定义的 test-gradient | 方案 A: 实现真正的 Null-E——对每个 fact，用其 CounterFact prompt 计算 `grad_W L(prompt; theta)` 代替 g_M。方案 B: 如果认为当前实现合理，将 experiment-design 中 Null-E 描述更新为 "sign-flipped gradient (structure destruction)" 并新增说明 | `experiments/full_scale/tecs_core_200.py` line 148-156 |
| F2 | 组件忠实度 | whitening_200.py 的 covariance hook 在 `c_fc` 输入（d_model=1600 维）而非 `c_proj` 输入（d_ff=6400 维），与 ROME 的 C^{-1} 空间不匹配 | 将 line 52 的 hook 从 `block.mlp.c_fc` 改为 `block.mlp.c_proj`，捕获 `input[0]`（6400 维），产生 (6400, 6400) 的 C。注意：6400x6400 矩阵的 eigendecomposition 计算量大，可改用 low-rank approximation（如 top-512 eigenvalues/vectors） | `experiments/full_scale/whitening_200.py` line 52 |

### 建议修复（影响可复现性或清晰度）

| # | 维度 | 问题 | 修复方案 | 文件 |
|---|------|------|---------|------|
| S1 | DL Bug / 可复现性 | Null-C/D/E 使用全局 torch RNG 而非显式 Generator | 在循环外创建 `gen = torch.Generator().manual_seed(seed)`，所有 `torch.randperm(..., generator=gen)` 和 `torch.randn(..., generator=gen)` 使用它 | `experiments/full_scale/tecs_core_200.py` line 133-156 |
| S2 | DL Bug / 效率 | `gradient_utils.py` 未冻结非目标参数，部分调用者也未外部冻结 | 在 `compute_gradient_at_layer` 开头添加全参数 freeze 逻辑，或在 `retrieval_ablation.py`、`pc1_removal.py`、`within_between_similarity.py` 的模型加载后添加 `for p in model.parameters(): p.requires_grad_(False)` | `core/gradient_utils.py` line 31-48; `experiments/gm_quality/retrieval_ablation.py`; `experiments/gm_quality/pc1_removal.py`; `experiments/gm_quality/within_between_similarity.py` |
| S3 | 可复现性 | 结果 JSON 不含 git commit hash | 在 `experiments/common.py::save_results()` 中添加 `results["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()` | `experiments/common.py` line 41-45 |
| S4 | 可复现性 | requirements.txt 缺少 scikit-learn | 添加 `scikit-learn>=1.3.0` | `Codes/requirements.txt` |
| S5 | 组件忠实度 | weighting_ablation.py 的 "tfidf" 方法实际是 rank-inverse weighting `1/(rank+1)` 而非 TF-IDF 分数 | 方案 A: 使用 sklearn TfidfVectorizer 计算真实 TF-IDF similarity 作为权重。方案 B: 将 method name 改为 "rank_inverse" | `experiments/ablation/weighting_ablation.py` line 109-110 |
