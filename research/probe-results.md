---
version: "1.0"
created: "2026-03-17"
last_modified: "2026-03-17"
---

# Probe Results: TECS 探针实验

## 1. 实验概览

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2-XL (1.5B) |
| 编辑层 | Layer 17 (ROME default) |
| 事实数 | 50 (CounterFact 随机采样, seed=42) |
| 训练样本来源 | OpenWebText (BM25 检索, 500K docs shard) |
| BM25 top-k candidates | 100 |
| 梯度 top-k samples | 10 |
| TDA 方法 | Raw gradient mean (无 Hessian) |
| Null-A unrelated directions | 10 per fact |
| Null-B placebo | 同层不相关事实梯度 (2 per fact) |
| GPU | 单张 24GB GPU |
| 总耗时 | ~6.4 分钟 |

## 2. 前置诊断结果

### 2.1 Sanity Check (Plan B Pilot, 5 facts)

| Fact | TECS | Edit Success | Angular Var |
|------|------|-------------|-------------|
| Angola → Antarctica | 0.0082 | ✗ | -0.016 |
| Shanghai → Dresden | 0.0020 | ✗ | 0.005 |
| 2011 Cannes Film Festival → Prescott | 0.0435 | ✗ | 0.027 |
| Delta Goodrem → India | -0.0011 | ✓ | 0.037 |
| Google Patents → Microsoft | 0.0333 | ✓ | 0.213 |

**VERDICT: PASS** — 管线产出非 NaN、非零 TECS 值，代码端到端正确。

### 2.2 SVD Projection Diagnostic (5 facts)

| Fact | Δθ proj (top-10 SVs) | g proj (top-10 SVs) | Risk |
|------|----------------------|---------------------|------|
| Roger Chartier | 0.0065 | 0.0555 | Low |
| Nirdoshi | 0.0081 | 0.0180 | Low |
| Peterloo Massacre | 0.0024 | 0.0227 | Low |
| Somali Region | 0.0057 | 0.0274 | Low |
| Libby Dam | 0.0054 | 0.0188 | Low |

- **Mean Δθ projection**: 0.56% (极低)
- **Mean g projection**: 2.85% (极低)
- **VERDICT**: Spectral confound risk **LOW** — 所有事实。Δθ 和 g 都不集中在 W 的 top-10 奇异方向上，TECS 的观测值不会被 spectral artifact 驱动。

## 3. 主探针实验结果

### 3.1 核心统计

#### Test 1: TECS(real) vs TECS(null-A)

| 指标 | 值 | Pass 阈值 | 结果 |
|------|-----|----------|------|
| Mean TECS(real) | **-0.00109** | > 0.05 | **FAIL** |
| Mean TECS(null-A) | -0.00081 | — | — |
| t-statistic | -0.476 | — | — |
| p-value | **0.637** | < 0.05 | **FAIL** |
| Cohen's d | **-0.067** | > 0.5 | **FAIL** |
| 95% CI for d | [-0.345, 0.210] | — | — |

#### Test 2: TECS(edit layer) vs TECS(placebo)

| 指标 | 值 | Pass 阈值 | 结果 |
|------|-----|----------|------|
| Mean TECS(edit) | -0.00109 | — | — |
| Mean TECS(placebo) | 0.00101 | — | — |
| t-statistic | -2.745 | — | — |
| p-value | 0.008 | < 0.05 | PASS (但方向相反) |
| Cohen's d | -0.388 | > 0.5 | FAIL |

#### Kill Gate

| 指标 | 值 | Kill 阈值 | 结果 |
|------|-----|----------|------|
| Mean angular variance | 0.0142 | < 0.001 | **NOT KILLED** |

#### Null-C: Edit Success vs Failure

| 组 | Mean TECS | n |
|----|-----------|---|
| Edit success | **+0.00275** | 7 |
| Edit failure | **-0.00172** | 43 |

### 3.2 Overall Verdict

**>>> OVERALL: FAIL <<<**

所有 4 个 pass criteria 均未满足：
1. ✗ 统计显著性: p = 0.637 (远超 0.05)
2. ✗ Effect size: Cohen's d = -0.067 (远低于 0.5)
3. ✗ Mean TECS: -0.00109 (低于 0.05 阈值，甚至为负)
4. ✗ Placebo test: Cohen's d = -0.388, 且方向相反（编辑层 TECS 低于不相关事实梯度 TECS）

Kill gate 未触发: angular variance = 0.014 > 0.001，说明梯度方向本身有意义的一致性（非纯噪声）。

## 4. 结果分析

### 4.1 核心发现

**探针假设被证伪**: TDA 归因方向（raw gradient mean of top-k training samples）与 ROME 编辑方向（rank-1 参数更新）在 GPT-2-XL 编辑层 (Layer 17) 的参数空间中**不存在统计显著的几何对齐**。

具体表现：
- TECS(real) 均值为 **负值** (-0.00109)，不仅不对齐，甚至有微弱的反对齐趋势
- TECS(real) 与 TECS(null-A) 无统计差异 (p = 0.637)，说明匹配事实的编辑方向与不匹配事实的编辑方向在梯度对齐度上没有区别
- Effect size 接近零 (d = -0.067)，不是"弱信号"而是"无信号"

### 4.2 Fail Mode 诊断

本次失败对应 problem-statement 中的 **Fail 模式 2**：

> g_M 有方向一致性（angular variance = 0.014 > 0.001，kill gate 未触发），但 TECS(real) ≈ TECS(null)（Cohen's d < 0.5）

这意味着：
- 梯度方向本身是有意义的（训练样本的梯度之间有正向对齐，非随机噪声）
- 但这些梯度方向与 ROME 编辑方向**无关**
- TDA 归因捕捉的参数空间结构与 ROME 假设的知识局部化结构**不同**

### 4.3 有趣的次级发现

1. **编辑成功 vs 失败分组**: 编辑成功的 7 个事实 TECS 均值 (+0.00275) 略高于编辑失败的 43 个事实 (-0.00172)。方向上一致但样本量太小 (n=7)，无法得出有意义的结论。

2. **Placebo test 反向显著**: 不相关事实的梯度在编辑层上与 ROME delta 的对齐 (+0.00101) 竟然高于匹配事实 (-0.00109)，且差异统计显著 (p = 0.008)。这是一个意外发现——可能暗示 ROME 的编辑方向更多反映"一般性参数空间操作方向"而非"特定事实的知识方向"。

3. **Spectral confound 排除**: SVD 诊断确认 spectral confound 风险极低，说明失败不是因为 spectral artifact 掩盖了真实信号，而是真的没有信号。

### 4.4 局限性

1. **ROME 编辑成功率低 (14%)**: 50 个事实中仅 7 个编辑成功。这可能反映简化的 ROME 实现（无 covariance 矩阵估计、仅 20 步优化）不够强，导致编辑方向本身质量不高。但即使只看编辑成功的子集 (n=7, mean TECS = +0.00275)，信号也极弱。

2. **OpenWebText shard 500K**: 仅使用了 500K 文档的子集做 BM25 检索。真实 OpenWebText 有 ~8M 文档。检索质量可能不够好，导致检索到的"训练样本"与事实相关性不高。但 angular variance > 0 说明检索到的样本梯度是有方向性的。

3. **Raw gradient 是最简 TDA 方法**: 未使用 Hessian 加权 (IF) 或 EK-FAC。更精确的 TDA 方法可能产出不同结果。

## 5. 结论与后续决策

### 5.1 科学结论

在 GPT-2-XL 上，使用 raw gradient TDA 和 ROME rank-1 editing，参数空间中的 TDA 归因方向与模型编辑方向**不存在几何一致性**。这提供了以下实证信息：

- **知识操作方向不共享参数子空间**: 训练归因方向（"哪些训练样本贡献了该知识"的梯度累积方向）与知识编辑方向（"修改该知识需要在参数空间中移动的方向"）指向参数空间的不同区域
- 这与近期关于知识非局部化的发现 (Hase et al., 2301.04213; 知识分离存储, 2409.00617) 一致——知识可能不以 ROME 假设的方式局部存储在单一层的参数子空间中

### 5.2 后续选项

按照 problem-statement §3.5 中预设的 fail 模式 2 后续动作：

1. **检查局部对齐**: 在特定层子集或特定关系类型上是否存在局部对齐（本次实验仅检查了 Layer 17）
2. **升级 TDA 端**: 从 raw gradient → EK-FAC IF，看是否改善对齐
3. **升级编辑端**: 从 ROME → MEMIT (batch editing)，看是否改善
4. **负面结果发表**: "TDA 归因方向与模型编辑方向不一致：知识分布式存储的参数空间实证" 可作为 Workshop 论文
