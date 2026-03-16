# Contribution Tracker: TECA

> 本文档跨阶段维护，记录项目贡献的演化过程。

---

## 贡献列表

### Startup 初始化

| # | 贡献 | 类型 | 来源阶段 | 状态 |
|---|------|------|---------|------|
| C0 | TDA 评估缺乏独立验证通道（不依赖重训练）的方法论空白 | Gap 识别 | Startup | 初始 |
| C1 | TECS（TDA-Editing Consistency Score）：参数空间几何一致性指标 | 方法创新 | Startup | 初始 |
| C2 | TDA 归因方向与模型编辑方向的几何关系实证分析 | 实验发现 | Startup | 初始 |

### C 阶段（问题锐化）

| # | 贡献 | 类型 | 来源阶段 | 状态 |
|---|------|------|---------|------|
| C0' | 精确化：RRO 范式的系统性脆弱性——重训练 ground truth 在非凸模型上不可靠，且所有评估 metric（Spearman、LDS、Counterfactual）共享此弱点 | Gap 深化 | C | 活跃 |
| C1' | TECS 定义精确化：cos(Δθ_E^{l*}, g_M^{l*})，需 null TECS 控制实验作为统计基础 | 方法精化 | C | 活跃 |
| C3 | **负面结果的贡献**：如果 TECS 全面失败（所有方法 ≈ random），这本身是关于知识分布式存储的实证证据——TDA 归因方向与模型编辑方向不一致意味着两者假设的知识结构不同 | 潜在发现 | C | 条件性 |
| C4 | 诊断工具：g_M angular variance 作为 TDA 梯度方向质量的前提检查指标 | 方法论工具 | C | 活跃 |

---

## 贡献评估

### 整体发表价值评估

| 评估维度 | 评级 | 论据 |
|---------|------|------|
| Novelty | 中-高 | 参数空间几何一致性作为 TDA 评估维度是全新的；RRO 范式的系统性批判在 Revisiting Fragility 的基础上进一步推进 |
| Significance | 中 | 方法论贡献，非 SOTA 突破；如果 TECS 有效，解锁非重训练评估范式具有基础设施级别的价值；即使失败，作为负面结果也提供知识分布式存储的参数空间证据 |
| 与目标会议/期刊的匹配度 | 中 | 方法论分析型工作，适合 EMNLP/ACL Findings 或 NeurIPS Workshop；在 ROME 可靠性争议（Hase et al.）未充分回应的情况下，main conference poster 需要 RQ2 全面成功 + 扩展实验 |

### Claim 层级（从高到低）

1. **最强 claim（需全面正面结果 + placebo pass）**：TECS 在编辑层特异性地捕捉到 TDA 归因方向与模型编辑方向的几何对齐，且能区分不同质量的 TDA 方法
2. **中间 claim（需部分正面结果）**：TDA 归因方向与模型编辑方向在参数空间中存在可测量的几何对齐，但 TECS 的方法区分度有限
3. **最弱 claim（负面结果也成立）**：参数空间几何一致性分析揭示了 TDA 归因结构与模型编辑假设之间的根本性差异，为知识分布式存储假说提供参数空间层面的实证证据

### 贡献定位（RS-Revise-2 后调整）

TECS 的定位从"TDA 验证工具"调整为"参数空间知识几何学的探索性实证研究"。降级后 TECS 回答的核心科学问题是：**知识在参数空间中的表征结构是否在不同知识操作（训练归因 vs 模型编辑）中呈现一致的几何特征？** 这个问题连接到参数空间知识几何学（knowledge geometry in parameter space）——一个尚无系统实证研究的领域。正面结果意味着不同知识操作共享参数子空间结构，为理解大模型中知识如何被参数化提供几何层面的证据；负面结果意味着训练归因与知识编辑操作假设了不同的参数空间知识组织方式，为知识分布式存储假说提供参数空间层面的实证。

与 RepSim（表征空间相似度）的区分：RepSim 已被证明是有效的非重训练 TDA 归因替代（Do IF Work on LLMs, 2409.19998），但它工作在表征空间（activations），测量的是训练样本与测试样本的内部表示相似性。TECS 的独特价值在于打开参数空间维度——直接探测 weights 层面的知识几何结构，这是 RepSim 无法提供的信息。两者互补而非竞争。

更诚实的一句话 pitch：**"We empirically investigate whether TDA attribution directions and model editing directions align in parameter space, revealing [positive/negative] evidence about the geometric structure of knowledge representations in neural network weights."**

---

## Metadata
- **目标会议/期刊**: EMNLP/ACL Findings 或 NeurIPS Workshop（RS 审查后下调；RQ2 全面成功可尝试 main conference）
- **上次更新**: C 阶段（RS-Revise-2）
- **当前状态**: 贡献框架已建立，科学问题锚点明确化，待探针验证
