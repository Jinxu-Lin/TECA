---
version: "1.2"
created: "2026-03-16"
last_modified: "2026-03-16"
entry_mode: "rs_revise"
iteration_major: 1
iteration_minor: 2
---

# Problem Statement

## 1. Gap 定义

### 1.1 现有方法概览

TDA 评估方法论的核心范式是 **"删除-重训练-观察"**（Remove-Retrain-Observe, RRO）：移除某些训练样本，从头或从 checkpoint 重训练模型，观察目标行为的变化，以此作为归因质量的 ground truth。这一范式的具体实例包括：

- **Spearman 相关**（经典 IF 评估）：逐一移除训练样本，重训练，计算 IF 估计与真实 loss difference 的秩相关
- **LDS（Linear Datamodeling Score）**：对大量随机子集做 leave-k-out 重训练，计算 IF 预测与真实 loss 变化的线性相关（TRAK 推广、MAGIC 声称 LDS ≈ 1.0）
- **Counterfactual retraining**：移除 top-k 归因样本，完全重训练，观察特定能力是否消失（SOURCE、DATE-LM 的 Counterfactual Rate）
- **Leave-one-out（LOO）**：移除单样本重训练的最简形式

所有这些评估方式共享同一个元假设：**重训练过程产出的行为变化是可靠的 ground truth**。

### 1.2 Gap 陈述

**一句话**：现有 TDA 评估方法 **全部** 依赖重训练获取 ground truth，而重训练本身在非凸大模型上系统性不可靠（随机噪声级别的 loss 轨迹、miss-relation 掩盖真实质量差异），导致整个 TDA 评估体系建立在一个 **自指的、不可验证的基础** 之上——我们无法用一个不可靠的度量来判断任何 TDA 方法是否有效。

**详细分析**：

知识库中对这一问题的证据链条如下：

1. **Revisiting Fragility (2303.12922)** 从两个独立角度瓦解 RRO 范式：
   - Spearman 秩相关存在 **miss-relation**——在凸和非凸场景下均给出 ≈0.85 的相关值，但实际 IF 质量截然不同（Figure 4）
   - 从最优点重训练在非凸大模型上 **脱离局部极小值**——重训练 loss 轨迹剧烈震荡（Figure 5），产出的"真实值"是随机噪声

2. **Do IF Work on LLMs? (2409.19998)** 从方法端补充证据：
   - LLM 的参数-行为断层——ASR 差 400 倍而 Δθ 仅差 1.6 倍（Table 5），说明即使参数空间的归因"正确"，也不一定反映行为因果关系
   - IF 在低信噪比下完全崩溃（从 97.5% 跌至 0.1-7.3%），而 RepSim 不依赖梯度/Hessian 却全面优于 IF

3. **KB 元分析**：H-TRAK1、H-RF1、H-MISS1 等多个独立识别的隐式假设均指向同一核心脆弱性——**LDS 是否也存在类似 Spearman 的 miss-relation？** MAGIC 的 LDS ≈ 1.0 可能在掩盖真实的归因质量差异。DATE-LM 发现"没有单一方法在所有任务上最优"，进一步暗示现有评估可能缺乏对真实因果关系的分辨力。

4. **Task-based evaluation 的部分绕道**：DATE-LM (2507.09424) 通过下游任务表现间接评估 TDA 质量（Counterfactual Rate），部分绕开了 RRO 对精确 loss difference 的依赖。但 task-based evaluation 仍然需要重训练（leave-k-out）来产出反事实模型，且其评估信号是行为层面的（"能力是否消失"），无法诊断参数空间层面的归因质量。TECS 关注的问题层面与 task-based evaluation 不同：TECS 探测参数空间几何结构（"归因方向是否指向知识操作区域"），task-based evaluation 探测行为因果性（"移除归因样本是否改变行为"）。两者互补而非竞争。

5. **缺失的独立验证通道**：G-RF2 明确指出 Revisiting Fragility "批判了现有评估方法，但没有提出应该用什么替代"。至今无人提出一个 **不依赖重训练** 的 TDA 验证路径。

### 1.3 Root Cause 分析

**类型**：错误假设 + 被忽视的维度

**逐层追问**：

- **Why-1**：为什么 TDA 评估不可靠？——因为 RRO 范式依赖重训练产出 ground truth，而重训练在非凸模型上是随机过程。
- **Why-2**：为什么所有评估方法都依赖重训练？——因为 TDA 定义的核心就是反事实因果："如果没有这个训练样本，模型会怎样？"回答这个反事实 **必须** 重训练，看似无法绕开。
- **Why-3**：为什么没有非重训练的验证路径？——因为社区将 TDA 评估等同于反事实实验，忽略了一个关键事实：**参数空间中的操作不只有"删除-重训练"一种**。模型编辑（ROME/MEMIT）提供了另一种参数空间干预方式，它直接修改参数以改变特定知识，且不需要重训练。如果 TDA 归因方向和模型编辑方向在参数空间中有可测量的几何关系，这就构成一个独立于重训练的验证信号。
- **Why-4**（最深层）：为什么参数空间几何一致性可能成为验证信号？——TDA 和模型编辑从相反方向操作同一对象：TDA 追溯"哪些训练样本影响了这个知识"（backward），ROME 主动修改"知识存储在哪些参数中"（forward）。两者在参数空间的交汇（或不交汇）本身就是关于知识表征几何结构的实证信号——这种信号完全不经过重训练。**降级后的核心科学问题**：知识在参数空间中的表征结构是否在不同知识操作（训练归因 vs 模型编辑）中呈现一致的几何特征？这个问题独立于"TECS 是否可以替代 RRO"——即使 TECS 不具备验证工具的实用价值，回答这个问题本身推进了我们对参数空间知识几何学（knowledge geometry in parameter space）的理解：正面结果意味着不同知识操作共享参数子空间结构，负面结果意味着训练归因与知识编辑假设了不同的参数空间知识组织方式。

**核心赌注与已知反证**：Why-4 的因果论证依赖一个关键中间假设——**知识在参数空间中是（至少部分）局部化的**，即 ROME 的编辑方向在某种程度上对应知识的存储位置。然而，这个假设已被多项工作动摇：

- **Hase et al. (2301.04213, "Does Localization Inform Editing?")** 发现 causal tracing 定位的层与编辑成功率之间的相关性接近零——causal tracing 的"知识位置"发现可能不是编辑成功的真正原因。
- **知识 entity/relation 分离存储 (2409.00617)** 表明知识可能以 entity 和 relation 分离的方式分布在不同层，而非集中在单一编辑层 l*。
- **Capability > knowledge localization (2502.20992)** 进一步表明 causal tracing 可能更多反映"capability 定位"而非"knowledge 定位"。

**在知识可能非局部化的情况下 TECS 为何仍有信息价值**：即使 ROME 的编辑方向不完美地代理知识的"真实位置"，它仍然是一个**定义明确的参数空间操作方向**——一个使模型从输出 o 变为输出 o' 的最小扰动方向。TDA 归因方向则是另一个定义明确的参数空间方向——训练样本对目标事实的梯度累积方向。两者的对齐或不对齐本身就是一个有信息的实证观察：对齐意味着"修改知识的参数方向"与"贡献知识的训练方向"存在几何联系；不对齐则提供知识分布式存储的参数空间证据。无论哪种结果，都不依赖"ROME 精确找到了知识的唯一真实位置"这个强假设。

**Oracle 思想实验验证**：假设存在一个完美的非重训练验证器，它可以直接告诉我们"这个 TDA 方法的归因方向是否指向知识存储的正确参数区域"。如果这样的验证器存在，我们就不再需要 RRO 范式。参数空间几何一致性是这种验证器的弱近似——TECS 不声称 ROME 方向 **等于** 知识位置，而是探测 ROME 方向与 TDA 方向是否共享参数空间结构。TECS 的 claim 因此被降级为"参数空间几何一致性度量"而非"独立验证工具"。

### 1.4 Gap 评价

| 维度 | 评级 | 论据 |
|------|------|------|
| **重要性** | **高** | 评估方法论是 TDA 领域的基础设施——如果评估本身不可靠，所有声称"方法 A 优于方法 B"的结论都是可疑的。KB 中 H-TRAK1、H-RF1、H-MISS1、H-ASTRA5 等 5+ 个独立条目指向同一问题，说明这是领域级别的系统性弱点，而非个别方法的局限。TDA 的可信度问题随 LLM 规模增长而加剧（重训练成本指数增长，ground truth 质量下降），是一个变得越来越重要的问题。 |
| **新颖性** | **中-高** | Revisiting Fragility 批判了 Spearman/重训练，但未提出替代方案（G-RF2）。无人提出过参数空间几何一致性作为独立验证维度。但需诚实承认：这不是"全新发现问题"，而是"为已知问题提供新的（部分）解法"。LDS 是否存在类似 Spearman 的 miss-relation（H-RF1）目前无人明确检验——MAGIC 的 LDS ≈ 1.0 可能在掩盖真实的归因质量差异，但这需要独立实证。 |
| **可解性** | **中**（有条件的） | 可解性取决于核心假设是否成立：TDA 归因方向与 ROME 编辑方向在参数空间中是否存在可测量的对齐。若不存在，方法论贡献仍成立（作为负面结果——"参数空间几何一致性不能用于 TDA 验证"本身也是有价值的发现），但研究天花板降低。Pilot 成本极低（< 1 GPU-day），风险-收益比有利。 |

**重要性深层判断**：

- 这个问题位于 **TDA 方法论 × 模型编辑 × 知识表征** 三个方向的交汇处，突破有 multiplier effect。
- 解决这个问题不仅提升已有能力的数字，而是 **解锁新的评估范式**——非重训练验证通道。
- 问题随模型规模增长变得更重要（大模型的重训练评估成本和 ground truth 噪声同时增加）。

### 1.5 Research Questions

**RQ1（核心）**：TDA 归因方向（聚合梯度向量 g_M）与 ROME 编辑方向（rank-1 参数更新 Δθ_E）在编辑层 l* 的参数空间中是否存在统计显著的几何对齐（余弦相似度显著高于 random baseline，Cohen's d > 0.5），且该对齐在编辑层特异性地存在（而非 spectral artifact）？

- **可证伪**：如果所有 TDA 方法（含梯度方法）的 TECS 与 random baseline 无统计差异（p > 0.05 且 Cohen's d < 0.5），则参数空间对齐不存在。
- **Placebo 子条件**：TECS 在编辑层 l* 应显著高于非编辑层（如 l* ± 5）的 TECS——若非编辑层 TECS 同样显著，则观测到的对齐可能是 spectral artifact 而非知识结构信号。
- **预测力**：如果 RQ1 为真，预测质量更高的 TDA 方法（LDS 排名更高的方法）应产出更高的 TECS——即 TECS 应能区分不同质量的 TDA 方法。
- **边界**：限于 GPT-2-XL/GPT-J 规模、ROME 作为编辑方法、CounterFact 数据集上的事实型知识。不声称对所有知识类型或所有模型规模成立。

**RQ2（条件性，RQ1 为正才有意义）**：TECS 能否区分不同质量的 TDA 方法？具体地，TECS 方法排序与 LDS 方法排序的 Spearman 秩相关是否 > 0.5？

- **可证伪**：TECS 排序与 LDS 排序 Spearman < 0.5，则 TECS 不是 LDS 的有效补充维度。
- **边界**：TECS 被定位为 LDS 的独立补充维度（不是替代），因为两者测量的是根本不同的属性（参数空间几何 vs 反事实行为变化）。

---

## 2. 攻击角度

### 2.1 候选攻击角度

| # | 攻击角度 | 核心 idea | 与 root cause 匹配度 | 可行性 | 评价 |
|---|---------|-----------|-------------------|--------|------|
| A | **参数空间几何一致性**（TECS） | 在 ROME 编辑层 l* 的参数空间上，计算编辑方向与归因方向的余弦相似度 | 直接——绕过重训练，用另一种参数操作作为验证信号 | 高（所有组件有开源实现） | **选定** |
| B | 行为空间一致性 | 比较 ROME 编辑后的行为变化与 TDA top-k 删除后的行为变化 | 间接——仍需重训练来获取 TDA 端的行为变化 | 中（需要 leave-k-out 重训练） | 未绕开核心问题 |
| C | 机械可解释性交叉验证 | 用 MDA 的 circuit-level 归因与 causal tracing 交叉验证 | 间接——比较两种不同的归因方法，但两者的 ground truth 都不独立 | 中低（MDA 仅在 ≤160M 验证） | 规模受限 |
| D | **表征空间相似度**（RepSim） | 比较训练样本与测试样本在模型中间层隐藏状态的余弦相似度，作为非重训练 TDA 评估 baseline | 部分——绕过重训练且有实证支持（Do IF Work on LLMs 报告 RepSim 全面优于 IF），但测量的是表征空间而非参数空间 | 高（实现简单，无需 Hessian 或梯度） | 有效 baseline，但回答不同问题 |

**TECS vs RepSim 的选择理由**：

RepSim（表征空间相似度）是已有实证支持的非重训练 TDA 评估替代路径——Do IF Work on LLMs (2409.19998) 报告 RepSim 在多项 TDA 任务上全面优于 IF，且计算成本极低。然而 RepSim 和 TECS 回答的是不同层面的问题：

- **RepSim** 在表征空间（activations）工作，测量"训练样本与测试样本是否产生相似的内部表示"。它是一个实用的归因替代工具，但不揭示参数空间中知识如何被存储和操作——它与 RRO 范式一样，本质上是一种行为层面的相关性度量（只是不需要重训练）。
- **TECS** 在参数空间（weights）工作，测量"训练归因方向与知识编辑方向是否共享参数子空间结构"。它的独特价值在于直接探测参数空间的知识几何学——这是 RepSim 无法提供的信息。

因此，TECS 与 RepSim 不是竞争关系，而是不同层面的互补视角。RepSim 已证明其实用价值；TECS 的科学贡献在于打开参数空间知识几何学这个新的分析维度。如果 TECS 探针失败，RepSim 作为 TDA 评估的实用替代仍然成立，但"参数空间中不同知识操作方向的几何关系"这个科学问题将获得负面结果的回答。

### 2.2 选定攻击角度：参数空间几何一致性（TECS）

**核心 idea**：定义 TECS（TDA-Editing Consistency Score）= cos(Δθ_E^{l*}, g_M^{l*})，其中 Δθ_E^{l*} 是 ROME 在编辑层 l* 上的 rank-1 参数更新，g_M^{l*} 是 TDA 方法 M 对同一事实的 top-k 训练样本的聚合梯度向量。方法级汇总：对多个事实取 TECS 均值。

**为什么可能有效**（因果论证）：

ROME 通过 causal tracing 定位关键层 l*，并通过 rank-1 update 改写事实知识。如果 TDA 归因方向确实反映了"哪些训练样本贡献了该知识"，那么这些样本的梯度方向应当在 l* 层的参数空间中指向与 ROME 编辑方向相关的子空间。

**重要限定**：Hase et al. (2301.04213) 发现 causal tracing 定位的层与编辑成功率相关性接近零，这意味着 ROME 选择编辑层 l* 的依据（causal tracing）可能不是编辑成功的真正原因。因此，TECS 的论证不能建立在"ROME 精确定位了知识存储位置"上，而应重新框定为：ROME 的 rank-1 update 是一个使模型在特定事实上改变行为的参数空间干预方向；TDA 归因是另一个参数空间方向。两者的几何关系是一个经验问题，其答案——无论正面还是负面——都提供关于知识参数表征结构的信息。

这个论证不要求两端各自完美：即使 ROME 的 rank-1 update 是知识存储位置的不完美近似，即使 TDA 的聚合梯度是训练影响的不完美估计，只要两者的不完美性是 **不相关的**，它们的对齐程度就能提供关于真实知识结构的信息。这类似于天文学中的多信使验证。但必须承认：两端噪声的独立性本身是一个假设（见 §2.3 第 5 点），需要 placebo test 来部分排除 spectral confound。

**与 root cause 的匹配**：Root cause 是"TDA 评估缺乏独立于重训练的验证通道"。TECS 提供一个这样的通道——ROME 编辑不涉及任何重训练过程。但 TECS 的定位应降级为"探索性实证研究"（探测参数空间中两种知识操作方向的几何关系），而非"验证工具"（断言 TECS 高意味着 TDA 归因可靠）。

**降级后回答的核心科学问题**：知识在参数空间中的表征结构是否在不同知识操作（训练归因 vs 模型编辑）中呈现一致的几何特征？这个问题连接到参数空间知识几何学——即使 TECS 不具备实用验证工具价值，回答这个问题本身有独立的科学意义：它提供了关于大模型参数空间中知识如何被组织的几何层面实证，这是现有工作（RepSim 在表征空间、RRO 在行为空间）未覆盖的分析维度。

### 2.3 攻击角度的局限性与风险

1. **循环论证风险**（反对者的核心质疑）：如果 TDA 和 ROME 都不可靠，它们的一致性可能是虚假的——两者可能被同一个 confound（如 loss landscape 几何结构）驱动而非真实知识位置。**缓解措施**：设计 null TECS 控制实验（随机方向编辑 baseline）、跨事实 shuffle 控制、以及 placebo test（非编辑层 TECS，见下文第 5 点）。

2. **ROME 可靠性争议（知识局部化假设动摇）**：这是 TECS 最核心的风险。Hase et al. (2301.04213) 发现 causal tracing 定位与编辑成功率相关性接近零；知识 entity/relation 分离存储 (2409.00617) 和 capability > knowledge localization (2502.20992) 进一步削弱了 ROME 编辑层 l* 作为"知识位置代理"的可信度。ROME 的 rank-1 update 是约束最小二乘解，可能更多反映优化方法的 artifact 而非知识的真实结构。**缓解措施**：(a) 分组控制——对 ROME 编辑成功/失败的事实分别计算 TECS，检查是否存在差异（编辑成功只说明优化目标被满足，不证明编辑层是知识的真实位置，但编辑成功子集 TECS 显著高于失败子集是间接支持信号）；(b) 降级 claim——从"验证工具"到"参数空间知识几何学的探索性研究"，减少对 ROME 可靠性假设的依赖。

3. **高维空间中的统计幻觉**：在 d >> 1000 维参数空间中，两个随机向量的余弦相似度趋近于 0，但极小的偏差可能产生统计显著但 effect size 微小的结果。**缓解措施**：预定义 practical significance threshold（Cohen's d > 0.5），不仅报告 p 值。

4. **贡献天花板**：即使 TECS 全面成功，它也只是"一个额外的评估维度"，不是 RRO 的替代。claim 必须降级为"参数空间几何一致性度量"而非"独立验证工具"。目标 venue 相应调整为 EMNLP/ACL Findings 或 NeurIPS Workshop（非 main conference poster+）。

5. **Spectral confound（两端噪声独立性假设）**：TDA 梯度方向和 ROME 编辑方向都受 weight matrix spectral structure 约束。如果该层 weight matrix 有强 low-rank tendency（前几个 singular vectors 主导），任何与该层相关的向量都可能有微弱对齐——TECS 可能测量的是 spectral bias 而非知识结构。当前的随机 shuffle 控制不足以排除此 confound。**缓解措施**：将 **placebo test** 升级为核心控制步骤——计算非编辑层（如 l* ± 5）的 TECS 作为 placebo；若 TECS(编辑层) 不显著高于 TECS(非编辑层)，则 spectral confound 不可排除。

---

## 3. 探针方案（Dim 0）

### 3.1 核心假设

**如果这一点不成立，整个方向就不成立**：

在 GPT-2-XL 的 ROME 编辑层 l* 上，梯度 TDA 方法（raw gradient mean of top-k training samples）产出的聚合梯度向量 g_M^{l*}，与 ROME 对同一事实的 rank-1 编辑方向 Δθ_E^{l*} 之间的余弦相似度，统计显著地高于随机方向 baseline。

**此假设的可证伪信号**：如果 20+ 个事实上的 TECS(real) 与 TECS(null) 的分布无统计差异（双侧 t 检验 p > 0.05 **且** Cohen's d < 0.5），假设被证伪。

### 3.2 最小实验方案

**模型**：GPT-2-XL（1.5B 参数，ROME 原始验证模型，开源实现成熟）

**数据集**：CounterFact（ROME 配套数据集），选取 50 个事实（覆盖不同关系类型）

**TDA 方法**：仅用 raw gradient（梯度点积方向，无 Hessian），这是最简单的 TDA 信号。如果 raw gradient 都不对齐，更复杂的 TDA 方法也不可能对齐。

**训练样本数据源（关键澄清）**：

CounterFact 不是 GPT-2 的训练集——CounterFact 是编辑评估数据集。探针所需的"训练样本"有两种操作定义：

- **方案 A（优先）**：从 OpenWebText（GPT-2 的实际训练数据近似）中检索包含目标事实 (s, r, o) 的文档，使用 BM25 检索 top-100 候选，再用梯度点积排序取 top-k。这是 TDA 的标准语义——追溯"真实训练样本对模型知识的贡献"。
- **方案 B（退化）**：如果 OpenWebText 检索不可行（如无法精确匹配事实到训练文档），则使用 CounterFact 中的 paraphrase prompts 作为代理样本。此时 TECS 的语义从"训练样本归因方向"退化为"事实相关文本的梯度方向"，claim 需相应降级。

**探针采用方案 A**。Justification：TDA 的核心问题是"训练数据归因"，使用非训练数据计算梯度会混淆"样本与模型知识的因果关系"和"样本与模型输出的相关性"。

**梯度 loss 函数定义（关键澄清）**：

对于训练样本 x_i 和目标事实 (s, r, o)，梯度定义为：g_i = ∇_θ L(θ; x_test)，其中 x_test 是目标事实的测试 prompt（如 "The capital of France is"），L 是该 prompt 的 next-token prediction loss（标准自回归 cross-entropy），θ 仅取编辑层 l* 的参数。注意：这里用 **测试 prompt 的 loss** 而非训练样本 x_i 的 loss——因为我们关心的是"训练样本 x_i 对模型在测试 prompt 上的行为的影响"，这与 IF 的标准定义一致：dL(θ; x_test)/dθ 在训练样本 x_i 方向的投影。

聚合方式：g_M = (1/k) Σ_{i=1}^{k} g_i（简单平均，raw gradient 无 Hessian 加权）。

**实验步骤**：

1. 对每个事实 (s, r, o)，运行 ROME 获取编辑层 l* 的 rank-1 更新 Δθ_E^{l*}
1b. **SVD 投影前置诊断**（~30 分钟，步骤 1 后立即执行）：对编辑层 l* 的 weight matrix W^{l*} 做 SVD，计算 Δθ_E^{l*} 和 g_M^{l*}（先用 5 个事实的快速计算）在 top-k singular vectors 上的投影比例。如果两者在 top-10 singular vectors 上的投影都 > 80%，说明观测到的对齐可能是 spectral structure 驱动而非知识信号——此时 TECS 的解释需要更强的控制（Null-B placebo 的重要性上升）。这一步不构成 kill gate，但提供 spectral confound 风险的前置估计，指导后续结果解释。
2. **Sanity check（方案 B pilot）**：在进入方案 A（OpenWebText 检索）之前，先用 CounterFact 的 paraphrase prompts 对 5 个事实计算 TECS，验证代码管线的端到端正确性（ROME 编辑方向提取 → 梯度计算 → 余弦相似度）。如果 paraphrase TECS 为 NaN 或全为零，说明实现有 bug，应先修复。这一步将"检索 pipeline 失败"与"TECS 概念失败"解耦。
3. 从 OpenWebText 中用 BM25 检索 top-100 候选训练文档，用梯度点积排序取 top-k（k=10）
4. 计算 top-k 样本在 l* 层的平均梯度向量 g_M^{l*}（对测试 prompt 的 next-token prediction loss 梯度）
5. 计算 TECS = cos(Δθ_E^{l*}, g_M^{l*})
6. 构建 null baselines（三层控制）：
   - **Null-A（无关事实编辑方向）**：对每个事实，用 10 个无关事实的编辑方向计算 TECS(null-A)
   - **Null-B（Placebo 层）**：在非编辑层（l* - 5, l* + 5）计算 TECS，检查对齐是否为编辑层特异性（排除 spectral confound）
   - **Null-C（编辑失败事实）**：对 ROME 编辑失败的事实计算 TECS，检查编辑成功是否与 TECS 相关
7. 统计检验：配对 t 检验 TECS(real) vs TECS(null-A)，报告 Cohen's d 和 effect size 95% CI；TECS(l*) vs TECS(非编辑层) 的配对比较

**不做的事**（控制范围）：
- 不比较多种 TDA 方法（探针只验证信号是否存在）
- 不用 EK-FAC/iHVP（raw gradient 足够回答核心假设）
- 不跨模型（GPT-2-XL 足够）

### 3.3 Pass 标准

| 指标 | Pass 阈值 | 理由 |
|------|----------|------|
| TECS(real) vs TECS(null-A) 配对 t 检验 | p < 0.05 | 统计显著性基础 |
| Cohen's d (real vs null-A) | > 0.5 | practical significance，排除"统计显著但效应量微小" |
| TECS(real) 均值 | > 0.05 | 高维空间中，即使小的正余弦也有意义，但不能是舍入误差级别 |
| **TECS(编辑层) > TECS(非编辑层)** | **配对 t 检验 p < 0.05** | **Placebo 条件：排除 spectral artifact，确认编辑层特异性** |

**额外诊断指标（不影响 pass/fail 但提供信息）**：
- g_M 的 angular variance（pairwise cosine among top-k gradients）：如果 < 0.05，说明梯度方向本身是噪声，TECS 失去意义（Kill Gate 1）
- ROME 编辑成功率：如果编辑成功的事实子集 TECS 显著高于编辑失败的事实，这是强力支持信号
- TECS(编辑成功) vs TECS(编辑失败) 的比较：间接支持 ROME 可靠性

### 3.4 时间预算

| 步骤 | 估计时间 |
|------|---------|
| ROME 编辑（50 个事实） | 1-2 小时 |
| SVD 投影前置诊断（5 个事实） | 0.5 小时 |
| Sanity check（方案 B pilot，5 个事实） | 0.5 小时 |
| OpenWebText 检索 + 训练样本梯度计算 | 2-4 小时 |
| Placebo 层梯度计算（l* ± 5） | 1-2 小时 |
| TECS 计算 + 三层 null baseline + 统计检验 | 1 小时 |
| **合计** | **6-10 小时**（单 A100 GPU） |

### 3.5 Fail 时的信息价值

**Fail 模式 1：g_M angular variance < 0.05**（梯度方向是噪声）
- 信息价值：确认 raw gradient 在 GPT-2-XL 上缺乏方向一致性。暗示 TDA 的梯度信号在这个规模上已经弱到无法提取有意义的方向信息。
- 后续动作：尝试 EK-FAC 加权梯度或 MDA 子空间投影是否改善方向一致性。如果仍然不行，方向需要根本性重思。

**Fail 模式 2：g_M 有方向一致性，但 TECS(real) ≈ TECS(null)（Cohen's d < 0.5）**
- 信息价值：TDA 归因方向和 ROME 编辑方向在参数空间中**不对齐**。这本身是关于知识表征的重要负面结果——暗示 TDA 归因捕捉的参数空间结构与 ROME 假设的知识局部化结构是不同的（可能反映知识的分布式存储本质）。
- 后续动作：检查是否在特定 layer 子集或特定关系类型上存在局部对齐。如果完全不存在，发表负面结果论文（"TDA 归因方向与模型编辑方向不一致：知识分布式存储的实证证据"）。

**Fail 模式 3：TECS 统计显著但 Cohen's d ∈ [0.2, 0.5]（弱效应）**
- 信息价值：信号存在但微弱。可能需要更精确的 TDA 方法（IF with proper Hessian）或更精确的编辑方法（MEMIT batch editing）来放大信号。
- 后续动作：升级 TDA 端（从 raw gradient → EK-FAC IF）和/或编辑端（从 ROME → MEMIT），看效应量是否增大。

---

## 4. 元数据

- **Gap 来源**：组合推导——Revisiting Fragility (2303.12922) 的 G-RF2（缺乏替代评估方案）+ ROME/MEMIT 的参数空间编辑方向 + MDA (2601.21996) 的子空间 IF 思路
- **攻击角度来源**：跨学科工具迁移——模型编辑的参数更新方向 → TDA 评估的验证信号
- **知识库消费**：
  - Gaps & Assumptions: G-RF2, H-RF1, H-TRAK1, H-MISS1, H-IF-LLM3
  - Cross-Paper Connections: C60 (Revisiting Fragility ↔ Do IF Work on LLMs?), C79 (TRAK ↔ Revisiting Fragility)
  - Methods Bank: MDA 子空间 IF 框架（启发子空间计算思路）
- **RS-Revise 新增文献**：Hase et al. (2301.04213), 知识分离存储 (2409.00617), Capability localization (2502.20992)
- **RS-Revise-2 新增文献/讨论**：DATE-LM (2507.09424) task-based evaluation 区分, RepSim (Do IF Work on LLMs, 2409.19998) 竞争路径分析
- **Startup 风险映射（RS-Revise-2 后更新）**：
  - 循环论证风险 → 三层 null baseline（无关编辑方向 + placebo 层 + 编辑失败事实）
  - ROME 可靠性争议（知识局部化假设动摇）→ 编辑成功/失败分组 + claim 降级为探索性研究
  - 高维统计幻觉 → Cohen's d > 0.5 硬阈值
  - Spectral confound → placebo test（非编辑层 TECS）+ SVD 投影前置诊断
  - 贡献天花板 → claim 降级为"参数空间知识几何学的探索性实证研究"，目标 venue 调整为 Findings/Workshop
  - RepSim 竞争路径 → 明确 TECS 与 RepSim 的层面区分（参数空间 vs 表征空间），两者互补非竞争
  - 工程风险与概念风险混淆 → 方案 B sanity check 前置解耦
