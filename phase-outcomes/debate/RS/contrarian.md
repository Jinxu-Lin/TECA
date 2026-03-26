## [Contrarian] 反对者视角

### 假设挑战

⚠️ **假设 1（TDA 梯度方向与 ROME 编辑方向在 l* 层对齐）**：这是整个 TECS 框架的根基假设，但在两轮 RS-Revise 后，支撑依然是"弱——无直接实验证据"。problem-statement.md §2.2 的因果论证承认"不要求两端各自完美"，但 Infusion (2602.09987) 报告 IF 在语言模型上仅 0.1% rank flip，表明参数空间梯度方向在 LM 上的语义信号极其微弱。如果 TDA 端输出的梯度方向本质上是高维噪声，那么 TECS 测量的是噪声向量与 ROME 编辑方向的余弦——期望值为 0，即使统计显著，effect size 也将极小。RS-Revise-2 中增加的 angular variance 诊断（§3.3 Kill Gate 1）是正确的缓解措施，但问题是：如果 angular variance 检查通过了（梯度有方向一致性），这种一致性是来自知识信号还是来自 loss landscape 的低秩结构？SVD 投影前置诊断（§3.2 步骤 1b）是关键鉴别手段，但它被定义为"不构成 kill gate"——如果 >80% 投影在 top-10 singular vectors 上，TECS 的解释就严重受限，应当升级为 soft kill gate。

⚠️ **假设 2（ROME rank-1 update 反映知识存储位置）**：problem-statement.md §1.3 诚实列出了三项反证（Hase et al.、知识分离存储、capability localization），但 §2.2 的重新框定——"ROME 编辑方向是一个定义明确的参数空间操作方向"——虽然降低了对假设 2 的依赖，却同时掏空了 TECS 的解释力。如果 ROME 方向不代理知识位置，TECS 对齐的含义从"两种知识操作共享参数子空间"退化为"两个参数空间向量碰巧有微弱余弦"。更严重的是，ICLR 2025 的 Precise Localization (2503.01090) 和 AlphaEdit 等新工作进一步表明知识编辑正在从 ROME 的 causal tracing + rank-1 update 范式转向更精确的定位方法——如果 ROME 本身正在被社区淘汰，以 ROME 编辑方向作为 TECS 一端的选择需要更强的 justification。

⚠️ **假设 5（仅编辑层 l* 比较足够）**：problem-statement.md §3.2 的 Null-B placebo test 在非编辑层（l* ± 5）计算 TECS 作为控制。但知识分离存储 (2409.00617) 表明 entity 和 relation 可能分布在不同层——如果知识是多层分布的，仅在 l* 层比较可能错过真实对齐发生的位置。placebo test 的反面解释是：如果 TECS(l*) > TECS(非编辑层)，可能不是因为 l* 是知识层，而是因为 ROME 的 rank-1 update 恰好在 l* 层注入了最大扰动——这是一个 tautological artifact。

### 反事实场景

**如果核心洞察是错的**：TDA 梯度方向和 ROME 编辑方向在参数空间中可能根本不共享知识相关信号。最可能的原因不是"知识是分布式的"（这只是一种解释），而是更基本的：TDA 的聚合梯度 g_M 反映的是 loss landscape 的局部曲率结构（dominated by top singular vectors of the layer weight matrix），ROME 的 rank-1 update 同样受制于相同的 spectral structure。两者的微弱对齐是 spectral bias 驱动的 artifact，而非知识信号。

**最可能的实验失败场景**：

- **场景 1（Spectral confound 主导）**：TECS(real) 统计显著高于 TECS(null-A)（无关事实编辑方向），Cohen's d > 0.5，但 SVD 投影诊断显示两个方向 >80% 投影在 top-10 singular vectors 上，且 TECS(编辑层) 仅 marginally 高于 TECS(非编辑层)（p 在 0.03-0.10 区间）。此时结果无法确定是知识信号还是 spectral artifact，论文将陷入无法下结论的灰色地带。

- **场景 2（TDA 端噪声主导）**：angular variance 检查通过（>0.05），但 TECS(real) 与 TECS(null-A) 的 Cohen's d 在 0.2-0.5 之间（弱效应）。研究者面临"升级 TDA 端"的选项（从 raw gradient 到 EK-FAC IF），但 EK-FAC 在 GPT-2-XL 规模上的近似质量本身存疑（Do IF Work on LLMs 的核心发现），形成无限后退。

### 被低估的竞争方法

**有** — **AirRep (2505.18513)**：这是 2025 年发表的表征优化 TDA 方法，通过可训练编码器学习任务特定的表征用于归因，直接优化归因质量。它代表了 RepSim 的进化版——不仅使用表征相似度，还通过排序目标显式优化归因精度。problem-statement.md 中将 RepSim 作为"已有实证支持的更简单路径"讨论（§2.1 攻击角度 D），但未提及 AirRep 这一更强的表征方法竞争者。如果目标是非重训练 TDA 评估，AirRep 的表征优化路径可能比 TECS 的参数空间路径更有实用价值。

**有** — **In-Run Data Shapley (Data Shapley in One Training Run)**：无需重训练的 TDA 评估方法，在单次训练过程中累积 Shapley 值。这直接绕开了 RRO 范式的核心限制，且不涉及参数空间几何假设。

### 生死线评估

**如果结果上限是"TECS(real) vs TECS(null) Cohen's d ≈ 0.5-0.8，但 SVD 投影诊断模糊（50-80% top-10 投影）且 placebo test borderline"**：边缘值得发表——作为 Workshop paper 或 Findings paper 可以接受（"参数空间中两种知识操作方向存在微弱但统计显著的几何关联，但无法排除 spectral confound"），但缺乏 main conference 所需的结论清晰度。问题在于：DL 社区对"探索性实证研究 + 模糊结论"的容忍度正在下降（2025-2026 年 NeurIPS/ICML 的接收标准越来越强调 clear takeaway），一个"可能是信号也可能是 artifact"的论文将在审稿中遭到严厉批评。
