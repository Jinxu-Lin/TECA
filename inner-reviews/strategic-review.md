# Strategic Review（战略审查）— TECA

**审查日期**：2026-03-16
**审查阶段**：RS（第 3 轮，RS-Revise-2 后）
**审查文档**：`research/problem-statement.md` (v1.2, entry_mode: rs_revise), `research/contribution.md`, `project-startup.md`

---

## 前轮审查跟踪

第 2 轮 RS 提出两项"必须修改"：
1. ~~Claim 降级后的科学问题锚点不足~~ → **已解决**（v1.2 §1.3 Why-4 明确框定为"参数空间知识几何学"，§2.2 详细阐述降级后回答的核心科学问题，contribution.md 更新了 pitch）
2. ~~RepSim 竞争路径未讨论~~ → **已解决**（v1.2 §2.1 攻击角度 D 新增 RepSim 讨论，§2.1 末尾 "TECS vs RepSim 的选择理由" 段落清晰区分了参数空间 vs 表征空间）

此外第 2 轮的两项"附条件"也已纳入：SVD 投影前置诊断（§3.2 步骤 1b）和方案 B sanity check（§3.2 步骤 2）。

---

## 审查维度评分

### 1. Gap 真实性

**评级：Pass**

评级不变。RRO 范式的系统性脆弱性有充分实证支持（Revisiting Fragility + Do IF Work on LLMs + KB 5+ 独立条目）。v1.2 新增了 DATE-LM task-based evaluation 的区分讨论（§1.2 第 4 点），明确 TECS 与 task-based evaluation 在不同层面互补——这回应了第 2 轮的注意事项。

### 2. Gap 重要性与贡献天花板

**评级：Pass（降级后锚点已建立）**

第 2 轮判 Revise 的原因是"降级后的 TECS 信息价值缺乏理论锚点"。v1.2 已充分回应：§1.3 Why-4 将核心科学问题框定为"知识在参数空间中的表征结构是否在不同知识操作中呈现一致的几何特征"；§2.2 详细阐述了降级后的独立科学意义（正面结果 → 不同知识操作共享参数子空间结构；负面结果 → 训练归因与知识编辑假设不同的参数空间组织方式）；contribution.md 的一句话 pitch 诚实且精确。

贡献天花板的现实约束不变：目标 venue 为 Findings/Workshop 是合理的，main conference 需要 RQ2 全面成功 + 跨模型验证。但这是 ceiling assessment 而非 blocker。

### 3. Gap 新颖性 + 竞争态势

**评级：Pass**

Web 搜索（本轮更新）再次确认无直接竞争——未发现任何将 TDA 归因方向与模型编辑方向做参数空间几何比较的工作。新发现的相关工作：
- AirRep (2505.18513) 是 RepSim 的进化版（可训练编码器优化表征归因），但在表征空间工作，不与 TECS 的参数空间定位竞争
- In-Run Data Shapley 绕开重训练但是归因计算方法而非评估方法
- Weight Space Learning survey (2603.10090) 和 ICLR 2025 Workshop 确认"weights as data modality"是新兴方向，TECS 可连接

并发风险仍然低。建议在 C-Revise 中补充 AirRep 引用。

### 4. Root Cause 深度

**评级：Pass**

评级不变。Root cause 分析（Why-1 到 Why-4）逻辑连贯，核心赌注与已知反证诚实呈现，Oracle 思想实验框架合理。v1.2 在 §1.3 末尾新增的"在知识可能非局部化的情况下 TECS 为何仍有信息价值"段落进一步强化了论证。

### 5. 攻击角度可信度

**评级：Pass（附注）**

第 2 轮判 Revise 的两个原因均已处理：(a) RepSim 对比讨论已在 §2.1 详细展开，TECS 的独特价值（参数空间 vs 表征空间）论证清晰；(b) SVD 投影前置诊断已纳入探针 §3.2 步骤 1b。

**附注**：两端不可靠性叠加的风险仍然存在（Infusion LM 0.1% rank flip + Hase et al. 的 causal tracing 质疑），但 v1.2 的缓解体系（三层 null baseline + placebo test + SVD 投影诊断 + claim 降级为"探索性研究"）已经是这个问题下能做到的最佳控制。Contrarian 指出 ICLR 2025 的 Precise Localization (2503.01090) 进一步削弱 ROME 基础，但 TECS 选择 ROME 的理由成立：ROME 的 rank-1 update 是最简单、最纯粹的参数空间编辑形式，且有最成熟的开源实现——探针阶段使用最简形式是合理的工程选择。

本轮 Interdisciplinary 提出的 **spectral residual TECS**（投影去除 top-k singular vectors 后计算余弦）是一个重要的方法论增强，应纳入探针方案。详见维度 6。

### 6. 探针方案合理性

**评级：Pass（附增强建议）**

v1.2 的探针方案在前两轮改进后已经扎实：
- 三层 null baseline（Null-A 无关事实 + Null-B placebo 层 + Null-C 编辑失败事实）
- SVD 投影前置诊断（步骤 1b）
- 方案 B sanity check（步骤 2）
- Pass 标准定量（Cohen's d > 0.5、placebo p < 0.05、TECS 均值 > 0.05）
- 时间预算合理（6-10 小时）
- Fail 模式分析完整（三种失败模式 + 对应后续动作）

**增强建议**（本轮新增）：
- 在 null baselines 中增加 **Null-D（spectral residual TECS）**：将 g_M 和 Δθ_E 投影到 W^{l*} 的 null space（去除 top-20 singular vectors），计算残差空间中的 TECS。如果 residual TECS 仍然显著，spectral confound 被有效排除。这比 SVD 投影诊断更有力（将诊断转变为干预），成本极低（~30 分钟额外计算，~10 行代码）。建议作为额外诊断指标纳入，不影响 pass/fail 判定但提供关键解释信息。
- 补充 AirRep (2505.18513) 在候选攻击角度讨论中的引用。

### 7. RQ 可回答性与可证伪性

**评级：Pass**

评级不变。RQ1 和 RQ2 的可证伪条件定量、具体、可实验验证。边界条件明确。v1.2 新增的 placebo 子条件（TECS 在编辑层特异性高于非编辑层）进一步增强了可证伪性。

---

## 综合判定

### 判定：**Pass**

### 判定理由

经过两轮 RS-Revise，problem-statement.md v1.2 已系统性地解决了所有先前提出的核心问题：

1. **科学问题锚点** ✅ — "参数空间知识几何学"框定清晰，正面和负面结果都有独立科学价值
2. **RepSim 竞争路径** ✅ — 参数空间 vs 表征空间的区分论证充分，TECS 的独特价值明确
3. **探针控制体系** ✅ — 三层 null baseline + placebo test + SVD 投影诊断 + 方案 B sanity check
4. **Claim 降级** ✅ — 从"验证工具"到"参数空间知识几何学的探索性实证研究"，诚实且有理据
5. **已知风险完整列举** ✅ — 循环论证、ROME 可靠性、高维统计幻觉、spectral confound、贡献天花板——均有对应缓解措施

本轮辩论发现的改进点（spectral residual TECS、AirRep 引用、Weight Space Learning 连接）均为增量完善，不构成方向性问题。这些建议可在进入 P（探针实验）前的 C-Revise 中快速处理，不需要再经过一轮完整 RS 审查。

**残余风险**（已知且已接受）：
- 两端不可靠性叠加——探针实验将直接检验信号是否存在
- ROME 正在被社区淘汰——探针阶段使用 ROME 作为最简形式是合理的工程选择，扩展阶段可引入 MEMIT/AlphaEdit
- 贡献天花板受限于 Findings/Workshop——这是 claim 降级后的合理期望

### 路由

**→ P（探针实验）**：战略审查通过。进入探针实验前，建议在 C 中做最后一次小幅修订：(1) 在探针方案 §3.2 中增加 spectral residual TECS 作为 Null-D；(2) 在 §2.1 中补充 AirRep 引用；(3) 在 contribution.md 中连接 Weight Space Learning 方向。这些修订量极小（<1 小时），可与探针代码编写并行完成。

---

## 附录：辩论摘要

| 辩论者 | 核心论点 | 关键发现 |
|--------|---------|---------|
| Contrarian | 两端不可靠性 + ROME 被淘汰趋势 + AirRep/In-Run Shapley 竞争 | Spectral confound 是最核心解释瓶颈；SVD 诊断应更有力；ICLR 2025 Precise Localization 进一步削弱 ROME |
| Comparativist | 无直接竞争 + 新文献覆盖漏洞 | AirRep (2505.18513) 未引用；Weight Space Learning survey (2603.10090) 提供 framing 机会；In-Run Shapley 应在 related work 提及 |
| Pragmatist | 工程可行、成本极低 + OpenWebText 检索风险 | BM25 召回率需前置评估；ROME 编辑层定位（固定 vs per-fact）需确认；内存管理需流式计算 |
| Interdisciplinary | 地球物理联合反演 + IV 弱工具 + Weight Space geometry | **Spectral residual TECS**（核心贡献）：投影去除 top-k singular vectors 后计算余弦，比诊断更有力；多编辑方法验证可作为扩展 |
