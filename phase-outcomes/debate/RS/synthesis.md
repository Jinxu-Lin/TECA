## [RS] 多视角辩论综合

### 交叉验证发现

**强信号问题**（多视角共识）：

- **Spectral confound 是 TECS 最核心的解释瓶颈**（Contrarian + Interdisciplinary 共同指出）：两者从不同角度论证了同一问题——Contrarian 指出 TDA 梯度和 ROME 编辑方向都可能被 weight matrix 的 low-rank structure 驱动，Interdisciplinary 进一步将此与 Weight Space Learning (2603.10090) 的 weight manifold 固有低维结构联系起来。problem-statement.md 的 SVD 投影诊断（步骤 1b）和 placebo test（Null-B）是正确的缓解方向，但 Interdisciplinary 建议的 **spectral residual TECS**（投影去除 top-k singular vectors 后计算余弦）比单纯的诊断更有力——它将诊断转变为干预。建议在探针中纳入 spectral residual TECS 作为核心 ablation。

- **ROME 作为编辑端的可靠性持续下降**（Contrarian + Comparativist 共同关注）：Contrarian 指出 ICLR 2025 Precise Localization (2503.01090) 和 AlphaEdit 进一步削弱 ROME 的定位基础；Comparativist 的文献搜索未发现直接竞争但确认了 ROME 正在被更新方法替代的趋势。缓解措施已在 problem-statement.md 中存在（claim 降级 + 编辑成功/失败分组），且 Interdisciplinary 提出的多编辑方法验证（MEMIT 作为第二工具变量）是强化方案，但超出探针最小范围。

**重要独立发现**：

- **[Comparativist] AirRep (2505.18513) 未被引用**：作为 RepSim 的进化版（通过可训练编码器显式优化归因质量），AirRep 应被纳入候选攻击角度讨论。但这不改变 TECS 的定位——AirRep 在表征空间工作，TECS 在参数空间工作，两者回答不同层面的问题。影响：related work 完整性问题，非方向性问题。

- **[Comparativist] Weight Space Learning (2603.10090) 提供了新的 framing 机会**：TECS 的参数空间知识几何学定位可以连接到 ICLR 2025 Weight Space Learning Workshop 的 "weights as data modality" 范式，增强 contribution 的理论锚点。

- **[Pragmatist] OpenWebText BM25 检索质量是核心工程风险**：BM25 对非常见事实的召回率可能不足。建议先评估 10 个事实的 BM25 召回率作为方案 A 可行性的前置检查。

- **[Contrarian] In-Run Data Shapley 作为竞争方法未被讨论**：虽然 In-Run Shapley 是归因计算方法而非评估方法（Comparativist 已区分），但作为另一种绕开重训练的 TDA 路径，应在 related work 中提及。

**分歧议题及裁判**：

- **Contrarian vs 其他三位：TECS 是否值得继续？** Contrarian 认为两端不可靠性叠加 + 生死线模糊使 TECS 的发表前景不确定。但 Pragmatist 确认 pilot 成本极低（0.5 GPU-day sanity check），Comparativist 确认无直接竞争且概念新颖，Interdisciplinary 提出了具体的增强措施（spectral residual TECS、多编辑方法验证）来应对核心风险。**裁判**：Contrarian 的质疑在技术层面成立，但在战略层面不构成 block——pilot 成本极低的项目应该先做实验再判断，而非在概念层面无限辩论。关键是探针设计必须包含足够强的控制来区分信号与 artifact。

- **SVD 投影诊断的地位：诊断 vs soft kill gate？** Contrarian 建议将 SVD 投影 >80% in top-10 升级为 soft kill gate；Interdisciplinary 建议更进一步，用 spectral residual TECS 替代纯诊断。**裁判**：采纳 Interdisciplinary 的建议——spectral residual TECS 作为核心 ablation 纳入探针（成本 ~30 分钟），SVD 投影诊断保留但不升级为 kill gate（因为 >80% 投影不一定意味着信号不存在，可能只是信号和 spectral structure 重叠）。

---

### 修订指引

**必须修改**：

1. **在探针方案中增加 spectral residual TECS** — 来源：Interdisciplinary（Contrarian 背书） — 具体修改：在 §3.2 步骤 6 的 null baselines 中增加 Null-D：将 g_M 和 Δθ_E 投影到 W^{l*} 的 null space（去除 top-20 singular vectors），计算 residual TECS。如果 residual TECS 仍然统计显著（p < 0.05，Cohen's d > 0.3），spectral confound 被有效排除。在 §3.3 Pass 标准中增加此项为额外诊断指标（不影响 pass/fail 但提供关键解释信息）。 — 影响范围：探针方案增加 ~30 分钟计算，不影响整体设计。

2. **在候选攻击角度中补充 AirRep 和 In-Run Data Shapley 的讨论** — 来源：Comparativist + Contrarian — 具体修改：在 §2.1 候选攻击角度表格中增加 AirRep 作为 RepSim 的进化版（攻击角度 D 的补充），并在 TECS vs RepSim 讨论段落中增加 AirRep 引用。在 §1.2 Gap 陈述或 §2.1 中简要提及 In-Run Data Shapley 作为另一种绕开重训练的路径。 — 影响范围：文献完整性，不改变方向选择。

**建议修改**：

- **连接到 Weight Space Learning 范式** — 来源：Comparativist — 说明：在 contribution.md 中将 TECS 的参数空间知识几何学定位与 ICLR 2025 Weight Space Learning 方向联系起来（"weights as data modality"），增强科学问题的理论锚点。 — 预期 cost：contribution.md 增加 1-2 段文字，约 30 分钟。

- **探针中增加 BM25 召回率前置评估** — 来源：Pragmatist — 说明：在 §3.2 步骤 3 前增加对 10 个事实的 BM25 召回率评估，如果 <50% 事实找到 10+ 相关文档，触发方案切换讨论。 — 预期 cost：额外 ~1 小时工程时间。

- **在 §2.3 风险列表中增加 ROME 被社区淘汰的趋势性风险** — 来源：Contrarian — 说明：引用 Precise Localization (2503.01090) 和 AlphaEdit 作为 ROME 后继方法，说明 TECS 选择 ROME 的理由（ROME 是参数空间编辑方向最简单、最纯粹的形式——rank-1 update，且有最成熟的开源实现）。 — 预期 cost：增加 2-3 句文字。

**已裁定可忽略**：

- Contrarian 建议将 SVD 投影 >80% 升级为 soft kill gate — 忽略理由：spectral residual TECS 已提供更直接的排除手段；SVD 投影诊断的 80% 阈值过于武断（信号和 spectral structure 可能重叠），升级为 kill gate 会过早终止有价值的探索。

- Contrarian 对"探索性实证研究 + 模糊结论"的发表前景担忧 — 忽略理由：这属于结果依赖的判断，在探针阶段讨论为时过早。pilot 成本极低（< 1 GPU-day），先做实验再判断发表价值。

---

### 综合判定

**小幅修订即可**

RS-Revise-2 后的 problem-statement.md (v1.2) 已经很好地处理了前两轮审查的核心问题：claim 降级后的科学问题锚点（参数空间知识几何学）已明确，RepSim 竞争路径已讨论并与 TECS 区分，三层 null baseline + placebo test + SVD 投影诊断构成了合理的控制体系。本轮辩论未发现方向性缺陷或需要重构的根本问题。两项"必须修改"——spectral residual TECS 和文献补充——均为增量完善，可在单次 C-Revise 中完成。
