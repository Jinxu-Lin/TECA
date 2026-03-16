# Pipeline Evolution Log

> 本文档记录每次 Phase 执行后的流程反思观察。

---

## 使用说明

- 每次执行阶段后，在此文档末尾追加一条 Entry
- Entry 编号递增，格式为 `Entry [序号]`

### 观察分类

| 类别 | 含义 |
|------|------|
| **缺失 (Missing)** | 流程中缺少的内容 |
| **冗余 (Redundant)** | 不需要或效果不佳的内容 |
| **改进 (Improve)** | 现有内容可以做得更好 |
| **确认 (Confirm)** | 好的实践，值得保留 |

---

<!-- 在此下方追加 Entry -->

## Entry 1 — C（问题锐化 / Crystallize） — 2026-03-16

**执行模式**: 首次
**时间分配观察**: Gap 候选生成和 Root Cause 分析耗时最多（约 50%），因为需要跨 5+ 篇 Episteme 深度阅读笔记做组合推理；探针设计约 20%；攻击角度选择较快（约 15%），因为 Startup 已经做了大量预工作。

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: crystallize-prompt] [中]** — Prompt 要求"从知识库中做组合推导"并"同时关联 10+ 篇论文"，但在实际执行中，本项目的核心 Gap 在 Startup 阶段已经相当精确——C 阶段主要是对已有 Gap 做 Root Cause 深化和探针精确化，而非重新生成候选。对于从 Startup 直接进入 C 的项目，prompt 中"Gap 候选生成"步骤有大量冗余。
  - 建议：增加一个快速分支判断——如果 Startup 已有明确 Gap + 攻击角度，C 阶段应聚焦于 Root Cause 深化 + 探针精确化，跳过"广撒网式"的 Gap 候选生成。

#### 边界 (Boundary)
- [ ] **[BOUNDARY] [跨阶段] [中]** — C 阶段的"攻击角度描述不超过 2 段话（防止越界成方法设计）"约束在本项目中恰好合适，但 TECS 的数学定义（cos、聚合方式、null baseline 设计）在 C 阶段已经不得不涉及一定的方法细节。C 和 D 阶段之间"攻击角度 vs 方法设计"的边界在实践中模糊——特别是当攻击角度本身就包含一个具体度量公式时。
  - 建议：明确 C 阶段可以定义度量的"接口"（输入输出规格），但不展开内部实现细节（如聚合权重策略、Hessian 近似选择）。

#### 缺失 (Missing)
- [ ] **[当前阶段] [中]** — Prompt 未提供关于如何处理"Startup 已有 Kill Gates"的指引。本项目的 Startup 已定义了 3 个 Kill Gates，C 阶段的探针方案需要与这些 Kill Gates 对齐但 prompt 未提及这种衔接。实际执行中我手动确保了探针 Pass 标准覆盖了 Startup Kill Gates，但这应该是显式要求。
  - 建议：C 阶段 prompt 增加"检查 Startup Kill Gates 并确保探针方案覆盖"步骤。

#### 确认 (Confirm)
- **[跨阶段]** — "Gap 评价三维矩阵"和"区分三类 Gap 价值层次"的指引非常有效——帮助快速识别出本项目的 Gap 属于"做了但方法有根本缺陷"型（高价值），而非"没人做过"型。这个框架节省了大量评估时间。
- **[当前阶段]** — "探针设计能在失败时区分方向错和实现问题"的要求迫使我设计了三种 Fail 模式及其诊断信号，这比简单的 pass/fail 标准有用得多。

## Entry 2 — RS（战略审查 / Strategic Review） — 2026-03-16

**执行模式**: 首次
**时间分配观察**: 辩论 Agent 输出占约 50%（4 个视角各需深度分析），综合裁判约 20%，正式报告生成约 20%，Codex 外部审查约 10%。最耗时的部分是 Comparativist 的在线文献搜索——需要执行多轮 WebSearch 并交叉验证结果的相关性。

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: strategic-review-prompt] [中]** — 审查配置 YAML 中的 7 个维度对于战略审查是全面的，但在实际执行中，维度之间的重叠较大——例如"Gap 真实性"和"Root Cause 深度"的评估严重依赖同一批文献证据，导致分析有冗余。"攻击角度可信度"和"探针方案合理性"也有交叉（攻击角度的可信度部分取决于探针能否有效验证它）。
  - 建议：考虑将 7 个维度精简为 5 个，合并"Gap 真实性 + Root Cause 深度"为"问题定义质量"，合并"攻击角度可信度 + 探针方案合理性"为"验证策略可行性"。

#### 缺失 (Missing)
- [ ] **[当前阶段] [高]** — RS 战略审查的 prompt 指令要求 Comparativist "在线搜索近 6 个月 arXiv 直接竞争工作"，但没有提供关于如何处理**上游文档遗漏关键反证文献**的指引。本次审查中最重要的发现（Hase et al. "Does Localization Inform Editing?"）不是竞争工作，而是直接质疑核心假设的反证——这类发现的处理流程应被 prompt 显式覆盖。
  - 建议：在 Comparativist 的任务中增加"核心假设反证搜索"步骤，与"竞争工作搜索"并列。

#### 边界 (Boundary)
- [ ] **[BOUNDARY] [跨阶段] [中]** — RS 的 Revise 路由回到 C，但 RS 发现的问题有些是 C 阶段可以修复的（补充文献、明确数据源），有些可能需要更深层的重新设计（如果知识局部化假设被完全推翻）。当前的路由不区分"小修"和"大改"——都回到 C。对于小修型 Revise，回到完整的 C 阶段可能是 overkill。
  - 建议：考虑在 RS Revise 路由中区分"minor revise"（直接修改 problem-statement.md 后重新提交 RS）和"major revise"（回到 C 完整重做）。

#### 确认 (Confirm)
- **[当前阶段]** — 4 Agent 并行辩论 + 综合者的架构非常有效。不同视角确实发现了不同的问题：Contrarian 聚焦假设脆弱性，Comparativist 发现关键文献遗漏，Pragmatist 暴露工程歧义，Interdisciplinary 提供跨域控制实验设计。如果只用单一审查者，至少 Hase et al. 的遗漏和训练样本数据源歧义可能被忽略。
- **[跨阶段]** — Codex 外部审查（GPT-5.3-codex）与内部审查的结论高度一致（都指出 ROME 可靠性、数据源歧义、spectral confound），这种独立验证增强了审查结论的可信度。Codex 额外提出了"researcher degrees of freedom"风险和"permutation + bootstrap CI"建议，补充了内部审查未覆盖的统计方法论建议。

## Entry 3 — C（RS-Revise / 基于战略审查修订） — 2026-03-16

**执行模式**: RS-Revise
**时间分配观察**: 审查意见消化约 20%，定位修改点约 10%，实际修改约 60%（其中最耗时的是重写 §1.3 Root Cause Why-4 和 §3.2 探针方案），contribution.md 更新约 10%。

### 观察

#### 确认 (Confirm)
- **[跨阶段]** — RS-Revise 模式"逐条理解审查意见 → 定位对应段落 → 针对性修改"的流程非常高效。审查意见的三个必修改项（Hase et al. 文献、数据源歧义、null baseline 不充分）都有明确的修改位置，不需要从零重写。保留已通过审查内容（Gap 定义、新颖性、RQ 可证伪性）的策略节省了大量工作。

#### 改进 (Improve)
- [ ] **[Prompt: crystallize-prompt] [中]** — RS-Revise 的指令说"不重新生成 Gap 候选列表（除非审查明确要求）"，但对于 Root Cause 分析的修改深度没有明确指引。本次修改中 Why-4 需要大幅重写以纳入知识局部化争议，但 Why-1 到 Why-3 完全不需要改动。一个更细粒度的指引——"哪些 Root Cause 层级需要修改"——可以减少判断成本。

#### 边界 (Boundary)
- [ ] **[BOUNDARY] [跨阶段] [中]** — 审查建议"重新框定为探索性实证研究而非验证工具"涉及 contribution.md 的定位调整，这超出了 problem-statement.md 的边界。实际执行中我同时修改了两个文档以保持一致性，但 RS-Revise 的 prompt 只提到"更新文档 frontmatter 版本号"而未明确说应同步更新 contribution.md。
  - 建议：RS-Revise 指令应明确列出"受影响的关联文档"需同步更新。

#### 缺失 (Missing)
- [ ] **[当前阶段] [低]** — 审查发现的关键反证文献（Hase et al.）不在 Episteme 知识库中。这意味着 Logos 的论文发现流程可能遗漏了与 ROME 可靠性相关的重要工作。这不是 C 阶段可以解决的问题，但暴露了 Logos 发现策略的覆盖盲区——对"核心假设的直接反证"类文献的搜索不够系统。

## Entry 4 — RS（战略审查 / Strategic Review 第 2 轮） — 2026-03-16

**执行模式**: RS-Revise 后第 2 轮审查
**时间分配观察**: 辩论 Agent 输出约 40%（第 2 轮可复用对项目的理解），Web 搜索约 15%，综合裁判约 20%，正式报告约 15%，反思约 10%。

### 观察

#### 确认 (Confirm)
- **[当前阶段]** — RS-Revise 后的文档质量显著提升。第 1 轮 RS 指出的三大问题（Hase et al. 文献遗漏、数据源歧义、null baseline 不充分）均在 C-Revise 中得到实质性处理。三层 null baseline + placebo test + claim 降级的组合是对原始方案的重大改进，说明 RS→C→RS 的迭代循环确实在提升研究质量。
- **[跨阶段]** — 4 Agent 并行辩论在第 2 轮仍然产出了新的有价值发现（SVD 投影诊断、RepSim 竞争路径、ROME 几何签名文献），说明每轮辩论不是简单重复，而是在更精细的层面上发现问题。

#### 改进 (Improve)
- [ ] **[Prompt: strategic-review-prompt] [中]** — 第 2 轮 RS 应有更明确的"增量审查"指引——聚焦于 C-Revise 修改是否充分回应了第 1 轮审查意见，而非重新完整审查所有维度。当前 prompt 不区分首次审查和迭代审查，导致部分维度（如 Gap 真实性、RQ 可证伪性）被重新评估但结论不变，产生冗余工作。
  - 建议：增加"迭代审查模式"——第 2 轮聚焦于 (a) 第 1 轮"必须修改"项是否解决；(b) 修改过程中是否引入新问题；(c) 新增辩论发现。已通过的维度可快速确认而非重新评估。

#### 缺失 (Missing)
- [ ] **[当前阶段] [低]** — Comparativist 的 Web 搜索在第 2 轮发现了 DATE-LM (2507.09424) 和 ROME 编辑几何签名等新文献，但 problem-statement.md 在第 1 轮 C-Revise 中未检索这些工作。这暴露了一个流程空白：C 阶段的文献搜索范围与 RS 阶段 Comparativist 的搜索范围不一致——C 阶段主要依赖 Episteme 知识库，而 RS 的 Comparativist 执行实时 Web 搜索。建议 C 阶段也执行一轮 targeted Web 搜索，覆盖核心假设相关的最新文献。

#### 边界 (Boundary)
- [ ] **[BOUNDARY] [跨阶段] [低]** — 本轮 RS 的 Revise 判定涉及 3 项"必须修改"，但这些修改的范围较小（补充 RepSim 讨论、增加 SVD 诊断步骤、明确降级后科学问题）。对于这种小范围修改，完整回到 C 阶段再经历一轮 C→RS 循环可能过重。考虑引入"conditional pass"——RS 列出必须修改项但不触发完整回退，C 阶段快速修改后直接进入 P（探针），RS 审查组对修改进行异步确认。

## Entry 5 — C（RS-Revise-2 / 基于第 2 轮战略审查修订） — 2026-03-16

**执行模式**: RS-Revise（第 2 轮）
**时间分配观察**: 审查意见消化约 15%（第 2 轮审查意见明确且聚焦），定位修改点约 10%，实际修改约 65%（核心工作：RepSim 竞争路径对比、科学问题锚点重写、SVD 诊断 + sanity check 步骤插入），contribution.md 同步更新约 10%。

### 观察

#### 确认 (Confirm)
- **[跨阶段]** — 第 2 轮 RS-Revise 的修改范围确实较小（3 项必修改），不涉及方向重构。RS 审查意见的定位精确（具体到段落和缺失内容），使修改过程高效且低风险。这验证了 Entry 4 中"conditional pass"建议的合理性——这类小修不需要完整 C 阶段重做。
- **[当前阶段]** — 被迫明确"降级后的核心科学问题"是非常有价值的审查压力。从"TDA 验证工具"到"参数空间知识几何学"的理论锚点转换，使项目的科学动机从工具导向变为认知导向，天花板虽然下降但基础更诚实。

#### 改进 (Improve)
- [ ] **[Prompt: crystallize-prompt] [低]** — RS-Revise 模式下，当审查要求"增加竞争路径对比"（如 RepSim）时，需要在候选攻击角度表和正文中同时修改，容易遗漏其中之一。建议 prompt 明确要求"候选攻击角度表与正文讨论必须同步更新"。

#### 缺失 (Missing)
- [ ] **[当前阶段] [低]** — 审查要求增加 SVD 投影诊断，但探针方案的 Pass 标准表没有相应更新——SVD 诊断结果的解释规则（"top-10 投影 > 80% 意味着什么"）以文本形式嵌入步骤描述而非结构化呈现。对于后续实现者，结构化的诊断判断标准可能更易操作。

## Entry 6 — RS（战略审查 / Strategic Review 第 3 轮） — 2026-03-16

**执行模式**: RS-Revise-2 后第 3 轮审查
**时间分配观察**: 前轮审查跟踪（确认修改已完成）约 15%，辩论 Agent 输出约 35%（第 3 轮对项目理解深入，辩论更聚焦），Web 搜索约 15%，综合裁判约 15%，正式报告约 10%，反思约 10%。

### 观察

#### 确认 (Confirm)
- **[跨阶段]** — 三轮 C→RS 迭代循环的质量递增模式清晰：第 1 轮发现核心假设反证（Hase et al.）和 null baseline 不充分；第 2 轮发现 claim 降级后的科学锚点缺失和 RepSim 竞争路径遗漏；第 3 轮发现仅增量性问题（spectral residual TECS、文献补充）。这验证了 RS 审查的收敛性——每轮发现的问题级别递减，最终收敛到 Pass。
- **[当前阶段]** — 第 3 轮辩论仍然产出了有价值的新洞察（Interdisciplinary 的 spectral residual TECS 方法、Comparativist 发现的 AirRep 和 Weight Space Learning survey），说明即使在趋近 Pass 时，多 Agent 辩论仍不是纯粹重复。

#### 改进 (Improve)
- [ ] **[Prompt: strategic-review-prompt] [高]** — 三轮 RS 的总计开销显著。Entry 4 已建议引入"增量审查模式"和"conditional pass"，本轮经验进一步强化这一建议。第 2 轮和第 3 轮的 RS 修改项都是增量完善（补充文献、增加一个控制步骤），不涉及方向重构——完整的 4-Agent 辩论 + 综合对于这类小修是 overkill。建议引入"fast-track re-review"：如果上一轮 RS 的判定理由仅涉及文献补充或控制步骤增加（而非方向质疑），下一轮 RS 可以跳过完整辩论，由综合者直接审查修改项是否解决即可。
- [ ] **[当前阶段] [中]** — Contrarian 在第 3 轮指出的"ROME 正在被社区淘汰"趋势是一个新发现（ICLR 2025 Precise Localization + AlphaEdit），但这类"技术趋势"类信息不在审查配置的 7 个维度中——它既不是 Gap 真实性问题，也不完全是攻击角度可信度问题。建议在审查维度中增加"技术时效性"维度，覆盖"所选技术栈是否面临被淘汰风险"。

#### 缺失 (Missing)
- [ ] **[当前阶段] [中]** — Comparativist 发现的 AirRep (2505.18513) 和 Weight Space Learning survey (2603.10090) 在前两轮 Web 搜索中未被发现，说明每轮 Web 搜索的关键词策略不够系统。第 1 轮搜索聚焦 TDA 评估竞争，第 2 轮搜索聚焦 ROME 可靠性，第 3 轮才搜索"表征优化 TDA"和"参数空间几何"。建议标准化 Comparativist 的搜索清单：每轮必须覆盖 (a) 直接竞争方法、(b) 核心假设反证、(c) 相邻领域新进展、(d) 所选技术栈的替代方案。
