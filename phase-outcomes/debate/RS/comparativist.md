## [Comparativist] 文献对标者视角

### SOTA 定位

**绝对 SOTA（非重训练 TDA 评估）**：目前不存在公认的非重训练 TDA 评估"金标准"——LDS 仍是社区认可的评估 metric，尽管其依赖重训练。DATE-LM (2507.09424) 是最新的 TDA 评估 benchmark，但其评估协议仍然基于重训练（counterfactual rate + leave-k-out）。

**最相近 approach**：
- **RepSim / AirRep (2505.18513)**：表征空间路径。RepSim 是 Do IF Work on LLMs (2409.19998) 中表现优于 IF 的非梯度方法；AirRep 进一步通过可训练编码器优化表征归因质量。两者都在表征空间（activations）工作，不涉及参数空间几何。与 TECS 的区别：TECS 在参数空间（weights）工作，探测的是 weight-level 的知识几何结构——这是 RepSim/AirRep 无法提供的信息维度。problem-statement.md §2.1 已正确区分了这一点。
- **In-Run Data Shapley**：在单次训练中累积 Shapley 值，绕开重训练。但它仍然是一种归因值计算方法，而非评估方法——它回答"每个样本的贡献值是多少"，不回答"这个归因值是否可靠"。
- **Infusion (2602.09987)**：IF 的反向应用（用 IF 指导训练数据构造来操纵模型行为）。在视觉域成功但在 LM 上失败。与 TECS 的区别：Infusion 用 IF 做参数空间操作（反向），TECS 比较两种参数空间操作方向的几何关系。

**最强简单 baseline**：直接使用 LDS 作为 TDA 评估标准（接受重训练成本）。NeurIPS 2025 的 "Taming Hyperparameter Sensitivity in Data Attribution" 专门优化了无需大量重训练的超参选择，降低了 LDS 的计算门槛。如果 LDS 的计算成本持续下降，TECS 作为"绕开重训练"的动机可能被削弱。

**其他关键竞争方法**：
- **LoRIF (2601.21929)**：低秩 IF，通过低秩近似大幅降低 IF 计算成本，使梯度 TDA 在 8B 参数规模上可行。与 TECS 不直接竞争，但扩大了梯度 TDA 的适用范围。
- **Scalable Influence and Fact Tracing for LLM Pretraining**：首次在 160B+ token 预训练语料上实现梯度 TDA，无需子采样。进一步降低了 TDA 的工程门槛。

### 文献覆盖漏洞

⚠️ **缺失关键工作**：
- **AirRep (2505.18513)**：RepSim 的进化版，通过可训练编码器显式优化表征归因质量。problem-statement.md 讨论了 RepSim 但未提及 AirRep——作为表征方法竞争者的最新进展，应被纳入 §2.1 候选攻击角度 D 的讨论。
- **Precise Localization (2503.01090, ICLR 2025)**：挑战 causal tracing 的精确性，提出更精确的知识定位方法。直接影响 TECS 对 ROME 编辑层 l* 的依赖。
- **Weight Space Learning survey (2603.10090, 2026.03)**：将 neural network weights 作为独立数据模态的系统综述，ICLR 2025 有专门 Workshop。TECS 的参数空间知识几何学定位可以连接到这个新兴方向，增强 framing。
- **NeurIPS 2025 "Taming Hyperparameter Sensitivity"**：降低 LDS 评估的重训练成本，间接削弱 TECS 的非重训练动机。

✅ **覆盖充分的方向**：
- RRO 范式批判（Revisiting Fragility、Do IF Work on LLMs）
- ROME/MEMIT 基础工作及其争议（Hase et al.、知识分离存储、capability localization）
- 相邻方法（MDA、Infusion）

### 贡献边际

**实际 delta**：TECS 提出的是一个全新的分析维度——参数空间中两种知识操作方向的几何关系。这不是对现有方法的增量改进（不声称"TECS 比 LDS 更好"），而是开辟了一个之前不存在的测量维度。实际 delta 取决于信号强度：强信号（Cohen's d > 0.8 + placebo clean pass）→ 开辟新评估范式的先驱工作；弱信号（0.5 < d < 0.8，placebo borderline）→ 仅为"有趣但初步的观察"。

**是否足够**：足够（作为 Findings/Workshop）— TECS 的概念新颖性独立于信号强度；无论正面还是负面结果都提供关于参数空间知识组织的实证。但 main conference 需要强信号 + 跨模型验证。

**创新类型**：**有意义的增量改进 / 新分析视角** — 不是新的 TDA 方法或新的编辑方法，而是在两个现有方向之间建立新的几何联系。类似于 lottery ticket hypothesis（不提升性能但提供新洞察）。

**核心差异点**：TECS 是唯一在参数空间权重层面直接测量 TDA 归因与知识操作几何关系的工作——RepSim/AirRep 在表征空间，LDS/RRO 在行为空间，TECS 打开了第三个维度。

### 并发工作风险

**风险等级**：低

**依据**：Web 搜索未发现任何将 TDA 归因方向与模型编辑方向做参数空间几何比较的工作。TDA 社区和模型编辑社区是两个独立且活跃的方向，但交叉极少——Infusion 是唯一明确的桥梁，但方向相反（用 IF 做编辑，而非用编辑验证 IF）。这种跨社区桥接需要特定的问题意识（"TDA 评估的独立性缺失"），不太可能被多组独立发现。ICLR 2025 Weight Space Learning Workshop 关注权重空间几何学但未涉及 TDA-编辑交叉。
