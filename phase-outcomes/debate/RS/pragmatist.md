## [Pragmatist] 务实者视角

### 工程组件拆解

✓ **GPT-2-XL 模型加载与推理** — HuggingFace Transformers (`transformers.GPT2LMHeadModel`)，成熟开源实现，无需改造。

✓ **ROME 编辑与 Δθ_E 提取** — 官方实现 `rome` (GitHub: kmeng01/rome)，支持 GPT-2-XL/GPT-J。Rank-1 update 提取需从 ROME 内部 `compute_v` 函数中间步骤截取，但代码结构清晰。

✓ **CounterFact 数据集** — ROME 配套数据集，已有标准化 JSON 格式，直接可用。

△ **OpenWebText 检索 pipeline（BM25 + 梯度排序）** — 需改造，估计 2-3 天。BM25 检索可用 `rank_bm25` 或 `pyserini`（Lucene 封装），但核心问题是：(a) OpenWebText 原始数据约 38GB 未压缩文本，需要预处理建立索引（~2-4 小时）；(b) BM25 检索返回 "document" 级别结果，但 GPT-2 训练时的上下文窗口是 1024 tokens，需要对检索结果做 sliding window 切分后再计算梯度——这个切分逻辑需要从头实现。改造点：索引构建 + 检索 + context window 切分 + 梯度排序。

△ **梯度计算 pipeline（g_M 提取）** — 需改造，估计 1-2 天。核心是在编辑层 l* 上提取测试 prompt 对 weight matrix 的梯度。PyTorch `torch.autograd.grad` 或 `register_hook` 可实现，但需要注意：(a) GPT-2-XL 的编辑层 weight matrix 是 [d_model, d_ffn] = [1600, 6400]，单个梯度向量约 40MB（float32），50 个事实 × 每个 10 个训练样本 = 500 个梯度向量，总计约 20GB——需要流式计算而非全部存储；(b) 梯度是对哪个参数？ROME 编辑的是 MLP 的 `fc_proj`（GPT-2 中 `mlp.c_proj`），需要确保梯度计算目标与 ROME 编辑目标一致。

✓ **TECS 计算 + 统计检验** — 纯 numpy/scipy，余弦相似度 + 配对 t 检验 + Cohen's d，无工程难度。

△ **SVD 投影诊断** — 需改造，估计 0.5 天。对编辑层 weight matrix 做 truncated SVD（`torch.svd_lowrank` 或 `scipy.sparse.linalg.svds`），计算 Δθ_E 和 g_M 在 top-k singular vectors 上的投影比例。实现简单但需要验证 SVD 的数值稳定性（GPT-2-XL weight matrix 条件数可能很大）。

✗ **Null-C 控制实验（ROME 编辑失败事实）** — 需从头设计，估计 0.5-1 天。ROME 对 CounterFact 的编辑成功率需要量化定义（paraphrase accuracy? neighborhood accuracy?），编辑失败的事实需要筛选和验证。ROME 官方代码有评估脚本但可能需要适配。

### 最小 Pilot 设计

**实验内容**：对 5 个 CounterFact 事实，提取 ROME Δθ_E 和 raw gradient g_M（先用 CounterFact paraphrase 作为方案 B sanity check，验证端到端代码正确性），计算 TECS + random baseline TECS，观察是否有定性差异。

**缩放策略**：5 个事实（而非 50 个）足以看到定性信号（TECS(real) 是否系统性高于 TECS(null)），但不足以做统计检验。方案 B（paraphrase 代替真实训练样本）消除了 OpenWebText 检索 pipeline 的工程依赖。

**所需已就位组件**：GPT-2-XL 加载 ✓、ROME 编辑 ✓、梯度计算 △（~0.5 天）、TECS 计算 ✓

**预计算力**：2-4 GPU-hours（A100），主要瓶颈是 ROME 编辑（每个事实约 5-10 分钟）+ 梯度计算（每个样本约 30 秒）。

### 工程陷阱

⚠️ **OpenWebText 检索质量**：BM25 检索"包含目标事实 (s, r, o) 的文档"的精度高度依赖查询构造。事实如 "The capital of France is Paris" 很容易匹配，但更复杂的事实（如 "Danielle Darrieux's native language is French"）可能在 OpenWebText 中出现次数极少或表述方式差异大。如果 top-100 BM25 候选中实际包含目标事实的文档 < 10 个，则梯度排序 top-k 的质量堪忧。**建议**：先对 10 个事实评估 BM25 召回率，如果 <50% 的事实能找到 10+ 相关文档，方案 A 的可行性存疑，需要考虑 dense retrieval（如 Contriever）或退回方案 B。

⚠️ **梯度计算的内存管理**：GPT-2-XL 本身约 6GB（float16），前向 + 反向传播峰值约 12-16GB。如果需要同时保留多个梯度向量做聚合，内存可能紧张。需要流式计算（每次计算一个样本的梯度，累加到 running mean，不保留中间结果）。

⚠️ **ROME 编辑层的精确定位**：ROME 的默认编辑层 l* 依赖 causal tracing 结果，在 GPT-2-XL 上通常是 layer 17（`transformer.h.17.mlp.c_proj`）。但不同事实的 causal tracing 峰值层可能不同——ROME 使用固定层还是 per-fact 自适应层？如果是固定层，需要确认 ROME 代码的默认设置；如果是 per-fact，每个事实都需要运行 causal tracing（额外 ~5 分钟/事实），增加 pilot 时间。

### 综合预估

⏱️ **日历时间（到第一个有意义结果）**：
- Sanity check（方案 B，5 个事实）：2-3 天（含代码编写 + debug）
- 完整探针（方案 A，50 个事实）：额外 5-7 天（含 OpenWebText 检索 pipeline）
- **合计：1-1.5 周**

💻 **算力（到第一个有意义结果）**：
- Sanity check：0.5 GPU-day（A100）
- 完整探针：2-3 GPU-days（A100）——ROME 编辑 50 事实 ~4h + 梯度计算 500 样本 ~4h + placebo 层梯度 ~4h + SVD + 统计

🔧 **主要工程风险**：OpenWebText BM25 检索对非常见事实的召回率不足，可能迫使退回方案 B 或引入 dense retrieval，增加 2-3 天工程成本。
