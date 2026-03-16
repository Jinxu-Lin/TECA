# 项目: TECA — TDA-Editing Consistency Analysis

## 研究主题
用模型编辑（ROME/MEMIT）的参数更新方向作为 TDA 归因方向的独立验证信号，提出 TECS（TDA-Editing Consistency Score）指标，探测参数空间中知识几何结构。

## 背景与动机
TDA 评估的根本困境：所有现有方法（LDS、Spearman、LOO）都依赖"删除-重训练-观察"（RRO）范式获取 ground truth，而重训练本身在非凸大模型上系统性不可靠（miss-relation、loss 轨迹震荡）。TECA 提出一个不依赖重训练的独立验证维度：在参数空间中比较 TDA 归因方向与模型编辑方向的几何一致性。

核心科学问题：知识在参数空间中的表征结构是否在不同知识操作（训练归因 vs 模型编辑）中呈现一致的几何特征？

Claim 定位（已降级）：从"TDA 验证工具"降级为"参数空间知识几何学的探索性实证研究"。正面结果意味着不同知识操作共享参数子空间结构；负面结果意味着训练归因与知识编辑假设了不同的参数空间知识组织方式——两种结果都有独立科学价值。

## 当前状态（从已有项目迁入）
- **已完成阶段**: Startup (Go with focus) → C (问题锐化, 3 轮 RS 审查通过) → 探针实验代码编写与执行
- **探针实验结果**: 全面失败（4 个 pass criteria 全部未通过），但审查发现严重实现问题，结论 inconclusive
- **关键实现问题待修复**:
  1. ROME 编辑成功率仅 14%（50 个事实中 7 个成功），疑似参数配置问题
  2. 代码中梯度定义与设计文档不一致（训练样本自身 loss vs test prompt loss）
  3. Null-B placebo 从跨层改为同层不同事实，缺失 spectral confound 控制
  4. BM25 检索仅覆盖 OpenWebText 6%（500K/8M docs）
- **迁入时需从 P0 修复开始，而非从头走 Sibyl 流水线**

## 初始想法
TECS = cos(Δθ_E^{l*}, g_M^{l*})：在 ROME 编辑层 l* 的参数空间上，计算编辑方向（rank-1 update）与归因方向（top-k 训练样本聚合梯度）的余弦相似度。

方法级汇总：对多个事实取 TECS 均值。

三层 null baseline：
- Null-A：无关事实的编辑方向
- Null-B：非编辑层的 TECS（placebo，检验编辑层特异性）
- Null-C：编辑失败事实的 TECS

### 修复后实验计划（P0 优先级）
1. **P0-a**: 在已有数据上分析编辑成功的 7 个事实子集 TECS（零成本验证）
2. **P0-b**: 用 EasyEdit 标准 ROME 实现替换简化版，目标编辑成功率 >60%
3. **P0-c**: 统一梯度定义（与 IF 标准定义对齐）
4. **P1**: 补做跨层 placebo（l* ± 5）
5. **P1**: 扩大 BM25 语料库到 2M+ docs
6. **决策点**: 修复后 Cohen's d > 0.2 → PROCEED; d < 0.2 → PIVOT 到负面结果论文

## 关键参考文献
- Meng et al. (2022), 2202.05262: ROME — causal tracing + rank-one editing
- Meng et al. (2022), 2210.07229: MEMIT — mass editing memory in transformers
- 2303.12922: Revisiting the Fragility of Influence Functions — Spearman miss-relation
- 2409.19998: Do Influence Functions Work on LLMs? — IF 三重失效 + RepSim 优于 IF
- 2602.09987: Infusion — IF 逆向优化，LM 上 0.1% rank flip
- 2601.21996: MDA — 参数子空间 IF
- 2301.04213: Hase et al. — Does Localization Inform Editing? causal tracing 与编辑成功率无关
- 2409.00617: 知识 entity/relation 分离存储
- 2502.20992: Capability > knowledge localization
- 2507.09424: DATE-LM — unified task-based TDA evaluation
- 2505.18513: AirRep — 可训练编码器优化表征归因

## 可用资源
- GPU: 1x NVIDIA RTX 4090 (24GB VRAM)
- 服务器: xuchang3 (SSH MCP)
- 远程路径: /home/jinxulin/sibyl_system

## 实验约束
- 实验类型: training-free（无需训练模型，仅前向/反向传播计算梯度和 ROME 编辑）
- 模型规模: GPT-2-XL (1.5B, ~6GB FP32) — 4090 24GB 充足
- 时间预算: 1-2 天（P0 修复 + 重跑实验 + 决策）
- 单实验时间: 每个实验任务控制在 1 小时内
- 内存注意: GPT-2-XL FP32 + 梯度计算峰值约 12-16GB VRAM

## 目标产出
- 顶会论文（NeurIPS 2026）
- 如果修复后 TECS 显著 → 正面结果论文
- 如果修复后仍不显著 → 负面结果论文（需扩展实验矩阵：多模型/多编辑方法）

## 特殊需求
- 已有代码在 ~/Research/TECA/Codes/，需要在此基础上修复而非重写
- 已有实验结果在 ~/Research/TECA/results/，可作为对照参考
- 已有完整的 problem-statement.md 和 contribution.md，保留并更新
- ROME 实现建议使用 EasyEdit 库（https://github.com/zjunlp/EasyEdit）的标准实现
