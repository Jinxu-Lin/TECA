## [Interdisciplinary] 跨学科者视角

### 跨域对应物

#### 类比 A — 地球物理联合反演（Joint Inversion）

**对应关系**：TECS 试图通过两种独立的"参数空间探针"（TDA 梯度方向 ↔ 地震波速度模型；ROME 编辑方向 ↔ 重力异常模型）来交叉验证地下结构（知识在参数空间的组织方式）。地球物理中，地震和重力两种信号由同一地下结构产生但通过不同物理过程传播，联合反演利用两者的结构一致性约束来提高解的可靠性。

**类比深度**：**深层类比** — 数学结构可以映射：设参数空间 Θ 中存在知识结构 K，TDA 产出的探测方向 g = P_TDA(K) + ε_1，ROME 产出的探测方向 d = P_ROME(K) + ε_2。如果 ε_1 ⊥ ε_2（两端噪声独立），则 cos(g, d) 的期望值正比于 K 在两种投影下的重叠度。这与联合反演的数学框架——通过结构耦合项（structural coupling term）连接两个前向模型——直接对应。

**该领域的已有解法**：Cross-gradient joint inversion (Gallardo & Meju, 2003) 通过约束两种物理场梯度的交叉乘积为零来强制结构一致性。更成熟的方法使用 petrophysical relationships 将两种模型参数链接到共同的岩性变量。

**可借鉴的核心洞察**：联合反演的成功关键是**两个信号的独立性**——如果地震和重力都主要由同一个dominant structure（如区域地壳厚度变化）驱动，联合反演不增加信息。类比到 TECS：如果 g_M 和 Δθ_E 都主要由 weight matrix 的 spectral structure 驱动（而非知识信号），TECS 的对齐就不提供关于知识结构的新信息。地球物理学通过**区域场去除（regional field removal）**来分离 dominant structure——TECS 的类比操作应该是**投影去除 top-k singular vectors**后再计算余弦相似度，这比简单的 SVD 投影诊断更有力。

#### 类比 B — 因果推理中的工具变量（Instrumental Variables, IV）

**对应关系**：ROME 编辑方向可被视为知识位置的"弱工具变量"（weak IV）。在因果推理中，IV 必须满足两个条件：(a) 与自变量相关（ROME 方向与知识位置相关，但 Hase et al. 质疑了这一点）；(b) 与误差项独立（ROME 方向不直接影响 TDA 归因质量，仅通过知识位置间接影响）。如果条件 (a) 弱（weak IV），估计结果的偏差和方差都会急剧增大——这对应 TECS 中 ROME 可靠性争议导致的信号稀释。

**类比深度**：**深层类比** — IV 框架提供了精确的统计诊断工具。弱工具变量检验（Cragg-Donald F-statistic < 10 表示弱 IV）可以映射到 TECS：如果 ROME 编辑成功率（作为 IV 强度的代理指标）低于某个阈值，TECS 的估计就不可靠。problem-statement.md §2.3 第 2 点的"编辑成功/失败分组控制"正是 IV 框架下的 first-stage regression 强度检验的类比。

**该领域的已有解法**：Weak IV 的标准应对策略包括：(a) 使用多个工具变量提高 first-stage F-statistic；(b) 报告 Anderson-Rubin 置信区间（对 IV 强度 robust）。

**可借鉴的核心洞察**：单一工具变量（仅 ROME）天然脆弱。如果能引入第二种独立的编辑方法（如 MEMIT 或 AlphaEdit (ICLR 2025, null-space constrained editing)），计算 TECS-MEMIT 和 TECS-ROME 的一致性，就构成多工具变量验证——如果两种编辑方法产出的 TECS 排名一致，spectral confound 假说被削弱（因为不同编辑方法的 spectral bias 不同）。

### 未被利用的工具

- **Spectral residual 对齐**（来自地球物理区域场去除）：在计算 TECS 前，将 g_M 和 Δθ_E 投影到 weight matrix 的 null space（去除 top-k singular vectors 方向），在残差空间中计算余弦。这直接排除 spectral confound，比 SVD 投影"诊断"更有力——它将诊断转变为干预。引入障碍：实现简单（仅需额外一步矩阵乘法），但增加了一个超参（k 的选择），且如果信号本身就在 top-k subspace 中，这步操作会消除真实信号。建议作为额外 ablation 而非替代主分析。

- **Weak IV 诊断统计量**（来自因果推理）：报告 ROME 编辑成功率作为"工具变量强度"的代理，设定 first-stage F > 10 的类比阈值。引入障碍：需要定义"编辑成功率"的精确操作定义（paraphrase accuracy > 0.5？neighborhood accuracy > 0.8？），且 F-statistic 的映射需要额外论证。

### 跨域盲点与教训

- **弱工具变量偏差（来自计量经济学）**：在 IV 框架中，弱工具变量不仅增大方差，还引入系统性偏差——估计结果会偏向 OLS 估计（即无工具变量的朴素估计）。类比到 TECS：如果 ROME 编辑方向作为"知识位置代理"太弱（Hase et al. 的核心发现），TECS 的对齐估计会系统性偏向 spectral structure 驱动的伪对齐值。problem-statement.md 的缓解措施（编辑成功/失败分组）是正确方向，但可能不充分——弱 IV 偏差在有限样本（50 个事实）中尤其严重。

- **Weight Space Learning (2603.10090) 的启示**：2026 年 3 月刚发表的综述将 weight space 作为独立数据模态来研究，识别出 weight space 的核心结构性质（permutation/scaling symmetries、low-rank subspace adaptation）。这意味着 TECS 观测到的对齐可能更多反映 weight space 的通用几何属性（如 Riemannian manifold 的低秩切空间结构），而非知识特异的信号。这是一个比 spectral confound 更深层的混淆因素——不仅是 top singular vectors 的问题，而是 weight manifold 固有的低维结构。

### 建议引入路径

1. **Spectral residual TECS**（成本最低）：在主 TECS 计算后，增加一步：将 g_M 和 Δθ_E 投影到 W^{l*} 的 null space（去除 top-20 singular vectors），计算 residual TECS。如果 residual TECS 仍然显著，spectral confound 被有效排除。实现：约 10 行代码 + 30 分钟额外计算。

2. **多编辑方法验证**（中等成本）：在探针中增加 MEMIT 编辑方向作为第二种"工具变量"——MEMIT 的 multi-layer update 与 ROME 的 single-layer rank-1 update 有不同的 spectral 特征，如果 TECS-ROME 和 TECS-MEMIT 方法排名一致，大幅增强结论可信度。成本：MEMIT 有开源实现（同一代码库），额外 ~2 小时编辑 + ~1 小时梯度计算。但这超出了探针的"最小"范围，建议作为探针 Pass 后的扩展。

3. **编辑成功率作为 IV 强度报告**（几乎零成本）：在探针结果中额外报告 ROME 编辑成功率（paraphrase accuracy + neighborhood accuracy），并将 TECS 分析按编辑成功率分层——高成功率子集的 TECS 应高于低成功率子集（类比 IV first-stage F-statistic）。
