TECS = cos(Δθ_E^{l*}, g_M^{l*})：在 ROME 编辑层 l* 的参数空间上，计算编辑方向（rank-1 update）与归因方向（top-k 训练样本聚合梯度）的余弦相似度。

方法级汇总：对多个事实取 TECS 均值。

三层 null baseline：
- Null-A：无关事实的编辑方向
- Null-B：非编辑层的 TECS（placebo，检验编辑层特异性）
- Null-C：编辑失败事实的 TECS