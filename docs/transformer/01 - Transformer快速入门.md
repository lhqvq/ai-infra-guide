# Transformer 快速入门

## 为什么 AI Infra 工程师必须吃透 Transformer

想进 AI Infra 这行——不管你是手写 CUDA kernel、折腾分布式训练，还是做推理部署落地——绕来绕去，你每天打交道的那个"主角"其实只有一个：**Transformer 模型**。

打个比方，这就跟赛车技师必须熟悉引擎一样。你用不着亲手去设计一台引擎（那是算法研究员操心的事），但你得清楚缸体长什么样、活塞是怎么往复运动的、油路和进气是怎么走的——不然，你凭什么判断该拧哪颗螺丝、该换哪个零件来压榨性能？

说得再直白一点：AI Infra 的每一层工作，几乎都能在 Transformer 里找到它"对口"的那个模块。你优化的从来不是抽象的"模型"，而是模型里某一块具体的结构。

下面这张表，就是把 AI Infra 的活儿和 Transformer 的零件一一"对号入座"：

| AI Infra 层级 | 你在做的事 | 盯着的 Transformer 模块 |
| --- | --- | --- |
| CUDA 算子优化 | FlashAttention、高效 GEMM kernel | Self-Attention 里的 $QK^T$ / $PV$ 矩阵乘法 |
| CUDA 算子优化 | Fused Softmax、Online Softmax | Attention 中的 softmax 计算 |
| CUDA 算子优化 | LayerNorm kernel 融合 | 每个 Block 里的归一化层 |
| 分布式训练 | 张量并行（Tensor Parallelism） | Attention 的多头切分、FFN 的矩阵切分 |
| 分布式训练 | 流水线并行（Pipeline Parallelism） | 模型里一层层堆叠的 Decoder Block |
| 分布式训练 | ZeRO 显存优化 | 所有参数矩阵的存储与通信 |
| 推理部署 | KV Cache 管理 | Self-Attention 里的 K、V 矩阵 |
| 推理部署 | PagedAttention | KV Cache 的显存碎片问题 |
| 推理部署 | 量化（INT4 / INT8 / FP8） | 所有权重矩阵和 KV Cache |

看懂这张表你就会发现：**所谓"优化模型"，本质上都是在优化 Transformer 里某个看得见、摸得着的具体结构。** 这也是为什么，入门 AI Infra 的第一步，不是先学 CUDA、也不是先学并行框架，而是先把 Transformer 这台"引擎"拆开看明白。
