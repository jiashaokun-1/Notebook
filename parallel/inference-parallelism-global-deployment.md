# 大模型推理并行：从全局视角理解 W、PP、TP、CP、DP、ETP、EP 与 EDP

## 1. 先建立一个最重要的全局模型

本文讨论以 Transformer + MoE 模型的推理部署为主，重点不是逐个解释缩写，而是回答三个问题：

1. 一张物理卡在全局系统里同时属于哪些并行组？
2. Attention 和 MoE Expert 为什么使用两套不同的并行坐标？
3. 给定固定卡数，增大一个并行度会挤压哪些并行度，又会引入什么通信？

设：

```text
W     = 全网总卡数
PP    = Pipeline Parallel，流水线并行度
TP    = Attention Tensor Parallel，Attention 张量并行度
CP    = Context Parallel，上下文并行度
DP    = Attention/Data Parallel，Attention 数据并行度
ETP   = Expert Tensor Parallel，单个专家内部的张量并行度
EP    = Expert Parallel，专家并行度
EDP   = Expert Data Parallel，专家副本数
```

在各 PP stage 使用相同卡数的常见部署中，物理卡总数满足：

```text
W = PP × TP × CP × DP
```

一个 PP stage 内的卡数为：

```text
stage_world = W / PP = TP × CP × DP
```

MoE Expert 层不会凭空再申请一套卡，而是把同一个 PP stage 内的这些卡重新解释成 Expert 坐标：

```text
stage_world = ETP × EP × EDP
```

因此最关键的恒等式是：

```text
                 W
TP × CP × DP = ──── = ETP × EP × EDP
                 PP
```

不要写成：

```text
W = PP × TP × CP × DP × ETP × EP × EDP    # 错误
```

`TP/CP/DP` 和 `ETP/EP/EDP` 不是七个互相独立、全部相乘的物理维度。它们是同一批 stage 内 Rank 在执行不同子层时使用的两套逻辑坐标：

```text
同一张 GPU
   │
   ├── 执行 Attention 时：坐标 (pp, dp, cp, tp)
   │
   └── 执行 MoE Expert 时：坐标 (pp, edp, ep, etp)
```

---

## 2. 一层 MoE Transformer 的全局数据流

下面是一条请求在某个 PP stage 内的典型推理路径：

```text
                         一个 PP stage
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  不同请求/Token                                                    │
│       │                                                            │
│       ▼                                                            │
│  DP：把请求或 Token 批次分给不同 Attention 副本                   │
│       │                                                            │
│       ▼                                                            │
│  CP：长序列沿 Context 维切分，分摊 KV Cache 和 Attention 计算      │
│       │                                                            │
│       ▼                                                            │
│  TP：Attention 的 Head、QKV/O Projection 等张量切到多张卡          │
│       │                                                            │
│       ▼                                                            │
│  Router：为每个 Token 选择 Top-K Expert                            │
│       │                                                            │
│       ▼                                                            │
│  EP Dispatch：把 Token 发送到持有目标 Expert 的 EP Rank            │
│       │                                                            │
│       ├── EDP：同一 Expert 有多个副本时，选择其中一个副本           │
│       │                                                            │
│       ▼                                                            │
│  ETP：一个 Expert 的矩阵由多张卡协同计算                            │
│       │                                                            │
│       ▼                                                            │
│  EP Combine：把 Expert 输出送回 Token 原来的逻辑位置                │
│       │                                                            │
│       ▼                                                            │
│  后续 Attention/Dense 计算                                         │
└───────┬────────────────────────────────────────────────────────────┘
        │ PP：把 stage 输出激活发送给下一个 stage
        ▼
     下一 PP stage
```

从通信角度看，一层里可能同时出现以下几类通信：

```text
PP  ：相邻 stage 之间传激活，通常是 P2P
TP  ：Attention 张量切分后的 AllReduce，或 Sequence Parallel 的 AllGather/ReduceScatter
CP  ：交换或轮转 K/V、Attention 中间结果，可能是 A2A、Ring 或 Gather
EP  ：Token Dispatch A2A + Expert 输出 Combine A2A
ETP ：单 Expert 内部的 AllReduce，或 ETP-SP 的 AllGather/ReduceScatter
DP  ：纯推理中通常不做逐层梯度同步；不同副本处理不同请求
EDP ：纯推理中通常不做梯度同步；用于专家副本选择与负载分摊
```

---

## 3. 每个并行特性是什么

### 3.1 W：全网总卡数

`W` 是物理资源预算。它只表示总 GPU 数量，并不自动说明这些 GPU 如何协作。

同样是 64 张卡，可以部署成：

```text
PP=1, TP=8, CP=2, DP=4
PP=2, TP=4, CP=2, DP=4
PP=4, TP=4, CP=1, DP=4
```

三者都满足 `W = PP × TP × CP × DP = 64`，但其模型容量、单请求延迟、长上下文能力和吞吐完全不同。

实际规划 W 时，还必须考虑物理拓扑：

- 单机内 GPU 数量和 NVLink/NVSwitch 拓扑；
- 跨机 InfiniBand/RoCE 的有效带宽和超售比；
- 一个并行组是否跨越节点、交换机或机柜；
- 通信库能否高效实现相应 Collective；
- 故障域和副本隔离要求。

并行度只给出“哪些 Rank 构成一组”，拓扑映射才决定这组通信到底快不快。

### 3.2 PP：Pipeline Parallel

PP 沿模型层方向切分模型。例如 80 层模型使用 `PP=4`，可以让每个 stage 大约持有 20 层：

```text
请求 → Stage 0：Layer 0~19
     → Stage 1：Layer 20~39
     → Stage 2：Layer 40~59
     → Stage 3：Layer 60~79 → 输出
```

主要收益：

- 降低每张卡需要常驻的模型权重；
- 让超大模型跨更多节点部署；
- 每个 stage 只保存自己所负责层的 KV Cache。

主要成本：

- stage 之间要传输激活；
- 单请求必须串行经过所有 stage，PP 不会直接缩短单 Token 的关键路径；
- 小 Batch 或调度不足时会出现流水线气泡；
- stage 计算量不均衡时，最慢 stage 决定流水线吞吐。

从全局约束看，固定 W 时增大 PP 会减少每个 stage 的卡数：

```text
stage_world = W / PP
```

这意味着 Attention 侧的 `TP×CP×DP` 和 Expert 侧的 `ETP×EP×EDP` 都必须随之缩小。

### 3.3 TP：Attention Tensor Parallel

TP 把同一个 Attention 层的张量计算切到多张卡上。常见切法包括：

- 按 Attention Head 切分；
- 按 Q/K/V 投影矩阵的列或行切分；
- 按输出投影矩阵切分；
- 与 Sequence Parallel 结合，把某些中间激活沿 Token 维分散保存。

例如 `TP=4` 时，一个 Attention 层由 4 张卡共同完成：

```text
                 Attention 输入
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼          ▼
       TP Rank 0    TP Rank 1    TP Rank 2   TP Rank 3
       一组 Head     一组 Head     一组 Head    一组 Head
          └────────────┴────────────┴──────────┘
                       │ Collective
                       ▼
                 Attention 输出
```

主要收益：

- 分摊 Attention 权重和部分 KV Cache；
- 分摊单层 GEMM 计算；
- 让单层宽度很大的模型可以落到多张卡上。

主要成本：

- 每层需要 TP Collective；
- TP 越大，单卡 GEMM 越小，计算效率可能下降；
- 同一个请求依赖更多卡，尾延迟和慢卡影响更明显；
- TP 跨节点时，频繁 Collective 往往非常昂贵。

因此 TP 通常优先放在节点内高速互联域中，只增大到模型能够装下或单请求计算能够有效加速所需的程度。

### 3.4 CP：Context Parallel

CP 沿序列或 KV Context 维度切分 Attention。不同 CP Rank 保存或处理同一请求的不同上下文片段：

```text
完整上下文：[Token 0 ................................ Token L-1]

CP Rank 0 ：[Token 0      ... Token L/4-1]
CP Rank 1 ：[Token L/4    ... Token L/2-1]
CP Rank 2 ：[Token L/2    ... Token 3L/4-1]
CP Rank 3 ：[Token 3L/4   ... Token L-1]
```

主要收益：

- 把长上下文 Attention 计算分摊到多张卡；
- 分摊 KV Cache 容量；
- 当单个请求的 Context 已无法放进一个 TP group 时，CP 能继续横向扩展。

主要成本：

- Attention 需要跨 CP Rank 获取相关 K/V 或交换部分结果；
- Prefill 的通信量可能很大；
- Decode 虽然每步 Query 很小，但仍要访问分散的历史 KV，容易受延迟影响；
- CP 算法不同，通信模式可能是 All-to-All、Ring P2P 或 AllGather，不能只看 CP 数字判断性能。

CP 主要解决“序列太长、KV 太大”，TP 主要解决“层太宽、矩阵太大”。两者虽然都作用于 Attention，但切分维度不同。

### 3.5 DP：Attention/Data Parallel

DP 复制 Attention 模型参数，让不同 DP 副本处理不同请求或不同 Token 批次：

```text
请求集合
   ├── DP 0：请求 A、B
   ├── DP 1：请求 C、D
   ├── DP 2：请求 E、F
   └── DP 3：请求 G、H
```

在纯推理中，DP 与训练时的经典 Data Parallel 不同：

- 没有反向传播；
- 没有每步梯度 AllReduce；
- 每个 DP 副本通常维护自己请求对应的 KV Cache；
- DP 的核心价值是通过处理互不相同的请求提高系统总吞吐。

主要收益：

- 提升并发请求容量和总吞吐；
- 副本之间独立性较强，调度灵活；
- 相比盲目增大 TP，DP 通常能保持更大的单卡 GEMM。

主要成本：

- Attention 权重会被复制；
- 各 DP 副本的请求长度、Prefill/Decode 比例不同会造成负载不均；
- Attention 的 DP 布局与 Expert 的 EP/EDP 布局不一致时，MoE 边界仍可能发生跨 Rank 重分布；
- 固定 stage_world 下，增大 DP 必须减小 TP 或 CP。

### 3.6 ETP：Expert Tensor Parallel

ETP 把“一个 Expert 自身”的矩阵切到多张卡上。它回答的是：一个专家太大，单卡放不下或算不动，怎么办？

```text
Expert 7 的权重
   ├── ETP Rank 0：Shard 0
   ├── ETP Rank 1：Shard 1
   ├── ETP Rank 2：Shard 2
   └── ETP Rank 3：Shard 3
```

一个 Token 被路由到 Expert 7 后，负责该 Expert 的 ETP Rank 要协同完成计算。

主要收益：

- 分摊单个 Expert 的权重和计算；
- 支持 Expert 中间维度很大的 MoE 模型；
- 可以利用节点内高速链路共同完成一个专家。

主要成本：

- 单 Expert 内部新增 Collective；
- ETP 增大后，每张卡分到的 Expert GEMM 变小，利用率可能下降；
- 如果 Dispatch 把完整 Token 复制给每个 ETP Rank，会增加 EP 通信量；
- 采用 ETP Sequence Parallel 时，可通过 Expert 前 AllGather、Expert 后 ReduceScatter 避免 Dispatch 阶段的输入复制，但会引入 ETP 组内通信。

ETP 和 TP 含义相似但作用对象不同：TP 切 Attention，ETP 切单个 Expert。两者可以取不同数值。

### 3.7 EP：Expert Parallel

EP 把不同 Expert 放到不同 Rank 或 Rank group 上。它回答的是：模型有很多专家，哪些 Rank 分别持有哪些专家？

假设一层有 16 个 Expert，`EP=4`：

```text
EP Rank 0：Expert 0~3
EP Rank 1：Expert 4~7
EP Rank 2：Expert 8~11
EP Rank 3：Expert 12~15
```

Router 为 Token 选择 Expert 后，Token 当前所在的 Rank 往往并不持有目标 Expert，于是需要两次核心通信：

```text
原始 Token 布局
      │
      ├── Dispatch All-to-All：Token 发往目标 Expert 所在 Rank
      ▼
Expert 计算
      │
      ├── Combine All-to-All：结果返回原 Token 的逻辑位置
      ▼
恢复 Token 布局
```

主要收益：

- 分摊所有 Expert 权重；
- 每张卡只计算路由到本地 Expert 的 Token；
- 对专家数量很多、总专家参数量很大的模型尤其重要。

主要成本：

- 每个 MoE 层通常有 Dispatch 和 Combine 两轮 A2A；
- EP 越大，目标 Expert 位于远端的概率通常越高；均匀路由近似下，远端比例为 `(EP-1)/EP`；
- 每个 Expert 收到的 Token 数可能不均，最热 Rank 决定该层完成时间；
- EP group 跨节点后，会受到共享网卡、交换机拥塞和有效双向带宽限制。

EP 不是越大越快。它首先解决专家权重容量，其次才可能通过并行 Expert 计算提高吞吐；通信、负载不均和小 GEMM 都可能抵消收益。

### 3.8 EDP：Expert Data Parallel

EDP 表示 Expert mesh 的副本数。更精确地说，一个完整的 `EP×ETP` Expert 布局会被复制 EDP 份：

```text
EDP 副本 0：完整的 EP×ETP Expert mesh
EDP 副本 1：完整的 EP×ETP Expert mesh
...
EDP 副本 EDP-1：完整的 EP×ETP Expert mesh
```

同一个 Expert 的同一个 ETP shard 在不同 EDP 副本中有多份拷贝。Router/Dispatcher 可以把发往该 Expert 的 Token 分配到其中一个副本。

主要收益：

- 提高 Expert 侧吞吐；
- 分散热门 Expert 的 Token 压力；
- 给负载均衡器更多选择空间；
- 可将副本分布到不同节点或故障域。

主要成本：

- Expert 权重按 EDP 复制，增加全网显存占用；
- 需要副本选择和负载均衡策略；
- 固定 stage_world 下，增大 EDP 必须减小 EP 或 ETP；
- 若副本选择不考虑拓扑，可能为了负载均衡把 Token 发到更远的位置。

在纯推理中，EDP 不需要像训练数据并行那样同步梯度。它的主要工作是承载相同 Expert 的多份推理服务能力。

---

## 4. 两套坐标如何映射到同一批 GPU

对一个 PP stage，可以把 Rank 看成一段连续的物理卡集合：

```text
Stage 0：Rank  0 ~ Rank 31
Stage 1：Rank 32 ~ Rank 63
```

在 Stage 0 内，同一个 local rank 有两种解释：

```text
Attention 坐标：local_rank ↔ (dp_i, cp_i, tp_i)
Expert 坐标：   local_rank ↔ (edp_i, ep_i, etp_i)
```

一种常见的 Expert 线性编号方式是：

```text
local_rank = etp_i + ETP × (ep_i + EP × edp_i)
```

即：

```text
ETP 是最内层、变化最快的维度
EP  是中间层
EDP 是最外层、变化最慢的维度
```

这不是唯一合法映射，但它体现了常见部署意图：把通信最频繁、粒度最细的 ETP Rank 放得更近，再组织 EP group 和 EDP 副本。

Attention 输出切分与 Expert 输入切分经常不同。因此进入 MoE 时，需要从 `(dp, cp, tp)` 布局转换为 `(edp, ep, etp)` 布局；完成 Expert 计算后，再转换回来。这就是理解 Dispatch/Combine 的全局视角：它不只是“把 Token 发给某个专家”，还承担两套逻辑布局之间的数据重排。

---

## 5. 各并行度如何相互影响

先记住固定 W、固定 PP 时的两个预算：

```text
TP × CP × DP = stage_world
ETP × EP × EDP = stage_world
```

所以任意一侧都不是自由扩张的。

| 增大的维度 | 直接收益 | 主要新增成本 | 固定 W、PP 下会挤压 |
|---|---|---|---|
| PP | 分摊层和权重，支持更大模型 | stage 间 P2P、流水线气泡、单请求路径变长 | 两侧全部 stage 内并行预算 |
| TP | 分摊 Attention 权重与计算 | 高频 Collective、小 GEMM、同步等待 | CP 或 DP |
| CP | 分摊长上下文计算和 KV Cache | K/V 或 Attention 结果通信 | TP 或 DP |
| DP | 增加独立请求副本、提高吞吐 | 权重复制、请求负载不均 | TP 或 CP |
| ETP | 分摊单个 Expert | Expert 内 Collective、小 GEMM | EP 或 EDP |
| EP | 分摊不同 Expert 权重 | Dispatch/Combine A2A、专家负载不均 | ETP 或 EDP |
| EDP | 增加 Expert 副本和吞吐 | Expert 权重复制、副本调度 | ETP 或 EP |

### 5.1 PP 与所有 stage 内并行度

固定 W 时：

```text
PP ↑  ⇒  stage_world ↓
```

PP 增大虽然让每张卡持有更少层，但也可能迫使：

- TP 下降，使单卡 Attention 权重变大；
- CP 下降，使单卡 KV Cache 变大；
- DP 下降，使并发吞吐降低；
- EP 下降，使每卡持有更多 Expert；
- ETP 下降，使单 Expert shard 变大；
- EDP 下降，使 Expert 副本和负载均衡空间减少。

所以“增加 PP 以省显存”必须同时验证每个 stage 剩余的横向并行预算是否足够。

### 5.2 TP、CP、DP 的竞争关系

固定 stage_world 时：

```text
DP = stage_world / (TP × CP)
```

- TP 增大：单请求使用更多卡，Attention 更易装下，但可并发的 DP 副本减少；
- CP 增大：长上下文能力增强，但也会减少 TP 或 DP；
- DP 增大：总吞吐通常提高，但每个请求可用于 TP/CP 的卡变少。

这是典型的“单请求纵向扩展”和“多请求横向扩展”竞争：

```text
TP、CP：让一个请求使用更多卡
DP：    让更多请求分别使用不同卡组
```

### 5.3 ETP、EP、EDP 的竞争关系

固定 stage_world 时：

```text
EDP = stage_world / (ETP × EP)
```

- EP 增大：每份 Expert mesh 中每卡持有的 Expert 更少，但副本数 EDP 下降；
- ETP 增大：单个 Expert 使用更多卡，但能部署的 EP 或 EDP 下降；
- EDP 增大：同一 Expert 的服务副本更多，但每份 mesh 可用的 EP×ETP 规模变小。

可把三者理解成三种不同目标：

```text
EP  ：解决“Expert 太多、总专家权重太大”
ETP ：解决“单个 Expert 太大”
EDP ：解决“Expert 请求太多、热点或吞吐不够”
```

### 5.4 DP 与 EDP 不是同一个概念

DP 是 Attention 侧副本，EDP 是 Expert 侧副本。它们可以不同：

```text
DP=4 不代表 EDP=4
```

如果 `DP > EDP`，多个 Attention DP lane 可能把 Token 汇聚到较少的 Expert 副本；如果 `EDP > DP`，一个 Attention DP lane 的 Token 也可能被分散到多个 Expert 副本。二者之间的映射由 Dispatch 和负载均衡策略决定。

### 5.5 TP 与 ETP 也不是同一个概念

```text
TP  = Attention 内部如何切
ETP = 单个 Expert 内部如何切
```

常见组合包括：

- `TP > ETP`：Attention 很宽，但单 Expert 可以用较少卡完成；
- `TP = ETP`：两侧切分粒度一致，布局转换可能更简单；
- `TP < ETP`：单 Expert 特别大，需要比 Attention 更多的张量切分。

某些实现还会把原 TP Rank 扩展进 EP group，使 `ETP=1` 而获得更大的 EP；这属于具体的 Rank 重映射策略，不改变两侧乘积必须等于 stage_world 的原则。

---

## 6. 显存、计算与通信的近似关系

以下只用于建立量级直觉，实际值还取决于 GQA/MQA、量化、激活格式、Top-K、容量因子和具体 Kernel。

### 6.1 Attention 权重

若一个 stage 的 Attention 权重为 `P_attn_stage`，则单 Rank 近似为：

```text
Attention 权重/Rank ≈ P_attn_stage / TP
```

这些权重通常会沿 CP 和 DP 复制，因此全 stage 的物理存储量近似乘上 `CP×DP`。

### 6.2 KV Cache

设 stage 内总并发 Token 对应的 KV Cache 为 `M_KV_stage`，粗略地：

```text
KV Cache/Rank ≈ M_KV_stage / (DP × CP × TP_KV)
```

其中 `TP_KV` 是否等于 TP 取决于 Attention Head/KV Head 的具体切法。DP 分摊不同请求，CP 分摊同一请求的 Context，TP 可能分摊 KV Head。

### 6.3 Expert 权重

若一个 stage 的全部 Expert 权重为 `P_expert_stage`，则单 Rank 近似为：

```text
Expert 权重/Rank ≈ P_expert_stage / (EP × ETP)
```

EDP 不进一步降低单 Rank 权重；它复制整个 Expert mesh。因此全 stage 的 Expert 权重物理存储量近似为：

```text
P_expert_stage × EDP
```

### 6.4 Expert Token 负载

设某层共有 `Q` 个输入 Token，每个 Token 选择 Top-K 个 Expert，则总路由 assignment 数约为：

```text
assignments = Q × K
```

如果路由完全均匀，单个 Expert 副本平均处理：

```text
assignments_per_expert_replica ≈ (Q × K) / (num_experts × EDP)
```

真实模型往往不均匀，尾部热点 Expert 的负载才是关键。EDP 只有配合有效的副本选择策略，才能真正降低热点。

### 6.5 通信不能只按“每对 Rank 的数据量”估算

以 EP A2A 为例，每个 Rank 可能同时向多个 Rank 发送，也同时从多个 Rank 接收。逻辑上并发，不代表拥有 N-1 条完全独占的物理链路。实际耗时受以下共享资源约束：

- GPU 注入/接收带宽；
- NIC 总带宽；
- NVSwitch 或 PCIe 上行带宽；
- 交换机双分带宽和链路超售；
- Collective 算法分轮次数；
- 最热源 Rank、目标 Rank 或 Expert 的负载。

更稳妥的 A2A 量级模型应同时考虑启动延迟、Rank 总发送/接收量和网络瓶颈，而不是只用单 pair payload：

```text
T_A2A ≈ rounds × α
        + max(
            max_rank_send_bytes / effective_injection_BW,
            max_rank_recv_bytes / effective_reception_BW,
            bottleneck_link_bytes / effective_link_BW
          )
```

对于 EP，一层通常要分别计算 Dispatch 和 Combine，再根据实现判断它们是否能与 Expert 计算或其他通信重叠。

---

## 7. 64 卡全局部署示例

假设：

```text
W  = 64
PP = 2
```

则每个 PP stage 有：

```text
stage_world = 64 / 2 = 32
```

选择 Attention 布局：

```text
TP = 4
CP = 2
DP = 4

TP × CP × DP = 4 × 2 × 4 = 32
```

选择 Expert 布局：

```text
ETP = 2
EP  = 8
EDP = 2

ETP × EP × EDP = 2 × 8 × 2 = 32
```

全局结构为：

```text
64 GPUs
│
├── PP Stage 0：32 GPUs，负责前半模型层
│   ├── Attention 视角：4 DP × 2 CP × 4 TP
│   └── Expert 视角：  2 EDP × 8 EP × 2 ETP
│
└── PP Stage 1：32 GPUs，负责后半模型层
    ├── Attention 视角：4 DP × 2 CP × 4 TP
    └── Expert 视角：  2 EDP × 8 EP × 2 ETP
```

在每个 stage 内：

- 4 个 DP lane 分别处理不同请求；
- 每个请求的长 Context 再分到 2 个 CP rank；
- 每个 Attention 计算由 4 个 TP rank 协作；
- 进入 MoE 后，32 张卡重组为 2 套 Expert 副本；
- 每套副本有 8 个 EP 位置，分摊不同 Expert；
- 每个 Expert 又由 2 个 ETP rank 协作计算。

数据流可以简化为：

```text
4 DP × 2 CP × 4 TP
        │
        │ Router + 布局转换
        ▼
2 EDP × 8 EP × 2 ETP
        │
        │ Dispatch → Expert → Combine
        ▼
4 DP × 2 CP × 4 TP
        │
        │ 完成本 stage 的层后进行 PP P2P
        ▼
下一 Stage
```

这个例子也说明：EDP 不是额外的 2 倍卡数。它使用的仍是 stage 内原来的 32 张卡。

---

## 8. 如何把逻辑并行映射到物理网络

并行配置正确只代表数学上可行；物理映射错误仍可能让性能非常差。

一个常见的优先级是：

```text
最频繁、最细粒度的通信  → 最快、最靠近的链路
低频、大粒度或可流水化通信 → 可以放到更远的链路
```

### 8.1 TP 与 ETP 优先放在节点内

TP 和 ETP 通常每层或每个 Expert 都需要 Collective，而且处在请求关键路径上。因此优先使用：

```text
NVLink / NVSwitch > PCIe > 跨节点网络
```

如果一个节点有 8 张高速互联 GPU，常见做法是让 TP 或 ETP group 尽量完整地落在节点内。

### 8.2 EP 需要高双分带宽

EP 是多对多流量。它不只需要某一条链路快，更需要整个网络在许多 Rank 同时收发时仍有足够的聚合带宽。

部署时应关注：

- EP group 是否跨 NIC；
- 多张 GPU 是否共享同一 NIC 上行；
- 多节点 A2A 是否经过超售交换层；
- Rank 到 NIC 的亲和性；
- 热门 Expert 是否集中在同一节点或同一网络入口。

### 8.3 CP 的放置取决于算法

- Ring CP：尽量让逻辑相邻 Rank 对应物理近邻；
- A2A CP：关注双分带宽，与 EP 类似；
- Gather 型 CP：关注接收端和共享入口是否成为瓶颈。

### 8.4 PP 可以跨较慢层级，但要控制 stage 边界

PP 通常只在 stage 边界发送激活，频率低于每层 TP Collective，因此更适合跨节点。不过边界激活很大、microbatch 很小或 Decode 强延迟场景下，PP P2P 仍可能成为关键路径。

### 8.5 DP 与 EDP 常作为较外层维度

纯推理的 DP/EDP 副本之间通常没有逐层梯度同步，因此可以比 TP/ETP 更自然地跨节点部署。但要注意：

- Attention DP 与 Expert EDP 不一致时，MoE Dispatch 仍可能跨节点；
- EDP 的“副本独立”不代表 Token 永远就近路由；
- 全局调度器应同时考虑负载和通信距离。

一种经验性的嵌套顺序是：

```text
节点内：优先容纳 TP / ETP
节点组：容纳 CP / EP
更外层：组织 PP stage、DP / EDP 副本
```

这只是起点，最终仍要根据实际 GPU、NIC、交换拓扑和通信实现测量。

---

## 9. 从 W 推导并行配置的实用步骤

### 第一步：确定模型容量与拓扑边界

先收集：

- 模型总层数、Attention 权重、Expert 权重；
- Expert 数量、Top-K、Expert 中间维度；
- 最大 Context、目标并发和 KV Cache 大小；
- 单机 GPU 数、显存、节点内和跨节点带宽；
- Prefill/Decode 流量比例以及延迟目标。

### 第二步：选择 PP

用 PP 分摊模型层，直到每个 stage 的权重和 KV Cache 有机会放入显存。不要只验证平均层数，还要验证 embedding、lm_head、共享 Expert 等不均匀模块。

### 第三步：选择 TP

选择能让 Attention 权重装下、GEMM 尺寸仍然高效的最小 TP。优先不跨高速互联域。

### 第四步：选择 CP

只有当长 Context 的 KV Cache 或 Attention 计算需要进一步拆分时增大 CP，并根据实际 CP 算法检查网络。

### 第五步：由预算确定 DP

```text
DP = W / (PP × TP × CP)
```

DP 必须是整数。然后检查这个 DP 是否能满足目标并发和吞吐。

### 第六步：选择 EP

根据 Expert 总权重、Expert 数量和网络 A2A 能力选择 EP。通常要求 Expert 数能够合理映射到 EP 位置；不整除时需要明确的非均匀分配规则。

### 第七步：选择 ETP

仅在单个 Expert 的权重或计算需要多卡协作时增大 ETP，同时确认 Expert 中间维度可切分，且 ETP Collective 能落在高速链路上。

### 第八步：由 Expert 预算确定 EDP

```text
EDP = W / (PP × EP × ETP)
    = stage_world / (EP × ETP)
```

EDP 必须是整数。它决定同一 Expert mesh 有多少份副本。

### 第九步：验证布局转换

确认：

- Attention `(dp, cp, tp)` 到 Expert `(edp, ep, etp)` 的 Rank 映射；
- Dispatch 和 Combine 的通信 group；
- 是否使用 EP、EP×ETP 联合 A2A，或 ETP Sequence Parallel；
- Token 是否在 ETP Rank 间被复制；
- EDP 副本如何选择；
- 输出如何准确恢复到原 Token 顺序。

### 第十步：按 Prefill 和 Decode 分别压测

不能用单一平均值代替两种阶段：

- Prefill：Token 多、GEMM 大，容易把带宽跑满，也更可能有足够计算窗口用于通信重叠；
- Decode：每步 Token 少、Kernel 小，Collective 启动延迟和尾延迟更突出；
- 在线混合流量：还要考虑 continuous batching、请求长度差异和 Expert 热点。

---

## 10. 不同目标下的配置倾向

### 10.1 单请求低延迟优先

- PP 尽量小，减少串行 stage 和边界传输；
- TP/ETP 只增大到容量或计算收益确实需要；
- 避免 TP、ETP 和 EP 跨越低速网络；
- EP 不宜仅为“看起来并行度更高”而盲目增大；
- CP 只在长 Context 确实需要时使用。

### 10.2 高吞吐优先

- 在模型可装下后，优先保留足够 DP；
- 用 continuous batching 提高 Expert GEMM 的 Token 数；
- 用 EDP 分摊 Expert 热点，但要考虑其权重复制成本；
- 让 EP 的单 Expert batch 足够大，避免大量小 GEMM；
- 通过分块 Dispatch/Expert/Combine 实现流水重叠。

### 10.3 超长上下文优先

- 先评估 KV Cache 是否可由 TP 和 DP 请求分片承载；
- 不足时增大 CP；
- CP 增大后重新评估 DP 是否仍满足并发；
- 验证 CP 通信与 EP A2A 是否竞争同一 NIC 或交换链路。

### 10.4 超大 MoE 模型容量优先

- 先用 PP 分摊层；
- 用 EP 分摊不同 Expert；
- 单 Expert 仍放不下时再用 ETP；
- 剩余 stage_world 才能形成 EDP；
- 确认 Expert 总权重复制 EDP 份后仍满足显存预算。

---

## 11. 通信与计算重叠在全局图中的位置

所谓“通信计算重叠”，不是把同一块数据的前后依赖消除，而是把 Token 分块或使用多个执行流，让不同数据块处于不同阶段：

```text
时间 ───────────────────────────────────────────────▶

Chunk 0：Dispatch ─ Expert Compute ─ Combine
Chunk 1：          Dispatch ─ Expert Compute ─ Combine
Chunk 2：                    Dispatch ─ Expert Compute ─ Combine
```

在稳态窗口中可能出现：

```text
网络正在 Dispatch Chunk 2
GPU   正在计算   Chunk 1 的 Expert GEMM
网络正在 Combine  Chunk 0
```

它依赖以下条件：

- Dispatch/Combine 与 GEMM 使用的硬件资源没有完全冲突；
- Token 足够多，可以切成多个有效 Chunk；
- Expert GEMM 时间足够长，能覆盖一部分网络时间；
- 通信库、CUDA stream、事件依赖和 Buffer 生命周期支持异步流水；
- 两轮通信没有把同一 NIC 带宽完全打满而互相争抢。

Prefill 通常比低 Batch Decode 更容易获得重叠收益，因为 Prefill 的 Token 和 Expert GEMM 更大。低 Batch Decode 的单步 Token 少，通信启动延迟占比高，能拿来覆盖通信的计算窗口也小。

重叠后的端到端耗时也不是简单取 `max(通信, 计算)`。只有完全稳态、资源独立且无流水首尾开销时才接近该下界；实际还包含 warm-up、drain、依赖空洞和资源竞争。

---

## 12. 部署前必须检查的不变量

### 12.1 规模恒等式

```text
W % PP == 0
stage_world = W / PP
TP × CP × DP == stage_world
ETP × EP × EDP == stage_world
```

### 12.2 模型维度可切分性

根据具体实现检查：

- Attention Head 数是否能被 TP 合理切分；
- KV Head 数与 TP 的映射是否受 GQA/MQA 限制；
- Context 长度或 Block 数能否被 CP 分配；
- Expert 数是否能被 EP 分配；
- Expert 中间维度能否被 ETP 切分；
- PP stage 的层数和计算量是否均衡。

### 12.3 通信组正确性

- TP/CP/EP/ETP group 是否只包含同一 PP stage 的 Rank；
- DP 和 EDP 的副本边界是否清晰；
- Dispatch 的目标 Rank 是否同时匹配 Expert ID、ETP shard 和 EDP replica；
- Combine 是否恢复原 Token 顺序和 Top-K 权重；
- Rank 映射是否符合物理拓扑。

### 12.4 性能验证

至少分别观测：

- Prefill 和 Decode 延迟；
- 每种 Collective 的时间和实际字节数；
- GPU GEMM 利用率；
- 每 Rank 发送量和接收量；
- 每 Expert Token 数、最大值、均值和偏斜度；
- NIC/NVLink 带宽与拥塞；
- PP stage 空闲时间和气泡；
- EDP 副本的负载分布。

---

## 13. 常见误区

### 误区一：所有并行度全部相乘

错误。`TP×CP×DP` 和 `ETP×EP×EDP` 是同一 stage_world 的两种分解。

### 误区二：DP 等于 EDP

错误。DP 复制 Attention 执行域，EDP 复制 Expert mesh；二者数值和调度目标都可以不同。

### 误区三：EP 越大，Expert 计算一定越快

错误。EP 会减少每 Rank 持有的 Expert，但可能增加远端 Token、A2A、热点和小 GEMM。

### 误区四：FullMesh 意味着每个 Rank 有 N-1 条独占物理链路

错误。FullMesh 通常描述逻辑可达关系；多个流仍会共享 GPU 入口、NIC、交换机和链路带宽。

### 误区五：只要有异步 API，通信就被完全隐藏

错误。重叠需要足够的独立计算窗口和硬件资源，且有流水线首尾成本。

### 误区六：只验证公式整除即可部署

错误。数学可行不等于性能可行，还必须验证 Rank 映射、拓扑、路由偏斜、Kernel 尺寸和真实通信带宽。

---

## 14. 一页总结

```text
第一层：按模型深度切
W = PP × stage_world

第二层：同一个 stage 内，Attention 看见
stage_world = TP × CP × DP

第三层：同一个 stage 内，Expert 看见
stage_world = ETP × EP × EDP

因此
TP × CP × DP = ETP × EP × EDP
```

每个维度解决的问题：

```text
PP  ：模型层太多、整模型放不下
TP  ：Attention 层太宽、单层权重或计算太大
CP  ：Context 太长、KV Cache 或 Attention 计算太大
DP  ：请求太多、需要更多 Attention 吞吐
EP  ：Expert 太多、总专家权重太大
ETP ：单个 Expert 太大
EDP ：Expert Token 太多或存在热点、需要更多专家副本
```

全局部署的本质，是在固定 W 下平衡四类目标：

```text
模型能否装下
单请求能否足够快
全网吞吐能否足够高
通信是否匹配物理拓扑
```

不存在脱离模型形状、请求流量和网络拓扑的“最大并行度最优解”。正确方法是先满足容量约束，再在真实 Prefill/Decode 工作负载下，用端到端测量选择最小必要的 TP/CP/ETP/EP，并把剩余资源用于能带来吞吐收益的 DP/EDP。

---

## 参考依据

本文的并行域恒等式、Rank 双坐标模型和通信语义与以下项目文档保持一致：

- `docs/training_inference_parallel_domains_zh.md`
- `docs/architecture/specs/communication.md`
- `ARCHITECTURE.md`
