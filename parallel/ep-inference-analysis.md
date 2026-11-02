# 推理场景下的 Expert Parallelism（EP）分析

> 本文整理推理场景中 Expert Parallelism、All-to-All、MoE Dispatch/Combine、FullMesh 通信、EP 扩展收益与代价，以及 SGLang 社区的通信计算重叠机制。

## 1. EP 解决什么问题

Mixture-of-Experts（MoE）模型包含大量专家，但每个 token 通常只激活其中 Top-K 个专家。Expert Parallelism（EP）把专家分布到不同 Rank 上：

```text
8 个专家，EP=4：

Rank 0：Expert 0、1
Rank 1：Expert 2、3
Rank 2：Expert 4、5
Rank 3：Expert 6、7
```

设：

```text
E = 专家总数
P = EP 大小
Q = 一个 MoE 层处理的全局 token 数
K = 每个 token 激活的专家数（Top-K）
```

在路由均衡的理想情况下：

```text
每卡专家数        ≈ E / P
每卡 routed token ≈ Q × K / P
每卡专家计算量    ≈ 总专家计算量 / P
每卡专家权重内存  ≈ 专家权重总量 / P
```

EP 的核心价值是降低每卡专家权重显存，并把专家计算分摊到更多设备上。代价是 token 必须根据 Router 的选择跨 Rank 重分布。

EP 只切分 routed expert 部分。Attention 权重、KV Cache 和 Shared Expert 是否切分取决于其他并行策略，因此模型总显存和总耗时不会严格按 `1/EP` 缩放。

## 2. 一层 MoE 的数据流

```text
输入 hidden states
        │
        ▼
      Router
  为 token 选择 Top-K 专家
        │
        ▼
Dispatch：复制、分桶、重排 token
        │
        ▼
第一次 All-to-All：发送到专家所在 Rank
        │
        ▼
各 Rank 执行本地 Routed Expert GEMM
        │
        ▼
第二次 All-to-All：专家结果发回 token 原 Rank
        │
        ▼
Combine：恢复顺序并按 Router 权重加权求和
        │
        ▼
输出 hidden states
```

对一个 token `t`，Combine 最终产生：

```text
output(t) = Σ router_weight(t, expert_i) × expert_i(t)
```

## 3. All-to-All 是什么

All-to-All（A2A）表示通信组中的每个 Rank 都可以向其他每个 Rank 发送不同的数据块。

发送前，每个 Rank 按目标 Rank 切分缓冲区：

```text
Rank 0 send = [0→0 | 0→1 | 0→2 | 0→3]
Rank 1 send = [1→0 | 1→1 | 1→2 | 1→3]
Rank 2 send = [2→0 | 2→1 | 2→2 | 2→3]
Rank 3 send = [3→0 | 3→1 | 3→2 | 3→3]
```

交换后，每个 Rank 得到所有来源发送给自己的部分：

```text
Rank 0 recv = [0→0 | 1→0 | 2→0 | 3→0]
Rank 1 recv = [0→1 | 1→1 | 2→1 | 3→1]
Rank 2 recv = [0→2 | 1→2 | 2→2 | 3→2]
Rank 3 recv = [0→3 | 1→3 | 2→3 | 3→3]
```

它可以理解为一次分布式数据转置。

与其他集合通信的区别：

- AllGather：每个 Rank 收集到所有 Rank 的完整数据。
- AllReduce：跨 Rank 做求和、最大值等归约，并把结果返回给所有 Rank。
- All-to-All：按目标 Rank 重排数据，不复制完整集合，也不做归约。

A2A 没有主 Rank，也不存在 Rank 0 固定先发。各 Rank 进入同一个 collective 后并发参与；先到的 Rank 可能等待晚到的 Rank，整个操作在所有必要传输完成后结束。

## 4. Dispatch 和 Combine

### 4.1 Dispatch

Router 为每个 token 给出 Top-K 专家 ID 和权重。Dispatch 负责：

1. 根据目标专家确定目标 EP Rank；
2. 按 Top-K 复制 token；
3. 按目标 Rank 和本地专家分桶；
4. 打包通信缓冲区；
5. 通过 A2A 把 token 发送到专家所在 Rank；
6. 在接收端按本地专家重新排列。

逻辑形状可表示为：

```text
hidden states：[B, S, H]
expert_ids：  [B, S, K]
routed data： [K, B, S, H]
```

若 Dispatch 使用低精度通信，则 payload 近似为：

```text
S_dispatch = B × S × K × H × dispatch_dtype_bytes
```

### 4.2 Combine

专家计算完成后，结果仍按专家分桶并位于专家所在 Rank。Combine 负责：

1. 按 token 的原始 Rank 重新分桶；
2. 通过第二次 A2A 发回；
3. 恢复原 token 顺序；
4. 按 Router 权重合并 Top-K 专家结果。

payload 近似为：

```text
S_combine = B × S × K × H × combine_dtype_bytes
```

如果 Dispatch 使用 FP8/INT8，而 Combine 使用 BF16/FP16，在形状相同时：

```text
S_combine ≈ 2 × S_dispatch
```

因此 Combine 的带宽项通常更大，但由于二者都包含固定启动延迟，总耗时不一定严格相差两倍。

## 5. A2A 的真实并发方式

### 5.1 发送和接收

现代 GPU/NPU 互联和 NIC 通常支持全双工，发送和接收可以同时进行：

```text
Rank 0 ───发送──→ Rank 1
Rank 0 ←──接收─── Rank 1
```

因此通信时间通常不是：

```text
发送时间 + 接收时间
```

而是受到较慢方向和共享资源的约束。收发字节可以相加做流量统计，但不能直接把 `2×` 再代入全双工耗时公式。

### 5.2 多 peer 接收

一张卡可以并发接收其他 `P-1` 个 Rank 的数据，但“并发”不表示每份数据都独占整卡带宽。多个流量可能共享：

- GPU/NPU 聚合入口和出口带宽；
- NVLink/HCCS/PCIe 端口；
- NIC；
- 交换网络；
- DMA/通信引擎；
- 片上网络和 HBM 带宽。

通信库可能采用：

- FullMesh/交换网络上的多路并发；
- Pairwise exchange：`P-1` 轮，每轮所有 Rank 并发交换；
- Bruck：较大规模下约 `ceil(log2 P)` 轮打包交换；
- 分层 A2A：节点内重排、节点间交换、节点内再分发。

## 6. FullMesh：独立链路还是共享入口

FullMesh 首先描述“任意两点可直接通信”的拓扑能力，不自动规定端点带宽口径。

### 6.1 物理直连 FullMesh

严格的物理 FullMesh 为每对设备配置独立链路。多个 peer 可以通过不同链路并发传输，但进入设备后仍会共享互联控制器、L2、片上网络和 HBM 等资源。

### 6.2 NVSwitch 等逻辑 FullMesh

现代 GPU 系统更常见的是经交换机构成的逻辑 FullMesh：

```text
GPU 1 ─┐
GPU 2 ─┼──→ NVSwitch Crossbar ──→ GPU 0 的多条 NVLink 端口
GPU 3 ─┘
```

NVSwitch 可以提供非阻塞或高二分带宽交换，但不同 peer 的流量最终仍共享目标 GPU 的有限聚合入口带宽。这里的“全带宽”通常表示交换 fabric 能让各端点达到自己的 line rate，而不是每个 peer 都同时独占目标 GPU 的全部聚合带宽。

参考：

- [NVIDIA DCGM：NVLink 端口与 GPU 聚合流量](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)
- [NVIDIA Fabric Manager](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)
- [NVIDIA NVLink 与 NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/)

## 7. A2A 通信量与耗时公式

设：

```text
P  = EP group size
S  = 每张卡参与一次 A2A 的逻辑 payload
α  = 通信启动延迟
BW = 单卡聚合单向有效带宽
```

均匀切分时，每张卡发给单个目标 Rank：

```text
peer_payload = S / P
```

其中发给自己的 `S/P` 不走网络，因此每张卡单向跨 Rank 字节为：

```text
remote_bytes = S × (P-1)/P
```

若 `BW` 是单卡聚合单向带宽，更合理的简化公式是：

```text
T_A2A ≈ α + remote_bytes / BW
      = α + S × (P-1)/P / BW
```

不要在时间公式中乘 2：发送和接收通常全双工并行。收发合计流量可以记录为：

```text
reported_send_recv_bytes = 2 × S × (P-1)/P
```

但它是流量指标，不是单向带宽公式的分子。

如果已知单 peer 独立链路带宽和单卡聚合带宽，可以同时施加两种约束：

```text
T_A2A = max(
    α_pair + (S/P) / BW_peer,
    α_gpu  + S×(P-1)/P / BW_gpu,
    fabric_traffic / BW_fabric,
    memory_traffic / BW_HBM
)
```

只有当每个 peer 都拥有独立的 `BW_peer`，且设备聚合入口至少达到 `(P-1)×BW_peer` 时，才能仅用：

```text
α + (S/P) / BW_peer
```

实际系统还应使用 A2A benchmark 校准启动延迟和有效带宽，因为小包效率、协议轮次、端口竞争、路由和拥塞不会被简化公式完全描述。

## 8. EP 增大的收益

### 8.1 专家权重显存下降

```text
expert_weight_memory_per_rank ≈ total_expert_weights / P
```

这是最确定的收益，也是大规模 MoE 能否部署的关键条件。

### 8.2 专家计算分摊

固定全局 token 数、路由均衡时：

```text
expert_compute_per_rank ≈ total_expert_compute / P
```

但整个 Transformer 层还包含 Attention、Router、通信和其他计算，因此端到端延迟不会按 `1/P` 缩短。

### 8.3 模型容量与吞吐扩展

EP 增大可以支持更多或更大的专家，并在增加设备、维持每卡负载时提高全局吞吐。

## 9. EP 增大的代价

### 9.1 远端路由比例升高

假设专家均匀分布，assignment 留在本 Rank 的概率约为 `1/P`：

```text
remote_fraction = (P-1)/P
```

```text
EP=1： 0
EP=2： 1/2
EP=4： 3/4
EP=8： 7/8
EP=16：15/16
```

EP 增大后，绝大多数 routed assignment 都需要跨卡。

一次 Dispatch 的全局跨 Rank 流量近似为：

```text
V_dispatch_global = Q × K × H × dtype_bytes × (P-1)/P
```

Combine 还需要一次反向重分布。

### 9.2 启动、打包与同步开销

除网络传输外，Dispatch/Combine 还包含：

- 路由计数和 prefix sum；
- token permutation；
- 通信 buffer 打包和解包；
- dtype 转换；
- collective 启动；
- Rank 间同步；
- HBM 读写。

整个 A2A 和 MoE 层通常受最慢 Rank 影响。

### 9.3 路由不均衡

平均每卡 routed token 为 `Q×K/P`，但实际层时间更接近最繁忙 Rank：

```text
T_expert ∝ max(tokens_on_rank_i)
```

EP 越大、token 越少，统计波动越明显，容易出现部分 Rank 繁忙、部分 Rank 空闲。

### 9.4 Expert GEMM 利用率下降

EP 增大使每 Rank、每专家得到的 token 减少，Grouped GEMM 的 `M` 维变小，可能降低 Tensor Core/NPU Cube 利用率。即使 FLOPs 减少一半，实际时间也可能只下降很少。

### 9.5 跨越物理网络边界

EP 从机内扩大到跨机时，性能可能发生非连续跳变：

```text
EP=8：机内 NVLink/NVSwitch
EP=16：开始跨机走 InfiniBand/RoCE
```

此时带宽下降、启动延迟上升，Dispatch/Combine 可能迅速成为瓶颈。

### 9.6 固定总卡数时挤压 DP

如果总设备数固定，增大 EP 往往会减少 DP 副本数：

```text
EP ↑ → 单请求专家并行度 ↑ → 独立请求副本数 ↓
```

它可能改善模型可部署性或单请求 MoE 计算，却降低高并发场景的整体 QPS。

## 10. 强扩展与弱扩展需要分开讨论

### 10.1 固定全局 token 数

全局工作量 `Q` 固定：

```text
每卡专家计算   ≈ Q×K/P
每卡平均远端量 ≈ Q×K×H×bytes×(P-1)/P²
全网远端总量   ≈ Q×K×H×bytes×(P-1)/P
```

每卡字节可能下降，但启动和同步时间不会按 `1/P` 缩短；计算下降得更快后，通信占比反而会上升。

### 10.2 每卡 token 数固定

若扩容时每卡保持 `q` 个 token，则 `Q=P×q`：

```text
每卡远端量 ≈ q×K×H×bytes×(P-1)/P
```

每卡通信量逐渐趋近 `q×K×H×bytes`，全网流量随规模增长。吞吐有机会扩展，但依赖网络、路由均衡和 Expert GEMM 效率。

## 11. Prefill 与 Decode

### 11.1 Prefill

Prefill 通常 token 多：

- Attention 计算窗口长；
- Expert Grouped GEMM 较大，计算效率高；
- A2A payload 大，带宽项占主导；
- 容易把 batch 拆成计算量较均衡的 micro-batch；
- 更有机会用独立计算覆盖通信。

### 11.2 Decode

Decode 每个请求每步通常只有一个 token：

- A2A payload 较小；
- 启动延迟和同步占比高；
- 每专家 token 少，GEMM 利用率低；
- 路由不均衡更明显；
- 小 batch 下缺少足够长的独立计算窗口。

Continuous batching 可以聚合大量并发请求，为每步形成较大的 token batch，从而改善 GEMM 效率，并为通信计算重叠创造条件。

## 12. 通信计算重叠是什么

通信计算重叠不是消除通信，而是在网络传输期间执行与该通信没有数据依赖的计算。

串行时：

```text
T_serial = T_compute + T_comm
```

理想重叠时：

```text
T_overlap ≈ max(T_compute, T_comm)

hidden_comm  = min(T_comm, T_independent_compute)
exposed_comm = max(0, T_comm - T_independent_compute)
```

例如通信 120 μs、独立计算 200 μs，则通信可以全部隐藏；若独立计算只有 40 μs，则仍暴露 80 μs。

### 12.1 同一个 micro-batch 的依赖限制

```text
Dispatch A → Routed Expert A → Combine A
```

三者有严格依赖，因此普通实现不能用 `Expert A` 隐藏自己的 `Dispatch A`，也不能在完整专家输出产生前完成自己的 `Combine A`。

需要寻找独立工作：

- 另一个 micro-batch 的 Attention、Router 或 Expert GEMM；
- 同一批次的 Shared Expert 分支；
- 其他无数据依赖的 kernel。

### 12.2 Two-Batch Overlap（TBO）

将一个 batch 拆成两个独立 micro-batch：

```text
A：Attention A → Dispatch A → Expert A → Combine A
B：Attention B → Dispatch B → Expert B → Combine B
```

交错后可以形成：

```text
Dispatch A || B 的 Attention/Router
Expert A   || Dispatch B
Combine A  || B 的独立计算
```

具体对齐关系取决于模型调度和 stream，原则是让 A 的通信与 B 的独立计算并发。

### 12.3 Single-Batch Overlap（SBO）

Routed Expert 和 Shared Expert 通常是独立分支：

```text
                     ┌→ Dispatch → Routed Expert → Combine ─┐
hidden states ───────┤                                     ├→ Add
                     └→ Shared Expert GEMM ─────────────────┘
```

Shared Expert 不依赖 Routed Expert 的 Dispatch 结果，因此其 GEMM 可以覆盖一部分 Dispatch 或 Combine。

### 12.4 重叠并非免费

通信和计算仍可能竞争 SM、HBM、PCIe/NVLink、片上网络和 NIC。实际通常满足：

```text
max(T_compute, T_comm)
    ≤ T_actual_overlap
    < T_compute + T_comm
```

## 13. SGLang 社区的当前实现

### 13.1 DeepEP 异步通信

SGLang 可使用 DeepEP 处理 EP Dispatch/Combine。DeepEP 提供通信 buffer、通信 stream、异步 event/receive hook，以及通信 SM 数量控制。

典型执行逻辑：

```text
启动 Dispatch
→ 返回 event/hook
→ 执行独立计算
→ 真正使用接收数据前等待 event
```

参考：[DeepEP](https://github.com/deepseek-ai/DeepEP)

### 13.2 Dispatch/Combine 分阶段

SGLang Dispatcher 提供：

```text
dispatch_a / dispatch_b
combine_a  / combine_b
```

`a/b` 是启动、准备、完成和等待的调度切点，不代表发送端和接收端。不同 DeepEP 模式中，通信真正启动的位置略有差异，但共同目标是把异步通信与等待点分开，为其他计算留下调度窗口。

参考：[SGLang DeepEP dispatcher](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)

### 13.3 TBO 调度

启用示例：

```bash
--moe-a2a-backend deepep
--deepep-mode auto
--enable-two-batch-overlap
```

TBO 通过 `YieldOperation` 把每个 micro-batch 切成多个 stage，并交错推进两个子 batch。GPU kernel 和通信运行在不同 stream 上，使前一阶段发出的异步通信可以与另一子 batch 的计算并发。

参考：

- [SGLang EP 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)
- [SGLang TBO stage executor](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_overlap/operations.py)
- [SGLang operation strategy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_overlap/operations_strategy.py)

### 13.4 Prefill 与 Decode 模式

`--deepep-mode auto` 的当前语义通常是：

```text
Prefill：DeepEP normal，高吞吐模式
Decode： DeepEP low_latency，低延迟模式
```

Prefill 追求大 payload 带宽，并可为 DeepEP 通信和 DeepGEMM 分配不同 SM 资源。Decode 的 Low-Latency 路径尽量在发出 RDMA 后释放 GPU SM，再通过 event/hook 等待通信完成。

参考：[SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)

### 13.5 支持边界

TBO/SBO 并非在所有模型、硬件和执行模式下都等价可用：

- 需要显式启用相应选项；
- 不同模型有独立的 operation strategy；
- CUDA Graph、投机解码、Context Parallel 和不同 A2A backend 可能限制组合；
- 某些模型当前只支持 Prefill TBO，Decode 路径可能尚未实现或收益不稳定。

建模与性能分析应分别记录：

```text
raw_dispatch_us
raw_combine_us
hidden_dispatch_us
hidden_combine_us
exposed_dispatch_us
exposed_combine_us
```

请求关键路径应计入 `exposed_comm_us`，而不是无条件把 Dispatch 和 Combine 的 raw latency 全部相加。

## 14. EP 的业务决策建议

适合增大 EP：

- 专家权重在较小 EP 下无法装入显存；
- MoE FFN 是主要瓶颈；
- batch/token 足够大；
- Router 分布较均衡；
- EP group 可以停留在高速互联域；
- Dispatch/Combine 有可重叠的独立计算窗口。

不宜继续增大 EP：

- 即将从机内互联跨到节点间网络；
- Decode batch 较小；
- 每专家 token 太少，Grouped GEMM 利用率明显下降；
- Dispatch 与 Combine 的暴露时间已接近或超过 Expert GEMM；
- 固定总卡数下 DP 被压缩，整体 QPS 开始下降；
- p99 延迟被热点专家或慢 Rank 主导。

建议优先选择能让专家权重安全装入显存的最小 EP，再在不跨越慢速互联、且 Expert GEMM 仍有足够 token 利用率的范围内逐步增大。

评估时至少观察：

```text
每 Rank 专家权重显存
每专家/每 Rank token 数
Expert GEMM 利用率与耗时
Dispatch/Combine raw latency
Dispatch/Combine exposed latency
路由负载不均衡程度
跨越的物理网络层
Prefill/Decode tokens/s
p50/p99 延迟
单位 token 成本
```

