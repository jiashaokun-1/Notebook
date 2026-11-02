# EP 扩展收益与代价：推理建模服务实测与正确性排查

## 1. 结论摘要

本次部署并实测了当前推理建模服务，使用 DeepSeek V4 Pro、Adevice03_POD，完成 22 个 EP 强扩展、弱扩展和 Prefill/Decode 对照任务，并补充 4 个并行配置探针。

结论分为两部分。

### 已正确或内部自洽的部分

1. EP 权重分片按 `1/EP` 生效。单 Rank 参数量可精确拟合为：

   ```text
   rank_params ≈ 26.86 GiB + 810.63 GiB / EP
   ```

2. routed-token 口径按 `ceil(B × S × top_k / EP)` 生效；本次 `EDP=1`。
3. Dispatch/Combine 的 logical payload 与 Tensor shape 一致。
4. A2A 报告字节满足：

   ```text
   comm_bytes = 2 × payload × (EP - 1) / EP
   ```

5. 通信域随物理规模切换正确：EP2 使用 `bwsio`，EP4~16 使用 `intra_node`，EP32 及以上使用 `pod_fabric`。
6. 单 Rank OOM 随 EP 增大而解除：本配置 EP16 仍 OOM，EP32 首次不 OOM。
7. 68 个 EP/通信相关定向测试通过。

### 不能直接相信或尚未完整建模的部分

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | 推理 API 固定 `EDP=1、ETP=1、EP=W/PP`；`edp_size/etp_size` 输入会被静默忽略 | 无法在固定 W 下单独扫描 EP，当前曲线混合了 DP 与 EP 扩展，无法验证一般的 `EP×ETP×EDP=stage_world` |
| P1 | FullMesh A2A 只按单 pair 的 `payload/EP` 计带宽 | 对共享 GPU/NIC 入口明显低估，EP 越大低估倍数越大；Prefill 的端到端偏差更突出 |
| P1 | GroupMatMul 单次 HBM 流量读取全部本地 Expert 权重，没有 active-expert 数 | 低 Batch Decode 下显著高估小 EP 的 Expert 访存，从而夸大增大 EP 的收益 |
| P1 | 推理链路没有 SGLang TBO/SBO 的 Dispatch/Expert/Combine 分块流水 | 真实启用 overlap 时会高估暴露通信；与 FullMesh 低估是两个方向相反、不能互相抵消的误差 |
| P2 | 通信算子的公开 `bound_type` 恒写 `communication`，而汇总重新按 `max(memory, communication)` 分类 | 同一算子在明细和汇总中可能出现互相矛盾的瓶颈标签 |
| P2 | Adevice03_POD 的 `intra_node` 与 `pod_fabric` 使用相同带宽和延迟 | EP 跨出节点后几乎没有拓扑跳变，需确认硬件资产是否真实 |

因此，当前服务可以用于检查“专家权重能否通过 full-stage EP 装入显存”和理想均衡下的趋势，但不应直接用来决定真实集群的最优 EP，尤其不能据此判断固定 W 下 EP 与 EDP/ETP 的取舍。

---

## 2. 实验环境

### 2.1 服务

```text
Backend API : http://127.0.0.1:8001
Worker      : 1 个 worker，内部 N=8/R=1
Python      : 3.13.9，仓库 .venv
代码分支    : main
代码版本    : d3f0cde4
```

服务已通过：

```text
GET /api/health → {"status":"ok"}
```

根启动脚本默认找到系统 Python 3.9，未满足 Python 3.10+；本次显式使用仓库 `.venv/bin/python` 启动，没有修改启动脚本。

### 2.2 模型与硬件

```text
模型：deepseek-v4-pro
层数：61
Hidden：7168
Routed Experts：384
Top-K：6
Expert Intermediate：3072

硬件：Adevice03_POD
显存：64 GiB/卡
```

通信层级：

| 层级 | 覆盖卡数 | 标称带宽 | 有效带宽 | Alpha | 拓扑 |
|---|---:|---:|---:|---:|---|
| bwsio | 2 | 224 GB/s | 179.2 GB/s | 2 us | FullMesh |
| intra_node | 16 | 196 GB/s | 156.8 GB/s | 2 us | FullMesh |
| pod_fabric | 768 | 196 GB/s | 156.8 GB/s | 2 us | FullMesh |
| scale_out | 无上限 | 50 GB/s | 35 GB/s | 4 us | FatTree |

本次 EP 最大为 128，没有跨出 pod_fabric。

### 2.3 量化与公共并行配置

```text
Attention / Shared Expert / Activation / KV：FP8
Routed Expert：FP4
Speculative Decode：关闭
PP=1，TP=1，CP=1，ETP=1，EDP=1
DP=EP=W
MoE A2A backend：DeepEP
```

注意：`ETP=1、EDP=1` 不是本次主动选择，而是当前推理后端的硬约束。

---

## 3. 实验设计

### 3.1 Decode 强扩展

目的：固定业务负载，观察加卡后的时延下降、通信占比、显存与卡效。

```text
Batch=128
Input Length=8192
W=EP=DP ∈ {1,2,4,8,16,32,64,128}
```

随着 W 增大，Attention 每 Rank Batch 与 Expert routed rows 同时减少，因此这条曲线不是纯 EP 曲线，而是 DP+EP 联合强扩展。

### 3.2 Decode 弱扩展

目的：固定每 Rank 工作量，突出 EP group 增大本身的通信与权重分片影响。

```text
Batch=4×W
Input Length=8192
W=EP=DP ∈ {1,2,4,8,16,32,64,128}
每 Rank Attention Batch=4
每 Rank routed assignments=4×6=24
```

这条曲线中 Attention 计算与 Expert FLOPs 近似不变。若通信入口共享，EP group 增大不应让固定 per-rank payload 的带宽耗时趋近于零。

### 3.3 Prefill 强扩展

目的：观察大 Token payload 下 EP 通信公式的影响。

```text
Batch=32
Input Length=2048
W=EP=DP ∈ {1,2,4,8,16,32}
```

---

## 4. 服务实测结果

### 4.1 Decode 强扩展

服务返回的 `tps` 是单卡 TPS；“全实例 TPS”按 `tps×W` 还原。

| EP/W | Rank 参数 GiB | TPOT ms | 加速比 | 并行效率 | 单卡 TPS | 全实例 TPS | EP 通信归因 ms | OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 837.489 | 686.32 | 1.000 | 100.0% | 186.50 | 186.50 | 0.000 | 是 |
| 2 | 432.173 | 358.27 | 1.916 | 95.8% | 178.63 | 357.26 | 2.625 | 是 |
| 4 | 229.515 | 191.95 | 3.576 | 89.4% | 166.71 | 666.84 | 1.622 | 是 |
| 8 | 128.186 | 108.99 | 6.297 | 78.7% | 146.80 | 1174.40 | 1.320 | 是 |
| 16 | 77.521 | 67.59 | 10.154 | 63.5% | 118.37 | 1893.92 | 1.245 | 是 |
| 32 | 52.189 | 46.90 | 14.634 | 45.7% | 85.28 | 2728.96 | 1.226 | 否 |
| 64 | 39.523 | 36.57 | 18.767 | 29.3% | 54.70 | 3500.80 | 1.222 | 否 |
| 128 | 33.190 | 31.40 | 21.857 | 17.1% | 31.85 | 4076.80 | 1.220 | 否 |

当前模型给出的业务含义：

- EP32 是本配置满足 64 GiB/卡的最小 EP；
- EP32→64：卡数翻倍，TPOT 仅下降 22.0%，全实例 TPS 增加 28.3%；
- EP64→128：卡数再次翻倍，TPOT 仅下降 14.1%，全实例 TPS 增加 16.5%；
- 固定 Batch 下卡效持续下降。

但这个结论仍受 FullMesh 低估和 inactive Expert 权重流量影响，不能直接用于真机选型。

### 4.2 Decode 弱扩展

| EP/W | 全局 Batch | Rank 参数 GiB | TPOT ms | Compute ms | Memory ms | EP 通信归因 ms | GroupMatMul ms | 单卡 TPS | OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 837.489 | 673.92 | 0.461 | 673.457 | 0.000 | 648.654 | 5.94 | 是 |
| 2 | 8 | 432.173 | 350.97 | 0.461 | 349.204 | 1.308 | 324.401 | 11.40 | 是 |
| 4 | 16 | 229.515 | 188.81 | 0.461 | 187.078 | 1.270 | 162.274 | 21.19 | 是 |
| 8 | 32 | 128.186 | 107.72 | 0.461 | 106.014 | 1.245 | 81.211 | 37.13 | 是 |
| 16 | 64 | 77.521 | 67.18 | 0.461 | 65.483 | 1.233 | 40.679 | 59.55 | 是 |
| 32 | 128 | 52.189 | 46.90 | 0.461 | 45.217 | 1.226 | 20.414 | 85.28 | 否 |
| 64 | 256 | 39.523 | 36.77 | 0.461 | 35.084 | 1.223 | 10.281 | 108.79 | 否 |
| 128 | 512 | 33.190 | 31.70 | 0.461 | 30.018 | 1.222 | 5.214 | 126.19 | 否 |

关键观测：

1. 每 Rank routed assignments 恒为 24，Compute 恒为 0.461 ms，说明 FLOPs 口径守恒。
2. GroupMatMul 耗时却随 EP 精确接近 `1/EP`，说明当前 Decode 主要由“全部本地 Expert 权重读取”决定，而不是 24 个 routed assignments 的 FLOPs。
3. 固定 payload 时，EP 通信归因从 EP2 的 1.308 ms 下降到 EP128 的 1.222 ms；这不是共享入口 FullMesh 应有的趋势，而是 `payload/EP` 公式的直接结果。

### 4.3 Prefill 强扩展

| EP/W | Rank 参数 GiB | TTFT ms | 加速比 | 并行效率 | Dispatch+Combine 算子 ms | 通信汇总归因 ms | 单卡 Prefill TPS | OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 837.489 | 18646.74 | 1.000 | 100.0% | 0.000 | 0.000 | 3514.61 | 是 |
| 2 | 432.173 | 10047.44 | 1.856 | 92.8% | 720.805 | 720.805 | 3261.33 | 是 |
| 4 | 229.515 | 4873.40 | 3.826 | 95.7% | 206.816 | 206.816 | 3361.92 | 是 |
| 8 | 128.186 | 2396.27 | 7.782 | 97.3% | 59.706 | 0.000 | 3418.65 | 是 |
| 16 | 77.521 | 1201.89 | 15.515 | 97.0% | 30.341 | 0.000 | 3407.97 | 是 |
| 32 | 52.189 | 604.70 | 30.836 | 96.4% | 15.659 | 0.000 | 3386.80 | 否 |

EP8 以后“通信汇总归因为 0”不是通信被重叠隐藏。逐算子中 Dispatch/Combine 仍执行，只是通信算子自身采用：

```text
op_time = max(local_HBM_time, network_time) + launch
```

EP8 时：

```text
Dispatch：HBM 350.15 us > Network 282.87 us
Combine ：HBM 612.65 us > Network 563.74 us
```

汇总逻辑将整个算子耗时记入 memory，因此 communication_ms 变成 0。

---

## 5. 建模观测点与判定

### 5.1 并行配置守恒

### 应观测

```text
W = PP × TP × CP × DP
stage_world = W / PP
stage_world = EDP × EP × ETP
```

### 实测

| 探针 | 结果 |
|---|---|
| `W8, EP8` | 成功，Run 22 |
| `W8, EP4` | 失败，`TASK.CONFIG.PARALLEL_INVALID`，Run 23 |
| `W8, EP2, ETP2, EDP2` | 数学乘积正确但失败，Run 46 |
| `W8, EP8` 并附带 `ETP2, EDP2` | 成功，但结果与 ETP1/EDP1 完全相同，Run 47 |

### 判定：P0，不合格

公共请求 Schema 没有 `edp_size` 和 `etp_size`；未知字段被 Pydantic 静默丢弃。后端解析器又硬编码：

```text
EDP=1
ETP=1
EP=stage_world
```

这与 `ARCHITECTURE.md` 中“保留 edp_size/etp_size 请求字段”和可配置 Expert mesh 的描述不一致。

建议观测项：

- API 回显 resolved EP/ETP/EDP；
- 非法乘积必须在 POST 校验阶段明确报错；
- 未支持字段必须 `extra=forbid`，不能静默忽略；
- 固定 W 的 EP↔EDP、EP↔ETP 交换实验必须可执行后，才能完整验证 EP 扩展。

### 5.2 routed-token 与 Expert 负载

### 当前公式

```text
routed_tokens_per_rank
  = ceil(ceil(B / EDP) × S × top_k / EP)
```

### 实测

Decode 弱扩展中：

```text
B = 4×EP, EDP=1, S=1, top_k=6
routed_tokens_per_rank = 24
```

所有 EP 下 GroupMatMul 输入行数保持 24，Compute 总量保持 0.461 ms。

### 判定：均匀路由公式合格，但缺少真实负载分布

当前只建模均匀分配或 Rank 0 最大切片，没有：

- 每 Expert token histogram；
- 最大 Rank/平均 Rank 比；
- 热门 Expert；
- capacity、drop、padding；
- EDP 副本选择；
- EPLB 动态迁移成本。

真实部署至少要采集：

```text
tokens_per_expert[p50/p90/p99/max]
tokens_per_rank[p50/p90/p99/max]
max_rank / mean_rank
padding_ratio
dropped_tokens
```

### 5.3 Expert 权重显存

### 实测公式

```text
rank_params ≈ 26.86 GiB + 810.63 GiB / EP
```

### 判定：合格

EP 权重分片准确反映了“每 Rank 持有更少 Expert”的容量收益。EP32 是本配置的首个非 OOM 点。

但 EDP 未实现，因此无法验证 Expert mesh 复制后：

```text
全网 Expert 存储量 ∝ EDP
单 Rank Expert 权重不因 EDP 再下降
```

### 5.4 GroupMatMul FLOPs 与 HBM 流量

### FLOPs 判定：合格

弱扩展 routed rows 恒定，Compute 恒定，说明没有再次错误乘 `expert_groups` 或 Top-K。

### HBM 判定：P1，高风险

`GroupMatMul.params` 的形状是：

```text
[num_experts / EP, D_out, D_in]
```

通用算子访存模型把整个 `param_bytes` 加入每次执行的 HBM bytes。这等价于假设一次 GroupMatMul 会读取该 Rank 的所有 Expert 权重，包括没有 Token 的 Expert。

Decode 弱扩展每 Rank 只有 24 个 assignment。均匀路由下，活跃 Expert 数的期望为：

```text
active ≈ local_experts × (1 - (1 - 1/local_experts)^24)
```

| EP | 本地 Expert | 期望活跃 Expert | 活跃比例 |
|---:|---:|---:|---:|
| 1 | 384 | 23.29 | 6.1% |
| 2 | 192 | 22.62 | 11.8% |
| 4 | 96 | 21.33 | 22.2% |
| 8 | 48 | 19.04 | 39.7% |
| 16 | 24 | 15.36 | 64.0% |
| 32 | 12 | 10.51 | 87.6% |
| 64 | 6 | 5.92 | 98.7% |
| 128 | 3 | 3.00 | 100.0% |

这会系统性高估小 EP、低 Batch Decode 的权重读取时间，并制造“EP 越大 GroupMatMul 越快”的强烈趋势。

需要增加或校准：

- `active_local_experts`；
- 只对活跃 Expert 计权重流量，或使用真机 grouped-GEMM latency table；
- 不同 tokens-per-expert 下的 Kernel launch、padding 和 tile 利用率；
- L2/HBM 权重缓存命中；
- routed rows 相同但 local experts 不同时的真机对照。

### 5.5 Dispatch/Combine payload

以 Decode 弱扩展为例，每 Rank Batch=4：

```text
Dispatch payload
  = 4 × 1 × top_k(6) × hidden(7168) × INT8(1 B)
  = 172,032 B

Combine payload
  = 4 × 1 × top_k(6) × hidden(7168) × FP16(2 B)
  = 344,064 B
```

服务输出与此完全一致。

报告字节也满足：

```text
EP8 Dispatch comm_bytes
  = 2 × 172,032 × 7/8
  = 301,056 B
```

### 判定：logical payload 与报告字节合格

应继续观测：

- Dispatch/Combine 是否含 scale、expert index、routing weight 和 alignment metadata；
- DeepEP low-latency/normal 模式的实际 wire bytes；
- Top-K 与 shared Expert 是否重复发送；
- TP>1 时是否发生重复 Dispatch；
- ETP classic 与 ETP-SP 的 payload 是否互斥。

### 5.6 FullMesh A2A 时间

### 当前公式

当前 `fullmesh` 实现为：

```text
T = alpha + (payload / EP) / BW
```

它选择最慢物理层，但只计算一个 Rank 到一个目标 Rank 的 pair payload。

### 弱扩展实测

Decode 弱扩展的 per-rank payload 固定不变：

| EP | Dispatch payload | 报告跨 Rank 字节 | 当前计费 traffic | 当前网络时间 |
|---:|---:|---:|---:|---:|
| 8 | 172,032 B | 301,056 B | 21,504 B | 2.137 us |
| 32 | 172,032 B | 333,312 B | 5,376 B | 2.034 us |
| 128 | 172,032 B | 341,376 B | 1,344 B | 2.009 us |

EP 越大：

- 实际每 Rank 远端总发送量趋近 payload；
- 报告 send+recv 字节趋近 `2×payload`；
- 当前计费 traffic 却按 `payload/EP` 趋近 0；
- 网络时间趋近纯 alpha。

### 判定：P1，对共享入口不合格

如果 GPU/NIC 使用共享注入与接收入口，更合理的下界应至少考虑：

```text
T_A2A ≥ alpha + max(
  rank_remote_send_bytes / injection_BW,
  rank_remote_recv_bytes / reception_BW,
  bottleneck_link_bytes / link_BW
)
```

在均匀 A2A、单向入口带宽口径下：

```text
rank_remote_send_bytes = payload × (EP-1)/EP
```

它相对当前 `payload/EP` 的倍率为 `EP-1`。如果硬件 bandwidth 是收发共享的双向总带宽，还要联合 send+recv 计价。

### 共享入口敏感性估算

下面只把 FullMesh 带宽项替换为 per-rank aggregate send，下游 HBM/Network 仍取 max；它不是硬件校准结果，只用于评估当前公式的偏差方向。

| Prefill EP | 当前 TTFT | 共享入口敏感性 TTFT | 增幅 |
|---:|---:|---:|---:|
| 4 | 4873.40 ms | 5284.59 ms | 8.4% |
| 8 | 2396.27 ms | 2697.58 ms | 12.6% |
| 16 | 1201.89 ms | 1365.52 ms | 13.6% |
| 32 | 604.70 ms | 689.85 ms | 14.1% |

Decode 小 payload 下变化只有约 0.1%~0.6%，因为 8 us 算子启动开销和 Expert 权重访存更大；Prefill 大 payload 对公式更敏感。

### 5.7 通信计算重叠

### 实测

- 所有任务的 `explainability_report.overlap` 均为 `null`；
- 推理请求 Schema 没有 TBO/SBO 或 EP chunk/wave 参数；
- 推理实现中没有 TBO/SBO 的 Dispatch/Expert/Combine 分块时间线；
- 当前只在单个通信算子内部计算 `max(network, local_HBM)`；这不是 Expert GEMM 与 A2A 的跨算子 overlap。

### 判定：P1，未建模

真实 SGLang + DeepEP 若启用 TBO/SBO，典型时间线是：

```text
Dispatch(chunk n+1) || Expert GEMM(chunk n) || Combine(chunk n-1)
```

当前服务则按图依赖串行累计：

```text
Dispatch → Expert GEMM → Combine
```

部署实测必须采集：

```text
raw_dispatch_us
raw_combine_us
raw_expert_us
hidden_dispatch_us
hidden_combine_us
exposed_ep_comm_us
chunk_count
pipeline_warmup_us
pipeline_drain_us
network/compute concurrent utilization
```

在实现 TBO/SBO 前，Prefill/Decode 的通信暴露时间不能从当前结果中可靠推导。

### 5.8 瓶颈归因与可观测性

当前通信算子无论 HBM 还是网络更慢，公开 `bound_type` 都写成 `communication`；Latency Breakdown 又重新比较 compute/memory/communication 后进行独占归因。

EP8 Prefill 的同一个 Dispatch：

```text
明细 bound_type：communication
HBM：350.15 us
Network：282.87 us
汇总归因：memory
```

### 判定：P2，观测口径不一致

建议同时输出：

```text
operator_kind=communication
dominant_bound=memory|communication
raw_network_us
raw_local_memory_us
effective_operator_us
```

不要用一个 `bound_type` 同时表达“算子类型”和“实际瓶颈”。

---

## 6. EP 扩展的收益与代价应该怎样观测

### 6.1 容量收益

| 观测点 | 期望趋势 | 当前状态 |
|---|---|---|
| routed Expert resident bytes/Rank | `1/(EP×ETP)` | 已覆盖 EP，未覆盖可配置 ETP |
| 全网 Expert resident bytes | 随 EDP 复制 | EDP 固定 1，无法验证 |
| KV/activation/comm buffer | 不一定随 EP 同比例下降 | 有总量，但需单列 DeepEP buffer |
| 首个非 OOM EP | 找到最小可部署 EP | 本配置为 EP32 |

### 6.2 计算收益

| 观测点 | 期望 |
|---|---|
| routed assignments/Rank | 均匀假设下约 `global_assignments/(EP×EDP)` |
| Expert FLOPs/Rank | 与 routed assignments 成正比，再除 ETP |
| active Experts/Rank | 由真实路由决定，不等于 local experts |
| tokens/active Expert | 决定 Grouped GEMM 尺寸与效率 |
| Expert Kernel 利用率 | EP 过大时可能因小 GEMM 下降 |

### 6.3 通信代价

| 观测点 | 说明 |
|---|---|
| logical payload | Tensor 语义字节 |
| remote send/recv bytes | 扣除本地 Expert 后的实际远端流量 |
| wire bytes | 包含 metadata、padding、对齐与协议开销 |
| group topology | Rank 到 NIC、节点、交换机的实际映射 |
| injection/reception BW | 每 GPU 或每 NIC 的共享入口上限 |
| fabric bottleneck bytes | 双分带宽、超售与热点链路 |
| alpha/rounds | 实际算法的轮数与启动延迟 |
| raw/hidden/exposed | TBO/SBO 前后的通信时间 |

### 6.4 负载均衡代价

| 观测点 | 说明 |
|---|---|
| expert token skew | 最热 Expert/平均 Expert |
| rank token skew | 最慢 Rank/平均 Rank |
| straggler time | A2A 和 Expert GEMM 的尾部 Rank |
| EDP replica choice | 本地优先、最短队列或全局均衡 |
| rebalancing cost | EPLB 迁移权重与路由表更新 |

### 6.5 业务指标

应同时报告：

```text
TTFT / TPOT p50、p95、p99
实例总 TPS
单卡 TPS
每请求占用卡数
卡时成本：GPU-seconds/request 或 GPU-seconds/1K tokens
OOM margin
Prefill/Decode 分阶段指标
```

只看 TPOT 会把“加了更多卡”误判为无成本的性能收益；只看单卡 TPS又会忽略容量约束。

---

## 7. 当前结果可用于哪些决策

### 可以使用

1. 判断 full-stage EP 下 Expert 权重是否能装入单卡。
2. 检查 routed-token、payload 和 A2A 报告字节的逻辑公式。
3. 做理想均衡、无 TBO/SBO 的趋势参考。
4. 找到需要重点真机采集的 Shape：
   - Decode：24 routed rows、3~384 local experts；
   - Prefill：大 payload A2A；
   - EP16→32 的 OOM 临界点。

### 暂时不能使用

1. 固定 W 下比较 EP 与 EDP。
2. 比较 ETP classic 与 ETP-SP。
3. 预测共享 NIC/交换网络上的 A2A 绝对时间。
4. 预测低 Batch Decode 的真实 Grouped GEMM 权重流量。
5. 预测启用 SGLang TBO/SBO 后的暴露通信。
6. 根据当前曲线直接确定生产最优 EP。

---

## 8. 建议的修复与验证顺序

本次仅排查，没有修改代码。若进入修复，建议按以下顺序进行。

### 第一步：修复公共并行语义

- 在请求 Schema 中显式加入 `edp_size/etp_size`，或在尚未支持时明确拒绝；
- 统一 `EP×ETP×EDP=stage_world` 的 resolver；
- 让响应回显 resolved 三元组；
- 增加固定 W 的合法乘积测试。

没有这一步，无法构造纯 EP 对照实验。

### 第二步：明确硬件带宽语义并修正 FullMesh

- 区分 `per_pair_link_bw` 与 `per_rank_injection_bw`；
- 同时约束 source、destination 和 fabric；
- 为共享 NIC、NVSwitch 和真正独立 direct-link 分别建模；
- 用真机 A2A sweep 标定 alpha、有效 BW 和并发退化。

### 第三步：补 active-expert GroupMatMul

- 运行期输入加入 active expert 数或 token histogram；
- 权重 traffic 只计算活跃 Expert；
- 用 `[routed rows, active experts, hidden, ffn, dtype]` 作为 Kernel profile key；
- 保持 resident weight 仍按全部本地 Expert 计算。

### 第四步：加入 TBO/SBO 时间线

- 显式记录 chunk/wave；
- 分离 raw、hidden、exposed；
- 建模首尾 drain 和共享 NIC 竞争；
- Prefill 与 Decode 分别标定。

### 第五步：统一可观测字段

- 分离 operator category 与 dominant bound；
- 报告 raw network、local HBM、effective op time；
- communication summary 不能用 0 表示“被 memory bound 覆盖”。

---

## 9. 验证记录

### 服务任务

```text
强/弱扩展与 Prefill 批量任务：Run 24~45，共 22 个，全部 succeeded
固定 W 非法 EP 探针：Run 23，failed（预期复现当前硬约束）
EP2×ETP2×EDP2 探针：Run 46，failed
EDP/ETP 静默忽略探针：Run 47，succeeded，结果仍为 EDP1/ETP1
```

### 定向测试

```text
68 passed, 1 deselected
```

覆盖：

- collective payload；
- communication domain；
- process group；
- SGLang layout；
- inference parallel API compatibility；
- DeepSeek V4 communication graph 中除 LM-head 特殊 TP 外的测试。

完整集合曾运行 69 项，其中 1 项失败：

```text
test_v4_special_tp_public_fields_activate_static_collectives
LMHEAD_TP expected comm_bytes=49,643,520
actual comm_bytes=387,840
```

该失败属于 LM-head TP payload 的旧期望或行为漂移，与本次 EP 路径无关，未修改。

---

## 10. 代码锚点

- 推理公共并行 Schema：`backend/web/services/infer/schemas/request.py:125`
- EP/EDP/ETP 固定解析：`backend/inference/kepler/engine/topology/expert_parallel.py:1`
- routed-token 公式：`backend/inference/kepler/engine/policies/op_input.py:150`
- GroupMatMul 参数 shape：`backend/inference/kepler/engine/layers/linear.py:17`
- 通用参数 HBM traffic：`backend/inference/kepler/engine/layers/op_base.py:241`
- A2A report bytes：`backend/inference/kepler/engine/layers/communication.py:9`
- FullMesh `payload/EP`：`backend/inference/kepler/engine/policies/comm.py:346`
- 通信算子 `max(network, HBM)`：`backend/inference/kepler/engine/layers/op_base.py:630`
- Latency Breakdown 重新归因：`backend/web/services/infer/kernel/response.py:463`
- 架构声明的 Expert mesh：`ARCHITECTURE.md:473`

完整机器可读结果见同目录 `ep-scaling-results.json`。
