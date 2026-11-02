# NPU 场景下的 Inductor 融合建模：工作纲领

> 来源：ChatGPT 会话《NPU场景下的inductor建模》的首轮回答。本文经过结构化整理，并移除了仅在原会话中有效的内部引用标记。

## 一、项目目标

不要只把目标限定为“寻找一个比 64 更好的固定值”，而应提升为：

> 建立 NPU-aware 的 Inductor Fusion Cost Model，使 Inductor 能根据算子图、Tensor Shape、访存特征和 NPU Kernel 特征，动态判断“该不该融合、先融合谁、融合到多大”。

`max_fusion_size = 64` 很适合作为第一个切入点和消融实验，但不应是最终建模目标。

## 二、先理解两个不同的 64

```python
max_fusion_size = 64
max_fusion_buffer_group_pairwise_attempts = 64
```

- `max_fusion_size`：限制一个标准 fused scheduler group 最多包含多少个底层 SchedulerNode。两个候选融合组的节点数之和超过该值时，拒绝继续融合。这是性能启发式，不是正确性硬约束。
- `max_fusion_buffer_group_pairwise_attempts`：限制在一个 buffer group 中，每个节点最多向后尝试多少个节点作为 fusion candidate，控制候选搜索开销。

两者影响不同，实验时必须解耦，否则无法判断收益来自候选搜索空间扩大，还是 fused group 大小变化。

Inductor 的融合也不是一次找到一个 64 节点子图，而是不断执行：

```text
Graph State
    ↓
Candidate Pair
    ↓
CanFuse
    ↓
Fusion Score
    ↓
Fuse / Reject
    ↓
New Graph State
```

所以真正的优化问题不是简单的 `64 → 32/128`，而是动态融合决策和融合顺序优化。

## 三、七阶段实施路线

### 阶段 1：固定阈值基线实验

先不使用机器学习，单独扫描：

```text
max_fusion_size = 8/16/24/32/48/64/80/96/128
```

同时固定：

```text
max_fusion_buffer_group_pairwise_attempts = 64
```

Workload 至少覆盖：

- LLM Prefill；
- LLM Decode；
- 不同 batch size、sequence length、hidden size；
- 若业务需要，再补充 CV 和训练场景。

每次实验记录：

- End-to-End latency / throughput；
- kernel latency 总和；
- kernel 数量；
- 平均、最大 fused node 数；
- 编译时间；
- 峰值内存；
- 编译或运行失败率；
- 正确性结果。

然后反向固定 `max_fusion_size = 64`，单独扫描 `max_fusion_buffer_group_pairwise_attempts`，最后再做小规模二维联合实验。

### 阶段 2：证明最佳 fusion size 不是常数

这是引入动态模型的必要性证明。

如果所有模型、Shape 和 Kernel 类型都在同一个固定值上最好，那么修改固定启发式即可，没有必要引入模型。更可能的情况是：

- 长 Pointwise chain 适合更大融合；
- Reduction、复杂 Indexing、多输入输出或高资源压力 Kernel 更早出现收益反转；
- 不同 Shape、Prefill/Decode、不同 NPU SoC 的最优值不同。

需要形成清晰的 baseline，例如：

```text
Model A / S=128   → 32 最优
Model A / S=4096  → 64 最优
Model B / Decode  → 16 最优
Model B / Prefill → 96 最优
```

这类结果才能有力说明固定 `max_fusion_size = 64` 无法表达 NPU 的最佳融合策略。

### 阶段 3：把 Scheduler 改造成 Fusion 数据采集器

不要只保存最终 fused graph，要采集每一步看到的 candidate pair。

建议每条样本至少记录：

```text
graph_id
fusion_round
node1 / node2
node1_size / node2_size / fused_size
fusion_type
current_score
shared_data
graph_distance
inputs / outputs
shape / dtype / stride
reduction / template 标记
ancestors / users
```

同时记录标签所需的真实性能：

```text
T_node1
T_node2
T_fused
```

自然的采集入口包括 Scheduler 的候选生成、`fuse_nodes_once`、`speedup_by_fusion` 以及融合前后 custom pass hook。

### 阶段 4：以 benchmark fusion 作为离线 Oracle

分别编译和测量两个独立 Kernel 与融合后的 Kernel：

```text
T_separate = T_node1 + T_node2
T_fused    = fused kernel latency
```

如果：

```text
T_fused < T_separate
```

则该候选在局部测量上值得融合。

这一机制可用于构造监督学习标签，但更适合离线采集和 Oracle 对照，不适合默认在线执行，因为会显著增加编译时间。

### 阶段 5：先训练简单、可解释的 Cost Model

第一版优先使用 GBDT、XGBoost 或 LightGBM 一类表格模型，不要一开始就上 GNN 或强化学习。

建议回归目标：

\[
y = \log \frac{T_1 + T_2}{T_{\text{fused}}}
\]

- `y > 0`：融合有收益；
- `y = 0`：基本持平；
- `y < 0`：融合变慢；
- `y` 越大，预测收益越高。

该模型可同时服务两个逻辑：

```text
should_fuse(node1, node2)      → true / false
predicted_gain(node1, node2)   → ranking score
```

即分别增强 Scheduler 中的融合准入和候选排序。

### 阶段 6：把模型接回 Inductor

集成原则：

1. 保留所有 correctness / legality 判断；
2. 将 `max_fusion_size` 提高到 128 或 256，作为 safety ceiling；
3. 由 NPU Cost Model 判断哪些融合真正值得做；
4. 用预测收益替代或增强 `score_fusion()`；
5. 尽量通过 `InductorChoices` 一类性能启发式扩展点注入，减少与上游 Scheduler 的耦合。

最终从固定阈值：

```text
NPU → max_fusion_size = 48
```

演进为动态决策：

```text
(node1, node2, graph, shape, NPU)
                ↓
          predict Δlatency
                ↓
       fusion gain = +7.2%
                ↓
              Fuse
```

### 阶段 7：回到整图评价

一个 pair 满足 `T_fused < T1 + T2`，不代表全图一定最优。一次融合会改变：

- 后续 candidate 集合；
- 融合顺序；
- 内存生命周期；
- 峰值内存；
- Codegen 结果；
- 后续 Kernel 的形态与资源占用。

最终至少比较四组策略：

1. PyTorch 默认固定 64；
2. NPU 调优后的固定 N；
3. `benchmark_fusion` Oracle；
4. Learned Cost Model。

首要 KPI 是端到端 latency / throughput，其次才是单候选预测准确率、AUC 或回归误差。

## 四、第一版特征设计

| 特征组 | 建议特征 |
|---|---|
| Graph | node1/node2 节点数、融合后节点数、graph distance、producer-consumer / horizontal、fan-in/out |
| Operator | Pointwise/Reduction/Template、算子类别、op count、reduction axis |
| Tensor | shape、numel、dtype、stride、broadcast 情况 |
| Memory | 输入 bytes、输出 bytes、共享 bytes、预计消除的 intermediate bytes |
| Fusion | 当前 `score_fusion_memory`、shared data score、buffer overlap |
| IO | unique input/output 数、融合后的总参数数量 |
| Kernel | Kernel 1/2 的估计工作量、融合后的估计工作量 |
| NPU | SoC 型号、Codegen Backend、编译器或 Profiler 可获得的资源占用与 Tiling 特征 |
| Context | fusion round、当前 fused group size、前后节点数量 |

关键原则：把 `fusion_size` 作为特征，而不是最终规则。

## 五、首轮实验数据格式

```python
import torch

torch._inductor.config.max_fusion_size = 32
```

建议结果表：

| Model | Shape | Fusion Size | Kernels | E2E ms | Compile s | Peak Mem | 最大融合节点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Case A | S=128 | 16 | ... | ... | ... | ... | ... |
| Case A | S=128 | 32 | ... | ... | ... | ... | ... |
| Case A | S=128 | 64 | ... | ... | ... | ... | ... |
| Case A | S=128 | 128 | ... | ... | ... | ... | ... |
| Case A | S=4096 | 16 | ... | ... | ... | ... | ... |

实验必须固定：

- 代码版本和编译参数；
- NPU 型号、驱动、固件、CANN/编译器版本；
- 输入数据和 Shape；
- Warmup 与测量轮数；
- 同步方式；
- 测量统计口径；
- 其他 Fusion 配置。

## 六、不要只研究“融合多少”，还要研究“先融合谁”

如果下列候选都合法：

```text
A + B
A + C
B + C
```

先选择哪一对会改变后续图结构。因此最终模型应同时解决：

```text
FusionCostModel
      ├── should_fuse(node1, node2)
      └── predicted_gain(node1, node2)
```

候选准入和候选排序都比单独寻找 `max_fusion_size = ?` 更有长期价值。

## 七、目标系统架构

```text
PyTorch FX / Inductor IR
          │
          ▼
   Scheduler Nodes
          │
          ▼
 Candidate Generator
          │
    ┌─────┴──────────┐
    ▼                ▼
Legality Check   NPU Cost Model
                       │
              ┌────────┴─────────┐
              ▼                  ▼
        Should Fuse?       Predicted Gain
              │                  │
              └────────┬─────────┘
                       ▼
                Fusion Ranking
                       │
                       ▼
                  Fuse Nodes
                       │
                       ▼
                  NPU Codegen
                       │
                       ▼
                  NPU Profiler
                       │
                       ▼
                Training Dataset
```

## 八、第一个工程交付物

第一步不直接训练模型，而是完成一个：

> Inductor Fusion Modeling Dataset Generator

它应在 `fuse_nodes_once`、候选生成与 `speedup_by_fusion` 等位置，持续输出每个 NPU fusion candidate 的：

- Graph 特征；
- IR 特征；
- Dependency 和 Memory 特征；
- Scheduler 特征；
- 融合前 latency；
- 融合后 latency；
- 编译、运行和正确性状态。

数据采集打通后，再依次完成：

```text
固定 64 的消融实验
    ↓
固定阈值调优
    ↓
GBDT Cost Model
    ↓
接入 should_fuse / score_fusion
    ↓
整图端到端验证
```

这条路线的每个阶段都有独立、可验证的工程产物。

## 九、第一阶段测试集的场景覆盖

第一阶段测试集不以“模型数量最多”为目标，而以覆盖会改变 Inductor Fusion 决策的典型计算图结构为目标。当前测试分为 Core、Extended 和 Stress 三组。

### 9.1 Core：第一轮主测试集

| Case | 粒度 | 主要覆盖点 | 选择原因 |
|---|---|---|---|
| `pointwise_fanout_24` | 合成片段 | 24 分支 Horizontal Pointwise Fusion | 检查多个节点共享同一输入时，是否适合横向融合，以及多输出 Kernel 的收益与资源压力 |
| `norm_pointwise_chain` | Transformer 片段 | Reduction + Pointwise | 检查归约域与逐元素域并存时的融合兼容性，以及大融合是否引起调度和资源问题 |
| `broadcast_residual` | Transformer 片段 | Broadcast、门控激活、Residual、RMSNorm | 覆盖 Transformer MLP Epilogue 和归一化附近常见的广播、旁路依赖及归约 |
| `qwen_prefill_block` | 完整 Block | RMSNorm、RoPE、Causal Attention、SwiGLU、Residual | 验证合成片段上的局部收益能否转化为长序列 Prefill 的端到端收益 |
| `qwen_decode_fragment` | Decode 片段 | 单 Token、KV Cache、Attention Epilogue | 覆盖 Kernel Launch 和 KV Cache 访存占比较高的低延迟 Decode 场景 |

### 9.2 Extended：扩大模型与算子结构覆盖

| Case | 粒度 | 主要覆盖点 |
|---|---|---|
| `index_softmax` | Attention 片段 | Gather、Mask、Softmax Reduction、Weighted Sum |
| `bert_encoder_2l` | 小型整模 | 双向 Attention、LayerNorm、GELU、Residual |
| `resnet_tiny` | 小型整模 | Conv-BN-ReLU、Residual、Global Reduction |
| `dlrm_tiny` | 小型整模 | Embedding、Feature Interaction、MLP Epilogue |

执行 Extended 时，应同时保留 Core，以便兼顾可解释片段和小型整模。

### 9.3 Stress：确认是否真正触及 64

| Case | 粒度 | 主要覆盖点 |
|---|---|---|
| `pointwise_fanout_80` | 合成压力片段 | 80 个共享输入、多输出分支，专门刺激超过 64 个底层 SchedulerNode 的融合分组 |

Stress Case 只用于验证阈值是否生效、观察超过 64 后的图结构和性能变化，不能单独代表真实模型结论。

### 9.4 按 Fusion 场景整理

| Fusion 场景 | 当前 Case |
|---|---|
| Vertical Pointwise Fusion | `broadcast_residual` |
| Horizontal Pointwise Fusion | `pointwise_fanout_24`、`pointwise_fanout_80` |
| Reduction + Pointwise | `norm_pointwise_chain`、`broadcast_residual` |
| Broadcast Fusion | `broadcast_residual` |
| Residual Epilogue | Qwen、BERT、ResNet Case |
| Indexing + Reduction | `index_softmax` |
| Attention Block | Qwen、BERT Case |
| LLM Prefill | `qwen_prefill_block` |
| LLM Decode / KV Cache | `qwen_decode_fragment` |
| Conv Epilogue | `resnet_tiny` |
| Embedding / Feature Interaction | `dlrm_tiny` |

### 9.5 当前尚未重点覆盖的场景

以下场景应在第一轮静态 Shape、单卡推理消融完成后，分别作为独立实验加入，避免一次改变过多变量：

| 未覆盖场景 | 为什么需要单独测试 |
|---|---|
| Backward 与训练 | 会引入反向图、保存中间激活、重计算和不同的内存生命周期 |
| Dynamic Shape | 会引入 Guard、Recompile、符号 Shape 和动态 Codegen，不能与静态 Shape 结果直接混合 |
| MoE | 包含路由、Top-K、Scatter/Gather、负载不均衡和多个 Expert，图结构与普通 Transformer 不同 |
| Scatter / Atomic | 存在写冲突和原子操作，合法性与性能判断不同于普通 Pointwise/Reduction |
| 大规模 Concatenation | 多输入和 Masked Load 可能导致参数量、Indexing 和 Kernel 代码显著膨胀 |
| 多 Stream | 需要考虑跨 Stream 依赖、同步以及 Fusion 对并发性的影响 |
| Collective Communication | AllReduce/AllGather 等通信与计算重叠不能仅用单 Kernel 局部收益评价 |
| 大型 Template Kernel Fusion | GEMM/Attention Template 的 Epilogue Fusion 受后端模板、Autotune 和资源占用约束 |

### 9.6 结果解释原则

只有当不同 `max_fusion_size` 确实改变了 Fused Group、生成 Kernel 或运行时 Kernel Launch，性能比较才具有明确的 Fusion 解释。

如果一个 Case 在 32、64、128 下生成完全相同的 Scheduler Graph 和 Kernel：

> 只能说明该 Case 对 `max_fusion_size` 不敏感，不能说明 64 是最优值。

因此每个性能结果都应同时保留 Fusion 前后 IR、Fused Group Size、Kernel 数量以及必要的 Profiler 证据。
