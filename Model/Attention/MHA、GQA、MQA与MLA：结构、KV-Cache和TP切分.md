# MHA、GQA、MQA 与 MLA：结构、KV Cache 和 TP 切分

## 问题：这四个缩写在改变什么

MHA、GQA 和 MQA 主要改变 Query head 与 KV head 的共享关系；MLA 则进一步改变 KV 的表示方法，
把每个 token 的 KV 压缩成低维 latent。它们都会影响模型结构、KV Cache、Decode 带宽和 Tensor
Parallelism（TP）切分，但不能简单放在同一条“head 越少越省”的轴上比较。

先记住一个判断依据：

```text
MHA / GQA / MQA：关注多少个 Query head 共享一套 K/V
MLA            ：关注 K/V 能否用低维 latent 表示和缓存
```

MQA 可以看成 GQA 的极端情况。MLA 不是“KV head 更少的 GQA”，它引入了低秩投影、拆分后的 RoPE
分量，以及不同的 Decode 计算路径。

## 共同基础：Attention 在计算什么

输入 hidden state 记为 `X[B,S,H]`：B 是 batch，S 是 sequence length，H 是 hidden size。标准
Attention 先产生 Query、Key 和 Value：

```text
Q = X · Wq
K = X · Wk
V = X · Wv

Attention(Q,K,V) = softmax(Q · Kᵀ / √D) · V
```

Query 表示“当前位置要寻找什么”，Key 表示“历史位置提供什么索引”，Value 表示“命中后读取什么
内容”。D 是每个 head 的维度。Multi-head 表示把这些向量拆成多个 head，让不同 head 学习不同关系。

自回归 Decode 每次只生成少量新 token，但每层都要读取所有历史 token 的 K/V。模型会把历史 K/V
保存在 KV Cache 中，避免每一步重新计算。随着上下文增长，KV Cache 容量和显存带宽往往成为 Decode
的主要瓶颈，这正是 GQA、MQA 和 MLA 要解决的问题。

## MHA：每个 Query head 都有自己的 K/V

MHA（Multi-Head Attention）让 Query、Key 和 Value 拥有相同数量的 head。若 head 数为 `N`，则
第 i 个 Query head 只和第 i 个 Key/Value head 配对：

```text
Q0 ↔ K0/V0
Q1 ↔ K1/V1
Q2 ↔ K2/V2
Q3 ↔ K3/V3
```

结构可以写成：

```text
X
├── Wq → [B,S,N,D] ─┐
├── Wk → [B,S,N,D] ─┼→ 每个 head 独立做 Attention
└── Wv → [B,S,N,D] ─┘
                         ↓
                    Concat → Wo → [B,S,H]
```

MHA 的结构直接，每个 Query head 都有独立 K/V 表达能力。代价是每个 token、每层需要缓存
`2×N×D` 个值，长上下文 Decode 的 KV Cache 和读取带宽都比较大。

## MQA：所有 Query head 共享一套 K/V

MQA（Multi-Query Attention）保留多个 Query head，但全体 Query head 只共享一个 Key head 和一个
Value head：

```text
Q0 ─┐
Q1 ─┤
Q2 ─┼→ K0/V0
Q3 ─┘
```

它的 projection shape 是：

```text
Q: [B,S,Nq,D]
K: [B,S,1,D]
V: [B,S,1,D]
```

执行 Attention 时，逻辑上让这一套 K/V 服务所有 Query head。Kernel 不一定真的复制 K/V；高效实现
会通过 stride 或专用 Kernel 直接共享。KV Cache 每个 token、每层只需保存 `2×D` 个值，Decode
带宽显著下降。

MQA 的取舍是共享程度很高。它是否影响模型效果取决于模型规模、训练方法和任务，不能仅凭结构断言
效果一定下降或不下降。

## GQA：一组 Query head 共享一套 K/V

GQA（Grouped-Query Attention）位于 MHA 和 MQA 之间。它把 Query head 分组，每组共享一个
Key/Value head。

假设有 8 个 Query head 和 4 个 KV head：

```text
Q0、Q1 → K0/V0
Q2、Q3 → K1/V1
Q4、Q5 → K2/V2
Q6、Q7 → K3/V3
```

若 Query head 数为 `Nq`，KV head 数为 `Nkv`，每个 KV head 服务的 Query head 数是：

```text
group_size = Nq / Nkv
```

所以三种结构可以用 KV head 数统一描述：

| 结构 | Query head 数 | KV head 数 | 共享关系 |
|---|---:|---:|---|
| MHA | `Nq` | `Nq` | 每个 Query 有独立 K/V |
| GQA | `Nq` | `1 < Nkv < Nq` | 每组 Query 共享 K/V |
| MQA | `Nq` | `1` | 全部 Query 共享 K/V |

GQA 的 KV Cache 每个 token、每层保存 `2×Nkv×D` 个值。它保留多套 KV 表达，同时降低 KV Cache
和 Decode 带宽，因此是效果、容量和吞吐之间的一种折中。

## MLA：缓存低维 latent，而不是完整多头 K/V

MLA（Multi-head Latent Attention）通过低秩投影压缩 K/V。这里的 latent 是比完整多头 K/V 更低维
的中间表示。不同模型实现会略有差异，下面描述的是 DeepSeek 系列常见结构，而不是所有名为 MLA
的实现都必须逐字段相同。

KV 路径先把 hidden state 下投影成低维 `c_kv`，再通过上投影恢复各 Attention head 使用的 K/V：

```text
X → KV down projection → c_kv[Rkv]
                           ├→ KV up projection → K_nope[Nh,D_nope]
                           └→ KV up projection → V[Nh,Dv]
```

RoPE（Rotary Position Embedding）是给 Q/K 注入位置信息的方法。MLA 通常把带 RoPE 的部分拆出来：

```text
Q = concat(Q_nope, Q_rope)
K = concat(K_nope, K_rope)
```

其中 `K_rope` 不能被普通低秩吸收完全替代，所以 Decode 时通常缓存：

```text
c_kv       低维 KV latent
K_rope     带位置信息的 Key 分量
```

而不是缓存每个 head 完整的 K 和 V：

```text
MHA/GQA/MQA Cache：完整 K + 完整 V
MLA Cache        ：c_kv + K_rope
```

Query 路径也可能使用低秩投影：

```text
X → Q down projection → c_q → Q up projection → Q_nope + Q_rope
```

是否存在 Q down projection，由模型配置中的 `q_lora_rank` 等字段决定，不能从“这是 MLA”直接推断。

### MLA Decode 为什么可以不恢复完整 K/V

朴素实现可以从 `c_kv` 上投影出完整 K/V，再执行 Attention。但 Decode 中这样做会增加中间激活和
带宽。Matrix absorption 会重新组合矩阵乘法：把 K 的上投影吸收到 Query 路径，把 V 的上投影吸收
到输出路径，从而直接让 Query 与缓存的 latent 交互。

```text
朴素路径：c_kv → 完整 K/V → Attention → output
吸收路径：Q 变换后直接读取 c_kv → latent Attention → 吸收后的 output projection
```

Matrix absorption 不改变模型的数学目标，但会改变实际算子、shape、权重布局和性能。因此建模时不能
看到逻辑 K/V shape，就假设 Decode 一定物化了完整 K/V。

## KV Cache 的数量级对比

下面用 BF16、`Nq=32`、`D=128` 举例。GQA 使用 8 个 KV head；MLA 假设
`Rkv=512、Drope=64`。这里只比较未量化的逻辑数据，不包含 page padding、scale、allocator 元数据和
后端特殊布局。

| 结构 | 每 token、每层缓存值 | BF16 字节数 |
|---|---:|---:|
| MHA | `2×32×128` | 16,384 B |
| GQA | `2×8×128` | 4,096 B |
| MQA | `2×1×128` | 512 B |
| MLA | `512+64` | 1,152 B |

这个例子不能用来断言 MLA 一定比 MQA 更省，因为实际维度由模型决定，量化 MLA 还可能带 block
scale，某些 backend 会使用 raw layout。正确做法是读取目标模型配置和运行时 KV pool manifest，
逐分量计算。

## 为什么 GQA 下的 TP 不能统一“除以 TP”

TP 把一个算子的权重或激活分给多个 rank。对 MHA，若 head 数可以整除 TP，Q/K/V 通常都能按 head
均分。例如 32 个 head、TP=4 时，每 rank 得到 8 个 Q head、8 个 K head 和 8 个 V head。

GQA 的 Query head 和 KV head 数不同。假设：

```text
Nq  = 32
Nkv = 8
TP  = 4
```

每 rank 可以得到 8 个 Query head 和 2 个 KV head，本地仍保持“4 个 Query 共享一套 K/V”。但当
TP=16 时：

```text
Nq / TP  = 2
Nkv / TP = 0.5
```

KV head 不能切成半个。框架可能把每个 KV head 复制到两个 TP rank，也可能拒绝该配置，具体选择由
目标框架的 QKV parallel Linear 实现决定。MQA 的唯一 K/V head 在 TP 下通常也需要复制，而不是
继续除以 TP。

因此 QKV 融合权重也不能只检查总输出维度。若 `Nq=32、Nkv=8、D=128`：

```text
Q 输出 = 32 × 128 = 4096
K 输出 =  8 × 128 = 1024
V 输出 =  8 × 128 = 1024
融合输出总维度       = 6144
```

直接计算 `6144/TP` 可能得到一个整数，却掩盖内部 KV head 被错误分片。可信断言应分别检查 Q、K、V
三个区段，并验证本地 Query head 能访问与它对应的本地或复制 KV head。

## MLA 下的 TP 又有什么不同

MLA 通常仍按 Query head 切分 Attention 计算，但 KV cache 的核心是低维 latent，不是
`num_key_value_heads` 套完整 K/V。需要分别判断：

- Q projection 的 head 维是 Column Parallel 还是其他切分；
- KV down projection 的 latent 是复制、按维切分，还是在 DCP 下按 token owner 切分；
- KV up projection 是否在当前阶段真实执行，还是已被 Matrix absorption 消除；
- RoPE 分量、latent、量化 scale 是否采用不同的缓存 dtype 和布局；
- 输出 projection 在吸收前后分别承担什么 Row Parallel 规约。

因此不能把 MLA 当成 `num_key_value_heads=1` 的 MQA。某些源码把融合的 KV down projection 命名为
`kv_a_proj_with_mqa`，其中的名字也不能替代结构检查。

## 建模和评审时怎样识别它们

模型配置通常会提供 `num_attention_heads`、`num_key_value_heads` 和 `head_dim`。若
`num_key_value_heads` 与 Query head 数相同，通常是 MHA；介于 1 和 Query head 数之间通常是 GQA；
等于 1 通常是 MQA。

识别 MLA 不能只看这三个字段，还要寻找 `kv_lora_rank`、`q_lora_rank`、`qk_nope_head_dim`、
`qk_rope_head_dim`、`v_head_dim` 等结构字段，并检查模型代码是否存在 KV down/up projection、RoPE
拆分和吸收式 Decode 路径。字段名会随模型与框架变化，最终依据应是锁定 revision 的模型代码和
实际执行 backend。

验证一份 Attention 手搓图时，至少要能回答：

- 本模型属于哪种结构，判断来自哪些配置和源码；
- Q head、KV head、group size 或 latent rank 分别是多少；
- 每个 TP rank 拥有哪些 Q/K/V 分片或副本；
- KV Cache 每个分量的 shape、dtype、scale 和 owner 是什么；
- Prefill 与 Decode 是否选择了不同的 Attention 路径；
- 是否真正执行 KV up projection，还是使用了 Matrix absorption；
- QKV 和输出 projection 的 collective 是否与本地 shape 对应。

## 为什么选择不同结构

MHA 结构最直接，适合 KV Cache 和 Decode 带宽不是主要约束的场景。MQA 最大程度压缩 KV，但高度
共享是否满足效果目标必须由训练和评测数据决定。GQA 在多套 KV 表达与推理成本之间折中，通常更易
兼顾效果和吞吐。MLA 进一步压缩 KV 表示，适合长上下文和带宽敏感的大模型，但它需要更复杂的模型
结构、专用 Attention backend 和更细的性能建模。

不能只根据显存公式决定采用哪一种结构。模型效果、训练稳定性、Kernel 支持、量化布局、TP/DCP
组合和硬件带宽都会改变选择。对于已经训练好的模型，推理系统通常也不能任意把 MHA 改成 GQA 或
MLA；这些结构属于模型权重和训练过程的一部分。

## 我不确定的地方

MLA 不是完全统一的行业接口。不同模型会选择不同的 latent rank、Q 压缩、RoPE 拆分和缓存布局；
SGLang、vLLM 及硬件专用 backend 也可能采用不同吸收和量化实现。因此本文的 MLA 图是通用理解框架，
不能替代某个模型 revision 的逐层源码核对。

## 需要你拍板的问题

目前不需要决策。若后续把这篇笔记发展成 Kepler 的正式验收依据，需要先指定目标模型和推理框架
revision，再决定是只验收逻辑结构，还是连 backend-specific KV layout、TP/DCP group 和通信算子一起
纳入断言。
