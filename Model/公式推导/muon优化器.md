我从"模型有多少参数"开始，一步一步算到 4.258 GB。

## 0. 目标先讲明

`AG operand = 4.258 GB` 是 Muon AllGather 一次要传的数据量。它的含义是：**每张卡在 ZeRO 之前应该拿到的非路由 Muon 参数，按 FP32 master 算的字节数**。

为什么是这个？因为 ZeRO-1+ 把 FP32 master copy 按 DP 切片存（每张卡只存 1/DP），但 Muon 做 Newton-Schulz 需要完整矩阵，所以每个 step 要把全部 DP 切片 AllGather 起来再做 NS，做完再 ReduceScatter 回去。**AG 的数据总量 = 一张卡应该完整看到的参数总量**。

下面拆步骤算这个量。

## 1. 模型总参数 = 1036.6 B

DeepSeek V4 Pro 是稀疏 MoE 模型。先把所有权重矩阵分两类数。

### Dense / 非专家部分（2.32 B）

这部分代码里叫 `_dense_params`。它包括 **embedding + lm_head + 一个 MTP 层（按 dense 公式估）**。

```
embed + lm_head = 2 × vocab × hidden
                = 2 × 129280 × 7168
                = 1.853 B           ← 词表查询和输出投影
1 个 MTP 层 (估为 dense 公式):
  attention   = 4 × hidden²              = 4 × 7168²     ≈ 205.5 M
  FFN         = 2 × hidden × ffn         = 2 × 7168 × 18432 ≈ 264.2 M
  norms/bias  = 4 × hidden                                ≈ 0.03 M
  小计                                                     ≈ 469.8 M
──────────────────────────────────────────
dense 部分 = 1.853 + 0.470 = 2.32 B ✓
```

> 注：61 个真正的 MoE 层里 attention 部分这里**没单独算**（被 `_dense_params` 漏掉了，但 `_moe_params` 也没算），是这个仓库的近似法。

### MoE 专家部分（1034.3 B）

这部分是大头。

```
每个专家 FFN 大小  = 2 × hidden × moe_ffn
                  = 2 × 7168 × 3072
                  = 44.04 M
每个 MoE 层有几个专家：384 个 routed + 1 个 shared = 385 个
每个 MoE 层参数    = 385 × 44.04 M = 16.96 B
61 个 MoE 层      = 61 × 16.96 B = 1034.3 B ✓
```

**总和**：`1036.6 B = 2.32 B (dense) + 1034.3 B (MoE)`

## 2. TP / EP / PP 怎么把参数分给每张卡

Row 1 配置：`tp=2, cp=4, pp=2, ep=384, dp=512`（cp 只切序列不切权重，所以这步不用考虑）。

每根轴的切分方式不一样：

| 轴 | 切什么 | 怎么切 |
|----|--------|--------|
| **TP** = 2 | **所有矩阵** | 每个 weight 矩阵按列切两半，每张 TP 卡拿一半 |
| **EP** = 384 | **只切 routed expert** | 384 个 routed expert 分到 384 张 EP 卡，每张卡只持有 1 个 expert（shared expert 不参与） |
| **PP** = 2 | **按层切** | 62 层切成两段，每个 PP stage 拿 31 层（embed/lm_head 也切两半） |

走一遍：

### 第一刀 TP=2：所有参数砍一半

```
TP 之后每张卡 = 1036.6 B / 2 = 518.3 B
```

(此时 dense=1.16B、shared=1.34B、routed=515.8B、其余 ≈ 0)

### 第二刀 EP=384：只切 routed expert

```
TP 之后 routed 部分 = 515.8 B
       shared 部分 =   1.34 B  (不被 EP 切)
       dense 部分  =   1.16 B  (不被 EP 切)

EP 切完 routed:  515.8 B / 384 = 1.34 B   ← 每张 EP 卡只拿 1 个 expert
所以 MoE 部分变成 = shared (1.34) + routed_after_ep (1.34) = 2.68 B
                                                              ↑
                                                   注意：从 517.1 → 2.68，大幅缩水

每张卡 = dense(1.16) + MoE_after_TP_EP(2.68) ≈ 3.84 B
```

代码里这步实现成：
```python
total = (total - moe_after_tp) + moe_after_tp_ep
      = 518.3 - 517.1 + 2.68
      = 3.84 B
```

### 第三刀 PP=2：按层切两半（embed 也分）

```
embedding + lm_head = 1.85 B   (固定不变，分给 PP 两端各一半)
非 embedding        = 3.84 - 1.85 = 1.99 B   (按层分)

PP=2 之后:
  non_embed = 1.99 × (31/62) = 0.995 B   ← 这个 PP stage 只持有 31 层
  embed     = 1.85 × (1/2)   = 0.925 B   ← embed/lm_head 一边一份

P_total / rank = 0.995 + 0.925 ≈ 1.92 B   ✓ 对得上 Python 输出 1.924 B
```

**关键结果**：每张卡（在 ZeRO 之前的视图）持有 ≈ **1.92 B 参数**。这是 ZeRO 切之前每个 rank 的"完整视图"。

## 3. 把 1.92 B 再拆成 routed vs non-routed

Muon 通信要区分这两类，因为它们分到不同的 collective group 里：
- **non-routed**（dense + shared + embed）：跨整个 DP=512 做 AG
- **routed**（每张卡上那 1 个 expert）：跨 expert_dp 组做 AG

`_routed_expert_params_on_rank` 单独算 routed 部分：

```
routed 总量(全模型)   = 61 × 384 × 44.04 M = 1031.6 B
TP=2 后              = 515.8 B
EP=384 后            = 1.34 B    (每张 EP 卡只拿 1 个 expert)
PP=2 后              = 1.34 × (31/62) = 0.672 B  ← 一个 PP stage 上的 routed expert 参数
                              ↑
                       公式: 1.34 × layers_per_stage / n_layers
                            = 1.34 × 31/62 = 0.672 B
```

**P_routed / rank = 0.672 B**

剩下的就是 non-routed：

```
P_non_routed = P_total - P_routed
             = 1.924 - 0.672
             = 1.252 B
```

物理意义：每张卡上有 1.252 B 个"需要跟整个 DP 组共同维护"的参数（embed + lm_head + shared expert + MTP），加 0.672 B 个"只跟 expert_dp 共同维护"的 routed expert 参数。

## 4. Muon 只取 85% 的参数做旋转

Muon 通信公式：

```python
P_muon_non_routed = int(P_non_routed × f_muon)
P_muon_routed     = int(P_routed × f_muon)
```

`f_muon = 0.85` 是默认值，意思是 **Muon 优化器只对 85% 的参数做 NS 旋转**（剩下 15% 像 LayerNorm/bias 之类的小张量走 Adam）。

```
P_muon_non_routed = 1.252 B × 0.85 = 1.065 B
P_muon_routed     = 0.672 B × 0.85 = 0.571 B
```

## 5. AG operand 字节数

Muon 用 **FP32 master copy** 做 NS，所以每个参数 4 字节：

```
bytes_for_non_routed_AG = P_muon_non_routed × 4 B
                       = 1.065 B × 4
                       = 4.258 GB    ← 这就是 4.258 GB 的来源！

bytes_for_routed_AG    = P_muon_routed × 4 B
                       = 0.571 B × 4
                       = 2.283 GB
```

## 6. 跨多大的 group AG？

- **non-routed**：跨整个 DP = 512 张卡做 AG（每张 DP 卡都要拿到完整 4.258 GB）
- **routed**：跨 EXPERT_DP 组做 AG

EXPERT_DP 是什么？它是"持有同一个 routed expert 复本的所有 DP 卡集合"。计算：

```python
expert_dp = max(1, dp // ep) = max(1, 512 // 384) = max(1, 1) = 1
```

**只有 1 张卡** —— 即每个 routed expert 在整个集群里只存在 1 份（因为 ep=384 已经把 384 个 expert 平铺到 384 个 EP rank，dp=512 没有完整覆盖 384 的整数倍，整除塌成 1）。

`expert_dp = 1` 意味着 routed expert 不需要 DP AG（自己就一份，无需 gather）→ **routed AG/RS = 0 ms**。

所以 Muon 通信只剩下 non-routed 那一支。

## 7. AG / RS 时间 = 3880 ms / 3880 ms

把 4.258 GB 放到 DP=512 的 ring 上做 AG。

按上一轮我讲的多层 ring 分解（你 YAML 当前的 buggy 配置）：

```
d_0=8,   network0 (504 GB/s)  →   1.34 ms
d_256,   network2 ( 50 GB/s)  →  3879 ms    ← bug 触发：本该是 d_1=16/d_2=4 的两层
                                  ─────────
                  AG 合计       =  3880 ms
```

RS 对称（同样的 ring，只是方向反过来）→ **3880 ms**。

```
total optimizer comm = AG + RS = 3880 + 3880 = 7761 ms   ✓ 对得上 CSV
```

## 一句话总结

```
模型总参数  1036.6 B
  ├─ TP=2          → /2
  ├─ EP=384 切 routed → routed /384
  ├─ PP=2 按层切   → /2
  ▼
每张卡持有 1.924 B 参数 (P_total)
  分两类：
    routed     = 0.672 B → 由 expert_dp=1 决定通信成本 0
    non_routed = 1.252 B → 由 DP=512 做 AG
  ×  Muon 比例 0.85
  ×  FP32 = 4 字节/参数
  ▼
AG operand = 1.252 × 0.85 × 4 = 4.258 GB
  ▼
跨 DP=512 ring AG（受 YAML tier1 bug 影响）= 3880 ms
RS 对称 = 3880 ms
       ── 合计 7761 ms ✓
```

我从"ring AllGather 的成本公式"讲起，再代数字。

## 0. 一个 ring AllGather 要花多少时间 —— α-β 模型

物理上 ring AG 长这样：N 张卡排成环，每张卡手里只有 1/N 的数据片段。然后做 N−1 次"传给下家"的循环，最后每张卡都凑齐了全部 N 片。

每一次"传一片"的成本 = 链路启动延迟 + 数据片大小 × 单字节传输时间：

```
单次传输时间 = α + (S/N) × β
```

- **α** = 链路一次握手 / 启动开销（latency_us，单位秒）
- **β** = 1 / 有效带宽（秒/字节）
- **S** = AG 完成后每张卡手里的完整数据量（字节）
- **S/N** = 每次环传送的片段大小（每张卡的分片）

一共做 `N−1` 轮，总时间：

```
T_ring = (N-1) × α  +  S × (N-1)/N × β
         └─ α 项 ─┘     └─── β 项（带宽项）────┘
```

- α 项：N−1 步的延迟串行累加（小，几十 µs 级别）
- β 项：俗称"NCCL bus-bandwidth"公式，传输总字节量 = `S × (N-1)/N`（每张卡总共要收/发这么多字节）

代码 [collective_time](python/zrt/training/models/comm.py:38) 就是这个公式：
```python
steps = N - 1                          # torus/ring
bw_term = S * (N - 1) / N * beta
return steps * alpha + bw_term         # AG/RS
```

## 1. 链路的"有效带宽"怎么算

每条链路在 YAML 里写：`bandwidth_gbps`、`kb_efficiency`、`latency_us`。代码 [LinkSpec.effective_bw_bps](python/zrt/hardware/spec.py:128)：

```python
base = bandwidth_gbps × 1e9 / 8 × kb_efficiency       # 单位：bytes/sec
```

> 注：变量名 `gbps` 在这里被代码当成 **Gigabits per second** 解释（`/8` 把比特换字节），尽管 docstring 写的是 "GB/s"。这是另一个文档与实现不一致的点，但不影响本次推导（一旦确定单位口径就一致）。

对你的三条链路：

| 链路 | bandwidth | × 1e9 / 8 × 0.7 | 有效带宽 |
|------|-----------|-----------------|----------|
| network0 | 504 | = 44.10 ×10⁹ | **44.1 GB/s** |
| network1 | 224 | = 19.60 ×10⁹ | 19.6 GB/s |
| network2 |  50 | =  4.375 ×10⁹ | **4.375 GB/s** |

延迟：`α_0 = 3.46 µs`, `α_2 = 4.66 µs`。

## 2. 多层 ring 的"嵌套"逻辑

`_tier_breakdown` 给出的 `[(d_0, link_0), (d_2, link_2)]` 不是简单地"分别跑两个 ring"，而是**串行接力**：

```
第 0 层：每个 d_0 人小圈先 AG 一轮，每人手里数据量从 S/N 变成 S × d_0/N
第 1 层：每个 d_1 人组（这里被跳过）
第 2 层：d_2 人外圈再 AG 一轮，每人手里数据量从 S × prod_inner_prev/N 变成最终 S × prod_inner/N
```

每一层结束时，每张卡手里的"完整片段"是这一层前面所有 d 的乘积：

```
out_bytes_i = S × (d_0 × d_1 × … × d_i) / N
```

代码 [collective_time_multi_tier](python/zrt/training/models/comm.py:208)：
```python
prod_inner = 1
for d_i, link_i in breakdown:
    prod_inner *= d_i
    out_bytes = S * prod_inner // N
    total += collective_time(sub_c=Collective(bytes_=out_bytes), d_i, link_i)
```

注意：传给子 `collective_time` 的 `bytes_` 是该层**完成后**的"全片"大小（不是片段大小）；而内部公式 `S × (N-1)/N × β` 里又会用 `(N-1)/N` 把它换算回每张卡实际传输的字节量。两步加在一起结果正确。

## 3. 代数字 —— buggy YAML 下的 breakdown

```
breakdown = [(8, network0), (256, network2)]   # tier1 被跳过
S = 4.258 × 10⁹ bytes
N = 512
```

### 第一段：tier 0 — d_0 = 8，network0 链路

- `prod_inner = 8`
- `out_bytes = 4.258e9 × 8 / 512 = 66.5 × 10⁶` bytes（66.5 MB）
- 在 d_0=8 的小 ring 里跑：

```
α 项  = (8-1) × 3.46 µs = 7 × 3.46 µs = 24.2 µs   ≈ 0.024 ms
β 项  = 66.5e6 × (7/8) / 44.1e9 bytes/s
      = 66.5e6 × 0.875 / 44.1e9
      = 1.32 × 10⁻³ s                              = 1.32 ms

T_tier0 = 0.024 + 1.32 ≈ 1.34 ms
```

含义：每张卡在自己 super-node 里和另外 7 张 DP 卡传 66.5 MB 的"小数据片段"，因为 super-node 内是 504 帧带宽（实际 44.1 GB/s effective），传得飞快。

### 第二段：tier 2 — d_2 = 256，network2 链路（**bug 触发**）

正常情况下应该有 tier 1（中等链路）夹在中间。但因为 `num_devices=16` 把 tier1 整除塌成 0 了，d_1 被跳过，**所有剩下的 256 个参与者直接挤进了最慢的 network2**。

- `prod_inner = 8 × 256 = 2048`  ← 这是错的，应该 ≤ N=512
- `out_bytes = 4.258e9 × 2048 / 512 = 4.258e9 × 4 = 17.03 × 10⁹` bytes（**17 GB**）

为什么 out_bytes 是原始 S 的 4 倍？数学上 `prod_inner` 超过了 N=512（理论不变量是 d 连乘等于 N），代码没有 clamp，于是公式输出"每张卡手里数据是 4 × S"这种物理上不合理的值。这是 bug 的算术表现。

然后在 d=256 的 ring 里跑：

```
α 项  = (256-1) × 4.66 µs = 255 × 4.66 µs        ≈ 1.19 ms
β 项  = 17.03e9 × (255/256) / 4.375e9 bytes/s
      = 17.03e9 × 0.99609 / 4.375e9
      = 3.878 s                                   = 3878 ms

T_tier2 = 1.19 + 3878 ≈ 3879 ms
```

β 项主导：17 GB 数据要在 256 人环上传 255 轮，带宽只有 4.375 GB/s。粗略口算：`17 GB / 4.375 GB/s = 3.89 s ≈ 3879 ms`。

### 加起来

```
T_AG = T_tier0 + T_tier2 = 1.34 + 3879 ≈ 3880 ms ✓
```

## 4. 把数字组合起来看放大倍数

```
                       理论正确              你的 buggy YAML
                       三层分解              两层(tier1 塌缩)
─────────────────────────────────────────────────────────────────
tier 0  504 GB/s        1.34 ms              1.34 ms
tier 1  224 GB/s       50.99 ms              ──── (跳过)
tier 2   50 GB/s      729.96 ms          3879.01 ms        ← bug 把这里放大 5.3×
─────────────────────────────────────────────────────────────────
T_AG (总)              782 ms              3880 ms
T_AG + T_RS           1564 ms              7761 ms
exposed (扣掉隐藏)     ~565 ms            ~6744 ms
占 step 比例           ~0.25%              ~3.0%
```

## 5. 为什么 bug 主要把时间灌到 β 项

α-β 公式里两项的比较：
- α 项：`(N-1) × latency`。哪怕 N=256 跑慢链，α 项也只有 1.2 ms 量级
- β 项：`S × (N-1)/N / bw`。数据量 17 GB / 带宽 4.375 GB/s ≈ 4 秒

**bug 的杀伤力来自三个叠加**：
1. **链路慢**：50 GB/s 而不是 224 GB/s（4.5× 慢）
2. **数据量被错算成 4×**（应该是 4.258 GB，算成了 17 GB）
3. **N=256 vs 应该 N=4**：(N-1)/N 因子从 3/4=0.75 变成 255/256≈1，不算大头

主要是 (1) 和 (2) 叠出来的：`(17/4.258) × (50/224 反过来) ≈ 4 × 4.5 ≈ 18×`。再考虑修正后还会跑一段 tier1（额外 51 ms），最终对比是 782 → 3880，约 **5×**。

## 一句话总结

```
T_AG = Σ_tier [ (d-1)·α + S·prod_inner/N · (d-1)/d · 1/bw ]

第 0 层  8 人圈   传   66.5 MB  @  44.1 GB/s  → 1.34 ms     (快链跑小数据)
第 2 层  256 人圈 传  17.03 GB  @  4.375 GB/s → 3879 ms     (慢链跑放大 4× 的数据)
                                              ──────────
                                                 3880 ms
```

数字本身严格按 α-β + ring 公式算出来的，没有计算 bug；但**输入到公式的 d_i 和 out_bytes 已经是错值**（因为 `_tier_breakdown` 不变量被 YAML 破坏），所以输出虽然"算对了"实际"问题大"。