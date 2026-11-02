是的。**如果你要研究“并行策略如何改变算子 shape、通信、调度和最终训练耗时”，最好一定通过训练框架来采。** 单算子 Roofline 校准可以脱离训练框架，但 TP/PP/CP/EP 的真实效果不能只靠手工改 shape 模拟。

MindSpeed-MM 很适合做这件事，因为当前版本已经把 Ascend PyTorch Profiler 集成到了训练流程中，内置模型通常只需要配置 `mindspeed_mm/tools/tools.json`，不用自己在训练代码里插 profiler。它的训练参数也直接支持 TP、PP、CP、EP 等并行配置。([github.com][1]) ([GitHub][2])

我建议你把采集工作固定成下面这套流程。

---

# 一、先明确：你最终要采两类数据

不要只做一套实验。

### A. 算子 Roofline 校准数据

目标是得到：

```text
parallel strategy
        ↓
local operator shape
        ↓
kernel
        ↓
duration
        ↓
compute / memory efficiency
```

例如：

```text
TP=1
MatMul [8192,4096] x [4096,16384]
duration = xxx us

TP=2
MatMul [8192,4096] x [4096,8192]
duration = xxx us

TP=4
MatMul [8192,4096] x [4096,4096]
duration = xxx us
```

这是你 Roofline 模型最核心的数据。

### B. 并行策略校准数据

目标是得到：

```text
TP/PP/CP/EP
   ↓
compute
communication
overlap
pipeline bubble
load imbalance
   ↓
step time
```

也就是最终验证：

[
T_{model}^{pred}
\quad vs \quad
T_{model}^{real}
]

这两类数据不要混成一个“硬件利用率”。

---

# 二、MindSpeed-MM 在这里扮演什么角色

你的建模程序负责：

```text
模型算子序
    ↓
并行策略
    ↓
预测 local shape
    ↓
Roofline
    ↓
预测算子耗时
```

MindSpeed-MM 真机训练负责提供 Ground Truth：

```text
               MindSpeed-MM
                    │
            TP / PP / CP / EP
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     实际算子shape          实际HCCL通信
          │                   │
      kernel time          comm time
          │                   │
      AI Core指标           overlap
          └─────────┬─────────┘
                    ↓
               real step time
```

所以最终你做的是：

```text
你的建模结果
      ↕
MindSpeed-MM真机结果
```

一一校准。

---

# 三、第一步：先把实验环境固定住

这个特别重要。

每次实验都建议生成一个自己的：

```text
run_meta.json
```

至少记录：

```json
{
  "model": "...",
  "model_commit": "...",
  "mindspeed_mm_commit": "...",
  "torch_version": "...",
  "torch_npu_version": "...",
  "cann_version": "...",

  "device": "Ascend A3",
  "world_size": 8,

  "micro_batch_size": 1,
  "global_batch_size": 64,
  "seq_length": 8192,
  "dtype": "bf16",

  "tp": 4,
  "pp": 1,
  "cp": 1,
  "ep": 1,

  "recompute": false,
  "distributed_optimizer": true,
  "fused_rmsnorm": true,
  "fused_swiglu": true
}
```

因为下面这些都会改变你的真机结果：

```text
CANN版本
torch_npu版本
MindSpeed-MM版本
融合算子开关
重计算
distributed optimizer
MBS
GBS
seq length
TP/PP/CP/EP
```

尤其不能出现：

```text
实验1 TP=1 + fused rmsnorm
实验2 TP=4 + unfused rmsnorm
```

然后把差异归因于 TP。

---

# 四、第二步：通过 MindSpeed-MM 配并行策略

MindSpeed-MM 当前参数里明确有：

```bash
--tensor-model-parallel-size ${TP}

--pipeline-model-parallel-size ${PP}

--context-parallel-size ${CP}

--expert-model-parallel-size ${EP}
```

CP 还可以进一步配置算法：

```bash
--context-parallel-algo ulysses_cp_algo
```

以及其它 CP 算法。MoE 开启 EP 后，MindSpeed-MM 文档建议 token dispatcher 使用 `alltoall`。([GitHub][2])

所以建议你的启动脚本统一写：

```bash
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS}
"

MOE_ARGS="
    --expert-model-parallel-size ${EP}
"
```

这样以后：

```bash
TP=4 PP=1 CP=1 EP=1 bash train.sh
```

就可以自动批量跑实验。

MindSpeed-MM 官方 examples 也是用 `${TP}`、`${PP}`、`${CP}` 变量传入并行参数。([GitHub][3])

---

# 五、并行策略不要一上来全组合

例如你有 8 卡。

第一轮做 **single-factor sweep**：

| Case     | TP | PP | CP | EP | 目的     |
| -------- | -: | -: | -: | -: | ------ |
| baseline |  1 |  1 |  1 |  1 | 基准     |
| TP2      |  2 |  1 |  1 |  1 | TP影响   |
| TP4      |  4 |  1 |  1 |  1 | TP影响   |
| TP8      |  8 |  1 |  1 |  1 | TP影响   |
| PP2      |  1 |  2 |  1 |  1 | PP影响   |
| PP4      |  1 |  4 |  1 |  1 | PP影响   |
| CP2      |  1 |  1 |  2 |  1 | CP影响   |
| CP4      |  1 |  1 |  4 |  1 | CP影响   |
| EP2      |  1 |  1 |  1 |  2 | MoE EP |
| EP4      |  1 |  1 |  1 |  4 | MoE EP |

当然前提是：

1. 卡数满足要求；
2. 模型本身支持对应策略。

MindSpeed-MM 的模型支持矩阵显示，不同模型支持的 TP/PP/CP 等组合不同，所以不能假设所有模型都支持所有并行方式。([github.com][4])

第二轮再研究：

```text
TP × PP
TP × EP
TP × CP
EP × CP
```

例如：

```text
TP2_PP2
TP4_PP2
TP2_EP4
TP2_CP2
```

这样数据会干净很多。

---

# 六、第三步：每个并行配置先跑“无 Profiler Baseline”

这是 **必须的**。

修改：

```text
mindspeed_mm/tools/tools.json
```

为：

```json
"profile": {
    "enable": false
}
```

然后正常训练。

建议至少：

```text
warmup 10~20 step
+
steady 20~100 step
```

采：

```text
step
step_time
tokens/s
samples/s
loss
```

例如：

```text
step 20: 865 ms
step 21: 860 ms
step 22: 871 ms
...
```

最终保存：

```text
mean_step_time
median_step_time
p20
p50
p80
std
```

这个才是：

[
T_{step}^{groundtruth}
]

**不要用开着 profiler 时的 step time 当 Ground Truth。**

---

# 七、第四步：使用 MindSpeed-MM 自带 Profiler 采 1 个稳定 step

这是最关键的一步。

MindSpeed-MM 当前的：

```text
mindspeed_mm/tools/tools.json
```

官方默认结构就是：

```json
{
    "profile": {
        "enable": false,
        "profile_type": "static",
        "ranks": [0, 1, 2, 3],

        "static_param": {
            "level": "level1",
            "with_stack": false,
            "with_memory": false,
            "record_shapes": false,
            "with_cpu": true,

            "save_path": "./npu_profiling",

            "start_step": 10,
            "end_step": 11,

            "data_simplification": false,

            "aic_metrics_type": "PipeUtilization",

            "analyse_flag": true
        }
    }
}
```

([GitHub][5])

对于你的建模任务，我建议改成：

```json
{
    "profile": {
        "enable": true,
        "profile_type": "static",
        "ranks": [-1],

        "static_param": {
            "level": "level1",

            "with_stack": false,
            "with_memory": false,
            "record_shapes": true,
            "with_cpu": true,

            "save_path": "./profiles/tp4_pp1_cp1_ep1",

            "start_step": 10,
            "end_step": 11,

            "data_simplification": false,

            "aic_metrics_type": "PipeUtilization",

            "analyse_flag": false
        },

        "dynamic_param": {
            "config_path": "./dynamic_profile"
        }
    }
}
```

这里有几个非常重要的选择。

---

# 八、为什么是 start_step=10，end_step=11

MindSpeed-MM 当前源码实际上计算的是：

```python
skip_first = start_step
active = end_step - start_step
```

所以：

```text
start_step = 10
end_step = 11
```

实际上就是：

> 跳过前 10 step，只抓 1 个稳定 step。

源码也是直接把它转换成 `torch_npu.profiler.schedule()`。([GitHub][6])

这正适合你的场景。

Ascend 官方也建议动态 shape 等情况下跳过前面的若干 step，避免初始化阶段污染 Profiling。([Hiascend][7])

---

# 九、为什么一定 `record_shapes=true`

因为你最重要的目标之一就是验证：

```text
理论并行切分后的shape
           ↕
真机实际shape
```

开启以后可以获得：

```text
Input Shapes
Input Data Types
Output Shapes
Output Data Types
```

最终 `kernel_details.csv` 中还能看到：

```text
Name
Type
Start Time
Duration
Wait Time
Block Dim
Input Shapes
Input Data Types
Output Shapes
Output Data Types
```

这是你建立：

```text
parallel config
→ op
→ shape
→ kernel
→ duration
```

映射的主数据文件。([Hiascend][8])

---

# 十、为什么用 Level1，而不是 Level0

对你这种并行建模，建议直接：

```json
"level": "level1"
```

因为 Level1 除了基本 kernel 数据之外，还会采 AI Core 指标以及 HCCL 数据，并生成：

```text
communication.json
communication_matrix.json
api_statistic.csv
```

Level2 更多是 Runtime、AI CPU 等底层数据，第一版训练建模暂时没必要。([Hiascend][7])

所以：

```text
Roofline / 并行建模
       ↓
Level1

复杂性能问题定位
       ↓
Level2
```

---

# 十一、第五步：同一个并行配置跑两遍 Profiler

这是我非常建议你采用的方法。

## Run A：PipeUtilization

```json
"aic_metrics_type": "PipeUtilization"
```

得到类似：

```text
mac_ratio
vec_ratio
mte ratio
...
```

用来判断：

```text
Cube忙不忙
Vector忙不忙
数据搬运忙不忙
```

---

## Run B：ArithmeticUtilization

再把：

```json
"aic_metrics_type": "ArithmeticUtilization"
```

重新跑完全相同的配置。

用来校准：

[
\eta_{compute}
]

比如：

```text
TP=1 MatMul shape A η=0.81
TP=2 MatMul shape B η=0.86
TP=4 MatMul shape C η=0.73
```

这里有一个你需要特别注意的地方：

**当前 MindSpeed-MM 的 `profiler.py` 只接受 `PipeUtilization` 和 `ArithmeticUtilization`。**

源码明确写了：

```python
if aic_metrics_type == 'PipeUtilization':
    ...
elif aic_metrics_type == 'ArithmeticUtilization':
    ...
else:
    raise ValueError(...)
```

([GitHub][6])

---

# 十二、这意味着你的 Memory Roofline 还需要再补一层

这是目前最关键的地方。

Ascend 底层 `torch_npu.profiler` 实际支持：

```text
PipeUtilization
ArithmeticUtilization
Memory
MemoryL0
MemoryUB
ResourceConflictRatio
L2 Cache
```

但是 MindSpeed-MM 当前 `tools.json → profiler.py` 这一层只暴露了前两种。([Hiascend][7])

所以我建议第一版：

```text
MindSpeed-MM
    │
    ├── shape
    ├── kernel time
    ├── compute utilization
    ├── communication
    ├── overlap
    └── step time

msprof op / 单算子 torch_npu.profiler
    │
    ├── Memory
    ├── MemoryL0
    ├── MemoryUB
    └── L2
```

不要为了 Memory 指标马上大改训练框架。

后面如果你觉得需要整网 Memory profile，我们再给 MindSpeed-MM 的 `profiler.py` 加：

```python
elif self.aic_metrics_type == 'Memory':
    aic_metrics_type = torch_npu.profiler.AiCMetrics.Memory
```

这样的支持即可。

---

# 十三、第六步：哪些 Rank 要采

这里建议分两种。

## 算子 Shape / Compute 校准

不用每次所有 rank 都采。

TP 场景通常可以先：

```json
"ranks": [0]
```

PP 场景建议：

```text
PP stage 0
PP 中间 stage
PP 最后一 stage
```

各选一个代表 rank。

因为 PP 的不同 stage 算子序本身不同。

---

## 通信 / overlap 校准

建议短时间：

```json
"ranks": [-1]
```

即所有 Rank 采 **一个 step**。

MindSpeed-MM 当前源码中：

```python
if self.ranks == [-1]:
    return True
```

代表所有 Rank 开 profiler。([GitHub][6])

这样你才能完整分析：

```text
TP AllReduce
DP ReduceScatter / AllGather
PP Send/Recv
CP通信
EP AllToAll
```

大集群不要采几十个 step。

**全 Rank × 一个稳定 step 就够你建模了。**

---

# 十四、第七步：采完之后离线解析

我更建议：

```json
"analyse_flag": false
```

这样训练期间尽量减少额外分析开销。

采集完之后执行：

```bash
python mindspeed_mm/tools/profiler.py \
    --profiler-path ./profiles/tp4_pp1_cp1_ep1
```

或者：

```bash
python mindspeed_mm/tools/profiler.py \
    --profiler-path ./profiles/tp4_pp1_cp1_ep1 \
    --export-type text \
    --export-type db
```

MindSpeed-MM 当前支持：

```text
text
db
```

两种离线导出。([GitHub][1])

对于你后续自动建模：

> **优先 CSV/Text；DB 留给 MindStudio Insight 分析。**

---

# 十五、最终你重点读取这几个文件

解析完成后的 `ASCEND_PROFILER_OUTPUT` 中会包含：

```text
ASCEND_PROFILER_OUTPUT/
│
├── kernel_details.csv
├── operator_details.csv
│
├── communication.json
├── communication_matrix.json
│
├── step_trace_time.csv
│
├── op_statistic.csv
├── api_statistic.csv
│
├── trace_view.json
│
└── analysis.db
```

官方当前文档给出的目录结构就是这些文件；其中通信 JSON 在 Level1/Level2 下产生。([Hiascend][9])

对你的用途，我建议优先级是：

### ① `kernel_details.csv`

**最高优先级。**

拿：

```text
Step Id
Name
Type

Start Time(us)
Duration(us)
Wait Time(us)

Block Dim

Input Shapes
Input Data Types

Output Shapes
Output Data Types
```

以及对应的 AI Core metric。([Hiascend][10])

最终形成：

```text
operator_calibration.csv
```

---

### ② `communication.json`

建立：

```text
collective type
communication time
wait time
bandwidth
```

---

### ③ `communication_matrix.json`

重点建立：

```text
rank A
rank B
communication
message size
topology-related result
```

这是 TP/EP/CP 建模的重要数据。

---

### ④ `step_trace_time.csv`

用于：

```text
整个step
compute time
communication time
```

官方将其定义为迭代中计算和通信时间的统计。([Hiascend][9])

---

### ⑤ `trace_view.json`

不是直接拿来训练你的 Roofline。

主要用它判断：

```text
compute
████████████████

communication
       ████████
```

到底是：

```text
串行
```

还是：

```text
overlap
```

---

# 十六、你的最终数据建议这样组织

例如一条：

```text
model=qwen_xxx
rank=0

TP=4
PP=1
CP=1
EP=1

phase=forward

op=MatMul
kernel=xxx

input_shape=[[2048,4096],[4096,4096]]
output_shape=[[2048,4096]]

dtype=bf16

M=2048
N=4096
K=4096

duration_us=72.4

block_dim=24

theoretical_flops=...
theoretical_bytes=...

achieved_tflops=...
compute_efficiency=...

mac_ratio=...
vec_ratio=...
mte_ratio=...
```

然后你的模型可以直接比较：

[
shape_{predicted}
\overset{?}{=}
shape_{real}
]

以及：

[
T_{roofline}
\overset{?}{\approx}
T_{kernel}
]

---

# 十七、并行策略分别重点看什么

这部分你后面分析数据时非常有用。

| 策略     | 重点校准                                              |
| ------ | ------------------------------------------------- |
| **DP** | 算子 shape 通常变化较少，重点看梯度通信                           |
| **TP** | MatMul shape + AllReduce/ReduceScatter/AllGather  |
| **PP** | 每个 rank 算子序不同 + Send/Recv + bubble                |
| **CP** | Attention local sequence shape + CP 通信            |
| **EP** | Expert GEMM shape + token distribution + AllToAll |

所以你现在说的：

> “并行策略改变算子 shape”

最典型的是 **TP、CP、EP**。

PP 更主要改变：

```text
这个 Rank 上有哪些 layer
```

以及 pipeline schedule。

DP 更主要改变：

```text
gradient communication
```

而不是 Transformer forward 单算子的 shape。

---

# 十八、EP 要多记录一个维度

MoE 场景千万不能只记录：

```text
EP=8
```

还要记录：

```text
expert_id
tokens_per_expert
```

因为：

```text
Expert0 GEMM M=300
Expert1 GEMM M=512
Expert2 GEMM M=110
Expert3 GEMM M=800
```

即使：

```text
N、K完全相同
```

由于 M 不同，算子效率完全可能不同。

最终建议：

[
\eta_{MoE}
==========

f(M_{tokens},N,K,dtype)
]

而不是：

[
\eta_{MoE}=f(EP)
]

---

# 十九、我建议你真正执行的实验流程就固定成下面这样

以后每一种并行策略都机械执行同样的流程：

```text
                   一种 Parallel Config
                   TP=x PP=x CP=x EP=x
                            │
                 ┌──────────┴──────────┐
                 │                     │
           Run 0 无Profiler       Run 1 Profiler
                 │                     │
          steady step time       PipeUtilization
                 │                record_shapes
                 │                Level1
                 │                     │
                 │                Run 2 Profiler
                 │                     │
                 │             ArithmeticUtilization
                 │                     │
                 └──────────┬──────────┘
                            │
                ┌───────────┼───────────┐
                ↓           ↓           ↓
           kernel CSV    HCCL JSON    trace
                │           │           │
                ↓           ↓           ↓
           Op Dataset   Comm Dataset  Overlap
                │           │           │
                └───────────┼───────────┘
                            ↓
                    Roofline Calibration
                            │
                            ↓
                     predicted step
                            │
                            ↕
                      real step time
```

---

# 二十、第一版你实际上只需要跑 3 次

例如：

```text
TP=4 PP=1 CP=1 EP=1
```

### Run 0

```text
Profiler OFF
```

得到：

```text
真实 step time
```

### Run 1

```text
Level1
record_shapes=true
PipeUtilization
```

得到：

```text
算子序
shape
dtype
kernel
duration
HCCL
pipeline/MTE指标
```

### Run 2

```text
Level1
record_shapes=true
ArithmeticUtilization
```

得到：

```text
算术利用指标
```

如果再加：

### Run 3

对训练中出现的重要 shape 做：

```text
msprof op
```

得到：

```text
Memory / HBM
L0
UB
L2
```

这样你的第一版 Roofline calibration 数据基本就齐了。

---

## 我比较推荐你下一步直接这样落地

不要先做几十个并行配置。先选 **一个你后续真正要建模的 MindSpeed-MM 模型**，例如某个 Qwen-VL / InternVL / 多模态模型，然后只做：

```text
TP=1
TP=2
TP=4
TP=8
```

先验证这一条链：

```text
我的建模器预测 MatMul shape
              ↓
MindSpeed-MM真实 shape
              ↓
Roofline预测 time
              ↓
kernel_details真实 time
```

**把 TP 这一条链打通以后，再扩 PP/CP/EP。** 这样工程风险最低，也最容易判断误差到底来自“并行 shape 推导错误”还是“Roofline 硬件效率错误”。

如果你后面告诉我你准备用 **MindSpeed-MM 的具体哪个模型、A3 是多少卡/多少节点**，我可以直接基于那个模型的官方 `train.sh`，给你整理成一套 `TP/PP/CP/EP` 可批量 sweep 的采集脚本，以及 `tools.json` 和最终 CSV 字段，不再停留在方法论层面。

[1]: https://github.com/Ascend/MindSpeed-MM/blob/master/docs/zh/tools.md "MindSpeed-MM/docs/zh/tools.md at master · Ascend/MindSpeed-MM · GitHub"
[2]: https://github.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/args_readme.md "MindSpeed-MM/docs/zh/pytorch/args_readme.md at master · Ascend/MindSpeed-MM · GitHub"
[3]: https://github.com/Ascend/MindSpeed-MM/blob/master/examples/wan2.1/feature_extract/feature_extraction.sh?utm_source=chatgpt.com "feature_extraction.sh - MindSpeed-MM"
[4]: https://github.com/Ascend/MindSpeed-MM "GitHub - Ascend/MindSpeed-MM · GitHub"
[5]: https://github.com/Ascend/MindSpeed-MM/blob/master/mindspeed_mm/tools/tools.json "MindSpeed-MM/mindspeed_mm/tools/tools.json at master · Ascend/MindSpeed-MM · GitHub"
[6]: https://github.com/Ascend/MindSpeed-MM/blob/master/mindspeed_mm/tools/profiler.py "MindSpeed-MM/mindspeed_mm/tools/profiler.py at master · Ascend/MindSpeed-MM · GitHub"
[7]: https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0033.html?utm_source=chatgpt.com "Ascend PyTorch Profiler APIs-Using ..."
[8]: https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/T%26ITools/Profiling/atlasprofiling_16_0035.html?utm_source=chatgpt.com "Ascend PyTorch Profiler接口采集数据-使用 ..."
[9]: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0035.html "数据目录说明-Ascend PyTorch Profiler接口采集数据-使用PyTorch框架接口采集和解析性能数据-性能数据其他采集方式-性能调优工具-训练&推理开发-CANN商用版8.1.RC1开发文档-昇腾社区"
[10]: https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0040.html?utm_source=chatgpt.com "使用PyTorch框架接口采集-其他采集方式介绍-性能分析工具"
