# Inductor 体系结构与算子融合概念

> 来源：ChatGPT 会话《NPU场景下的inductor建模》的后续两轮问答。本文按知识主题重新编排，合并了重复内容，并纳入了后续回答对早期简化示意的纠正。

## 1. 最重要的结论

在 Inductor 中，算子融合不是直接把 FX Graph 上两个 `aten` 算子拼在一起。真正进行融合决策的对象，是 Lowering 后形成的 Inductor IR、Buffer、Dependency 和 SchedulerNode。

需要始终区分：

```text
FX Node 数
≠ ATen op 数
≠ Pointwise / Reduction 数
≠ ComputedBuffer 数
≠ SchedulerNode 数
≠ 最终 Kernel 数
```

## 2. 从 PyTorch 程序到 NPU Kernel

```text
PyTorch Python Model
        │
        ▼
TorchDynamo
        │
        ▼
FX Graph
        │
        ▼
AOTAutograd / Decomposition
        │
        ▼
ATen / Prims Graph
        │
        ▼
Inductor Lowering
        │
        ▼
┌─────────────────────────────────┐
│          Inductor IR            │
│ TensorBox / StorageBox          │
│ Buffer / ComputedBuffer         │
│ Pointwise / Reduction / ...     │
└─────────────────────────────────┘
        │
        ▼
Dependency Analysis
MemoryDep / ReadWrites / producer-consumer
        │
        ▼
┌─────────────────────────────────┐
│          Scheduler IR           │
│ SchedulerBuffer                 │
│ SchedulerNode                   │
│ FusedSchedulerNode              │
│ ExternKernelSchedulerNode       │
│ ForeachKernelSchedulerNode      │
└─────────────────────────────────┘
        │
        ▼
Fusion Candidate Generation
buffer grouping / can_fuse / score_fusion
        │
        ▼
FusedSchedulerNode
        │
        ▼
Scheduling / Codegen
        │
        ▼
NPU / Triton / C++ Kernel
        │
        ▼
Kernel Launch
```

## 3. Node、Buffer 与 Kernel

假设用户代码是：

```python
a = x + 1
b = torch.relu(a)
c = b * 2
```

在用户层看到的是算子链：

```text
x → add → relu → mul → c
```

在 Scheduler 关心的抽象中，更接近：

```text
compute op0 → logical buffer → compute op1 → logical buffer → compute op2
```

- Node：表示谁在计算或哪个调度任务要执行。
- Buffer：表示计算结果对应的逻辑数据对象，带有名称、Layout、Shape、Stride、Device 等信息。
- Kernel：Codegen 后真正在 NPU/GPU/CPU 上执行的程序。

三者不是同一个概念。

## 4. TensorBox、StorageBox 与 Buffer

可以先用如下层级理解：

```text
TensorBox
   │
   ▼
StorageBox
   │
   ▼
Buffer / View / Loops / ...
```

### TensorBox

更接近 PyTorch Tensor 的逻辑包装。

### StorageBox

支持 Tensor 背后的可变 Storage 语义、别名和物化过程。

### Buffer

更接近参与调度、依赖和内存关系分析的逻辑数据对象。一个 Python Tensor 不一定简单对应一个独立 Buffer，因为还存在 View、Alias、Mutation、Reinterpret 等关系。

## 5. Pointwise 是什么

`Pointwise` 不是某个具体的 `add` 算子，而是 Loop-level computation：它描述“对每一个输出位置，如何算出该位置的值”。

概念上：

```python
Pointwise(
    device=...,
    dtype=torch.float32,
    ranges=[N],
    inner_fn=...,
)
```

其中 `inner_fn` 是核心。例如：

```python
def inner_fn(index):
    value = load(x, index)
    result = value + 1
    return result
```

它表示：

```text
for i in range(N):
    y[i] = inner_fn(i)
```

一个 `Pointwise` 可以包含多个低级运算。例如：

```python
y = torch.relu(x + 1) * 2
```

其单个 `inner_fn` 可能包含：

```text
load x → add 1 → relu → mul 2 → return
```

因此不能认为：

```text
1 Pointwise = 1 ATen op
```

## 6. Pointwise 与 Reduction

两者都属于 Loop IR，但依赖关系不同。

### Pointwise

```python
y = x + 1
```

```text
for i:
    y[i] = f(x[i])
```

一个输出位置通常只依赖对应输入位置。

### Reduction

```python
y = x.sum(dim=-1)
```

```text
for i:
    acc = 0
    for r:
        acc += x[i, r]
    y[i] = acc
```

一个输出位置依赖多个 reduction-range 输入位置，因此具有额外的归约范围和并行约束。

## 7. ComputedBuffer 是什么

`ComputedBuffer` 表示在 Kernel 执行期间通过计算产生的 Buffer。可以把它理解为：

```text
ComputedBuffer
= 计算方式
+ 输出 Buffer 信息
```

其关键关系是：

```python
class ComputedBuffer(OperationBuffer):
    data: Loops
```

而 `Pointwise`、`Reduction` 等属于 `Loops`。因此：

```text
ComputedBuffer buf0
│
├── name = buf0
├── layout
│   ├── device
│   ├── dtype
│   ├── size
│   └── stride
│
└── data
     │
     ▼
Pointwise / Reduction / ...
```

`ComputedBuffer` 与 `Pointwise` 不是并列对象。`Pointwise` 是 `ComputedBuffer.data` 所保存的计算内容。

## 8. SchedulerNode 是什么

`SchedulerNode` 是供 Scheduler 进行依赖分析、排序和融合的调度节点。普通情况下，它封装一个 `ComputedBuffer` 或 `TemplateBuffer`。

准确层级是：

```text
SchedulerNode op0
│
└── node
     │
     ▼
ComputedBuffer buf0
│
└── data
     │
     ▼
Pointwise
│
└── inner_fn(index)
     ├── load x[index]
     ├── add 1
     └── return
```

所以对问题“ComputedBuffer buf0 和 Pointwise(x + 1) 是否共同组成 SchedulerNode op0”，严格回答是：

> 不是共同组成。`Pointwise` 是 `ComputedBuffer` 内部的 `data`，而整个 `ComputedBuffer` 再被 `SchedulerNode` 封装。

## 9. Lazy Pointwise 与 realize

还有一个重要细节：Pointwise 起初不一定已经对应一个 `ComputedBuffer`。

### realize 之前

```text
TensorBox
   ↓
StorageBox
   ↓
Pointwise
```

此时是可以继续内联或组合的 lazy computation。

### realize 之后

```text
TensorBox
   ↓
StorageBox
   ↓
ComputedBuffer buf0
   ↓
Pointwise
```

`StorageBox.realize()` 会把尚未物化的 Pointwise、Reduction 等 Loop IR 包装成 ComputedBuffer，注册 Buffer，使其进入后续 Graph 和 Scheduler 管理。

这意味着 Inductor 中可能存在两个容易混淆的融合层次：

1. Scheduler 之前的 lazy Pointwise inlining / fusion；
2. Scheduler 阶段的 SchedulerNode fusion。

例如：

```python
a = x + 1
b = torch.relu(a)
c = b * 2
```

不一定先形成三个 ComputedBuffer 再由 Scheduler 融合。纯 Pointwise 链可能在 `realize()` 之前先组合为一个较大的 `inner_fn`，随后才物化为一个 ComputedBuffer。

这也是分析“64 个 SchedulerNode”时必须特别小心的原因。

## 10. SchedulerBuffer 是什么

`SchedulerBuffer` 是 Buffer 在 Scheduler 世界中的包装对象。它关联：

```text
node: ir.Buffer
defining_op
users
```

依赖图可表示为：

```text
SchedulerNode op0
        │ produces
        ▼
SchedulerBuffer buf0
        │ used by
        ▼
SchedulerNode op1
```

## 11. MemoryDep 与 ReadWrites

### MemoryDep

只知道某个节点读取 `buf0` 还不够；Fusion 还要知道它怎样访问 `buf0`。

`MemoryDep` 主要描述：

```text
name       访问哪个 Buffer
index      如何索引
var_names  迭代变量
size       访问范围
mode       读写模式
```

例如：

```text
MemoryDep(
    name="buf0",
    index=i * 1024 + j,
    var_names=(i, j),
    size=(128, 1024)
)
```

两个节点都访问 `buf0`，并不代表访问模式相同：

```text
buf0[i]
buf0[i * 2]
```

因此：

```text
共享同一个 Buffer ≠ 拥有相同 Memory Dependency
```

### ReadWrites

一个 SchedulerNode 的 `read_writes` 通常包含：

```text
reads
writes
index_exprs
```

例如：

```text
SchedulerNode op1
reads:  MemoryDep(buf0, i)
writes: MemoryDep(buf1, i)
```

这些是 Fusion Cost Model 最重要的输入特征之一。

## 12. used_buffer_names

`used_buffer_names()` 从节点的读写依赖中提取它使用的 Buffer 名称。

```text
op0:
  reads  = {x}
  writes = {buf0}
  used_buffer_names = {x, buf0}

op1:
  reads  = {buf0}
  writes = {buf1}
  used_buffer_names = {buf0, buf1}
```

两个节点都使用 `buf0`，因此值得被放进同一个候选搜索集合。

## 13. Buffer Group 是什么

`buffer group` 不是正式的 IR Node 类型，而是 Scheduler 搜索 Fusion Candidate 时建立的临时索引结构。

概念代码：

```python
buffer_names_grouping = defaultdict(list)

for node in nodes:
    for buf in node.used_buffer_names():
        buffer_names_grouping[buf].append(node)
```

例如：

```text
op0: reads x,    writes buf0
op1: reads buf0, writes buf1
op2: reads buf0, writes buf2
op3: reads y,    writes buf3
```

得到：

```text
buffer x:    [op0]
buffer buf0: [op0, op1, op2]
buffer buf1: [op1]
buffer buf2: [op2]
buffer y:    [op3]
buffer buf3: [op3]
```

其中 `[op0, op1, op2]` 就是基于 `buf0` 建立的 buffer group：因为使用同一个 Buffer 而进入同一候选搜索集合的一组 SchedulerNode。

它的目的，是避免对全图 N 个节点做完全的 `O(N²)` 两两检查。

## 14. `max_fusion_buffer_group_pairwise_attempts`

```python
max_fusion_buffer_group_pairwise_attempts = 64
```

这个参数控制 buffer group 内的候选搜索预算。

如果一个 group 中有：

```text
N0, N1, N2, ..., N200
```

对 `N0` 不一定尝试所有 `N1...N200`，而是最多向后尝试 64 个节点。

因此该参数控制的是：

> 要不要尝试这对节点。

它不是一个 fused kernel 最多能容纳多少节点。

## 15. Fusion Candidate 是什么

Fusion Candidate 是 Scheduler 当前认为值得尝试融合的两个调度节点组成的 Pair：

```text
(op0, op1)
```

候选中的对象也可以是已经融合的节点：

```text
A + B → AB
AB + C → ABC
ABC + D → ABCD
```

所以融合是逐轮、贪心地生长，而不是一次找到一个完整大子图。

可进一步区分：

```text
raw candidate
      ↓
can_fuse
      ↓
legal / possible fusion candidate
```

## 16. can_fuse

`Scheduler.can_fuse(node1, node2)` 判断两个节点是否可以组合成一个 fused node。它通常同时包含：

### 正确性与合法性约束

- Dependency 是否允许；
- 是否会形成环；
- Device、Stream 是否兼容；
- Loop / Reduction 域是否兼容；
- Template 或特殊 Kernel 是否支持。

### 性能启发式

- Fusion Size；
- Memory Access 是否匹配；
- Peak Memory 风险；
- Horizontal Fusion 的共享数据收益；
- 图距离等。

因此 `can_fuse()` 不纯粹等于“语义上能否融合”，也可能包含“理论上可以，但启发式认为不值得”的判断。

建模接入时，应保留 correctness / legality 检查，仅替换或增强性能启发式。

## 17. `max_fusion_size = 64`

典型判断是：

```python
len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
```

对普通 SchedulerNode：

```text
get_nodes() → [self]
```

对：

```text
FusedSchedulerNode(A, B, C)
```

则：

```text
get_nodes() → [A, B, C]
```

因此 `max_fusion_size = 64` 更准确的解释是：

> 单个标准 fused scheduler group 最多包含约 64 个底层 SchedulerNode。

它不是：

- 64 个 FX Node；
- 64 个 ATen op；
- 64 个 Pointwise；
- 64 条后端指令。

该上限是性能启发式，不是正确性要求。

## 18. FusedSchedulerNode

当 Scheduler 决定融合 A 与 B，会创建一个代表融合组的 `FusedSchedulerNode`：

```text
FusedSchedulerNode
┌─────────────────┐
│ A               │
│ B               │
└─────────────────┘
```

它是编译期 Scheduler IR 中的逻辑容器，内部保存 constituent scheduler nodes。随着迭代，它可以继续变大。

## 19. FusedSchedulerNode 与 Fused Kernel

两者不是同一个对象：

```text
FusedSchedulerNode
    编译期 Scheduler IR
           │
           │ Codegen
           ▼
Fused Kernel
    后端可执行程序
```

普通 Pointwise/Reduction Fusion 中，可以暂时用“一个 FusedSchedulerNode 生成一个 fused kernel”建立直觉，但 Template、Extern Kernel、Foreach/Combo Kernel 等特殊路径可能打破简单的一一对应。

## 20. Fused Kernel 的收益

未融合时：

```text
Kernel 0:
  tmp0 = x + 1
  store tmp0 → HBM

Kernel 1:
  load tmp0 ← HBM
  tmp1 = relu(tmp0)
  store tmp1 → HBM

Kernel 2:
  load tmp1 ← HBM
  y = tmp1 * 2
  store y → HBM
```

融合后：

```text
Kernel fused_0:
  x = load(...)
  a = x + 1
  b = relu(a)
  y = b * 2
  store y
```

主要收益来源：

- 减少 Kernel Launch；
- 减少中间 Buffer 的 HBM 写回和重新读取；
- 中间值可能保留在寄存器、片上缓存或 SSA value 中；
- 可能提高 Cache Locality。

潜在代价包括寄存器压力、片上资源压力、Tiling 变差、并行度下降、编译时间增加和 Peak Memory 变化，因此并非越大越好。

## 21. score_fusion

当有多个合法候选时：

```text
A+B
A+C
B+C
C+D
```

`score_fusion()` 为它们排序。当前启发式通常会考虑：

- Fusion 类型；
- 预计节省的 Memory Operations；
- Shared Data / Buffer Overlap；
- 原图中的 Proximity。

先融合哪一对会改变后续图。因此 `score_fusion()` 往往比单独优化 `max_fusion_size` 更重要，也是 Learned Cost Model 的关键接入点。

## 22. Vertical Fusion 与 Horizontal Fusion

### Vertical Fusion

生产者与消费者融合：

```text
A
│
▼
B
```

典型收益是消除 A 到 B 的中间 Buffer HBM 写回与读取。

### Horizontal Fusion

两个节点没有直接 producer-consumer 关系，但共享输入或相似迭代域：

```text
       x
      / \
     A   B
```

例如：

```python
a = x.sin()
b = x.cos()
```

收益可能来自共享 Input Load、减少 Launch、改善 Cache Locality；代价可能来自更大的 Kernel、资源压力或 Peak Memory 增长。

Buffer Group 对发现 Horizontal Fusion 尤其重要。例如 `sin(x)`、`cos(x)`、`exp(x)` 都读取 `x`，会进入以 `x` 为键的同一候选组。

## 23. 四种容易混淆的 Group

| 名称 | 含义 |
|---|---|
| buffer group | 使用同一 Buffer 的节点集合，用于 Candidate Search |
| `node.group` | SchedulerNode 的 Device 与 Iteration Space / Loop Grouping 信息 |
| `GroupedSchedulerNode` | 一种特殊逻辑 Scheduler Container，通常不等同于普通可继续融合节点 |
| `FusedSchedulerNode` | 已经决定融合在一起的节点集合 |

尤其要记住：

```text
buffer group ≠ GroupedSchedulerNode
```

前者只是搜索索引结构，后者是具体的 Scheduler 容器类型。

## 24. 完整 Fusion 流程

```text
                 Scheduler Nodes
                        │
                        ▼
                 Buffer Analysis
                        │
                used_buffer_names()
                        │
                        ▼
              buffer_names_grouping
              bufA → [N1, N2, N4]
              bufB → [N2, N3]
                        │
                        ▼
                  Pair Generation
                        │
        max_fusion_buffer_group_pairwise_attempts
                        │
                        ▼
               Raw Fusion Candidate
                     (N1, N2)
                        │
                        ▼
                     can_fuse
                 ┌──────┴──────┐
                 │             │
               False          True
                 │             │
               Reject          ▼
                     Possible Fusions
                              │
                              ▼
                       score_fusion()
                              │
                              ▼
                     Candidate Ranking
                              │
                              ▼
                      speedup_by_fusion
                      （可选 Benchmark）
                              │
                              ▼
                            Fuse
                              │
                              ▼
                    FusedSchedulerNode
                        [N1, N2]
                              │
                              ▼
                       下一轮 Fusion
                              │
                              ▼
                    FusedSchedulerNode
                      [N1, N2, N3]
                              │
                              ▼
                           Codegen
                              │
                              ▼
                        Fused Kernel
```

## 25. 两个 64 的最终对照

| 配置 | 控制的问题 | 所处阶段 |
|---|---|---|
| `max_fusion_buffer_group_pairwise_attempts = 64` | “要不要尝试这对节点？” | Search Space / Candidate Generation |
| `max_fusion_size = 64` | “合并后的 fused scheduler group 是否太大？” | Fusion Group Size Heuristic |

实验时不要同时修改两者。

## 26. 面向 NPU 融合建模的五层数据模型

| 层级 | 核心对象 | 建模作用 |
|---|---|---|
| Graph 层 | FX Node、ATen op | 算子语义、原始模型结构 |
| IR 层 | TensorBox、StorageBox、Buffer、ComputedBuffer、Pointwise、Reduction | 运算内容、Shape、Layout |
| Dependency 层 | MemoryDep、ReadWrites | 数据依赖、Indexing、访存模式 |
| Scheduler 层 | SchedulerNode、SchedulerBuffer、FusedSchedulerNode | Candidate、Fusion 决策与顺序 |
| Kernel 层 | Generated Kernel、Tiling、资源占用、Runtime | NPU 真实性能标签 |

数据集字段名必须明确所指数值，例如：

```text
fx_node_count
aten_op_count
loop_ir_count
computed_buffer_count
scheduler_node_count
fused_scheduler_node_size
kernel_launch_count
```

不要用含义模糊的 `node_count`。

## 27. 一张可长期记忆的对象关系图

```text
SchedulerNode op0
│
├── group
├── read_writes
│   ├── reads:  MemoryDep(...)
│   └── writes: MemoryDep(...)
├── unmet_dependencies
├── users
│
└── node
     │
     ▼
ComputedBuffer buf0
│
├── name = buf0
├── layout
│   ├── device
│   ├── dtype
│   ├── size
│   └── stride
│
└── data
     │
     ▼
Pointwise / Reduction
│
├── device
├── dtype
├── ranges
│
└── inner_fn(index)
     ├── ops.load(...)
     ├── arithmetic ops
     └── return ...
```

最精炼的职责划分是：

| 对象 | 回答的问题 |
|---|---|
| Pointwise / Reduction | “怎么算？” |
| ComputedBuffer | “这个计算产生什么逻辑结果和 Layout？” |
| MemoryDep / ReadWrites | “读写什么、怎样访问？” |
| SchedulerNode | “这个任务怎样调度、能否与谁融合？” |
| FusedSchedulerNode | “Scheduler 已决定把哪些任务作为一组？” |
| Fused Kernel | “NPU 最终执行什么程序？” |
