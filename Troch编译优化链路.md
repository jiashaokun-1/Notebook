# 从 PyTorch 到 Triton kernel 的完整算子分解和融合路径
好，这一块我给你画一条**工程级真实路径**，尽量贴近 PyTorch + TorchInductor + Triton 的实际执行链路。

我用「**一条具体算子链 + 每一步发生什么**」来讲，比纯概念更有用。

---

# 🚀 一条完整路径（从 PyTorch → Triton kernel）

我们以一个典型例子：

```python
y = torch.nn.functional.layer_norm(x + bias)
```

---

# 🧭 总体路径（先给你全景图）

```text
PyTorch eager
   ↓
TorchDynamo（抓图）
   ↓
FX Graph（高阶 ATen）
   ↓
AOTAutograd（前后向拆分）
   ↓
core_aten_decompositions（关键！）
   ↓
低阶 ATen Graph（原子算子）
   ↓
Inductor IR（调度 / 融合）
   ↓
Fusion（生成 kernel 计划）
   ↓
Triton codegen
   ↓
GPU kernel 执行
```

---

# 🧩 Step 1：Eager → Graph 捕获

```python
y = layer_norm(x + bias)
```

进入 TorchDynamo 后：

👉 变成 FX Graph：

```text
add → layer_norm
```

---

# 🧩 Step 2：AOTAutograd（拆前后向）

在 AOTAutograd：

👉 forward / backward 被拆开

forward graph：

```text
add → layer_norm
```

backward graph：

```text
layer_norm_backward → add_backward
```

---

# 🔥 Step 3：core_aten_decompositions（核心步骤）

现在开始**真正关键的分解**

原来：

```text
layer_norm
```

👉 被拆成：

```text
mean
sub
pow
mean
add
rsqrt
mul
mul
add
```

完整链路（简化）：

```text
x + bias
   ↓
mean(x)
   ↓
x - mean
   ↓
(x - mean)^2
   ↓
var = mean(...)
   ↓
rsqrt(var + eps)
   ↓
normalize
   ↓
scale + shift
```

👉 此时图变成：

```text
add → mean → sub → pow → mean → add → rsqrt → mul → mul → add
```

---

# ⚙️ Step 4：变成“Core ATen IR”

现在所有算子都是：

👉 **小颗粒 primitive ops**

例如：

* `aten.add`
* `aten.mul`
* `aten.rsqrt`
* `aten.mean`

💡 这一步的意义：

> backend 不再需要理解 layer_norm，只需要理解这些基础 op

---

# 🧠 Step 5：Inductor IR（开始优化）

进入 TorchInductor

👉 做三件大事：

---

## ① Pattern 分析

识别：

```text
(x - mean) * rsqrt(var + eps)
```

👉 是 normalize 模式

---

## ② 内存访问分析

* 哪些 tensor 是：

  * elementwise
  * reduction（mean）

👉 分块策略（tile/block）

---

## ③ 构建调度（scheduler）

生成类似：

```text
for each block:
    load x
    compute mean
    compute var
    normalize
    write y
```

---

# 🔥 Step 6：Fusion（最关键优化）

此时会发生：

## 👉 多算子融合成一个 kernel

原始：

```text
add
mean
sub
pow
mean
rsqrt
mul
mul
add
```

👉 被融合成：

```text
🔥 一个 Triton kernel
```

---

### Fusion 原则：

#### ✅ 可以融合

* elementwise ops
* 同一 reduction 依赖链

#### ❌ 不可融合

* 不同 reduction 维度
* 有数据依赖冲突

---

# ⚡ Step 7：生成 Triton Kernel

进入 Triton codegen：

生成类似👇

```python
@triton.jit
def layer_norm_kernel(X, Y, ...):
    pid = tl.program_id(0)

    x = tl.load(...)
    
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    
    rstd = 1 / tl.sqrt(var + eps)
    
    y = (x - mean) * rstd
    y = y * weight + bias
    
    tl.store(Y, y)
```

💡 注意：

👉 这里已经是**融合后的完整逻辑**

---

# 🚀 Step 8：Kernel 执行（GPU）

最终：

* Triton → LLVM → PTX → CUDA kernel
* 在 GPU 上执行

---

# 📊 一次对比（你会更直观）

## ❌ 没 decomposition + fusion

```text
add kernel
layer_norm kernel（黑盒）
```

👉 多 kernel launch
👉 多次显存读写
👉 性能差

---

## ✅ 有 decomposition + fusion

```text
🔥 1 个 Triton kernel
```

👉 只读一次 x
👉 全部计算在寄存器完成
👉 显存访问最小化

---

# 🧠 本质总结（非常重要）

这条路径本质是：

---

## 🔹 Step A：语义降级（decomposition）

```text
复杂算子 → 原子算子
```

---

## 🔹 Step B：结构重组（fusion）

```text
原子算子 → 高效 kernel
```

---

👉 听起来像绕一圈，其实是：

> “先打碎，再重组，但这次是按硬件最优方式”

---

# 🎯 给你一个工程结论（重点）

如果你在做：

* Triton 算子优化
* Inductor hack
* AI 编译器

你要盯住三件事：

---

## 1️⃣ decomposition 是否合理

* 有没有拆太碎？
* 有没有破坏数值稳定？

---

## 2️⃣ fusion 是否成功

* 有没有被 graph break？
* 有没有多 kernel？

---

## 3️⃣ kernel 访存模式

* coalesced？
* tile 是否合理？

---

# 如果你想更深入（我可以带你做）

我可以帮你做一个更硬核的👇：

✅ **真实打印 Inductor graph + fusion 结果**
✅ 分析一个 attention（flash_attn）是怎么被拆 + 融合失败的
✅ 教你怎么“强行控制 decomposition / fusion”

直接说你想看哪一块（建议下一步看 attention，会很值🔥）



# Inductor graph + fusion 结果

好，这一步我直接带你做**“可落地调试”**：把 TorchInductor 的 **graph + fusion 结果真实打印出来**。你可以在自己环境里一跑就看到。

我会给你三层：

1️⃣ 最小可复现代码
2️⃣ 打印 FX graph / Inductor IR / Triton kernel
3️⃣ 怎么看 fusion 是否成功（重点）

---

# 🚀 一、最小示例（直接跑）

```python
import torch
import torch.nn.functional as F

torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

def fn(x, bias, weight):
    return F.layer_norm(x + bias, x.shape[-1:], weight)

x = torch.randn(1024, 1024, device="cuda")
bias = torch.randn(1024, device="cuda")
weight = torch.randn(1024, device="cuda")

compiled_fn = torch.compile(fn, backend="inductor")

y = compiled_fn(x, bias, weight)
```

---

# 🔍 二、打印 FX Graph（Dynamo 捕获）

在代码前加：

```python
import torch._dynamo as dynamo

dynamo.config.log_level = logging.DEBUG
dynamo.config.output_graph_code = True
```

你会看到类似：

```python
def forward(self, x, bias, weight):
    add = x + bias
    layer_norm = torch.ops.aten.layer_norm.default(add, ...)
    return layer_norm
```

👉 此时 **还没 decomposition**

---

# 🔥 三、打印 Decomposition 之后的 Graph（关键）

加这个：

```python
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True
```

运行后会在目录生成：

```bash
torch_compile_debug/
```

里面有：

```text
fx_graph_runnable.py
ir_pre_fusion.txt
ir_post_fusion.txt
output_code.py   ← ⭐ 最重要
```

---

# 🧩 四、看 decomposition 结果（ir_pre_fusion）

打开：

```text
ir_pre_fusion.txt
```

你会看到类似（简化版）：

```text
tmp0 = add(x, bias)
tmp1 = mean(tmp0)
tmp2 = sub(tmp0, tmp1)
tmp3 = mul(tmp2, tmp2)
tmp4 = mean(tmp3)
tmp5 = add(tmp4, eps)
tmp6 = rsqrt(tmp5)
tmp7 = mul(tmp2, tmp6)
tmp8 = mul(tmp7, weight)
```

👉 说明：

✅ `layer_norm` 已被拆掉
❗ 全是 core aten ops

---

# ⚡ 五、看 fusion 结果（ir_post_fusion）

打开：

```text
ir_post_fusion.txt
```

你会看到类似：

```text
fused_kernel_0:
    loads: x, bias, weight
    compute:
        mean
        var
        normalize
        scale
    stores: output
```

👉 如果你看到：

```text
fused_kernel_0
fused_kernel_1
```

❗ 说明 fusion **失败/不完全**

---

# 🔥 六、最关键：看 Triton Kernel（output_code.py）

打开：

```text
output_code.py
```

你会看到真实生成的 Triton：

```python
@triton.jit
def triton_kernel_0(X, BIAS, WEIGHT, OUT, ...):
    pid = tl.program_id(0)

    x = tl.load(...)
    bias = tl.load(...)

    x = x + bias

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean

    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    y = x_centered * rstd
    y = y * weight

    tl.store(OUT, y)
```

---

# 🧠 七、如何判断 fusion 成功（实战标准）

你重点看这三点👇

---

## ✅ 1. kernel 数量

```text
只有一个 triton_kernel_0
```

✔️ 完全融合

---

## ❌ 2. 是否出现中间 store/load

如果你看到：

```python
tmp = tl.store(...)
tmp = tl.load(...)
```

❗ 说明：

👉 中间结果写回显存 → fusion 断了

---

## ✅ 3. 是否跨 op 融合

比如：

```python
x = x + bias
...
y = x * weight
```

👉 add + norm + mul 全在一个 kernel

✔️ fusion 成功

---

# 🚨 八、常见 fusion 失败原因（你调优会用到）

---

## 1️⃣ reduction 维度不同

```python
mean(dim=1)
sum(dim=0)
```

👉 ❌ 很难融合

---

## 2️⃣ graph break（最常见）

比如：

```python
if x.shape[0] > 0:
```

👉 Dynamo break → fusion 断

---

## 3️⃣ 不支持的 op

比如：

* 自定义 CUDA op
* 某些 view/stride 操作

---

## 4️⃣ 内存布局问题

```python
x = x.transpose(...)
```

👉 stride 复杂 → fusion 失败

---

# 🧪 九、一个“强力调试技巧”（推荐你用）

加这个环境变量：

```bash
TORCH_LOGS="+inductor,+fusion"
```

或者：

```python
import os
os.environ["TORCH_LOGS"] = "+inductor,+fusion"
```

你会看到：

```text
Fusing node add + mul + rsqrt → fused_kernel_0
```

👉 直接看到 fusion 决策过程

---

# 🎯 十、给你一个工程级总结

你现在可以做到：

---

## ✔️ 看 decomposition

→ `ir_pre_fusion.txt`

---

## ✔️ 看 fusion

→ `ir_post_fusion.txt`

---

## ✔️ 看最终 kernel

→ `output_code.py`

---

## ✔️ 判断性能

→ kernel 数量 + memory load/store

---

# 如果你想继续深入（强烈建议🔥）

下一步可以做：

👉 我帮你分析一个你真实遇到的 case（比如你之前的 `flash_attn`）
👉 找出为什么 fusion 没发生
👉 甚至教你**手动干预 decomposition / fusion**

直接把你的模型 / kernel 贴出来，我可以帮你“拆到 Triton 级别” 👇


好，我们来一把**完整实战拆解（从 graph → decomposition → fusion → Triton）**。我选一个非常典型、又能体现问题的例子：

> 🔥 `softmax`（比 layer_norm 更容易看到“分解 + 融合”的全过程）

---

# 🧪 目标：我们要看 3 层东西

对同一个函数：

```python
def fn(x):
    return torch.softmax(x, dim=-1)
```

我们依次看：

---

## ✅ ① 原始 ATen Graph（make_fx）

## ✅ ② Decomposition 后 Graph

## ✅ ③ Inductor 融合 + Triton kernel

---

# 🚀 Step 0：准备代码

```python
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
```

```python
def fn(x):
    return torch.softmax(x, dim=-1)

x = torch.randn(128, 1024, device="cuda")
```

---

# 🔍 Step 1：看“原始 graph”（未 decomposition）

```python
gm = make_fx(fn)(x)
print(gm.code)
```

---

## 🧾 你会看到类似：

```python
def forward(self, x):
    softmax = torch.ops.aten.softmax.int(x, -1)
    return softmax
```

---

## 🧠 解释

👉 当前 graph：

```text
aten.softmax
```

✔️ 高阶算子
❗ 黑盒（还不能很好 fusion）

---

# 🔥 Step 2：看“decomposition 后 graph”

```python
decomp = get_decompositions([
    torch.ops.aten.softmax.int
])

gm = make_fx(fn, decomposition_table=decomp)(x)
print(gm.code)
```

---

## 🧾 你会看到类似（非常关键👇）：

```python
def forward(self, x):
    max_val = torch.ops.aten.amax.default(x, [-1], True)
    x_sub = torch.ops.aten.sub.Tensor(x, max_val)
    
    exp = torch.ops.aten.exp.default(x_sub)
    sum_exp = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    
    out = torch.ops.aten.div.Tensor(exp, sum_exp)
    return out
```

---

# 🧠 解释（重点！！！）

原来一个：

```text
softmax
```

👉 被拆成：

```text
amax   （数值稳定）
sub
exp
sum    （reduction）
div
```

---

## ⚠️ 注意一个关键点

👉 这里已经出现：

```text
reduction（sum / max）
```

这会**直接影响 fusion**

---

# ⚡ Step 3：进入 TorchInductor

```python
compiled_fn = torch.compile(fn, backend="inductor")
compiled_fn(x)
```

然后打开：

```bash
torch_compile_debug/
```

---

# 🔍 Step 4：看 fusion（ir_post_fusion.txt）

你会看到类似：

```text
fused_kernel_0  ← compute max
fused_kernel_1  ← compute exp + sum
fused_kernel_2  ← compute div
```

---

# 🧠 解释（非常关键）

为什么不是 1 个 kernel？

👉 因为：

## ❗ softmax 有“多阶段 reduction”

---

### 阶段 1：max（reduction）

```text
x → max
```

---

### 阶段 2：exp + sum（reduction）

```text
(x - max) → exp → sum
```

---

### 阶段 3：normalize

```text
exp / sum
```

---

👉 不同 reduction 阶段：

❌ 很难完全 fusion 成一个 kernel
（除非 specialized kernel，比如 flash attention）

---

# 🔥 Step 5：看 Triton kernel（output_code.py）

你会看到类似👇

---

## kernel 1（max）

```python
max_val = tl.max(x, axis=...)
```

---

## kernel 2（exp + sum）

```python
x = x - max
exp = tl.exp(x)
sum_exp = tl.sum(exp, axis=...)
```

---

## kernel 3（normalize）

```python
out = exp / sum_exp
```

---

# 📊 对比总结（三层变化）

---

## 🧩 层 1：原始 graph

```text
softmax
```

👉 ❌ 不可优化（黑盒）

---

## 🧩 层 2：decomposition

```text
amax → sub → exp → sum → div
```

👉 ✅ 可优化
👉 ❗ 引入 reduction 边界

---

## 🧩 层 3：fusion

```text
kernel 0: max
kernel 1: exp + sum
kernel 2: div
```

👉 ⚠️ 部分融合（不是一个 kernel）

---

# 🧠 最关键 insight（你做优化必须知道）

---

## 🔥 1️⃣ decomposition ≠ 一定更快

它只是：

```text
让优化成为可能
```

---

## 🔥 2️⃣ fusion 的“天花板”由 reduction 决定

| 类型            | fusion 能力 |
| ------------- | --------- |
| elementwise   | ✅ 全融合     |
| 单一 reduction  | ✅ 部分融合    |
| 多阶段 reduction | ❌ 很难完全融合  |

---

## 🔥 3️⃣ 为什么 flash_attn 更快？

因为它：

👉 **手写 kernel，把多阶段 reduction 融成一个 kernel**

（绕过 Inductor 的通用策略）

---

# 🎯 你现在已经掌握的能力

你现在可以：

✅ 看一个 op 是否被 decomposition
✅ 判断 fusion 是否成功
✅ 理解 kernel 为什么被拆开
✅ 识别性能瓶颈（reduction！）

---

# 🚀 下一步（强烈建议）

如果你愿意，我们可以直接进阶到你之前遇到的真实问题：

👉 `flash_attn` 为什么：

* 有时候不能被 Inductor fusion
* 或者 fallback / 报错

或者更狠一点👇

👉 我带你**手写一个 Triton softmax kernel**
对比 Inductor 生成的版本（直接看性能差距）

你选一个方向，我带你继续往下拆 🔥
