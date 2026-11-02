`call_id` 是"模块前向调用实例编号"——同一个 nn.Module 每次被调用都拿到一个全新的递增 ID。它是上一轮修 row 36 时新加的字段，专门用来识别"哪几个 aten op 是同一次 forward 调用产生的"。

来源链路一共 5 跳：

```
ModuleTracker._call_counter / _call_id_stack
    │  (forward pre/post hook 维护)
    ▼
ModuleTracker.current_call_id          ← graph/tracker.py
    │
    ▼
RecordingDispatch 在每条 record 里写 "call_id" 字段     ← graph/dispatch.py
    │
    ▼
records_to_opgraph 把 record["call_id"] 塞进 OpNode    ← ir/adapter.py
    │
    ▼
OpNode.call_id (dataclass 字段)        ← ir/node.py
    │
    ▼
fusion/algorithm.py 的 bucket_into_groups 拿来切桶
```

---

## 1. 起点：`ModuleTracker` 维护栈

文件：[python/zrt/graph/tracker.py:9-46](python/zrt/graph/tracker.py)

模型加载后，`ModuleTracker(model)` 给每个子模块挂 forward pre/post hook。每次某个模块的 forward 触发，pre-hook 让一个全局计数器自增并入栈：

```python
class ModuleTracker:
    def __init__(self, root: nn.Module):
        self._stack: list[str] = []
        self._class_stack: list[str] = []
        ...
        self._call_counter: int = 0          # ← 全局递增计数
        self._call_id_stack: list[int] = []  # ← 当前嵌套 forward 的 call_id 栈
        self._registered_ids: set[int] = set()   # 防止 weight tying 时重复挂 hook
        self._install(root, "")

    def _install(self, module: nn.Module, prefix: str):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            ...
            if id(child) in self._registered_ids:
                self._install(child, full_name)
                continue                      # 同一个对象别名出现两次 → 跳过重复注册
            self._registered_ids.add(id(child))

            def _pre_hook(m, inp, _fn=full_name, _cls=class_name, _cls_obj=class_obj):
                self._stack.append(_fn)
                self._class_stack.append(_cls)
                self._class_obj_stack.append(_cls_obj)
                self._call_counter += 1                      # ★ 每次进 forward +1
                self._call_id_stack.append(self._call_counter)  # ★ 入栈
                self._forward_depth += 1

            def _post_hook(m, inp, out, _fn=full_name):
                if self._stack:           self._stack.pop()
                if self._class_stack:     self._class_stack.pop()
                if self._class_obj_stack: self._class_obj_stack.pop()
                if self._call_id_stack:   self._call_id_stack.pop()   # ★ 出栈
                self._forward_depth = max(0, self._forward_depth - 1)
            ...
            child.register_forward_pre_hook(_pre_hook)
            child.register_forward_hook(_post_hook)
```

栈顶就是"当前正在执行的 forward 调用实例"：

```python
@property
def current_call_id(self) -> int:
    return self._call_id_stack[-1] if self._call_id_stack else 0
```

### 举个具体例子

当 PyTorch 跑 `transformer.layers[0].attn.forward(...)` 时：

```
事件                                   _call_counter   _call_id_stack
────────────────────────────────────────────────────────────────────
attn 进 pre_hook                            1            [1]
  q_norm 进 pre_hook                        2            [1, 2]
  q_norm aten ops 触发                      2            [1, 2]   ← 这些 op call_id=2
  q_norm 出 post_hook                       2            [1]
  inline RMSNorm aten ops 触发(在 attn 体内) 2           [1]      ← 这些 op call_id=1
  wq_b 进 pre_hook                          3            [1, 3]
  wq_b aten ops                             3            [1, 3]   ← call_id=3
  wq_b 出 post_hook                         3            [1]
  ...
attn 出 post_hook                           N            []
```

注意：**attn 进出之间，控制权多次回到 attn 体内**（在子模块调用之间），那些 inline 算子全部拿到 `call_id=1`，但被切割在 `_call_id_stack` 顶部曾经是 `[1,2]` / `[1,3]` 的子调用之间。

---

## 2. 第二跳：dispatch 把当前 call_id 写到 record

文件：[python/zrt/graph/dispatch.py](python/zrt/graph/dispatch.py)（约 145-210 行）

`RecordingDispatch.__torch_dispatch__` 截每个 aten op，从 tracker 读 `current_call_id` 写进 record：

```python
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    out = func(*args, **kwargs)
    ...
    if not self.active:
        return out
    func_name = str(func.overloadpacket) + "." + func._overloadname

    module_path = ""
    module_class = ""
    call_id = 0
    if self._module_tracker:
        module_path  = self._module_tracker.current_module
        module_class = self._module_tracker.current_module_class
        call_id      = getattr(self._module_tracker, "current_call_id", 0)   # ★

        if getattr(self._module_tracker, '_in_backward_phase', False):
            pending = getattr(self._module_tracker, '_bwd_expected_pop', 0)
            if pending == 0:
                module_path = ""
                module_class = ""
                call_id = 0                   # ← 反向时无 backward hook 在跑就清零
    ...
    self.records.append({
        "node_id": len(self.records),
        "aten_op": func_name,
        "module_path": module_path,
        "module_class": module_class,
        "layer": extract_layer_idx(module_path),
        ...
        "call_id": call_id,                   # ★ 落到 record dict
    })
```

到这一步，`records: list[dict]` 里每条记录都有自己的 `call_id`。

---

## 3. 第三跳：record → OpNode

文件：[python/zrt/ir/adapter.py:138-155](python/zrt/ir/adapter.py)

`records_to_opgraph` 从 record 读出来塞进 `OpNode`：

```python
node = OpNode(
    id=node_id,
    op_type=op_type,
    inputs=inputs, outputs=outputs,
    scope=rec.get("module_path", ""),
    module_class=rec.get("module_class", ""),
    layer=str(rec.get("layer", "")),
    component=component,
    src_file=rec.get("src_file", ""),
    src_line=rec.get("src_line", 0),
    ...
    call_id=int(rec.get("call_id", 0) or 0),    # ★
)
```

而 `OpNode` 的字段定义在 [python/zrt/ir/node.py](python/zrt/ir/node.py)：

```python
@dataclass
class OpNode:
    id: str
    op_type: str
    ...
    src_file: str = ""
    src_line: int = 0
    src_code: str = ""
    call_id: int = 0     # ★ 新增字段，缺省 0 表示"无归属"
```

---

## 4. 第四跳：fusion 用它切桶

到了 `algorithm.py` 的 `bucket_nodes_by_leaf_module`：

```python
node_call_id = getattr(node, "call_id", 0) or 0
if node_call_id > 0:
    key = ("call", node_call_id)        # ★ 同 call_id 的连续 op 进同一个桶
else:
    key = ("legacy", node.scope, node.module_class, node.layer)  # 兜底
```

`getattr(node, "call_id", 0)` 写得保守是因为：
- torch.compile 路径走的是 `_compile_graph_to_records`（不经过 ModuleTracker hook），那条路径的 record 里没有 `call_id` 键，OpNode 的 `call_id` 取默认值 0。
- 老 fixture / 老测试 graph 里手工构造的 OpNode 也可能不带 `call_id`，`getattr` 让那条路径不至于崩。
- `or 0` 同时处理 `call_id` 字段被显式写成 None 的情况。

`call_id == 0` 的节点走 legacy 分支（按 scope/class/layer 切），保持向后兼容。

---

## `call_id` 解决了什么具体问题

回到 row 36 的场景。`Attention.forward` 内部代码大致是：

```python
def forward(self, x, ...):
    q = self.q_norm(self.wq_a(x))           # ← q_norm 是子模块（pre_hook 入栈，新 call_id）
    pow_x = x.pow(2)                        # ← 这几个 inline 算子在 attn 体内
    mean_x = pow_x.mean(-1, keepdim=True)
    rsqrt = (mean_x + eps).rsqrt()
    x = x * rsqrt
    x = x * self.weight
    kv = self.wkv(x)                        # ← wkv 是子模块（再次入栈）
    ...
```

dispatch 看到的 op 序列、对应的 call_id 长这样（假设 attn 这次调用的 call_id=10）：

```
op_type                       module_path                              call_id
aten.mm.default               transformer.layers.0.attn.wq_a            11
aten.pow.Tensor_Scalar        transformer.layers.0.attn.q_norm          12  ← q_norm
aten.mean.dim                 transformer.layers.0.attn.q_norm          12
aten.add.Tensor               transformer.layers.0.attn.q_norm          12
aten.rsqrt.default            transformer.layers.0.attn.q_norm          12
aten.mul.Tensor               transformer.layers.0.attn.q_norm          12
aten.mul.Tensor               transformer.layers.0.attn.q_norm          12
aten.pow.Tensor_Scalar        transformer.layers.0.attn                 10  ← inline RMSNorm
aten.mean.dim                 transformer.layers.0.attn                 10
aten.add.Tensor               transformer.layers.0.attn                 10
aten.rsqrt.default            transformer.layers.0.attn                 10
aten.mul.Tensor               transformer.layers.0.attn                 10
aten.mm.default               transformer.layers.0.attn.wkv             13
...
```

按 call_id 切桶得到：
- 桶 A：6 个 `q_norm` ops（call_id=12）→ 完整 q_norm 调用 → `is_full_forward=True` → 命中 RMSNorm rule，融合为 `rms_norm`。
- 桶 B：5 个 attn 体内 inline ops（call_id=10）→ 但 graph 里 call_id=10 的 op **总共 N 个**（attn 整次调用产生的所有 op），这 5 个只是其中一段 → `is_full_forward=False` → `Attention` 的 `class_only` 规则被拒绝 → 不融合，保留 raw aten。
- 桶 C：1 个 `wq_b mm` op（call_id=13）等等。

**没有 `call_id` 之前**：按 `(scope, module_class, layer)` 切，桶 B 的 5 个 inline op 与 attn scope 下其他 fragment 全部分到一个桶，scope=`attn` 又匹中 `Attention class_only` 规则，于是被错贴成 `Attention / sub=5`。

**有了 `call_id`**：每次 forward 都有自己的编号，能区分"完整一次调用"和"调用中间的某个片段"，class_only 规则只对完整调用生效。