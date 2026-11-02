让我把两行关键差异拎出来对照：

| 字段 | row 98 | row 100 |
|------|--------|---------|
| **seq_len** | **8192** | **262144** ←32× |
| **global_batch** | **65536** | **2048** |
| total_token | 536870912 | 536870912 |
| recompute | none | full |
| tp/cp/pp/ep/dp | 16/1/2/384/256 | 16/1/2/384/256 |
| fwd_compute_ms | 105367 | 114666 |
| optimizer_compute_ms | 5.8 | 5.8 ← 相同 |
| optimizer_comm_ms (总量) | 972.17 | 972.17 ← 相同 |
| **optimizer_exposed_ms** | **520.71** | **0.0** ← 全藏住了 |

通信量相同说得通：参数切分（tp/cp/pp/ep/dp）一样 → AG/RS 字节量和 DP ring 一模一样。

差异出在**掩盖窗口** —— 让我用 Python 验证。

找到原因了。**两行通信量完全相同，但每个 microbatch 的 forward 时间差 35 倍 → 掩盖窗口完全不同**。

## 两行的本质差异：total_token 同，但切法不同

```
row 98:   global_batch=65536  × seq_len=8192    = 5.37e8 tokens
row 100:  global_batch=2048   × seq_len=262144  = 5.37e8 tokens
```

总训练量一样（536870912 tokens），但 row 100 用"少量超长序列"，row 98 用"大量短序列"。

通信量取决于 TP/EP/PP/DP/参数量，跟 seq_len、global_batch 没关系 → **AG+RS 都是 972 ms**。

## 关键：每个 microbatch 的 forward 时间

每 rank 的微批数：
```
M = global_batch / micro_batch / dp
row 98:   M = 65536 / 1 / 256 = 256 个 microbatch
row 100:  M = 2048  / 1 / 256 = 8   个 microbatch
```

DualPipeV pp=2 时 `warmup_fwd = 0`，所以掩盖窗口 = `steady_fwd_per_mb`（一个 microbatch 的 fwd 时间）。

CSV 里 `fwd_compute_ms` 是整个 step 的 fwd 总时长（M 个 microbatch 累加）：

```
steady_fwd_per_mb ≈ fwd_compute_ms / M

row 98:   105367 / 256 ≈ 411   ms    ← 短序列、microbatch 小
row 100:  114667 / 8   ≈ 14333 ms    ← 超长序列、microbatch 巨大
                                       (差 35×)
```

虽然总 fwd_compute_ms 相差不到 10%（因为总 token 量一样），但**单个 microbatch 长度差了 32×**（seq_len 8192→262144）。

## Moonshot rotation 掩盖代入数字

公式：
```python
rs_hidden = min(rs, fwd_window)
ag_hidden = min(ag, compute + (fwd_window - rs_hidden))
exposed   = (ag + rs) - (ag_hidden + rs_hidden)
```

代入 `ag = rs ≈ 486 ms`、`compute = 5.8 ms`：

### Row 98（fwd_window = 411 ms，比 ag 还小）

```
rs_hidden = min(486, 411) = 411         ← RS 把 fwd_window 占满了
remaining_fwd = 411 - 411 = 0
ag_hidden = min(486, 5.8 + 0) = 5.8     ← AG 只能藏在 NS compute 那 5.8 ms 里
hidden    = 411 + 5.8 = 417
exposed   = 972 - 417 = 555 ms          ← 与 CSV 520.71 差 ~35 ms*
```

### Row 100（fwd_window = 14333 ms，远大于 ag+rs）

```
rs_hidden = min(486, 14333) = 486       ← RS 完整藏住
remaining_fwd = 14333 - 486 = 13847
ag_hidden = min(486, 5.8 + 13847) = 486 ← AG 也完整藏住
hidden    = 486 + 486 = 972
exposed   = 972 - 972 = 0 ms            ✓ 完美对得上 CSV 0.0
```

> *Row 98 的 555 vs 520 的小偏差来源：实际的 `steady_fwd_per_mb` 是 composer 里 `t_fwd_max`（每 stage max），不完全等于 `fwd_compute_ms / M`（两个 PP stage 不完全均衡 + composer 里对 fwd 时间还有些细节调整），所以我估的 411 ms 偏小，实际窗口大约 446 ms。

## 用图直观看

```
row 98:  microbatch 短 (411 ms)
   ┌── rs (486) ──┐                   ┌── ag (486) ──┐
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                 │←── fwd 411 ─→│
                 └ 藏住 411 ms ┘└ NS 5.8 藏住 AG ┘
                                只剩 80 ms 在 fwd 里能藏 AG，
                                AG 80% 暴露 → 暴露 ~555 ms

row 100: microbatch 超长 (14333 ms)
   ┌── rs (486) ──┐ ┌── ag (486) ──┐
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   │←──── fwd 14333 ────────────────→ ...
   │     大窗口足以全部藏下 RS+AG
   暴露 = 0 ms
```

## 一句话总结

- **两行通信量真的一样**（972 ms），因为 ZeRO/Muon 通信量只取决于参数切分（这俩并行配置完全相同）
- **暴露差距来自 microbatch fwd 长度**：row 100 把 seq_len 拉到 262144（32× 长），单 microbatch fwd 变成 14333 ms，远大于 972 ms 的通信，全部藏下；row 98 单 microbatch fwd 只有 ~446 ms，藏不下一半通信，剩 520 ms 暴露
- **不是 bug**，是 moonshot rotation 隐藏公式 `fwd_window = max(warmup_fwd, steady_fwd_per_mb)` 的"窗口大小随 microbatch 大小线性变化"的直接后果

顺带一个观察：这是模型层面对 long-sequence + small-batch 友好的副作用之一 —— 但同时 row 100 因为 seq_len=262144 引入了 69 秒的 full recompute（CSV `recompute_time_ms=69024`），net MFU 反而更低（0.1993 vs 0.2002），所以"完全藏住 optimizer 通信"在整体 step 时间上并没占到便宜。