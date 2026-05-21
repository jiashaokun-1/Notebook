# Claude Code Worktree 隔离工作区

## 什么是 Worktree

Git Worktree 是 Git 的原生功能，允许在同一个仓库中同时检出多个分支到不同目录。Claude Code 对此做了深度集成，将其作为 **子 Agent 的隔离沙盒** 使用。

当你用 `isolation: "worktree"` 启动一个子 Agent 时，Claude Code 会：

1. 从当前 HEAD 自动创建一个临时 Git worktree（新目录 + 新分支）
2. 将子 Agent 的所有文件操作限制在该目录内
3. Agent 完成后，**如果没有文件变更，自动清理 worktree**；如果有变更，返回 worktree 路径和分支名

这意味着：主会话不受干扰，多个子 Agent 可以真正并行修改不同文件而不冲突。

---

## 核心价值

| 场景 | 不用 Worktree | 用 Worktree |
|---|---|---|
| 并行子 Agent 修改同一文件 | 冲突覆盖 | 各自独立分支，安全并行 |
| 探索性重构 | 污染主工作区 | 隔离试验，无副作用 |
| Code Review / 安全审计 | 可能误改代码 | 只读性探索，天然安全 |
| 大规模多步骤任务 | 中途失败难回滚 | 分支隔离，失败直接丢弃 |

---

## 使用方式

### 在 Agent 工具中启用

在调用 `Agent` 工具时，添加 `isolation: "worktree"` 参数：

```
Agent({
  description: "重构认证模块",
  isolation: "worktree",
  prompt: "将 auth/legacy.py 中的所有函数重构为类方法，保持接口不变。"
})
```

### 返回值

- **无变更**：worktree 自动清理，Agent 返回普通结果
- **有变更**：结果中包含 `worktree_path`（临时目录路径）和 `branch`（新分支名），用于后续合并

---

## 典型使用模式

### 1. 并行独立任务

同时在隔离环境中运行多个互不依赖的任务，主会话收集结果后统一决策：

```
# 同一条消息中发起两个并行 Agent
Agent({ isolation: "worktree", description: "前端单元测试", prompt: "为 components/Button 补全测试" })
Agent({ isolation: "worktree", description: "后端单元测试", prompt: "为 api/users.py 补全测试" })
```

两个 Agent 各自在独立分支上工作，互不干扰，完成后你决定合并哪个。

### 2. 探索性研究（只读）

让 Agent 深入探索代码库而不担心误修改：

```
Agent({
  isolation: "worktree",
  subagent_type: "Explore",
  description: "分析性能瓶颈",
  prompt: "分析 python/zrt/transform/ 下的热路径，找出 O(n²) 复杂度的函数并报告"
})
```

### 3. 危险操作沙盒

在 worktree 里试验破坏性修改，满意后再合并到主分支：

```
Agent({
  isolation: "worktree",
  description: "试验性 API 重设计",
  prompt: "将所有 REST 接口改为 GraphQL，确保测试通过后停止"
})
# 返回 branch 名，评估后决定是否 git merge
```

---

## 底层机制

```
主仓库 (.git/)
├── 主工作区 /Users/sky/Code/myproject/        ← 你正在工作的地方
└── 临时 worktree /tmp/claude-worktree-abc123/  ← Agent 的沙盒
    └── 指向新分支 claude/worktree-abc123
```

等价的手动 Git 命令：

```bash
git worktree add /tmp/claude-worktree-abc123 -b claude/worktree-abc123
# ... Agent 在此目录工作 ...
git worktree remove /tmp/claude-worktree-abc123  # 无变更时自动执行
```

---

## 注意事项

- **需要 Git 仓库**：worktree 依赖 Git，非 git 仓库无法使用
- **分支从当前 HEAD 创建**：子 Agent 看到的是触发时刻的代码快照，之后主分支的变更它感知不到
- **合并由你决定**：Claude Code 不会自动合并 worktree 的变更，你需要手动 `git merge` 或 `git cherry-pick`
- **并行有上限**：同时运行过多 worktree Agent 会占用大量磁盘和 API 并发，按需使用
- **子 Agent 从零开始**：worktree Agent 没有主会话的上下文，prompt 必须自包含，交代清楚背景

---

## 与普通子 Agent 的对比

```
# 普通 Agent（共享主工作区）
Agent({ description: "修复 bug", prompt: "..." })
→ 直接修改当前工作区文件，主会话可见

# Worktree Agent（隔离沙盒）
Agent({ isolation: "worktree", description: "修复 bug", prompt: "..." })
→ 在独立目录/分支操作，主工作区零影响
```

选择原则：**需要并行、需要隔离、需要可回滚** → 用 worktree；单步骤、顺序执行、结果需要立即用于下一步 → 用普通 Agent。

---

## 快速参考

```
isolation: "worktree"   # 在 Agent 工具中启用隔离
```

适用于：并行重构、探索审计、危险试验、大规模多步任务。
