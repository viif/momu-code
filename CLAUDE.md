# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

一个极简 Python Agent Harness（单文件 `agent.py`，约 2500 行），实现了类似 Claude Code 的 agent 编排运行时。刻意保持单文件结构——除非需求规模明显超出，否则不要拆分为多模块。

## 常用命令

```bash
uv sync                    # 安装依赖
uv run python agent.py     # 启动交互式 REPL
```

当前仓库没有配置测试框架、linter 或 formatter。

## REPL 内置命令

- `/compact` — 压缩对话历史为连续性摘要
- `/team` — 查看 teammate 列表与状态
- `/inbox` — 读取并清空 lead agent 收件箱
- `/tasks` — 查看所有持久化任务

## 环境变量（从 `.env` 加载）

- `MODEL_ID`（必填）— 传给 Anthropic client 的模型 ID
- `ANTHROPIC_API_KEY`（官方 API 必填）— 认证凭据
- `ANTHROPIC_BASE_URL`（选填）— 自定义 API Base URL；设置后 client 走此地址，同时清除 `ANTHROPIC_AUTH_TOKEN`

## 架构总览

### 管理器单例

运行时由模块级实例化的管理器类组成：

| 管理器 | 职责 |
|---------|------|
| `SkillLoader` | 扫描 `skills/*/SKILL.md`，解析 YAML frontmatter，按需加载 |
| `TodoManager` | session 级 todo 列表——模型写入结构化状态，harness 负责渲染 |
| `TaskManager` | 持久化任务，以 JSON 文件存在 `.tasks/`，支持依赖图（`blockedBy`）、owner、状态、worktree 绑定 |
| `EventBus` | 追加式生命周期事件日志，写入 `.worktrees/events.jsonl` |
| `WorktreeManager` | git worktree 的创建/列表/运行/移除，支持任务绑定，索引入口在 `.worktrees/` |
| `BackgroundManager` | 线程化 shell 执行，下一轮注入通知 |
| `MessageBus` | 基于 JSONL 的队友收件箱，位于 `.team/inbox/` |
| `TeammateManager` | 持久化队友配置（`.team/config.json`），spawn/状态/shutdown，自动认领任务 + idle 轮询循环 |

### 工具权限分层

工具定义为 `ToolParam` 字典，按权限范围组合：

- **Lead agent**（`LEAD_AGENT_TOOLS`）：全部工具——base、todo、tasks、background、subagent、teammate 管理、messaging、worktree
- **Explore 子代理**（`EXPLORE_SUBAGENT_TOOLS`）：`bash`、`read_file`、`load_skill`、`compact`——只读
- **General-purpose 子代理**（`GENERAL_SUBAGENT_TOOLS`）：base 工具（`bash`、`read_file`、`write_file`、`edit_file`、`load_skill`、`compact`）
- **Teammate**（`TEAMMATE_AGENT_TOOLS`）：base + 消息（send/read inbox、shutdown/plan 审批）+ 运行时（idle、claim_task）+ worktree 执行（status、run）

### 上下文压缩（三层）

1. **micro_compact** — 每次调用 LLM 前执行；将旧工具结果替换为 `[Compacted previous result from ...]` 占位符，保留最近 3 条结果，并豁免 `read_file`、`load_skill`、`todo`、`task_list`、`task_get`、`check_background`、`subagent` 等工具
2. **auto_compact** — Token 估算超过 50,000 时触发；调用模型生成全文摘要，然后替换全部消息
3. **手动 compact** — 由 `compact` 工具或 `/compact` 命令触发；机制同 auto_compact，可附带 focus 字符串

### 主循环（`agent_loop`）

核心模式是 `while True` 循环：
1. 排空收件箱消息 → 注入为 user content
2. 排空后台任务通知 → 注入为 user content
3. 执行 micro_compact
4. 若 token 估算超过阈值，执行 auto_compact
5. 使用 `LEAD_AGENT_TOOLS` 调用模型
6. 若 `stop_reason != "tool_use"`，返回（任务结束）
7. 执行每个工具调用（subagent 通过 `execute_subagent_block` 特殊处理）
8. 若 todo 连续 3 轮未更新且有未完成任务，注入提醒
9. 若调用了 `compact` 工具，执行 auto_compact

### 子代理执行

子代理获得全新上下文（不含对话历史）。运行相同的 `agent_loop`，使用受限工具集。最终 assistant 消息的文本内容作为工具结果返回。

### Teammate 系统

Teammate 以 daemon 线程运行各自的 agent loop。使用 `TEAMMATE_AGENT_TOOLS`，通过收件箱消息通信，支持：

- **Shutdown 协议**：lead 发出 `shutdown_request`，teammate 回复 `shutdown_response`；lead 的 tracker 通过 `request_id` 关联
- **Plan 审批**：teammate 通过 `plan_approval` 提交计划，落入 lead 收件箱（`plan_approval_response`）；lead 批准/拒绝
- **自动认领**：idle 时 teammate 轮询未认领的持久化任务
- **Idle 轮询**：发出 `idle` 后，harness 每 5 秒检查收件箱和可认领任务（60 秒超时）

### Worktree 隔离

Worktree 是在 `.worktrees/` 下管理的 git worktree。每个可绑定到持久化任务。生命周期事件（create/remove/bind）发送到事件总线。Teammate 可通过 `worktree_run` 在绑定的 worktree 内执行命令。

## 关键文件

- `agent.py` — 全部 harness（工具定义、管理器、agent loop、REPL）
- `skills/*/SKILL.md` — 技能定义（YAML frontmatter），按需加载
- `pyproject.toml` — 项目元数据与依赖（`anthropic`、`python-dotenv`、`pyyaml`）
- `.env.example` — 环境变量模板
- `.tasks/` — 持久化任务 JSON 存储（自动维护）
- `.team/` — teammate 配置与收件箱数据（自动维护）
- `.worktrees/` — worktree 索引、实际 git worktree 与事件日志（自动维护）

## 设计约束

- 保持单文件结构，不要过早提取模块
- 复用已有的管理器与工具分层，不要为一次性逻辑新增抽象
- 统一使用 `uv` 管理依赖与运行
- 提交信息格式：`type: description`（type 可取 `feat`、`fix`、`refactor`、`chore`、`style`、`docs`）