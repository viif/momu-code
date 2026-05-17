# momu-code

一个受 Learn Claude Code 启发的极简 Python Agent Harness 示例项目。

当前实现以单文件 `agent.py` 为主，但已经支持一套相对完整的 agent orchestration 能力，包括：

- 基础 bash / 文件读写 / 精确替换工具调用
- session 级 todo 跟踪
- 带依赖、owner 与 worktree 绑定的持久化任务系统
- Explore / general-purpose 两类 subagent 委派
- skill 按需加载
- micro compact、自动上下文压缩与 `/compact` 手动压缩
- 后台任务执行
- teammate 协作、空闲轮询、自动认领任务与 inbox 消息传递
- plan approval / shutdown 协议
- 基于 git worktree 的隔离执行、任务绑定与生命周期事件

## 安装依赖

```bash
uv sync
```

当前核心依赖：
- `anthropic`
- `python-dotenv`
- `pyyaml`

## 环境变量

程序会自动从 `.env` 加载环境变量。当前需要关注的变量有：

- `ANTHROPIC_API_KEY`
  - 必填，用于身份认证。
- `MODEL_ID`
  - 必填，用于指定调用的模型 ID。
- `ANTHROPIC_BASE_URL`
  - 选填，用于指定自定义 API Base URL。

可参考 `.env.example` 创建本地 `.env` 文件。

```bash
cp .env.example .env
```

## 运行方式

```bash
uv run python agent.py
```

REPL 内置命令：

- `/compact`：压缩当前对话历史，保留连续性摘要。
- `/team`：查看 teammate 列表与状态。
- `/inbox`：读取并清空 lead 收件箱。
- `/tasks`：查看持久化任务列表。

输入 `q`、`exit` 或空行会退出 REPL。

## 工具与协作模型

- lead agent 拥有完整工具集：基础工具、todo、持久化任务、后台任务、subagent、teammate 管理、消息与 worktree 管理。
- Explore subagent 只使用只读工具，适合代码探索和信息汇总。
- general-purpose subagent 可使用基础文件编辑工具，适合隔离的小范围修改。
- teammate 可使用基础工具、消息/审批工具、任务认领与 worktree 执行工具；进入 idle 后会轮询 inbox 和未认领任务。
- skill 会从 `skills/*/SKILL.md` 扫描 frontmatter，并在模型判断相关时通过 `load_skill` 按需加载。

当前内置 skill：

- `agent-builder`
- `code-review`
- `mcp-builder`
- `pdf`

## 目录说明

- `agent.py`：主程序入口，包含工具定义、主循环以及任务/队友/worktree 管理逻辑。
- `skills/`：按目录组织的技能定义，运行时会扫描 `SKILL.md` 并按需加载。
- `.tasks/`：持久化任务存储目录，由 harness 自动维护。
- `.team/`：teammate 配置与 inbox 数据目录，由 harness 自动维护。
- `.worktrees/`：worktree 索引、实际 worktree 目录与事件日志，由 harness 自动维护。

## 致谢

感谢 shareAI-lab 提供的 [《Learn Claude Code》](https://github.com/shareAI-lab/learn-claude-code) 教程。
