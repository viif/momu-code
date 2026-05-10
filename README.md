# momu-code

一个受 Learn Claude Code 启发的极简 Python Agent Harness 示例项目。

当前实现以单文件 `agent.py` 为主，但已经支持一套相对完整的 agent orchestration 能力，包括：

- 基础 bash / 文件工具调用
- session 级 todo 跟踪
- 持久化任务系统
- subagent 委派
- skill 按需加载
- 后台任务执行
- teammate 协作与 inbox 消息传递
- plan approval / shutdown 协议
- 基于 git worktree 的隔离执行

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

## 目录说明

- `agent.py`：主程序入口，包含工具定义、主循环以及任务/队友/worktree 管理逻辑。
- `skills/`：按目录组织的技能定义，运行时会扫描 `SKILL.md` 并按需加载。
- `.tasks/`：持久化任务存储目录，由 harness 自动维护。
- `.team/`：teammate 配置与 inbox 数据目录，由 harness 自动维护。
- `.worktrees/`：worktree 索引与事件目录，由 harness 自动维护。

## 致谢

感谢 shareAI-lab 提供的 [《Learn Claude Code》](https://github.com/shareAI-lab/learn-claude-code) 教程。
