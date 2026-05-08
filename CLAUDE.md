# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

这是一个受 Learn Claude Code 启发的极简 Python Agent Harness。
当前仓库以单文件实现为主，修改时优先保持简单直接，避免过早抽象。

## 常用开发命令

- `uv sync`
  - 优先使用的依赖安装方式。
- `uv run python agent.py`
  - 启动交互式命令行 REPL。

当前仓库还没有提交测试框架、lint 配置或格式化配置，因此也没有项目内标准的单测运行命令。

## 环境变量

程序会自动从 `.env` 加载环境变量。当前需要关注的变量有：

- `MODEL_ID`
  - 必填，用于指定调用的模型 ID。
- `ANTHROPIC_API_KEY`
  - 直连 Anthropic 官方 API 时通常必填，用于身份认证。
- `ANTHROPIC_BASE_URL`
  - 选填，用于指定自定义 API Base URL。

可参考 `.env.example` 创建本地 `.env` 文件。

## 仓库协作约定

- 这是一个有意保持极简的 Harness，除非需求明显扩大，否则不要主动拆分为多模块结构。
- `CLAUDE.md` 只保留仓库级协作信息，不记录过多实现细节。
- 本项目统一使用 `uv` 管理依赖与运行。
- 如果后续引入测试、lint、格式化或新的开发命令，需同步更新本文件。

## 关键文件

- `agent.py`：当前主要实现入口。
- `pyproject.toml`：项目元数据与依赖定义。
- `uv.lock`：`uv` 锁文件。
- `README.md`：面向人类读者的项目说明。
- `.env.example`：本地环境变量示例。

## Git 提交规范

提交信息格式：`type: description`

常用 type：
- `feat`：新功能
- `fix`：缺陷修复
- `refactor`：重构（非功能变更、非缺陷修复）
- `chore`：构建、依赖、CI 等维护性改动
- `style`：代码格式调整（不影响逻辑）
- `docs`：文档变更
