# momu-code

一个极简的 Python Agent Harness 示例项目。

## 环境要求

- Python >= 3.13

## 安装依赖

```bash
uv sync
```

## 环境变量

程序会自动从 `.env` 加载环境变量。当前需要关注的变量有：

- `MODEL_ID`
  - 必填，用于指定调用的模型 ID。
- `ANTHROPIC_API_KEY`
  - 直连 Anthropic 官方 API 时通常必填，用于身份认证。
- `ANTHROPIC_BASE_URL`
  - 选填，用于指定自定义 API Base URL。

可参考 `.env.example` 创建本地 `.env` 文件。

## 运行方式

```bash
uv run python agent.py
```
