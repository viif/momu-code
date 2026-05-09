import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


class BashToolInput(TypedDict):
    command: str


class ReadToolInput(TypedDict):
    path: str
    limit: NotRequired[int]


class WriteToolInput(TypedDict):
    path: str
    content: str


class EditToolInput(TypedDict):
    path: str
    old_text: str
    new_text: str


TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(cast(BashToolInput, kw)["command"]),
    "read_file": lambda **kw: run_read(
        cast(ReadToolInput, kw)["path"], cast(ReadToolInput, kw).get("limit")
    ),
    "write_file": lambda **kw: run_write(
        cast(WriteToolInput, kw)["path"], cast(WriteToolInput, kw)["content"]
    ),
    "edit_file": lambda **kw: run_edit(
        cast(EditToolInput, kw)["path"],
        cast(EditToolInput, kw)["old_text"],
        cast(EditToolInput, kw)["new_text"],
    ),
}


def iter_text_blocks(content: object) -> Iterator[str]:
    if not isinstance(content, list):
        return

    for block in content:
        if isinstance(block, dict):
            typed_block = cast(dict[str, object], block)
            if typed_block.get("type") == "text":
                yield cast(str, typed_block["text"])
        elif block.type == "text":
            yield block.text


def execute_tool_block(block: ToolUseBlock) -> ToolResultBlockParam:
    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
    print(f"> {block.name}:")
    print(output[:200])
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def create_response(messages: list[MessageParam]) -> Message:
    return client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=messages,
        tools=TOOLS,
        max_tokens=8000,
    )


# -- 核心模式：一个循环调用工具的 while 循环，直到模型停止 --
def agent_loop(messages: list[MessageParam]) -> None:
    while True:
        response = create_response(messages)
        # 追加助手的回复内容
        messages.append({"role": "assistant", "content": response.content})
        # 如果模型没有调用工具，说明任务结束
        if response.stop_reason != "tool_use":
            return
        # 执行每个工具调用，并收集结果
        results: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type == "tool_use":
                results.append(execute_tool_block(block))
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history: list[MessageParam] = []
    while True:
        try:
            query = input("\033[36muser >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for text in iter_text_blocks(history[-1]["content"]):
            print(text)
        print()
