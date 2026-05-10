import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    TextBlockParam,
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

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "Act, don't explain. Use the todo tool to plan multi-step tasks. "
    "Mark in_progress before starting and completed when done."
)


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


TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItemInput(TypedDict):
    id: str
    text: str
    status: TodoStatus


class TodoToolInput(TypedDict):
    items: list[TodoItemInput]


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
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]


class TodoManager:
    def __init__(self) -> None:
        self.items: list[TodoItemInput] = []

    def update(self, items: list[TodoItemInput]) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated: list[TodoItemInput] = []
        in_progress_count = 0
        for i, item in enumerate(items):
            item_id = str(item.get("id", str(i + 1)))
            text = str(item.get("text", "")).strip()
            status = cast(str, item.get("status", "pending")).lower()

            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            validated.append(
                {
                    "id": item_id,
                    "text": text,
                    "status": cast(TodoStatus, status),
                }
            )

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines: list[str] = []
        for item in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        done = sum(1 for item in self.items if item["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


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
    "todo": lambda **kw: TODO.update(cast(TodoToolInput, kw)["items"]),
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
    rounds_since_todo = 0
    while True:
        response = create_response(messages)
        # 追加助手的回复内容
        messages.append({"role": "assistant", "content": response.content})
        # 如果模型没有调用工具，说明任务结束
        if response.stop_reason != "tool_use":
            return
        # 执行每个工具调用，并收集结果
        results: list[ToolResultBlockParam | TextBlockParam] = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                results.append(execute_tool_block(block))
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append(
                {"type": "text", "text": "<reminder>Update your todos.</reminder>"}
            )
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
