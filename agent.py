"""极简 Python Agent Harness：支持基础工具调用、todo 跟踪、skill 按需加载与 subagent 委派。"""

import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import yaml
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
SKILLS_DIR = WORKDIR / "skills"


class SkillMeta(TypedDict):
    name: str
    description: str


class SkillRecord(TypedDict):
    meta: SkillMeta
    body: str
    path: str


# -- SkillLoader: 扫描 skills/<name>/SKILL.md 文件，并解析其 YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills: dict[str, SkillRecord] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for file_path in sorted(self.skills_dir.rglob("SKILL.md")):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            meta, body = self._parse_frontmatter(text)
            name = str(meta.get("name") or file_path.parent.name)
            description = str(meta.get("description") or "No description")
            self.skills[name] = {
                "meta": {"name": name, "description": description},
                "body": body,
                "path": str(file_path),
            }

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, object], str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text.strip()
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return cast(dict[str, object], meta), match.group(2).strip()

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        return "\n".join(
            f"  - {name}: {skill['meta']['description']}"
            for name, skill in self.skills.items()
        )

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            available = ", ".join(self.skills.keys()) or "(none)"
            return f"Error: Unknown skill '{name}'. Available: {available}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'


# Skills 第 1 层：注入到系统提示词中的技能元数据
def build_system_prompt(base: str) -> str:
    return (
        base
        + "Use load_skill to load specialized instructions only when relevant.\n\n"
        + f"Skills available:\n{SKILL_LOADER.get_descriptions()}"
    )


SKILL_LOADER = SkillLoader(SKILLS_DIR)
SYSTEM = build_system_prompt(
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "Act, don't explain. Use the todo tool to plan multi-step tasks. "
    "Use the task tool to delegate exploration or scoped subtasks. "
    "Mark in_progress before starting and completed when done. "
)
SUBAGENT_SYSTEM = build_system_prompt(
    f"You are a coding subagent at {WORKDIR}. Complete the given task "
    "with fresh context, then return a concise summary. "
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


class TaskToolInput(TypedDict):
    prompt: str
    description: NotRequired[str]


class LoadSkillToolInput(TypedDict):
    name: str


TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItemInput(TypedDict):
    id: str
    text: str
    status: TodoStatus


class TodoToolInput(TypedDict):
    items: list[TodoItemInput]


BASE_TOOLS: list[ToolParam] = [
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
        "name": "load_skill",
        "description": "Load specialized knowledge by skill name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to load",
                }
            },
            "required": ["name"],
        },
    },
]

TODO_TOOL: ToolParam = {
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
}

TASK_TOOL: ToolParam = {
    "name": "task",
    "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "description": {
                "type": "string",
                "description": "Short description of the task",
            },
        },
        "required": ["prompt"],
    },
}

# 父代理可用所有工具，子代理不可用 task 工具以避免无限递归
CHILD_TOOLS: list[ToolParam] = [*BASE_TOOLS]
PARENT_TOOLS: list[ToolParam] = [*BASE_TOOLS, TODO_TOOL, TASK_TOOL]


# -- 待办管理器：大语言模型写入的结构化状态 --
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


# -- 父代理与子代理共享的工具实现 --
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
            text=False,
            timeout=120,
        )
        stdout = r.stdout.decode("utf-8", errors="replace") if r.stdout else ""
        stderr = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
        out = (stdout + stderr).strip()
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


# -- 调度映射表：{工具名称: 处理函数} --
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
    # 技能加载工具：按需加载技能内容，避免一次性注入过多信息
    "load_skill": lambda **kw: SKILL_LOADER.get_content(
        cast(LoadSkillToolInput, kw)["name"]
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


# -- 子代理：全新上下文、过滤后的工具、仅返回摘要 --
def run_subagent(prompt: str) -> str:
    sub_messages: list[MessageParam] = [
        {"role": "user", "content": prompt}
    ]  # 全新上下文
    response: Message | None = None
    for _ in range(30):  # 避免死循环，最多调用工具30次
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"Unknown tool: {block.name}"
                )
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        sub_messages.append({"role": "user", "content": results})

    if response is None:
        return "(no summary)"
    # 返回所有文本块的内容拼接，子代理的最后输出应该是一个摘要文本块
    return "".join(iter_text_blocks(response.content)) or "(no summary)"


def execute_tool_block(block: ToolUseBlock) -> ToolResultBlockParam:
    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
    print(f"> {block.name}:")
    print(output[:200])
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def execute_task_block(block: ToolUseBlock) -> ToolResultBlockParam:
    task_input = cast(TaskToolInput, block.input)
    description = task_input.get("description", "subtask")
    prompt = task_input["prompt"]
    print(f"> task ({description}):")
    output = run_subagent(prompt)
    print(output[:200])
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def create_response(messages: list[MessageParam]) -> Message:
    return client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=messages,
        tools=PARENT_TOOLS,
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
                if block.name == "task":
                    # 任务工具需要特殊处理，调用子代理
                    results.append(execute_task_block(block))
                else:
                    results.append(execute_tool_block(block))
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            # 增加催促更新进度的提醒
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
