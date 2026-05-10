"""
极简 Python Agent Harness

支持：
- 基础 bash 工具调用
- todo 跟踪
- subagent 委派
- skill 按需加载
- 上下文压缩
- 持久化任务系统
- 后台任务
- 自治智能体团队
- 基于 worktree 的任务隔离
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
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


def detect_repo_root(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        root = Path(result.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"
TASKS_DIR = REPO_ROOT / ".tasks"
TEAM_DIR = REPO_ROOT / ".team"
WORKTREES_DIR = REPO_ROOT / ".worktrees"
INBOX_DIR = TEAM_DIR / "inbox"
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- 请求追踪器：通过 request_id 进行关联 --
shutdown_requests: dict[str, dict[str, object]] = {}
plan_requests: dict[str, dict[str, object]] = {}
_tracker_lock = threading.Lock()


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


SubagentAgentType = Literal["Explore", "general-purpose"]


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
    "Act, don't explain. Use the todo tool to plan current session work. "
    "Use task_* tools to manage persistent tasks across conversations. "
    "For parallel or risky changes, create tasks and use worktree_* tools to isolate execution lanes when needed. "
    "Use background_run for long-running shell commands and check_background to inspect them. "
    "Background task completions are delivered in <background-results>. "
    "Use the subagent tool to delegate exploration or scoped subtasks. "
    "Use agent_type='Explore' for read-only research and agent_type='general-purpose' only when the subagent must edit files. "
    "Use spawn_teammate to create named teammates and use send_message/read_inbox to coordinate through inboxes. "
    "Manage shutdown and plan approval workflows for teammates when needed. "
    "Teammate messages arrive in <inbox>. "
    "Mark in_progress before starting and completed when done. "
)


def build_subagent_system(agent_type: SubagentAgentType) -> str:
    if agent_type == "general-purpose":
        mode_note = "You may edit files when needed, but keep changes minimal and scoped to the task. "
    else:
        mode_note = "Stay read-only: investigate, read files, and summarize findings without modifying files. "
    return build_system_prompt(
        f"You are a coding subagent at {WORKDIR}. Complete the given task "
        "with fresh context, then return a concise summary. " + mode_note
    )


TEAMMATE_SYSTEM = build_system_prompt(
    f"You are a coding teammate at {WORKDIR}. Complete the assigned task, "
    "use send_message to report back to lead or other teammates, and check read_inbox for follow-up messages. "
    "Submit plans via plan_approval before major work and respond to shutdown_request with shutdown_response. "
    "Use worktree_run/worktree_status when a task is assigned to a worktree lane. "
    "Use idle when you have no more work; the harness will poll for inbox messages and auto-claim new tasks for you. "
)
COMPACT_THRESHOLD = 50000
COMPACT_KEEP_RECENT_RESULTS = 3
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
COMPACT_PRESERVE_RESULT_TOOLS = {
    "read_file",
    "load_skill",
    "todo",
    "task_list",
    "task_get",
    "check_background",
    "subagent",
}
ENABLE_SUBAGENT_AUTO_COMPACT = False


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


class BackgroundRunToolInput(TypedDict):
    command: str


class CheckBackgroundToolInput(TypedDict):
    task_id: NotRequired[str]


class SubagentToolInput(TypedDict):
    prompt: str
    description: NotRequired[str]
    agent_type: NotRequired[SubagentAgentType]


class TaskCreateToolInput(TypedDict):
    subject: str
    description: NotRequired[str]


class TaskUpdateToolInput(TypedDict):
    task_id: int
    status: NotRequired[Literal["pending", "in_progress", "completed"]]
    addBlockedBy: NotRequired[list[int]]
    removeBlockedBy: NotRequired[list[int]]


class TaskGetToolInput(TypedDict):
    task_id: int


class LoadSkillToolInput(TypedDict):
    name: str


class CompactToolInput(TypedDict):
    focus: NotRequired[str]


class SpawnTeammateToolInput(TypedDict):
    name: str
    role: str
    prompt: str


class SendMessageToolInput(TypedDict):
    to: str
    content: str
    msg_type: NotRequired[str]


class BroadcastToolInput(TypedDict):
    content: str


class ShutdownRequestToolInput(TypedDict):
    teammate: str


class ShutdownResponseToolInput(TypedDict):
    request_id: str
    approve: NotRequired[bool]
    reason: NotRequired[str]


class PlanApprovalToolInput(TypedDict):
    request_id: NotRequired[str]
    approve: NotRequired[bool]
    feedback: NotRequired[str]
    plan: NotRequired[str]


class ReadInboxToolInput(TypedDict):
    pass


class TaskBindWorktreeToolInput(TypedDict):
    task_id: int
    worktree: str


class WorktreeCreateToolInput(TypedDict):
    name: str
    task_id: NotRequired[int]
    base_ref: NotRequired[str]


class WorktreeNameToolInput(TypedDict):
    name: str


class WorktreeRunToolInput(TypedDict):
    name: str
    command: str


class WorktreeRemoveToolInput(TypedDict):
    name: str
    force: NotRequired[bool]
    complete_task: NotRequired[bool]


class WorktreeEventsToolInput(TypedDict):
    limit: NotRequired[int]


class ClaimTaskToolInput(TypedDict):
    task_id: int


TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItemInput(TypedDict):
    id: str
    text: str
    status: TodoStatus
    activeForm: NotRequired[str]


class TodoToolInput(TypedDict):
    items: list[TodoItemInput]


COMPACT_TOOL: ToolParam = {
    "name": "compact",
    "description": "Compress the conversation context and keep only a continuity summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "What to preserve in the summary",
            }
        },
    },
}

BASH_TOOL: ToolParam = {
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

READ_FILE_TOOL: ToolParam = {
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
}

WRITE_FILE_TOOL: ToolParam = {
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
}

EDIT_FILE_TOOL: ToolParam = {
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
}

LOAD_SKILL_TOOL: ToolParam = {
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
}

TODO_TOOL: ToolParam = {
    "name": "todo",
    "description": "Update the session todo list for current execution progress.",
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
                        "activeForm": {
                            "type": "string",
                            "description": "Present continuous status for the in_progress item",
                        },
                    },
                    "required": ["id", "text", "status"],
                },
            }
        },
        "required": ["items"],
    },
}

TASK_CREATE_TOOL: ToolParam = {
    "name": "task_create",
    "description": "Create a persistent task that survives compression and restarts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["subject"],
    },
}

TASK_UPDATE_TOOL: ToolParam = {
    "name": "task_update",
    "description": "Update a persistent task status or dependencies.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
            },
            "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
            "removeBlockedBy": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["task_id"],
    },
}

TASK_LIST_TOOL: ToolParam = {
    "name": "task_list",
    "description": "List all persistent tasks with status summary.",
    "input_schema": {"type": "object", "properties": {}},
}

TASK_GET_TOOL: ToolParam = {
    "name": "task_get",
    "description": "Get full details of a persistent task by ID.",
    "input_schema": {
        "type": "object",
        "properties": {"task_id": {"type": "integer"}},
        "required": ["task_id"],
    },
}

BACKGROUND_RUN_TOOL: ToolParam = {
    "name": "background_run",
    "description": "Run a shell command in a background thread and return a task ID immediately.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

CHECK_BACKGROUND_TOOL: ToolParam = {
    "name": "check_background",
    "description": "Check one background task, or list all background tasks.",
    "input_schema": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
    },
}

SUBAGENT_TOOL: ToolParam = {
    "name": "subagent",
    "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "description": {
                "type": "string",
                "description": "Short description of the task",
            },
            "agent_type": {
                "type": "string",
                "enum": ["Explore", "general-purpose"],
                "description": "Explore is read-only. general-purpose can edit files.",
            },
        },
        "required": ["prompt"],
    },
}

SPAWN_TEAMMATE_TOOL: ToolParam = {
    "name": "spawn_teammate",
    "description": "Spawn a named teammate that can report back through inbox messages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "prompt": {"type": "string"},
        },
        "required": ["name", "role", "prompt"],
    },
}

LIST_TEAMMATES_TOOL: ToolParam = {
    "name": "list_teammates",
    "description": "List all teammates with their role and status.",
    "input_schema": {"type": "object", "properties": {}},
}

SEND_MESSAGE_TOOL: ToolParam = {
    "name": "send_message",
    "description": "Send a message to a teammate inbox.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "content": {"type": "string"},
            "msg_type": {"type": "string", "enum": sorted(VALID_MSG_TYPES)},
        },
        "required": ["to", "content"],
    },
}

READ_INBOX_TOOL: ToolParam = {
    "name": "read_inbox",
    "description": "Read and drain your inbox.",
    "input_schema": {"type": "object", "properties": {}},
}

BROADCAST_TOOL: ToolParam = {
    "name": "broadcast",
    "description": "Send a message to all teammates.",
    "input_schema": {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
}

SHUTDOWN_REQUEST_TOOL: ToolParam = {
    "name": "shutdown_request",
    "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
    "input_schema": {
        "type": "object",
        "properties": {"teammate": {"type": "string"}},
        "required": ["teammate"],
    },
}

SHUTDOWN_RESPONSE_TOOL: ToolParam = {
    "name": "shutdown_response",
    "description": "Respond to or check the status of a shutdown request by request_id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "approve": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["request_id"],
    },
}

PLAN_APPROVAL_TOOL: ToolParam = {
    "name": "plan_approval",
    "description": "Submit a plan for review, or approve/reject a teammate plan request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "approve": {"type": "boolean"},
            "feedback": {"type": "string"},
            "plan": {"type": "string"},
        },
    },
}

IDLE_TOOL: ToolParam = {
    "name": "idle",
    "description": "Signal that you have no more work and enter idle polling mode.",
    "input_schema": {"type": "object", "properties": {}},
}

TASK_BIND_WORKTREE_TOOL: ToolParam = {
    "name": "task_bind_worktree",
    "description": "Bind a persistent task to a worktree lane.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "worktree": {"type": "string"},
        },
        "required": ["task_id", "worktree"],
    },
}

WORKTREE_CREATE_TOOL: ToolParam = {
    "name": "worktree_create",
    "description": "Create a git worktree and optionally bind it to a task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "task_id": {"type": "integer"},
            "base_ref": {"type": "string"},
        },
        "required": ["name"],
    },
}

WORKTREE_LIST_TOOL: ToolParam = {
    "name": "worktree_list",
    "description": "List tracked worktrees.",
    "input_schema": {"type": "object", "properties": {}},
}

WORKTREE_STATUS_TOOL: ToolParam = {
    "name": "worktree_status",
    "description": "Show git status for one worktree.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}

WORKTREE_RUN_TOOL: ToolParam = {
    "name": "worktree_run",
    "description": "Run a shell command in a named worktree.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "command": {"type": "string"},
        },
        "required": ["name", "command"],
    },
}

WORKTREE_KEEP_TOOL: ToolParam = {
    "name": "worktree_keep",
    "description": "Mark a worktree as kept without removing it.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}

WORKTREE_REMOVE_TOOL: ToolParam = {
    "name": "worktree_remove",
    "description": "Remove a worktree and optionally complete its bound task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "force": {"type": "boolean"},
            "complete_task": {"type": "boolean"},
        },
        "required": ["name"],
    },
}

WORKTREE_EVENTS_TOOL: ToolParam = {
    "name": "worktree_events",
    "description": "List recent worktree lifecycle events.",
    "input_schema": {
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    },
}

CLAIM_TASK_TOOL: ToolParam = {
    "name": "claim_task",
    "description": "Claim a persistent task by ID and mark it in progress.",
    "input_schema": {
        "type": "object",
        "properties": {"task_id": {"type": "integer"}},
        "required": ["task_id"],
    },
}

FILE_TOOLS: list[ToolParam] = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
]

TASK_TOOLS: list[ToolParam] = [
    TASK_CREATE_TOOL,
    TASK_UPDATE_TOOL,
    TASK_LIST_TOOL,
    TASK_GET_TOOL,
]

BACKGROUND_TOOLS: list[ToolParam] = [
    BACKGROUND_RUN_TOOL,
    CHECK_BACKGROUND_TOOL,
]

MESSAGE_TOOLS: list[ToolParam] = [
    SEND_MESSAGE_TOOL,
    READ_INBOX_TOOL,
    BROADCAST_TOOL,
    SHUTDOWN_REQUEST_TOOL,
    SHUTDOWN_RESPONSE_TOOL,
    PLAN_APPROVAL_TOOL,
]

TEAMMATE_MESSAGE_TOOLS: list[ToolParam] = [
    SEND_MESSAGE_TOOL,
    READ_INBOX_TOOL,
    SHUTDOWN_RESPONSE_TOOL,
    PLAN_APPROVAL_TOOL,
]

TEAMMATE_RUNTIME_TOOLS: list[ToolParam] = [
    IDLE_TOOL,
    CLAIM_TASK_TOOL,
]

TEAMMATE_MANAGEMENT_TOOLS: list[ToolParam] = [
    SPAWN_TEAMMATE_TOOL,
    LIST_TEAMMATES_TOOL,
    *TEAMMATE_RUNTIME_TOOLS,
]

WORKTREE_TOOLS: list[ToolParam] = [
    TASK_BIND_WORKTREE_TOOL,
    WORKTREE_CREATE_TOOL,
    WORKTREE_LIST_TOOL,
    WORKTREE_STATUS_TOOL,
    WORKTREE_RUN_TOOL,
    WORKTREE_KEEP_TOOL,
    WORKTREE_REMOVE_TOOL,
    WORKTREE_EVENTS_TOOL,
]

TEAMMATE_WORKTREE_TOOLS: list[ToolParam] = [
    WORKTREE_STATUS_TOOL,
    WORKTREE_RUN_TOOL,
]

BASE_TOOLS: list[ToolParam] = [
    COMPACT_TOOL,
    BASH_TOOL,
    *FILE_TOOLS,
    LOAD_SKILL_TOOL,
]

# Explore 子代理可用的只读工具：允许压缩、bash、读文件、加载 skill，不允许写文件或进入任务/协作工具。
EXPLORE_SUBAGENT_TOOLS: list[ToolParam] = [
    COMPACT_TOOL,
    BASH_TOOL,
    READ_FILE_TOOL,
    LOAD_SKILL_TOOL,
]

# 通用子代理可用的基础工具：在 Explore 基础上允许写文件与编辑文件，但仍不暴露任务/协作工具。
GENERAL_SUBAGENT_TOOLS: list[ToolParam] = [*BASE_TOOLS]

# teammate 可用工具：基础工具 + 队友间消息/审批 + 运行时工具 + worktree 执行工具。
TEAMMATE_AGENT_TOOLS: list[ToolParam] = [
    *BASE_TOOLS,
    *TEAMMATE_MESSAGE_TOOLS,
    *TEAMMATE_RUNTIME_TOOLS,
    *TEAMMATE_WORKTREE_TOOLS,
]

# lead 主 agent 可用工具：完整基础工具集 + todo/task/background/subagent + teammate/worktree 管理工具。
LEAD_AGENT_TOOLS: list[ToolParam] = [
    *BASE_TOOLS,
    TODO_TOOL,
    *TASK_TOOLS,
    *BACKGROUND_TOOLS,
    SUBAGENT_TOOL,
    *TEAMMATE_MANAGEMENT_TOOLS,
    *MESSAGE_TOOLS,
    *WORKTREE_TOOLS,
]


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
            active_form = str(item.get("activeForm", "")).strip()

            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
                if not active_form:
                    active_form = text

            validated_item: TodoItemInput = {
                "id": item_id,
                "text": text,
                "status": cast(TodoStatus, status),
            }
            if active_form:
                validated_item["activeForm"] = active_form
            validated.append(validated_item)

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
            suffix = ""
            active_form = item.get("activeForm", "")
            if item["status"] == "in_progress" and active_form:
                suffix = f" <- {active_form}"
            lines.append(f"{marker} #{item['id']}: {item['text']}{suffix}")

        done = sum(1 for item in self.items if item["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return any(item["status"] != "completed" for item in self.items)


# -- 任务管理器：带有依赖图的增删改查，以 JSON 文件形式持久化，支持可选工作区绑定 --
class TaskManager:
    def __init__(self, tasks_dir: Path) -> None:
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1
        self._claim_lock = threading.Lock()

    def _max_id(self) -> int:
        ids = [
            int(file_path.stem.split("_")[1])
            for file_path in self.dir.glob("task_*.json")
        ]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict[str, object]:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        task = cast(
            dict[str, object],
            json.loads(path.read_text(encoding="utf-8", errors="replace")),
        )
        task.setdefault("blockedBy", [])
        task.setdefault("owner", "")
        task.setdefault("worktree", "")
        created_at = task.get("created_at")
        if not isinstance(created_at, (int, float)):
            created_at = time.time()
            task["created_at"] = created_at
        updated_at = task.get("updated_at")
        if not isinstance(updated_at, (int, float)):
            task["updated_at"] = created_at
        return task

    def _save(self, task: dict[str, object]) -> None:
        now = time.time()
        if not isinstance(task.get("created_at"), (int, float)):
            task["created_at"] = now
        task["updated_at"] = now
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(
            json.dumps(task, indent=2, ensure_ascii=False),
            encoding="utf-8",
            errors="replace",
        )

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "owner": "",
            "worktree": "",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def exists(self, task_id: int) -> bool:
        return (self.dir / f"task_{task_id}.json").exists()

    def update(
        self,
        task_id: int,
        status: str | None = None,
        add_blocked_by: list[int] | None = None,
        remove_blocked_by: list[int] | None = None,
    ) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            blocked_by = cast(list[int], task["blockedBy"])
            task["blockedBy"] = list(set(blocked_by + add_blocked_by))
        if remove_blocked_by:
            blocked_by = cast(list[int], task["blockedBy"])
            task["blockedBy"] = [x for x in blocked_by if x not in remove_blocked_by]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task.get("status") == "pending":
            task["status"] = "in_progress"
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    # -- 任务面板扫描 --
    def scan_unclaimed(self) -> list[dict[str, object]]:
        tasks: list[dict[str, object]] = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda file_path: int(file_path.stem.split("_")[1]),
        )
        for file_path in files:
            task = cast(
                dict[str, object],
                json.loads(file_path.read_text(encoding="utf-8", errors="replace")),
            )
            if (
                task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")
            ):
                tasks.append(task)
        return tasks

    def claim(self, task_id: int, owner: str) -> str:
        with self._claim_lock:
            task = self._load(task_id)
            existing_owner = cast(str, task.get("owner", ""))
            if existing_owner:
                return f"Error: Task {task_id} has already been claimed by {existing_owner}"
            status = cast(str, task.get("status", "pending"))
            if status != "pending":
                return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
            if task.get("blockedBy"):
                return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
            task["owner"] = owner
            task["status"] = "in_progress"
            self._save(task)
        return f"Claimed task #{task_id} for {owner}"

    def _clear_dependency(self, completed_id: int) -> None:
        for file_path in self.dir.glob("task_*.json"):
            task = cast(
                dict[str, object],
                json.loads(file_path.read_text(encoding="utf-8", errors="replace")),
            )
            blocked_by = cast(list[int], task.get("blockedBy", []))
            if completed_id in blocked_by:
                blocked_by.remove(completed_id)
                task["blockedBy"] = blocked_by
                self._save(task)

    def list_all(self) -> str:
        tasks: list[dict[str, object]] = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda file_path: int(file_path.stem.split("_")[1]),
        )
        for file_path in files:
            tasks.append(cast(dict[str, object], json.loads(file_path.read_text())))
        if not tasks:
            return "No tasks."
        lines: list[str] = []
        for task in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(cast(str, task["status"]), "[?]")
            blocked = (
                f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            )
            owner = f" @{task['owner']}" if task.get("owner") else ""
            worktree = f" wt={task['worktree']}" if task.get("worktree") else ""
            lines.append(
                f"{marker} #{task['id']}: {task['subject']}{owner}{worktree}{blocked}"
            )
        return "\n".join(lines)


# -- 事件总线：用于可观测性的仅追加生命周期事件 --
class EventBus:
    def __init__(self, event_log_path: Path) -> None:
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8", errors="replace")

    def emit(
        self,
        event: str,
        task: Mapping[str, object] | None = None,
        worktree: Mapping[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        size = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = lines[-size:]
        items: list[dict[str, object]] = []
        for line in recent:
            try:
                items.append(cast(dict[str, object], json.loads(line)))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2, ensure_ascii=False)


# -- 工作区管理器：Git 工作区的创建/列表/运行/移除 + 生命周期索引 --
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus) -> None:
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps({"worktrees": []}, indent=2, ensure_ascii=False),
                encoding="utf-8",
                errors="replace",
            )
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=False,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=False,
            timeout=120,
        )
        output = decode_process_output(result)
        if result.returncode != 0:
            raise RuntimeError(output or f"git {' '.join(args)} failed")
        return output or "(no output)"

    def _load_index(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(self.index_path.read_text(encoding="utf-8", errors="replace")),
        )

    def _save_index(self, data: dict[str, object]) -> None:
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
            errors="replace",
        )

    def _find(self, name: str) -> dict[str, object] | None:
        idx = self._load_index()
        for worktree in cast(list[dict[str, object]], idx.get("worktrees", [])):
            if worktree.get("name") == name:
                return worktree
        return None

    def _validate_name(self, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    def create(
        self, name: str, task_id: int | None = None, base_ref: str = "HEAD"
    ) -> str:
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"
        task_ref = {"id": task_id} if task_id is not None else {}
        self.events.emit(
            "worktree.create.before",
            task=task_ref,
            worktree={"name": name, "base_ref": base_ref},
        )
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])
            entry: dict[str, object] = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }
            idx = self._load_index()
            worktrees = cast(list[dict[str, object]], idx.setdefault("worktrees", []))
            worktrees.append(entry)
            self._save_index(idx)
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)
            self.events.emit(
                "worktree.create.after",
                task=task_ref,
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            return json.dumps(entry, indent=2, ensure_ascii=False)
        except Exception as e:
            self.events.emit(
                "worktree.create.failed",
                task=task_ref,
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        idx = self._load_index()
        worktrees = cast(list[dict[str, object]], idx.get("worktrees", []))
        if not worktrees:
            return "No worktrees in index."
        lines: list[str] = []
        for worktree in worktrees:
            suffix = (
                f" task={worktree['task_id']}"
                if worktree.get("task_id") is not None
                else ""
            )
            lines.append(
                f"[{worktree.get('status', 'unknown')}] {worktree['name']} -> "
                f"{worktree['path']} ({worktree.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        worktree = self._find(name)
        if not worktree:
            return f"Error: Unknown worktree '{name}'"
        path = Path(cast(str, worktree["path"]))
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=False,
            timeout=60,
        )
        output = decode_process_output(result)
        return output or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(item in command for item in dangerous):
            return "Error: Dangerous command blocked"

        worktree = self._find(name)
        if not worktree:
            return f"Error: Unknown worktree '{name}'"
        path = Path(cast(str, worktree["path"]))
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=False,
                timeout=300,
            )
            output = decode_process_output(result)
            return output[:50000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"
        except (FileNotFoundError, OSError) as e:
            return f"Error: {e}"

    def remove(
        self, name: str, force: bool = False, complete_task: bool = False
    ) -> str:
        worktree = self._find(name)
        if not worktree:
            return f"Error: Unknown worktree '{name}'"

        task_id = worktree.get("task_id")
        task_ref = {"id": task_id} if task_id is not None else {}
        self.events.emit(
            "worktree.remove.before",
            task=task_ref,
            worktree={"name": name, "path": worktree.get("path")},
        )
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(cast(str, worktree["path"]))
            self._run_git(args)

            if complete_task and isinstance(task_id, int):
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                self.events.emit(
                    "task.completed",
                    task={"id": task_id, "status": "completed"},
                    worktree={"name": name},
                )

            idx = self._load_index()
            for item in cast(list[dict[str, object]], idx.get("worktrees", [])):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)
            self.events.emit(
                "worktree.remove.after",
                task=task_ref,
                worktree={
                    "name": name,
                    "path": worktree.get("path"),
                    "status": "removed",
                },
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task=task_ref,
                worktree={"name": name, "path": worktree.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        worktree = self._find(name)
        if not worktree:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept: dict[str, object] | None = None
        for item in cast(list[dict[str, object]], idx.get("worktrees", [])):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)
        self.events.emit(
            "worktree.keep",
            task={"id": worktree.get("task_id")}
            if worktree.get("task_id") is not None
            else {},
            worktree={"name": name, "path": worktree.get("path"), "status": "kept"},
        )
        return (
            json.dumps(kept, indent=2, ensure_ascii=False)
            if kept
            else f"Error: Unknown worktree '{name}'"
        )


# -- 后台任务管理器：线程执行 shell 命令并在下一轮注入通知 --
class BackgroundManager:
    def __init__(self) -> None:
        self.tasks: dict[
            str, dict[str, object]
        ] = {}  # task_id -> {status, command, result, returncode, started_at, finished_at}
        self._notification_queue: list[
            dict[str, str]
        ] = []  # {"task_id": str, "status": str, "command": str, "result": str}
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {
            "status": "running",
            "command": command,
            "result": None,
            "returncode": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=False,
                timeout=300,
            )
            output = decode_process_output(result)[:50000]
            status = "completed"
            returncode: int | None = result.returncode
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
            returncode = None
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
            returncode = None

        task = self.tasks[task_id]
        task["status"] = status
        task["result"] = output or "(no output)"
        task["returncode"] = returncode
        task["finished_at"] = time.time()

        with self._lock:
            self._notification_queue.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],
                    "result": (output or "(no output)")[:500],
                }
            )

    def check(self, task_id: str | None = None) -> str:
        if task_id:
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            command = cast(str, task["command"])
            status = cast(str, task["status"])
            result = cast(str | None, task.get("result"))
            return f"[{status}] {command[:60]}\n{result or '(running)'}"

        lines: list[str] = []
        for current_id, task in self.tasks.items():
            command = cast(str, task["command"])
            status = cast(str, task["status"])
            lines.append(f"{current_id}: [{status}] {command[:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list[dict[str, str]]:
        with self._lock:
            notifications = list(self._notification_queue)
            self._notification_queue.clear()
        return notifications


# -- 消息总线：每位队友一个 JSONL 格式的收件箱 --
class MessageBus:
    def __init__(self, inbox_dir: Path) -> None:
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict[str, object] | None = None,
    ) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg: dict[str, object] = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with inbox_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list[dict[str, object]]:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages: list[dict[str, object]] = []
        text = inbox_path.read_text(encoding="utf-8", errors="replace")
        for line in text.strip().splitlines():
            if line:
                messages.append(cast(dict[str, object], json.loads(line)))
        inbox_path.write_text("", encoding="utf-8", errors="replace")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# -- 上下文压缩后的身份重注 --
def make_identity_block(name: str, role: str, team_name: str) -> MessageParam:
    return {
        "role": "user",
        "content": (
            f"<identity>You are '{name}', role: {role}, team: {team_name}. "
            "Continue your work.</identity>"
        ),
    }


def should_reinject_identity(messages: list[MessageParam]) -> bool:
    if len(messages) <= 3:
        return True
    first_content = messages[0].get("content") if messages else None
    return isinstance(first_content, str) and first_content.startswith(
        "[Conversation compressed"
    )


def apply_identity_reinjection(
    messages: list[MessageParam], name: str, role: str, team_name: str
) -> None:
    if not should_reinject_identity(messages):
        return
    messages.insert(0, make_identity_block(name, role, team_name))
    messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})


# -- 自治队友管理器：基于 config.json 持久化存储的命名智能体，支持协议关闭与计划审批，可自动认领任务并执行 --
class TeammateManager:
    def __init__(self, team_dir: Path) -> None:
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads: dict[str, threading.Thread] = {}

    def _load_config(self) -> dict[str, object]:
        if self.config_path.exists():
            return cast(
                dict[str, object],
                json.loads(
                    self.config_path.read_text(encoding="utf-8", errors="replace")
                ),
            )
        return {"team_name": "default", "members": []}

    def _save_config(self) -> None:
        self.config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8",
            errors="replace",
        )

    def _members(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], self.config["members"])

    def _find_member(self, name: str) -> dict[str, object] | None:
        for member in self._members():
            if member.get("name") == name:
                return member
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            status = cast(str, member.get("status", "idle"))
            if status not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {status}"
            member["role"] = role
            member["status"] = "working"
        else:
            member = cast(
                dict[str, object],
                {"name": name, "role": role, "status": "working"},
            )
            self._members().append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _set_status(self, name: str, status: str) -> None:
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict[str, object]) -> str:
        return execute_named_tool(
            tool_name,
            args,
            sender=sender,
            allowed_tools={tool["name"] for tool in TEAMMATE_AGENT_TOOLS},
        )

    def _teammate_loop(self, name: str, role: str, prompt: str) -> None:
        team_name = cast(str, self.config.get("team_name", "default"))
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        should_exit = False

        try:
            while True:
                # -- 工作阶段：标准智能体循环 --
                idle_requested = False
                for _ in range(50):
                    inbox = BUS.read_inbox(name)
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                    if inbox:
                        messages.append(
                            {
                                "role": "user",
                                "content": render_inbox_messages(inbox),
                            }
                        )
                    micro_compact(messages)
                    if estimate_tokens(messages) > COMPACT_THRESHOLD:
                        auto_compact(messages, focus=f"Continue teammate {name} work")
                    response = client.messages.create(
                        model=MODEL,
                        system=(
                            TEAMMATE_SYSTEM
                            + f"\n\nYour name is {name}. Your role is {role}. Team: {team_name}."
                        ),
                        messages=messages,
                        tools=TEAMMATE_AGENT_TOOLS,
                        max_tokens=8000,
                    )
                    messages.append({"role": "assistant", "content": response.content})
                    if response.stop_reason != "tool_use":
                        break

                    results: list[ToolResultBlockParam] = []
                    manual_compact = False
                    compact_focus: str | None = None
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        if block.name == "compact":
                            manual_compact = True
                            compact_focus = cast(CompactToolInput, block.input).get(
                                "focus"
                            )
                        if block.name == "idle":
                            idle_requested = True
                        try:
                            output = self._exec(
                                name, block.name, cast(dict[str, object], block.input)
                            )
                        except Exception as e:
                            output = f"Error: {e}"
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output,
                            }
                        )
                        if block.name == "shutdown_response" and cast(
                            bool | None, block.input.get("approve")
                        ):
                            should_exit = True
                    messages.append({"role": "user", "content": results})
                    if manual_compact:
                        auto_compact(
                            messages,
                            compact_focus or f"Continue teammate {name} work",
                        )
                    if should_exit or idle_requested:
                        break

                if should_exit:
                    self._set_status(name, "shutdown")
                    return

                # -- 空闲阶段：轮询收件箱消息与未认领任务 --
                self._set_status(name, "idle")
                resume = False
                polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
                for _ in range(polls):
                    time.sleep(POLL_INTERVAL)
                    inbox = BUS.read_inbox(name)
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                    if inbox:
                        apply_identity_reinjection(messages, name, role, team_name)
                        messages.append(
                            {
                                "role": "user",
                                "content": render_inbox_messages(inbox),
                            }
                        )
                        resume = True
                        break

                    unclaimed = TASKS.scan_unclaimed()
                    if unclaimed:
                        task = unclaimed[0]
                        task_id = cast(int, task["id"])
                        result = TASKS.claim(task_id, name)
                        if result.startswith("Error:"):
                            continue
                        apply_identity_reinjection(messages, name, role, team_name)
                        task_prompt = (
                            f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                            f"{task.get('description', '')}</auto-claimed>"
                        )
                        messages.append({"role": "user", "content": task_prompt})
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"Claimed task #{task['id']}. Working on it.",
                            }
                        )
                        resume = True
                        break

                if not resume:
                    self._set_status(name, "shutdown")
                    return

                self._set_status(name, "working")
        finally:
            if not should_exit:
                member = self._find_member(name)
                if member and member.get("status") == "working":
                    self._set_status(name, "idle")

    def list_all(self) -> str:
        members = self._members()
        if not members:
            return "No teammates."
        lines = [f"Team: {self.config.get('team_name', 'default')}"]
        for member in members:
            lines.append(
                f"  {member.get('name')} ({member.get('role')}): {member.get('status')}"
            )
        return "\n".join(lines)

    def member_names(self) -> list[str]:
        return [
            cast(str, member.get("name"))
            for member in self._members()
            if member.get("name")
        ]


TODO = TodoManager()
TASKS = TaskManager(TASKS_DIR)
EVENTS = EventBus(WORKTREES_DIR / "events.jsonl")
WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)
BG = BackgroundManager()
BUS = MessageBus(INBOX_DIR)
TEAM = TeammateManager(TEAM_DIR)


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead",
        teammate,
        "Please shut down gracefully.",
        "shutdown_request",
        {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_idle_tool() -> str:
    return "Lead does not idle."


def handle_claim_task(task_id: int, owner: str = "lead") -> str:
    return TASKS.claim(task_id, owner)


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        request = plan_requests.get(request_id)
    if not request:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        request["status"] = "approved" if approve else "rejected"
    teammate = cast(str, request["from"])
    BUS.send(
        "lead",
        teammate,
        feedback,
        "plan_approval_response",
        {
            "request_id": request_id,
            "approve": approve,
            "feedback": feedback,
        },
    )
    return f"Plan {request['status']} for '{teammate}'"


def check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        request = shutdown_requests.get(request_id, {"error": "not found"})
    return json.dumps(request, ensure_ascii=False)


def handle_task_bind_worktree(task_id: int, worktree: str) -> str:
    return TASKS.bind_worktree(task_id, worktree)


def handle_worktree_create(
    name: str, task_id: int | None = None, base_ref: str = "HEAD"
) -> str:
    return WORKTREES.create(name, task_id, base_ref)


def handle_worktree_list() -> str:
    return WORKTREES.list_all()


def handle_worktree_status(name: str) -> str:
    return WORKTREES.status(name)


def handle_worktree_run(name: str, command: str) -> str:
    return WORKTREES.run(name, command)


def handle_worktree_keep(name: str) -> str:
    return WORKTREES.keep(name)


def handle_worktree_remove(
    name: str, force: bool = False, complete_task: bool = False
) -> str:
    return WORKTREES.remove(name, force, complete_task)


def handle_worktree_events(limit: int = 20) -> str:
    return EVENTS.list_recent(limit)


def decode_process_output(result: subprocess.CompletedProcess[bytes]) -> str:
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    return (stdout + stderr).strip()


def safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode("utf-8", errors="replace")
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(encoded + b"\n")
        else:
            print(encoded.decode("utf-8", errors="replace"))


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
        out = decode_process_output(r)
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text(encoding="utf-8", errors="replace")
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
        file_path.write_text(content, encoding="utf-8", errors="replace")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(
            content.replace(old_text, new_text, 1),
            encoding="utf-8",
            errors="replace",
        )
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def execute_named_tool(
    tool_name: str,
    args: dict[str, object],
    *,
    sender: str = "lead",
    allowed_tools: set[str] | None = None,
) -> str:
    if allowed_tools is not None and tool_name not in allowed_tools:
        return f"Error: Tool '{tool_name}' is not allowed in this context"
    if tool_name == "send_message":
        return BUS.send(
            sender,
            cast(str, args["to"]),
            cast(str, args["content"]),
            cast(str, args.get("msg_type", "message")),
        )
    if tool_name == "read_inbox":
        return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
    if tool_name == "claim_task":
        return TASKS.claim(cast(ClaimTaskToolInput, args)["task_id"], sender)
    if tool_name == "shutdown_response":
        request_id = cast(str, args["request_id"])
        approve = bool(args.get("approve"))
        reason = cast(str, args.get("reason", ""))
        with _tracker_lock:
            if request_id in shutdown_requests:
                shutdown_requests[request_id]["status"] = (
                    "approved" if approve else "rejected"
                )
        BUS.send(
            sender,
            "lead",
            reason,
            "shutdown_response",
            {"request_id": request_id, "approve": approve},
        )
        return f"Shutdown {'approved' if approve else 'rejected'}"
    if tool_name == "plan_approval":
        if sender == "lead":
            return handle_plan_review(
                cast(str, args.get("request_id", "")),
                bool(args.get("approve", False)),
                cast(str, args.get("feedback", "")),
            )
        plan_text = cast(str, args.get("plan", ""))
        request_id = str(uuid.uuid4())[:8]
        with _tracker_lock:
            plan_requests[request_id] = {
                "from": sender,
                "plan": plan_text,
                "status": "pending",
            }
        BUS.send(
            sender,
            "lead",
            plan_text,
            "plan_approval_response",
            {"request_id": request_id, "plan": plan_text},
        )
        return f"Plan submitted (request_id={request_id}). Waiting for lead approval."
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"
    return handler(**args)


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
    "compact": lambda **kw: "Manual compression requested.",
    "todo": lambda **kw: TODO.update(cast(TodoToolInput, kw)["items"]),
    "task_create": lambda **kw: TASKS.create(
        cast(TaskCreateToolInput, kw)["subject"],
        cast(TaskCreateToolInput, kw).get("description", ""),
    ),
    "task_update": lambda **kw: TASKS.update(
        cast(TaskUpdateToolInput, kw)["task_id"],
        cast(TaskUpdateToolInput, kw).get("status"),
        cast(TaskUpdateToolInput, kw).get("addBlockedBy"),
        cast(TaskUpdateToolInput, kw).get("removeBlockedBy"),
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(cast(TaskGetToolInput, kw)["task_id"]),
    "task_bind_worktree": lambda **kw: handle_task_bind_worktree(
        cast(TaskBindWorktreeToolInput, kw)["task_id"],
        cast(TaskBindWorktreeToolInput, kw)["worktree"],
    ),
    "background_run": lambda **kw: BG.run(cast(BackgroundRunToolInput, kw)["command"]),
    "check_background": lambda **kw: BG.check(
        cast(CheckBackgroundToolInput, kw).get("task_id")
    ),
    "spawn_teammate": lambda **kw: TEAM.spawn(
        cast(SpawnTeammateToolInput, kw)["name"],
        cast(SpawnTeammateToolInput, kw)["role"],
        cast(SpawnTeammateToolInput, kw)["prompt"],
    ),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send(
        "lead",
        cast(SendMessageToolInput, kw)["to"],
        cast(SendMessageToolInput, kw)["content"],
        cast(SendMessageToolInput, kw).get("msg_type", "message"),
    ),
    "read_inbox": lambda **kw: json.dumps(
        BUS.read_inbox("lead"), indent=2, ensure_ascii=False
    ),
    "broadcast": lambda **kw: BUS.broadcast(
        "lead",
        cast(BroadcastToolInput, kw)["content"],
        TEAM.member_names(),
    ),
    "shutdown_request": lambda **kw: handle_shutdown_request(
        cast(ShutdownRequestToolInput, kw)["teammate"]
    ),
    "shutdown_response": lambda **kw: check_shutdown_status(
        cast(ShutdownResponseToolInput, kw)["request_id"]
    ),
    "plan_approval": lambda **kw: handle_plan_review(
        cast(str, cast(PlanApprovalToolInput, kw).get("request_id", "")),
        bool(cast(PlanApprovalToolInput, kw).get("approve", False)),
        cast(PlanApprovalToolInput, kw).get("feedback", ""),
    ),
    "idle": lambda **kw: handle_idle_tool(),
    "claim_task": lambda **kw: handle_claim_task(
        cast(ClaimTaskToolInput, kw)["task_id"]
    ),
    "worktree_create": lambda **kw: handle_worktree_create(
        cast(WorktreeCreateToolInput, kw)["name"],
        cast(WorktreeCreateToolInput, kw).get("task_id"),
        cast(WorktreeCreateToolInput, kw).get("base_ref", "HEAD"),
    ),
    "worktree_list": lambda **kw: handle_worktree_list(),
    "worktree_status": lambda **kw: handle_worktree_status(
        cast(WorktreeNameToolInput, kw)["name"]
    ),
    "worktree_run": lambda **kw: handle_worktree_run(
        cast(WorktreeRunToolInput, kw)["name"],
        cast(WorktreeRunToolInput, kw)["command"],
    ),
    "worktree_keep": lambda **kw: handle_worktree_keep(
        cast(WorktreeNameToolInput, kw)["name"]
    ),
    "worktree_remove": lambda **kw: handle_worktree_remove(
        cast(WorktreeRemoveToolInput, kw)["name"],
        bool(cast(WorktreeRemoveToolInput, kw).get("force", False)),
        bool(cast(WorktreeRemoveToolInput, kw).get("complete_task", False)),
    ),
    "worktree_events": lambda **kw: handle_worktree_events(
        cast(WorktreeEventsToolInput, kw).get("limit", 20)
    ),
}


def render_background_notifications(notifications: list[dict[str, str]]) -> str:
    lines = [
        f"[bg:{item['task_id']}] {item['status']} | {item['command']}: {item['result']}"
        for item in notifications
    ]
    return "<background-results>\n" + "\n".join(lines) + "\n</background-results>"


def render_inbox_messages(messages: list[dict[str, object]]) -> str:
    return (
        "<inbox>\n" + json.dumps(messages, indent=2, ensure_ascii=False) + "\n</inbox>"
    )


def get_block_type(block: object) -> str | None:
    if isinstance(block, dict):
        return cast(str | None, cast(dict[str, object], block).get("type"))
    return cast(str | None, getattr(block, "type", None))


def get_block_text(block: object) -> str | None:
    if isinstance(block, dict):
        return cast(str | None, cast(dict[str, object], block).get("text"))
    return cast(str | None, getattr(block, "text", None))


def get_block_name(block: object) -> str | None:
    if isinstance(block, dict):
        return cast(str | None, cast(dict[str, object], block).get("name"))
    return cast(str | None, getattr(block, "name", None))


def get_block_id(block: object) -> str | None:
    if isinstance(block, dict):
        return cast(str | None, cast(dict[str, object], block).get("id"))
    return cast(str | None, getattr(block, "id", None))


def get_block_input(block: object) -> dict[str, object]:
    if isinstance(block, dict):
        return cast(dict[str, object], cast(dict[str, object], block).get("input", {}))
    return cast(dict[str, object], getattr(block, "input", {}))


def estimate_tokens(messages: list[MessageParam]) -> int:
    return len(json.dumps(messages, default=str, ensure_ascii=False)) // 4


# 通过匹配之前的 assistant 消息中的 tool_use_id，查找每个结果对应的 tool_name
def build_tool_name_map(messages: list[MessageParam]) -> dict[str, str]:
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if get_block_type(block) != "tool_use":
                continue
            block_id = get_block_id(block)
            block_name = get_block_name(block)
            if block_id and block_name:
                tool_name_map[block_id] = block_name
    return tool_name_map


# -- 压缩第 1 层：micro_compact - 用占位符替换旧的工具结果 --
def micro_compact(messages: list[MessageParam]) -> None:
    # 收集所有 tool_result 条目的 (msg_index, part_index, tool_result_dict)
    tool_results: list[dict[str, object]] = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_results.append(cast(dict[str, object], block))

    if len(tool_results) <= COMPACT_KEEP_RECENT_RESULTS:
        return

    tool_name_map = build_tool_name_map(messages)
    # 清理旧的结果（保留最近的 KEEP_RECENT 个）
    for result in tool_results[:-COMPACT_KEEP_RECENT_RESULTS]:
        content = result.get("content")
        if not isinstance(content, str) or len(content) <= 100:
            continue
        tool_use_id = cast(str, result.get("tool_use_id", ""))
        tool_name = tool_name_map.get(tool_use_id, "unknown")
        if tool_name in COMPACT_PRESERVE_RESULT_TOOLS:
            continue
        result["content"] = f"[Compacted previous result from {tool_name}]"


def summarize_messages(
    messages: list[MessageParam], focus: str | None = None, *, is_subagent: bool = False
) -> str:
    conversation_text = str(messages)[-80000:]
    focus_line = f"Focus: {focus}\n\n" if focus else ""
    scope = "subagent" if is_subagent else "agent"
    response = client.messages.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this coding-agent conversation for continuity. "
                    "Include: 1) what was accomplished, 2) current state, "
                    "3) active todos or pending work, 4) key decisions, "
                    "5) important file paths and findings. "
                    f"This summary is for a {scope} conversation.\n\n"
                    + focus_line
                    + conversation_text
                ),
            }
        ],
        max_tokens=2000,
    )
    return "".join(iter_text_blocks(response.content)) or "No summary generated."


# -- 压缩第 2 层：auto_compact - 保存对话记录，生成摘要，替换消息 --
def auto_compact(
    messages: list[MessageParam], focus: str | None = None, *, is_subagent: bool = False
) -> None:
    summary = summarize_messages(messages, focus, is_subagent=is_subagent)
    label = "Subagent" if is_subagent else "Conversation"
    messages[:] = [
        {
            "role": "user",
            "content": (
                f"[{label} compressed. Continue from this summary.]\n\n{summary}"
            ),
        }
    ]


def iter_text_blocks(content: object) -> Iterator[str]:
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return

    for block in content:
        if get_block_type(block) == "text":
            text = get_block_text(block)
            if text:
                yield text


# -- 子代理：全新上下文、过滤后的工具、仅返回摘要 --
def run_subagent(
    prompt: str,
    agent_type: SubagentAgentType = "Explore",
) -> str:
    sub_messages: list[MessageParam] = [
        {"role": "user", "content": prompt}
    ]  # 全新上下文
    sub_tools = (
        EXPLORE_SUBAGENT_TOOLS if agent_type == "Explore" else GENERAL_SUBAGENT_TOOLS
    )
    response: Message | None = None
    for _ in range(30):  # 避免死循环，子代理最多调用工具30次
        # 压缩第 1 层：在每次调用 LLM 之前执行 micro_compact
        micro_compact(sub_messages)
        # 压缩第 2 层：如果预估 Token 数超过阈值，则执行 auto_compact
        if (
            ENABLE_SUBAGENT_AUTO_COMPACT
            and estimate_tokens(sub_messages) > COMPACT_THRESHOLD
        ):
            auto_compact(
                sub_messages,
                focus=f"Continue {agent_type} subagent work",
                is_subagent=True,
            )
        response = client.messages.create(
            model=MODEL,
            system=build_subagent_system(agent_type),
            messages=sub_messages,
            tools=sub_tools,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results: list[ToolResultBlockParam] = []
        manual_compact = False
        compact_focus: str | None = None
        allowed_tool_names = {tool["name"] for tool in sub_tools}
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    compact_focus = cast(CompactToolInput, block.input).get("focus")
                try:
                    output = execute_named_tool(
                        block.name,
                        cast(dict[str, object], block.input),
                        allowed_tools=allowed_tool_names,
                    )
                except Exception as e:
                    output = f"Error: {e}"
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        sub_messages.append({"role": "user", "content": results})
        # 压缩第 3 层：由 compact 工具触发的手动压缩
        if manual_compact:
            auto_compact(
                sub_messages,
                compact_focus or f"Continue {agent_type} subagent work",
                is_subagent=True,
            )

    if response is None:
        return "(no summary)"
    # 返回所有文本块的内容拼接，子代理的最后输出应该是一个摘要文本块
    return "".join(iter_text_blocks(response.content)) or "(no summary)"


def execute_tool_block(block: ToolUseBlock) -> ToolResultBlockParam:
    try:
        output = execute_named_tool(block.name, cast(dict[str, object], block.input))
    except Exception as e:
        output = f"Error: {e}"
    safe_console_print(f"> {block.name}:")
    safe_console_print(output[:200])
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def execute_subagent_block(block: ToolUseBlock) -> ToolResultBlockParam:
    task_input = cast(SubagentToolInput, block.input)
    description = task_input.get("description", "subtask")
    prompt = task_input["prompt"]
    agent_type = task_input.get("agent_type", "Explore")
    safe_console_print(f"> subagent ({description}, {agent_type}):")
    output = run_subagent(prompt, agent_type)
    safe_console_print(output[:200])
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def create_response(messages: list[MessageParam]) -> Message:
    return client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=messages,
        tools=LEAD_AGENT_TOOLS,
        max_tokens=8000,
    )


# -- 核心模式：一个循环调用工具的 while 循环，直到模型停止 --
def agent_loop(messages: list[MessageParam]) -> None:
    rounds_since_todo = 0
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(
                {
                    "role": "user",
                    "content": render_inbox_messages(inbox),
                }
            )
        # 清空后台通知队列，并在调用 LLM 前将其注入
        notifications = BG.drain_notifications()
        if notifications:
            messages.append(
                {
                    "role": "user",
                    "content": render_background_notifications(notifications),
                }
            )
        # 压缩第 1 层：在每次调用 LLM 之前执行 micro_compact
        micro_compact(messages)
        # 压缩第 2 层：如果预估 Token 数超过阈值，则执行 auto_compact
        if estimate_tokens(messages) > COMPACT_THRESHOLD:
            auto_compact(messages)
        response = create_response(messages)
        # 追加助手的回复内容
        messages.append({"role": "assistant", "content": response.content})
        # 如果模型没有调用工具，说明任务结束
        if response.stop_reason != "tool_use":
            return
        # 执行每个工具调用，并收集结果
        results: list[ToolResultBlockParam | TextBlockParam] = []
        used_todo = False
        manual_compact = False
        compact_focus: str | None = None
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    compact_focus = cast(CompactToolInput, block.input).get("focus")
                if block.name == "subagent":
                    # subagent 工具需要特殊处理，调用子代理
                    results.append(execute_subagent_block(block))
                else:
                    results.append(execute_tool_block(block))
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3 and TODO.has_open_items():
            # 增加催促更新进度的提醒
            results.append(
                {"type": "text", "text": "<reminder>Update your todos.</reminder>"}
            )
        messages.append({"role": "user", "content": results})
        # 压缩第 3 层：由 compact 工具触发的手动压缩
        if manual_compact:
            auto_compact(messages, compact_focus)


if __name__ == "__main__":
    history: list[MessageParam] = []
    while True:
        try:
            query = input("\033[36muser >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/compact":
            if history:
                auto_compact(history, focus="Continue the current REPL session")
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        if query.strip() == "/tasks":
            print(TASKS.list_all())
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for text in iter_text_blocks(history[-1]["content"]):
            safe_console_print(text)
        print()
