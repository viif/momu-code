import os
import subprocess
from collections.abc import Iterator
from typing import TypedDict, cast

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


class BashToolInput(TypedDict):
    command: str


TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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
    tool_input = cast(BashToolInput, block.input)
    command = tool_input["command"]
    print(f"\033[33m$ {command}\033[0m")
    output = run_bash(command)
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
