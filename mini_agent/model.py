# -*- coding: utf-8 -*-
"""模型接口 -- 纯文本协议 + THOUGHT/ACTION 解析"""
import ast
import json
import time
import uuid
from openai import OpenAI

def _get_graph_actions():
    """延迟导入，避免循环依赖"""
    from .graph_tool import GraphTool
    return GraphTool.ACTIONS


def _parse_action_json(text: str, start: int) -> tuple[str | None, str | None]:
    """从指定位置提取完整 JSON 对象（括号计数）。

    返回 (json_str, error) — 成功时 error 为 None。
    """
    if start < 0 or start >= len(text) or text[start] != "{":
        return None, "未找到 { 起始"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], None
    return None, "括号不闭合"


def _robust_json_parse(json_str: str) -> dict | None:
    """多级 JSON 解析：json.loads → ast.literal_eval → None"""
    stripped = json_str.strip()
    # 去 markdown 包裹
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[1:end]).strip()
    for parser in [json.loads, ast.literal_eval]:
        try:
            result = parser(stripped)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    return None


def _make_tool_calls(parsed_list: list[dict]) -> list[dict]:
    """将解析后的工具调用转为标准 tool_calls 格式"""
    graph_actions = _get_graph_actions()
    tool_calls = []
    for parsed in parsed_list:
        name = parsed.get("name", "")
        args_raw = parsed.get("arguments", {})
        if isinstance(args_raw, dict):
            args_str = json.dumps(args_raw)
        else:
            args_str = str(args_raw)

        if name in graph_actions:
            actual_args = json.dumps({"action": name, **json.loads(args_str)})
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "query_graph", "arguments": actual_args},
            })
        else:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
    return tool_calls


def _split_thought_and_actions(content: str) -> tuple[str, list[dict]]:
    """从文本中分离 THOUGHT 和 ACTION JSON。也处理裸 JSON（无 ACTION: 前缀）。"""
    import re

    parts = re.split(r"\nACTION:\s*", content, maxsplit=1)
    thought = parts[0].strip()

    if thought.startswith("THOUGHT:"):
        thought = thought[len("THOUGHT:"):].strip()

    # 如果 THOUGHT 部分本身就是裸 JSON（LLM 忘了写 THOUGHT 前缀）
    if thought.startswith("{") and '"name"' in thought and '"arguments"' in thought:
        parsed = _robust_json_parse(thought)
        if parsed and "name" in parsed and "arguments" in parsed:
            return "", _make_tool_calls([parsed])

    if len(parts) < 2:
        # 无 ACTION: 前缀，检查是否有裸 JSON
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', content)
        if json_match:
            brace = content.find("{", json_match.start())
            json_str, _ = _parse_action_json(content, brace)
            if json_str:
                parsed = _robust_json_parse(json_str)
                if parsed and "name" in parsed and "arguments" in parsed:
                    return thought[:brace].strip(), _make_tool_calls([parsed])
        return thought, []

    remaining = parts[1]
    graph_actions = _get_graph_actions()
    tool_calls = []
    pos = 0

    while pos < len(remaining):
        brace = remaining.find("{", pos)
        if brace < 0:
            break
        json_str, err = _parse_action_json(remaining, brace)
        if err or not json_str:
            pos = brace + 1
            continue

        parsed = _robust_json_parse(json_str)
        if parsed and "name" in parsed and "arguments" in parsed:
            name = parsed["name"]
            args = parsed["arguments"]
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args)

            # 图操作包装为 query_graph(action=..., ...)
            if name in graph_actions:
                actual_args = json.dumps({"action": name, **json.loads(args_str)})
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": "query_graph", "arguments": actual_args},
                })
            else:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })
            pos = brace + len(json_str)
        else:
            # JSON 括号完整但内容非法时也必须推进，否则会在同一位置死循环。
            pos = brace + len(json_str)

    return thought, tool_calls


class ModelConfig:
    def __init__(self, base_url: str, api_key: str, model_name: str, quiet: bool = False, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.quiet = quiet
        self.model_kwargs = kwargs


class Model:
    """纯文本协议模型接口，解析 THOUGHT/ACTION/FINAL 格式"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.quiet = config.quiet
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def query(self, messages: list[dict]) -> dict:
        """流式查询，实时打印，解析 THOUGHT/ACTION"""
        kwargs = {"temperature": 0.1, "max_tokens": 3000, **self.config.model_kwargs}
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=True,
            **kwargs,
        )

        content_parts = []
        finish_reason = None
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                if not self.quiet:
                    print(delta.content, end="", flush=True)
                content_parts.append(delta.content)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        content = "".join(content_parts).strip()
        if content and not self.quiet:
            print()

        thought, tool_calls = _split_thought_and_actions(content)

        # 如果只有 ACTION 没有 THOUGHT，用 content 作为 thought
        if not thought and tool_calls:
            thought = content

        return {
            "role": "assistant",
            "content": thought,
            "raw_content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "extra": {"model": self.config.model_name, "timestamp": time.time()},
        }

    def generate(self, messages: list[dict]) -> str:
        """纯文本生成，不解析 THOUGHT/ACTION，用于答案合成"""
        kwargs = {"temperature": 0.3, "max_tokens": 4000, **self.config.model_kwargs}
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=True,
            **kwargs,
        )
        parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                if not self.quiet:
                    print(delta.content, end="", flush=True)
                parts.append(delta.content)
        content = "".join(parts).strip()
        if content and not self.quiet:
            print()
        return content
