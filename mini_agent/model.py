# -*- coding: utf-8 -*-
"""模型接口 -- 支持 OpenAI function calling 的流式调用"""
import json
import time

from openai import OpenAI


class ModelConfig:
    def __init__(self, base_url: str, api_key: str, model_name: str, quiet: bool = False, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.quiet = quiet
        self.model_kwargs = kwargs


class Model:
    """模型接口，支持 OpenAI function calling"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.quiet = config.quiet
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def query(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """流式查询，支持 tools 参数和 tool_calls 解析。

        Args:
            messages: 消息列表
            tools: OpenAI function calling 工具定义列表（可选）

        Returns:
            包含 content, raw_content, tool_calls, finish_reason 的字典
        """
        kwargs = {"temperature": 0.1, "max_tokens": 3000, **self.config.model_kwargs}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=True,
            **kwargs,
        )

        # 流式收集：处理 content 和 tool_calls 的增量
        content_parts = []
        tool_calls_acc: dict[int, dict] = {}  # {index: {id, name, arguments}}
        finish_reason = None

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # 收集 content
            if delta.content:
                if not self.quiet:
                    print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            # 收集 tool_calls（增量拼接）
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        content = "".join(content_parts).strip()
        if content and not self.quiet:
            print()

        # 构建 tool_calls 列表（按 index 排序）
        tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            })

        # 回退方案：如果流式响应中没有 tool_calls，尝试从 content 中解析
        if not tool_calls and tools and content:
            tool_calls = self._parse_tool_calls_from_content(content, tools)

        return {
            "role": "assistant",
            "content": content,
            "raw_content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "extra": {"model": self.config.model_name, "timestamp": time.time()},
        }

    def _parse_tool_calls_from_content(
        self, content: str, tools: list[dict]
    ) -> list[dict]:
        """从 content 中解析 tool_calls（回退方案）。

        当 Ollama 流式响应不支持 tool_calls 时，
        尝试从 content 中提取 JSON 格式的工具调用。
        """
        import re
        import uuid

        # 尝试从 content 中提取 JSON
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', content)
        if not json_match:
            return []

        brace = content.find("{", json_match.start())
        if brace < 0:
            return []

        # 提取完整的 JSON 对象
        depth = 0
        for i in range(brace, len(content)):
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = content[brace:i + 1]
                    try:
                        parsed = json.loads(json_str)
                        if "name" in parsed and "arguments" in parsed:
                            # 验证工具名称是否在 tools 列表中
                            tool_names = {t["function"]["name"] for t in tools}
                            if parsed["name"] in tool_names:
                                return [{
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": parsed["name"],
                                        "arguments": json.dumps(parsed["arguments"]),
                                    },
                                }]
                    except json.JSONDecodeError:
                        pass
                    break

        return []

    def generate(self, messages: list[dict]) -> str:
        """纯文本生成，不使用 tools，用于答案合成"""
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
