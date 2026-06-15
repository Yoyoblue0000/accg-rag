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

        return {
            "role": "assistant",
            "content": content,
            "raw_content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "extra": {"model": self.config.model_name, "timestamp": time.time()},
        }

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
