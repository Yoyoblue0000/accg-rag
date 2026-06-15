# -*- coding: utf-8 -*-
"""Model 层 function calling 测试 — 验证 OpenAI tools 参数和流式 tool_calls 处理。"""

import json
from unittest.mock import MagicMock, patch

from mini_agent.model import Model, ModelConfig


def _make_model():
    """创建测试用的 Model 实例。"""
    config = ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="test",
        model_name="test-model",
        quiet=True,
    )
    return Model(config)


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    """创建模拟的流式响应 chunk。"""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = tool_calls
    chunk.choices[0].finish_reason = finish_reason
    return chunk


def _make_tool_call_delta(index, id=None, name=None, arguments=None):
    """创建 tool_calls 的增量 delta。"""
    tc = MagicMock()
    tc.index = index
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class TestModelQueryWithTools:
    """测试 Model.query() 方法的 tools 参数支持。"""

    @patch("mini_agent.model.OpenAI")
    def test_query_accepts_tools_parameter(self, mock_openai_class):
        """query() 方法应接受 tools 参数。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        # 模拟流式响应：无 tool_calls
        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(content="回答", finish_reason="stop"),
        ]

        messages = [{"role": "user", "content": "test"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        result = model.query(messages, tools=tools)

        # 验证 tools 参数被传递给 OpenAI API
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("tools") == tools or call_kwargs[1].get("tools") == tools

    @patch("mini_agent.model.OpenAI")
    def test_query_returns_tool_calls(self, mock_openai_class):
        """当 LLM 返回 tool_calls 时，query() 应正确解析。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        # 模拟流式响应：包含 tool_calls
        tc_delta = _make_tool_call_delta(
            index=0,
            id="call_123",
            name="contextualize",
            arguments='{"name": "foo"}',
        )
        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(tool_calls=[tc_delta], finish_reason="tool_calls"),
        ]

        messages = [{"role": "user", "content": "test"}]
        tools = [{"type": "function", "function": {"name": "contextualize"}}]

        result = model.query(messages, tools=tools)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"]["name"] == "contextualize"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"name": "foo"}

    @patch("mini_agent.model.OpenAI")
    def test_query_no_tools_returns_empty_tool_calls(self, mock_openai_class):
        """当不传入 tools 时，返回空的 tool_calls。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(content="回答", finish_reason="stop"),
        ]

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["tool_calls"] == []
        assert result["content"] == "回答"

    @patch("mini_agent.model.OpenAI")
    def test_query_streaming_tool_calls_concatenation(self, mock_openai_class):
        """流式 tool_calls 的增量 arguments 应正确拼接。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        # 模拟流式响应：arguments 分多个 chunk 发送
        tc_delta_1 = _make_tool_call_delta(index=0, id="call_123", name="contextualize")
        tc_delta_1.function.arguments = '{"na'

        tc_delta_2 = _make_tool_call_delta(index=0)
        tc_delta_2.function.arguments = 'me": "foo"}'

        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(tool_calls=[tc_delta_1]),
            _make_stream_chunk(tool_calls=[tc_delta_2], finish_reason="tool_calls"),
        ]

        messages = [{"role": "user", "content": "test"}]
        tools = [{"type": "function", "function": {"name": "contextualize"}}]

        result = model.query(messages, tools=tools)

        assert len(result["tool_calls"]) == 1
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"name": "foo"}

    @patch("mini_agent.model.OpenAI")
    def test_query_multiple_tool_calls(self, mock_openai_class):
        """多个 tool_calls 应正确解析。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        tc_delta_1 = _make_tool_call_delta(index=0, id="call_1", name="contextualize")
        tc_delta_1.function.arguments = '{"name": "a"}'

        tc_delta_2 = _make_tool_call_delta(index=1, id="call_2", name="read_file")
        tc_delta_2.function.arguments = '{"path": "x.py"}'

        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(tool_calls=[tc_delta_1, tc_delta_2], finish_reason="tool_calls"),
        ]

        messages = [{"role": "user", "content": "test"}]
        tools = [
            {"type": "function", "function": {"name": "contextualize"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]

        result = model.query(messages, tools=tools)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "contextualize"
        assert result["tool_calls"][1]["function"]["name"] == "read_file"

    @patch("mini_agent.model.OpenAI")
    def test_query_finish_reason_preserved(self, mock_openai_class):
        """finish_reason 应正确保留。"""
        model = _make_model()
        mock_client = MagicMock()
        model.client = mock_client

        mock_client.chat.completions.create.return_value = [
            _make_stream_chunk(content="回答", finish_reason="stop"),
        ]

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["finish_reason"] == "stop"
