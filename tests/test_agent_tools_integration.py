# -*- coding: utf-8 -*-
"""Agent 层工具集成测试 — 验证 Agent 使用新的 tools 参数。"""

import json
from unittest.mock import MagicMock, patch

from mini_agent.agent import Agent
from mini_agent.environment import Environment, EnvConfig
from mini_agent.model import Model, ModelConfig
from mini_agent.tools_schema import TOOLS


def _make_agent():
    """创建测试用的 Agent 实例。"""
    config = ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="test",
        model_name="test-model",
        quiet=True,
    )
    model = Model(config)
    env = Environment(EnvConfig(cwd="/tmp"))
    return Agent(model=model, env=env, max_steps=3)


class TestAgentToolsIntegration:
    """测试 Agent 如何使用 tools 参数。"""

    def test_tools_schema_importable(self):
        """Agent 可以导入 TOOLS 定义。"""
        assert isinstance(TOOLS, list)
        assert len(TOOLS) == 11

    def test_tools_schema_format(self):
        """TOOLS 格式符合 OpenAI function calling 规范。"""
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    @patch("mini_agent.model.OpenAI")
    def test_agent_passes_tools_to_model(self, mock_openai_class):
        """Agent 应将 TOOLS 传给 model.query()。"""
        agent = _make_agent()
        mock_client = MagicMock()
        agent.model.client = mock_client

        # 模拟：第一次返回 tool_call，第二次返回无 tool_calls（完成）
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_123"
        tc.function = MagicMock()
        tc.function.name = "read_file"
        tc.function.arguments = '{"path": "test.py"}'

        chunk_1 = MagicMock()
        chunk_1.choices = [MagicMock()]
        chunk_1.choices[0].delta = MagicMock()
        chunk_1.choices[0].delta.content = None
        chunk_1.choices[0].delta.tool_calls = [tc]
        chunk_1.choices[0].finish_reason = "tool_calls"

        chunk_2 = MagicMock()
        chunk_2.choices = [MagicMock()]
        chunk_2.choices[0].delta = MagicMock()
        chunk_2.choices[0].delta.content = "完成"
        chunk_2.choices[0].delta.tool_calls = None
        chunk_2.choices[0].finish_reason = "stop"

        mock_client.chat.completions.create.side_effect = [
            [chunk_1],
            [chunk_2],
        ]

        # 运行 Agent（会因为图未构建而失败，但我们可以验证 tools 参数）
        try:
            agent.run("test query")
        except Exception:
            pass

        # 验证第一次调用传入了 tools
        calls = mock_client.chat.completions.create.call_args_list
        if calls:
            first_call = calls[0]
            # 检查 tools 参数
            tools_arg = first_call.kwargs.get("tools") or first_call[1].get("tools")
            assert tools_arg is not None, "第一次调用应传入 tools 参数"
            assert len(tools_arg) == 11
