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

        # 模拟：_analyze_question 的 generate 调用 + query 调用
        # generate 用于问题分析（不传 tools），query 用于 ReAct 循环（传 tools）
        generate_chunk = MagicMock()
        generate_chunk.choices = [MagicMock()]
        generate_chunk.choices[0].delta = MagicMock()
        generate_chunk.choices[0].delta.content = '{"key_entities": [], "question_type": "unknown", "exploration_targets": [], "analysis": "test"}'
        generate_chunk.choices[0].delta.tool_calls = None
        generate_chunk.choices[0].finish_reason = "stop"

        query_chunk = MagicMock()
        query_chunk.choices = [MagicMock()]
        query_chunk.choices[0].delta = MagicMock()
        query_chunk.choices[0].delta.content = "FINAL: 完成"
        query_chunk.choices[0].delta.tool_calls = None
        query_chunk.choices[0].finish_reason = "stop"

        mock_client.chat.completions.create.side_effect = [
            [generate_chunk],  # _analyze_question 的 generate
            [query_chunk],     # ReAct 循环的 query
        ]

        # 运行 Agent（graph_tool 为 None，会走无图路径）
        try:
            agent.run("test query")
        except Exception:
            pass

        # 验证 query 调用传入了 tools
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) >= 2, f"应有至少 2 次调用，实际 {len(calls)} 次"
        query_call = calls[1]  # 第二次调用是 query
        tools_arg = query_call.kwargs.get("tools")
        assert tools_arg is not None, "query 调用应传入 tools 参数"
        assert len(tools_arg) == 11


class TestExecuteToolRouting:
    """测试 _execute_tool 对图工具名的路由能力。"""

    def _make_agent_with_graph_tool(self):
        """创建带 mock graph_tool 的 Agent。"""
        config = ModelConfig(
            base_url="http://localhost:11434/v1",
            api_key="test",
            model_name="test-model",
            quiet=True,
        )
        model = Model(config)
        env = Environment(EnvConfig(cwd="/tmp"))
        graph_tool = MagicMock()
        graph_tool.is_ready = True
        graph_tool.ensure_built.return_value = "ok"
        graph_tool.execute_full.return_value = '{"items": []}'
        return Agent(model=model, env=env, graph_tool=graph_tool, max_steps=3)

    def test_contextualize_dispatches_to_graph_tool(self):
        """_execute_tool('contextualize', ...) 应调用 graph_tool.execute_full。"""
        agent = self._make_agent_with_graph_tool()
        result = agent._execute_tool("contextualize", {"name": "Foo"})
        agent.graph_tool.execute_full.assert_called_once_with(
            action="contextualize", name="Foo"
        )
        assert '{"items": []}' in result

    def test_transitive_callers_dispatches_to_graph_tool(self):
        """_execute_tool('transitive_callers', ...) 应调用 graph_tool.execute_full。"""
        agent = self._make_agent_with_graph_tool()
        result = agent._execute_tool("transitive_callers", {"symbol": "Bar"})
        agent.graph_tool.execute_full.assert_called_once_with(
            action="transitive_callers", symbol="Bar"
        )

    def test_class_hierarchy_dispatches_to_graph_tool(self):
        """_execute_tool('class_hierarchy', ...) 应调用 graph_tool.execute_full。"""
        agent = self._make_agent_with_graph_tool()
        result = agent._execute_tool("class_hierarchy", {"class_name": "MyClass"})
        agent.graph_tool.execute_full.assert_called_once_with(
            action="class_hierarchy", class_name="MyClass"
        )

    def test_query_graph_still_works(self):
        """_execute_tool('query_graph', ...) 原有行为不变。"""
        agent = self._make_agent_with_graph_tool()
        result = agent._execute_tool(
            "query_graph", {"action": "contextualize", "name": "Foo"}
        )
        agent.graph_tool.execute_full.assert_called_once_with(
            action="contextualize", name="Foo"
        )

    def test_read_file_still_works(self):
        """_execute_tool('read_file', ...) 原有行为不变。"""
        agent = self._make_agent_with_graph_tool()
        agent.env = MagicMock()
        agent.env.read_file.return_value = "file content"
        result = agent._execute_tool("read_file", {"path": "test.py"})
        assert result == "file content"

    def test_unknown_tool_returns_error(self):
        """未知工具名应返回错误信息。"""
        agent = self._make_agent_with_graph_tool()
        result = agent._execute_tool("nonexistent_tool", {})
        assert "未知工具" in result


class TestMainLoopNormalization:
    """测试主循环中图工具名的归一化和重复检测。"""

    def _make_graph_tool_mock(self):
        """创建 mock graph_tool，返回有效的 contextualize 结果。"""
        graph_tool = MagicMock()
        graph_tool.is_ready = True
        graph_tool.ensure_built.return_value = "ok"
        graph_tool.execute_full.return_value = json.dumps({
            "results": [{
                "id": "src/foo.py::bar",
                "name": "bar",
                "type": "FUNCTION",
                "file": "src/foo.py",
                "source_context": "def bar():\n    pass\n",
                "start_line": 1,
                "end_line": 2,
                "calls": [],
                "called_by": [],
            }],
        })
        graph_tool.search.return_value = MagicMock(
            candidates=[], stages_attempted=[], stages_succeeded=[],
            diagnostics=[], status="ok",
        )
        graph_tool.select_query_anchors.return_value = []
        return graph_tool

    def test_contextualize_creates_evidence(self):
        """model 返回 contextualize tool_call 时，证据应正确写入账本。"""
        import networkx as nx
        from accg.models import NodeType
        from accg.query import GraphQuery

        graph = nx.MultiDiGraph()
        graph.add_node(
            "src/foo.py::bar",
            node_type=NodeType.FUNCTION,
            name="bar",
            file_path="src/foo.py",
            start_line=1,
            end_line=2,
            docstring="",
            signature="def bar()",
            parent_id=None,
            extra={},
        )

        graph_tool = self._make_graph_tool_mock()
        graph_tool._graph = graph
        graph_tool._query = GraphQuery(graph)
        graph_tool._built = True

        class _ToolCallModel:
            """第一次返回 contextualize tool_call，第二次返回完成。"""
            def __init__(self):
                self._call_count = 0

            def query(self, messages, tools=None):
                self._call_count += 1
                if self._call_count == 1:
                    return {
                        "content": "需要查看 bar 函数",
                        "raw_content": "需要查看 bar 函数",
                        "tool_calls": [{
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "contextualize",
                                "arguments": json.dumps({"name": "bar"}),
                            },
                        }],
                    }
                return {
                    "content": "bar 函数是一个简单的空函数",
                    "raw_content": "bar 函数是一个简单的空函数",
                    "tool_calls": [],
                }

            def generate(self, messages):
                return "bar 函数是一个简单的空函数"

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                _ToolCallModel(),
                Environment(EnvConfig(cwd=tmp)),
                graph_tool=graph_tool,
                max_steps=3,
            )
            result = agent.run("bar 函数做什么？")

        # 验证：graph_tool.execute_full 被正确调用
        graph_tool.execute_full.assert_called_with(action="contextualize", name="bar")

        # 验证：证据写入了账本
        source_items = [e for e in result.evidence if e.kind == "source"]
        assert len(source_items) > 0, "应有 source 类型证据"

    def test_duplicate_contextualize_is_intercepted(self):
        """连续两次调用同一 contextualize 应被拦截。"""
        graph_tool = self._make_graph_tool_mock()

        class _DupModel:
            """连续两次返回相同的 contextualize tool_call。"""
            def __init__(self):
                self._call_count = 0

            def query(self, messages, tools=None):
                self._call_count += 1
                if self._call_count <= 2:
                    return {
                        "content": "查看 bar",
                        "raw_content": "查看 bar",
                        "tool_calls": [{
                            "id": f"call_{self._call_count:03d}",
                            "type": "function",
                            "function": {
                                "name": "contextualize",
                                "arguments": json.dumps({"name": "bar"}),
                            },
                        }],
                    }
                return {
                    "content": "完成",
                    "raw_content": "完成",
                    "tool_calls": [],
                }

            def generate(self, messages):
                return "完成"

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                _DupModel(),
                Environment(EnvConfig(cwd=tmp)),
                graph_tool=graph_tool,
                max_steps=5,
            )
            result = agent.run("bar 函数做什么？")

        # execute_full 只应被调用一次（第二次被拦截）
        assert graph_tool.execute_full.call_count == 1


class TestGateAndCompressionMessages:
    """测试门控失败消息和压缩消息不引用旧协议格式。"""

    def test_gate_failure_message_no_old_format(self):
        """门控失败消息不应包含 THOUGHT/ACTION/FINAL 指令。"""
        from mini_agent.agent import _format_gate_failure_message
        from mini_agent.sufficiency import GateDecision

        decision = GateDecision(
            passed=False,
            missing_requirements=["缺少源码证据"],
            expansion_requests=[],
            reasons=["no_source"],
        )
        msg = _format_gate_failure_message(decision, "test draft", [])
        assert "THOUGHT:" not in msg
        assert "ACTION:" not in msg
        assert "FINAL:" not in msg
        # 应该有新的指导语
        assert "调用工具" in msg or "输出你的回答" in msg

    def test_compress_messages_no_old_format(self):
        """压缩消息不应包含 FINAL: 引用。"""
        config = ModelConfig(
            base_url="http://localhost:11434/v1",
            api_key="test",
            model_name="test-model",
            quiet=True,
        )
        model = Model(config)
        env = Environment(EnvConfig(cwd="/tmp"))
        agent = Agent(model=model, env=env, max_steps=3)

        # 设置一些消息让压缩逻辑有机会触发
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "thought", "tool_calls": []},
            {"role": "assistant", "content": "FINAL: draft answer", "tool_calls": []},
        ]
        agent.last_query_plan = {"anchors": [], "relation_expansions": []}
        agent._compress_messages(keep_recent_turns=1)

        # 检查压缩后的消息不含旧格式
        for msg in agent.messages:
            content = msg.get("content", "")
            assert "FINAL:" not in content, f"压缩消息不应引用 FINAL: {content}"


class TestSystemPrompt:
    """测试 SYSTEM_PROMPT 与 function calling 协议一致。"""

    def test_system_prompt_no_old_protocol(self):
        """SYSTEM_PROMPT 不应包含 THOUGHT/ACTION/FINAL 格式指令。"""
        from mini_agent.agent import SYSTEM_PROMPT
        assert "THOUGHT:" not in SYSTEM_PROMPT
        assert "ACTION:" not in SYSTEM_PROMPT
        assert "FINAL:" not in SYSTEM_PROMPT

    def test_system_prompt_describes_tool_usage(self):
        """SYSTEM_PROMPT 应描述工具调用方式。"""
        from mini_agent.agent import SYSTEM_PROMPT
        # 应该提到工具或调用
        assert "工具" in SYSTEM_PROMPT or "tool" in SYSTEM_PROMPT.lower()

    def test_system_prompt_has_placeholders(self):
        """SYSTEM_PROMPT 应保留环境占位符。"""
        from mini_agent.agent import SYSTEM_PROMPT
        assert "__CWD__" in SYSTEM_PROMPT
        assert "__GRAPH_STATUS__" in SYSTEM_PROMPT


class TestQuestionAnalysisIntegration:
    """测试 QuestionAnalysis 结果连接到检索流程。"""

    def test_analysis_entities_bypass_entity_extractor(self):
        """当问题分析返回有效 key_entities 时，应跳过 EntityExtractor 的 LLM 调用。"""
        from mini_agent.agent import Agent, QuestionAnalysis

        class _AnalysisModel:
            """generate 返回有效的问题分析 JSON，query 返回完成。"""
            def query(self, messages, tools=None):
                return {
                    "content": "answer",
                    "raw_content": "FINAL: answer",
                    "tool_calls": [],
                }

            def generate(self, messages):
                prompt = messages[-1].get("content", "") if messages else ""
                if "代码问题分析" in prompt or "问题分析" in prompt:
                    return json.dumps({
                        "key_entities": ["format_header", "OutputFormatter"],
                        "question_type": "relation",
                        "exploration_targets": [
                            {"type": "function", "name": "format_header", "reason": "需要查看实现"},
                            {"type": "class", "name": "OutputFormatter", "reason": "需要理解类结构"},
                        ],
                        "analysis": "分析两个符号之间的关系",
                    })
                return "answer"

        mock_graph_tool = MagicMock()
        mock_graph_tool.is_ready = True
        mock_graph_tool.ensure_built.return_value = "ok"
        mock_graph_tool.search.return_value = MagicMock(
            candidates=[], stages_attempted=[], stages_succeeded=[],
            diagnostics=[], status="ok",
        )
        mock_graph_tool.select_query_anchors.return_value = []

        mock_entity_extractor = MagicMock()
        mock_entity_extractor.extract.return_value = []  # 不应被调用

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                _AnalysisModel(),
                Environment(EnvConfig(cwd=tmp)),
                graph_tool=mock_graph_tool,
                max_steps=3,
                entity_extractor=mock_entity_extractor,
            )
            result = agent.run("format_header 和 OutputFormatter 的关系？")

        # EntityExtractor.extract 不应被调用
        mock_entity_extractor.extract.assert_not_called()

        # 实体应来自问题分析
        assert len(result.entities) >= 2
        entity_names = {e["name"] for e in result.entities}
        assert "format_header" in entity_names
        assert "OutputFormatter" in entity_names

    def test_analysis_failure_falls_back_to_entity_extractor(self):
        """当问题分析失败时，应回退到 EntityExtractor。"""
        from mini_agent.agent import Agent

        class _NoAnalysisModel:
            """generate 返回无法解析的内容。"""
            def query(self, messages, tools=None):
                return {
                    "content": "answer",
                    "raw_content": "FINAL: answer",
                    "tool_calls": [],
                }

            def generate(self, messages):
                return "not valid json"

        mock_graph_tool = MagicMock()
        mock_graph_tool.is_ready = True
        mock_graph_tool.ensure_built.return_value = "ok"
        mock_graph_tool.search.return_value = MagicMock(
            candidates=[], stages_attempted=[], stages_succeeded=[],
            diagnostics=[], status="ok",
        )
        mock_graph_tool.select_query_anchors.return_value = []

        mock_entity_extractor = MagicMock()
        mock_entity_extractor.extract.return_value = [
            MagicMock(
                name="fallback_entity",
                query="fallback_entity",
                description="fallback",
                type_hint="CONCEPT",
                to_dict=lambda: {
                    "name": "fallback_entity",
                    "query": "fallback_entity",
                    "description": "fallback",
                    "type_hint": "CONCEPT",
                },
            ),
        ]

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                _NoAnalysisModel(),
                Environment(EnvConfig(cwd=tmp)),
                graph_tool=mock_graph_tool,
                max_steps=3,
                entity_extractor=mock_entity_extractor,
            )
            result = agent.run("test question")

        # EntityExtractor.extract 应被调用（回退）
        mock_entity_extractor.extract.assert_called_once()
