# -*- coding: utf-8 -*-
"""model.py 解析器测试 —— 覆盖 THOUGHT/ACTION 协议的全体变体"""

import json
import re
import uuid
import pytest
from unittest.mock import patch

from mini_agent.model import (
    _parse_action_json,
    _robust_json_parse,
    _make_tool_calls,
    _split_thought_and_actions,
)


# ═══════════════════════════════════════════
# _parse_action_json
# ═══════════════════════════════════════════

class TestParseActionJson:
    def test_simple_json(self):
        text = '{"name": "test"}'
        result, err = _parse_action_json(text, 0)
        assert err is None
        assert result == '{"name": "test"}'

    def test_nested_json(self):
        text = '{"name": "test", "args": {"k": [1,2,3]}}'
        result, err = _parse_action_json(text, 0)
        assert err is None
        assert result == text

    def test_unbalanced_brackets(self):
        result, err = _parse_action_json('{"name": "test"', 0)
        assert result is None
        assert err == "括号不闭合"

    def test_no_opening_brace(self):
        result, err = _parse_action_json('no brace here', 0)
        assert result is None
        assert "未找到" in err

    def test_start_index_negative(self):
        result, err = _parse_action_json('{"k":1}', -1)
        assert result is None

    def test_adjacent_jsons_first_only(self):
        text = '{"a":1}{"b":2}'
        result, _ = _parse_action_json(text, 0)
        assert result == '{"a":1}'

    def test_string_containing_braces(self):
        text = '{"name": "f{oo}", "args": {"pattern": "x{y}"}}'
        result, err = _parse_action_json(text, 0)
        assert err is None
        assert result == text

    def test_escaped_quotes(self):
        text = r'{"name": "{\"key\": \"val\"}"}'
        result, err = _parse_action_json(text, 0)
        assert err is None
        assert result == text


# ═══════════════════════════════════════════
# _robust_json_parse
# ═══════════════════════════════════════════

class TestRobustJsonParse:
    def test_valid_json(self):
        result = _robust_json_parse('{"name": "test", "arguments": {}}')
        assert result == {"name": "test", "arguments": {}}

    def test_single_quoted(self):
        result = _robust_json_parse("{'name': 'test', 'arguments': {}}")
        assert result == {"name": "test", "arguments": {}}

    def test_markdown_code_block(self):
        text = '```\n{"name": "test", "arguments": {}}\n```'
        result = _robust_json_parse(text)
        assert result == {"name": "test", "arguments": {}}

    def test_markdown_json_block(self):
        text = '```json\n{"name": "test", "arguments": {}}\n```'
        result = _robust_json_parse(text)
        assert result == {"name": "test", "arguments": {}}

    def test_invalid_json(self):
        result = _robust_json_parse('not json')
        assert result is None

    def test_empty_string(self):
        result = _robust_json_parse('')
        assert result is None

    def test_array_returns_none(self):
        result = _robust_json_parse('[1, 2, 3]')
        assert result is None  # 只要 dict

    def test_trailing_comma(self):
        # Python 3.9+ 的 ast.literal_eval 支持尾逗号
        result = _robust_json_parse('{"name": "test",}')
        assert result == {"name": "test"}

    def test_boolean_int_none(self):
        result = _robust_json_parse('{"a": true, "b": 42, "c": null}')
        assert result == {"a": True, "b": 42, "c": None}


# ═══════════════════════════════════════════
# _make_tool_calls
# ═══════════════════════════════════════════

class TestMakeToolCalls:
    def test_graph_action_wrapped(self):
        """图操作应被包装为 query_graph"""
        parsed = [{"name": "contextualize", "arguments": {"name": "get_environ_proxies"}}]
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678123456781234567812345678')):
            result = _make_tool_calls(parsed)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "query_graph"
        args = json.loads(result[0]["function"]["arguments"])
        assert args["action"] == "contextualize"
        assert args["name"] == "get_environ_proxies"

    def test_non_graph_tool_passthrough(self):
        """非图工具直接透传"""
        parsed = [{"name": "read_file", "arguments": {"path": "test.py"}}]
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678123456781234567812345678')):
            result = _make_tool_calls(parsed)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["arguments"] == '{"path": "test.py"}'

    def test_unknown_tool_name(self):
        """未知工具名直接透传"""
        parsed = [{"name": "list_dir", "arguments": {"path": "/"}}]
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678123456781234567812345678')):
            result = _make_tool_calls(parsed)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "list_dir"

    def test_multiple_calls(self):
        parsed = [
            {"name": "contextualize", "arguments": {"name": "foo"}},
            {"name": "transitive_callers", "arguments": {"symbol": "X::Y::bar"}},
        ]
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678123456781234567812345678')):
            result = _make_tool_calls(parsed)
        assert len(result) == 2
        assert all(c["type"] == "function" for c in result)

    def test_string_arguments(self):
        """arguments 可能是字符串形式"""
        parsed = [{"name": "contextualize", "arguments": '{"name": "foo"}'}]
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678123456781234567812345678')):
            result = _make_tool_calls(parsed)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "query_graph"

    def test_call_id_format(self):
        parsed = [{"name": "read_file", "arguments": {"path": "x"}}]
        result = _make_tool_calls(parsed)
        assert result[0]["id"].startswith("call_")
        assert len(result[0]["id"]) == 13  # call_ + 8 hex


# ═══════════════════════════════════════════
# _split_thought_and_actions
# ═══════════════════════════════════════════

class TestSplitThoughtAndActions:
    def test_thought_and_action(self):
        content = 'THOUGHT: 需要查函数\nACTION: {"name": "contextualize", "arguments": {"name": "foo"}}'
        thought, actions = _split_thought_and_actions(content)
        assert "需要查函数" in thought
        assert len(actions) == 1
        assert actions[0]["function"]["name"] == "query_graph"

    def test_thought_only_no_action(self):
        content = 'THOUGHT: 已经足够'
        thought, actions = _split_thought_and_actions(content)
        assert "已经足够" in thought
        assert actions == []

    def test_thought_only_with_final(self):
        content = 'THOUGHT: ok\nFINAL: answer text here'
        thought, actions = _split_thought_and_actions(content)
        assert "ok" in thought
        assert "FINAL" in thought
        assert "answer text here" in thought
        assert actions == []

    def test_no_thought_prefix(self):
        content = '直接思考内容\nACTION: {"name": "read_file", "arguments": {"path": "x.py"}}'
        thought, actions = _split_thought_and_actions(content)
        assert "直接思考内容" in thought
        assert len(actions) == 1
        assert actions[0]["function"]["name"] == "read_file"

    def test_bare_json_no_prefix(self):
        """裸 JSON 无 THOUGHT/ACTION 前缀"""
        content = '{"name": "contextualize", "arguments": {"name": "foo"}}'
        thought, actions = _split_thought_and_actions(content)
        assert thought == ""
        assert len(actions) == 1
        assert actions[0]["function"]["name"] == "query_graph"

    def test_bare_json_after_thought(self):
        """先有 THOUGHT 部分，再接裸 JSON"""
        content = 'THOUGHT: 想到了\n{"name": "contextualize", "arguments": {"name": "bar"}}'
        thought, actions = _split_thought_and_actions(content)
        assert "想到了" in thought
        assert len(actions) == 1
        assert actions[0]["function"]["name"] == "query_graph"

    def test_multiple_actions(self):
        """多个独立 ACTION 时解析为多个 tool_calls"""
        content = (
            'THOUGHT: 需要两个独立查询\n'
            'ACTION: {"name": "contextualize", "arguments": {"name": "a"}}\n'
            'ACTION: {"name": "contextualize", "arguments": {"name": "b"}}'
        )
        thought, actions = _split_thought_and_actions(content)
        assert "需要两个独立查询" in thought
        assert len(actions) == 2
        assert actions[0]["function"]["name"] == "query_graph"
        assert actions[1]["function"]["name"] == "query_graph"

    def test_multiple_actions_bare_json(self):
        """裸 JSON（无 ACTION: 前缀）作为第二个 action 也能解析"""
        content = (
            'THOUGHT: 需要两个查询\n'
            'ACTION: {"name": "contextualize", "arguments": {"name": "a"}}\n'
            '{"name": "transitive_callers", "arguments": {"symbol": "X::Y::b"}}'
        )
        thought, actions = _split_thought_and_actions(content)
        assert "需要两个查询" in thought
        assert len(actions) == 2
        assert actions[0]["function"]["name"] == "query_graph"
        assert actions[1]["function"]["name"] == "query_graph"

    def test_non_graph_action_no_wrap(self):
        """read_file 不应被包装为 query_graph"""
        content = 'THOUGHT: 读文件\nACTION: {"name": "read_file", "arguments": {"path": "x.py"}}'
        _, actions = _split_thought_and_actions(content)
        assert actions[0]["function"]["name"] == "read_file"

    def test_empty_content(self):
        thought, actions = _split_thought_and_actions("")
        assert thought == ""
        assert actions == []

    def test_content_with_braces_not_json(self):
        content = "THOUGHT: use {variable} here"
        thought, actions = _split_thought_and_actions(content)
        assert "variable" in thought
        assert actions == []

    def test_invalid_json_in_action(self):
        content = 'THOUGHT: try\nACTION: {invalid json here}}'
        thought, actions = _split_thought_and_actions(content)
        assert "try" in thought
        assert actions == []

    def test_multi_line_thought(self):
        content = (
            'THOUGHT: 第一行\n'
            '第二行思考\n'
            '第三行\n'
            'ACTION: {"name": "read_file", "arguments": {"path": "x.py"}}'
        )
        thought, actions = _split_thought_and_actions(content)
        assert "第一行" in thought
        assert "第二行" in thought
        assert len(actions) == 1

    def test_two_action_blocks(self):
        """第二个 ACTION: 前缀的处理"""
        content = (
            'THOUGHT: first\nACTION: {"name": "contextualize", "arguments": {"name": "a"}}\n'
            'THOUGHT: second\nACTION: {"name": "read_file", "arguments": {"path": "x.py"}}'
        )
        thought, actions = _split_thought_and_actions(content)
        assert actions  # 只解析第一个 ACTION 块

    def test_content_surrounding_json(self):
        """裸 JSON 前后有额外文本"""
        content = 'before {"name": "read_file", "arguments": {"path": "x.py"}} after'
        thought, actions = _split_thought_and_actions(content)
        assert actions

    def test_graph_action_wrapping_transitive(self):
        """图操作包装测试: transitive_callers → query_graph"""
        content = 'ACTION: {"name": "transitive_callers", "arguments": {"symbol": "X::Y::func"}}'
        _, actions = _split_thought_and_actions(content)
        assert actions[0]["function"]["name"] == "query_graph"

    def test_explicit_query_graph_passthrough(self):
        """name=query_graph 的直接透传（已在外部包装好）"""
        content = (
            'ACTION: {"name": "query_graph", "arguments": {"action": "contextualize", "name": "x"}}'
        )
        _, actions = _split_thought_and_actions(content)
        assert actions[0]["function"]["name"] == "query_graph"
