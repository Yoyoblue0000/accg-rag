# -*- coding: utf-8 -*-
"""查询计划、锚点验证与预取预算测试。"""

import json

import networkx as nx
from accg.models import NodeType
from accg.query import GraphQuery

from mini_agent.agent import Agent
from mini_agent.environment import EnvConfig, Environment
from mini_agent.graph_tool import GraphTool
from mini_agent.retrieval import Candidate, RetrievalResult, tokenize


def test_anchor_selection_rejects_test_candidate_and_records_reason(tmp_path):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "tests/test_format.py::format_header",
            "name": "format_header",
            "type": "FUNCTION",
            "file": "tests/test_format.py",
            "score": 500.0,
            "sources": ["exact_symbol"],
            "matched_terms": ["format", "header"],
            "matched_fields": ["name"],
        },
        {
            "id": "src/format.py::format_header",
            "name": "format_header",
            "type": "FUNCTION",
            "file": "src/format.py",
            "score": 120.0,
            "sources": ["lexical"],
            "matched_terms": ["format", "header"],
            "matched_fields": ["name"],
        },
        {
            "id": "src/format.py::OutputFormatter",
            "name": "OutputFormatter",
            "type": "CLASS",
            "file": "src/format.py",
            "score": 110.0,
            "sources": ["lexical"],
            "matched_terms": ["format"],
            "matched_fields": ["name"],
        },
    ]

    anchors = tool.select_query_anchors(
        "Compare the header formatting function and formatter class.",
        candidates,
        max_anchors=2,
    )

    assert [item["id"] for item in anchors] == [
        "src/format.py::format_header",
        "src/format.py::OutputFormatter",
    ]
    assert all(item["selection_reason"] for item in anchors)
    assert anchors[0]["covered_terms"] == ["format", "header"]
    assert anchors[0]["candidate_sources"] == ["lexical"]


def test_exact_node_id_anchor_has_priority_over_type_diversity(tmp_path):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "src/service.py::handler",
            "name": "handler",
            "type": "FUNCTION",
            "file": "src/service.py",
            "score": 150.0,
            "sources": ["lexical"],
        },
        {
            "id": "src/service.py::Service",
            "name": "Service",
            "type": "CLASS",
            "file": "src/service.py",
            "score": 1000.0,
            "sources": ["exact_id"],
        },
    ]

    anchors = tool.select_query_anchors(
        "src/service.py::Service",
        candidates,
        max_anchors=1,
    )

    assert [item["id"] for item in anchors] == [
        "src/service.py::Service",
    ]
    assert anchors[0]["selection_reason"] == "精确 Node ID 匹配"


def test_explicit_method_request_prioritizes_method_anchor(tmp_path):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "src/service.py::helper",
            "name": "helper",
            "type": "FUNCTION",
            "file": "src/service.py",
            "score": 200.0,
            "sources": ["lexical"],
        },
        {
            "id": "src/service.py::Service::handle",
            "name": "handle",
            "type": "METHOD",
            "file": "src/service.py",
            "score": 150.0,
            "sources": ["lexical"],
        },
    ]

    anchors = tool.select_query_anchors(
        "Which method handles the service request?",
        candidates,
        max_anchors=1,
    )

    assert [item["id"] for item in anchors] == [
        "src/service.py::Service::handle",
    ]
    assert "METHOD" in anchors[0]["selection_reason"]


def test_single_anchor_preserves_top_rank_without_explicit_type(tmp_path):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "src/rule.py::Rule::_get_indentation",
            "name": "_get_indentation",
            "type": "METHOD",
            "file": "src/rule.py",
            "score": 274.0,
            "sources": ["lexical", "embedding"],
            "matched_terms": ["indent", "indentation", "unit"],
        },
        {
            "id": "src/respace.py::_construct_alignment_whitespace",
            "name": "_construct_alignment_whitespace",
            "type": "FUNCTION",
            "file": "src/respace.py",
            "score": 268.0,
            "sources": ["lexical", "embedding"],
            "matched_terms": ["indent", "unit", "whitespace"],
        },
    ]

    anchors = tool.select_query_anchors(
        "What is the propagation path through indentation checking?",
        candidates,
        max_anchors=2,
    )

    assert anchors[0]["id"] == "src/rule.py::Rule::_get_indentation"


def test_comparison_prioritizes_distinct_entity_terms_over_unrelated_type(
    tmp_path,
):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "src/io.py::read_payload",
            "name": "read_payload",
            "type": "FUNCTION",
            "file": "src/io.py",
            "score": 200.0,
            "sources": ["lexical"],
            "matched_terms": ["read", "payload"],
        },
        {
            "id": "src/io.py::PayloadConfig",
            "name": "PayloadConfig",
            "type": "CLASS",
            "file": "src/io.py",
            "score": 190.0,
            "sources": ["lexical"],
            "matched_terms": ["payload"],
        },
        {
            "id": "src/io.py::write_payload",
            "name": "write_payload",
            "type": "FUNCTION",
            "file": "src/io.py",
            "score": 180.0,
            "sources": ["lexical"],
            "matched_terms": ["write", "payload"],
        },
    ]

    anchors = tool.select_query_anchors(
        "Compare the read payload function and write payload function.",
        candidates,
        max_anchors=2,
    )

    assert [item["id"] for item in anchors] == [
        "src/io.py::read_payload",
        "src/io.py::write_payload",
    ]


def test_explicit_function_type_is_covered_before_second_exact_class(
    tmp_path,
):
    tool = GraphTool(str(tmp_path))
    candidates = [
        {
            "id": "src/formatters.py::OutputStreamFormatter",
            "name": "OutputStreamFormatter",
            "type": "CLASS",
            "file": "src/formatters.py",
            "score": 1028.0,
            "sources": ["exact_symbol", "lexical"],
            "matched_terms": ["class", "format", "output", "stream"],
        },
        {
            "id": "src/outputstream.py::OutputStream",
            "name": "OutputStream",
            "type": "CLASS",
            "file": "src/outputstream.py",
            "score": 1019.0,
            "sources": ["exact_symbol", "lexical"],
            "matched_terms": ["class", "output", "stream"],
        },
        {
            "id": "src/formatters.py::colorize",
            "name": "colorize",
            "type": "METHOD",
            "file": "src/formatters.py",
            "score": 131.0,
            "sources": ["lexical"],
            "matched_terms": ["format", "output", "stream", "string", "use"],
        },
        {
            "id": "src/formatters.py::format_linting_result_header",
            "name": "format_linting_result_header",
            "type": "FUNCTION",
            "file": "src/formatters.py",
            "score": 125.0,
            "sources": ["lexical"],
            "matched_terms": ["format", "function", "head", "output"],
        },
    ]

    anchors = tool.select_query_anchors(
        "What is the relationship between the standalone header formatting "
        "function and the output stream formatter class?",
        candidates,
        max_anchors=3,
    )

    assert [item["name"] for item in anchors] == [
        "OutputStreamFormatter",
        "format_linting_result_header",
        "OutputStream",
    ]


def test_invalid_node_id_is_rejected_and_next_anchor_is_prefetched(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "valid.py").write_text(
        "class ValidService:\n"
        "    pass\n",
        encoding="utf-8",
    )
    graph = nx.MultiDiGraph()
    graph.add_node(
        "src/valid.py::ValidService",
        node_type=NodeType.CLASS,
        name="ValidService",
        file_path="src/valid.py",
        start_line=1,
        end_line=2,
        docstring="A valid service.",
        signature="",
        parent_id=None,
        extra={},
    )

    class _StaleCandidateTool(GraphTool):
        def search(self, query, limit=12, use_embeddings=None):
            return RetrievalResult(
                candidates=[
                    Candidate(
                        id="src/missing.py::missing_handler",
                        name="missing_handler",
                        type="FUNCTION",
                        file="src/missing.py",
                        score=1000.0,
                        sources=["exact_id"],
                    ),
                    Candidate(
                        id="src/valid.py::ValidService",
                        name="ValidService",
                        type="CLASS",
                        file="src/valid.py",
                        score=120.0,
                        sources=["lexical"],
                    ),
                ],
                stages_attempted=["exact_id", "lexical"],
                stages_succeeded=["exact_id", "lexical"],
                diagnostics=[],
            )

    class _FinalModel:
        def query(self, messages):
            return {
                "content": "done",
                "raw_content": "FINAL: done",
                "tool_calls": [],
            }

        def generate(self, messages):
            return "verified"

    tool = _StaleCandidateTool(str(tmp_path))
    tool._graph = graph
    tool._query = GraphQuery(graph)
    tool._built = True
    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run(
        "Compare src/missing.py::missing_handler with ValidService."
    )

    # P4 门控：比较问题只有 1 个有效锚点，证据不足
    assert result.error is not None
    assert "证据不足" in result.error
    assert [item["id"] for item in agent.last_query_plan["anchors"]] == [
        "src/valid.py::ValidService",
    ]
    rejected = agent.last_query_plan["rejected_anchors"]
    assert rejected[0]["reason"] == "node_id_not_found"
    assert "suggestions" in rejected[0]
    assert agent.last_query_plan["prefetch_evidence_ids"]
    json.dumps(agent.last_query_plan, ensure_ascii=False)


def test_prefetch_budget_previews_large_class_but_keeps_full_ledger(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    class_lines = ["class LargeFormatter:"]
    class_lines.extend(
        f"    field_{index} = {index}"
        for index in range(80)
    )
    late_marker = "LATE_CLASS_MARKER = 999"
    class_lines.append(f"    {late_marker}")
    source_lines = [
        "def small_helper():",
        "    return 1",
        "",
        *class_lines,
    ]
    (source_dir / "formatters.py").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )

    graph = nx.MultiDiGraph()
    graph.add_node(
        "src/formatters.py::small_helper",
        node_type=NodeType.FUNCTION,
        name="small_helper",
        file_path="src/formatters.py",
        start_line=1,
        end_line=2,
        docstring="Small helper.",
        signature="def small_helper()",
        parent_id=None,
        extra={},
    )
    graph.add_node(
        "src/formatters.py::LargeFormatter",
        node_type=NodeType.CLASS,
        name="LargeFormatter",
        file_path="src/formatters.py",
        start_line=4,
        end_line=len(source_lines),
        docstring="Large formatter.",
        signature="",
        parent_id=None,
        extra={},
    )

    class _CaptureModel:
        def __init__(self):
            self.first_messages = None

        def query(self, messages):
            self.first_messages = json.loads(json.dumps(messages))
            return {
                "content": "done",
                "raw_content": "FINAL: done",
                "tool_calls": [],
            }

        def generate(self, messages):
            return "verified"

    tool = GraphTool(str(tmp_path))
    tool._graph = graph
    tool._query = GraphQuery(graph)
    tool._built = True
    model = _CaptureModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run(
        "Compare the small helper function and LargeFormatter class."
    )

    user_message = model.first_messages[1]["content"]
    assert "def small_helper():" in user_message
    # 大类使用结构化视图，只展示前 40 行源码，late_marker 不在展示中
    # 但完整源码仍在 evidence payload 中
    assert "LargeFormatter" in user_message
    large_evidence = next(
        item for item in result.evidence
        if item.node_id == "src/formatters.py::LargeFormatter"
    )
    assert late_marker in large_evidence.payload["source_context"]
    display_levels = {
        item["name"]: item["display_level"]
        for item in agent.last_query_plan["anchors"]
    }
    assert display_levels == {
        "small_helper": "complete",
        "LargeFormatter": "complete",
    }
    large_anchor = next(
        item for item in agent.last_query_plan["anchors"]
        if item["name"] == "LargeFormatter"
    )
    assert large_anchor["omitted_reason"] == ""


def test_tokenize_preserves_cjk_characters():
    """中文查询中的 CJK 字符应作为独立 token 参与检索，不应被丢弃。"""
    tokens = tokenize("parse 和 render 的调用关系是什么")
    cjk = [t for t in tokens if any("一" <= c <= "鿿" for c in t)]
    assert len(cjk) > 0, f"期望至少一个 CJK token，实际 tokens: {tokens}"
    assert "调" in cjk or "调用" in tokens, f"期望 '调用' 相关 token，实际: {tokens}"


def test_english_query_still_works():
    """纯英文查询的 tokenize 行为不应被 CJK 修改破坏。"""
    tokens = tokenize("how does OutputStreamFormatter handle errors")
    assert "output" in tokens
    assert "stream" in tokens
    assert "error" in tokens
