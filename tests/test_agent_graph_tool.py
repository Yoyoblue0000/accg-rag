# -*- coding: utf-8 -*-
"""query 候选排序与自动预取测试。"""

import copy
import json

import networkx as nx
import pytest

from accg.models import EdgeType, NodeType
from accg.query import GraphQuery
from mini_agent.agent import Agent
from mini_agent.environment import EnvConfig, Environment
from mini_agent.graph_tool import GraphTool


def _add_symbol(
    graph,
    node_id,
    node_type,
    name,
    file_path,
    docstring="",
    signature="",
    parent_id=None,
    extra=None,
):
    graph.add_node(
        node_id,
        node_type=node_type,
        name=name,
        file_path=file_path,
        start_line=1,
        end_line=3,
        docstring=docstring,
        signature=signature,
        parent_id=parent_id,
        extra=extra or {},
    )


def _graph_tool(graph, project_path):
    tool = GraphTool(str(project_path))
    tool._graph = graph
    tool._query = GraphQuery(graph)
    tool._built = True
    return tool


def test_embedding_augmentation_is_opt_in(tmp_path):
    tool = GraphTool(str(tmp_path))

    assert tool.enable_embeddings is False


def test_rank_query_candidates_prefers_multi_term_source_symbols(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/formatters.py::format_linting_result_header",
        NodeType.FUNCTION,
        "format_linting_result_header",
        "src/formatters.py",
        docstring="Format the header of a linting result output.",
        signature="def format_linting_result_header() -> str",
    )
    _add_symbol(
        graph,
        "src/formatters.py::OutputStreamFormatter",
        NodeType.CLASS,
        "OutputStreamFormatter",
        "src/formatters.py",
        docstring="Formatter which writes formatted output to an OutputStream.",
    )
    _add_symbol(
        graph,
        "docs/build.py::build_global_headers",
        NodeType.FUNCTION,
        "build_global_headers",
        "docs/build.py",
    )
    _add_symbol(
        graph,
        "test/test_formatters.py::test_output_stream",
        NodeType.FUNCTION,
        "test_output_stream",
        "test/test_formatters.py",
    )
    tool = _graph_tool(graph, tmp_path)
    query = (
        "What is the relationship between the standalone header formatting "
        "function and the output stream formatter class?"
    )

    first = tool.rank_query_candidates(query)
    second = tool.rank_query_candidates(query)

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert {item["name"] for item in first[:2]} == {
        "format_linting_result_header",
        "OutputStreamFormatter",
    }
    assert first[0]["matched_terms"]
    assert first[0]["score"] != 100


def test_select_query_anchors_covers_function_and_class(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/a.py::format_header",
        NodeType.FUNCTION,
        "format_header",
        "src/a.py",
        docstring="Format a header.",
    )
    _add_symbol(
        graph,
        "src/a.py::OutputFormatter",
        NodeType.CLASS,
        "OutputFormatter",
        "src/a.py",
        docstring="Manage output formatting.",
    )
    _add_symbol(
        graph,
        "src/a.py::other_output",
        NodeType.FUNCTION,
        "other_output",
        "src/a.py",
    )
    tool = _graph_tool(graph, tmp_path)
    query = "Compare the header formatting function and output formatter class."
    candidates = tool.rank_query_candidates(query)

    anchors = tool.select_query_anchors(query, candidates, max_anchors=3)

    assert "FUNCTION" in {item["type"] for item in anchors}
    assert "CLASS" in {item["type"] for item in anchors}
    assert "format_header" in {item["name"] for item in anchors}
    assert "OutputFormatter" in {item["name"] for item in anchors}


def test_static_method_metadata_supports_indirect_query(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/layout.py::Rule_LT09",
        NodeType.CLASS,
        "Rule_LT09",
        "src/layout.py",
        docstring="Select targets should be placed on separate lines.",
    )
    _add_symbol(
        graph,
        "src/layout.py::Rule_LT09::_get_indexes",
        NodeType.METHOD,
        "_get_indexes",
        "src/layout.py",
        signature="def _get_indexes(context: RuleContext) -> SelectTargetsInfo",
        parent_id="src/layout.py::Rule_LT09",
        extra={"decorators": ["staticmethod"]},
    )
    _add_symbol(
        graph,
        "src/other.py::OtherRule::_generate_violations",
        NodeType.METHOD,
        "_generate_violations",
        "src/other.py",
        signature="def _generate_violations()",
        extra={"decorators": ["staticmethod"]},
    )
    tool = _graph_tool(graph, tmp_path)
    query = (
        "What is the responsibility distribution between the static method "
        "that extracts select layout information and methods that generate fixes?"
    )

    candidates = tool.rank_query_candidates(query)

    assert candidates[0]["name"] == "_get_indexes"
    assert "decorator" in candidates[0]["matched_fields"]


def test_exact_node_id_ranks_first(tmp_path):
    graph = nx.MultiDiGraph()
    target_id = "src/utils.py::get_environ_proxies"
    _add_symbol(
        graph,
        target_id,
        NodeType.FUNCTION,
        "get_environ_proxies",
        "src/utils.py",
    )
    _add_symbol(
        graph,
        "src/utils.py::get_proxies",
        NodeType.FUNCTION,
        "get_proxies",
        "src/utils.py",
    )
    tool = _graph_tool(graph, tmp_path)

    candidates = tool.rank_query_candidates(target_id)

    assert candidates[0]["id"] == target_id
    assert candidates[0]["sources"][0] == "exact_id"


def test_source_symbol_beats_test_symbol_by_default(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/utils.py::normalize_headers",
        NodeType.FUNCTION,
        "normalize_headers",
        "src/utils.py",
    )
    _add_symbol(
        graph,
        "tests/test_utils.py::normalize_headers",
        NodeType.FUNCTION,
        "normalize_headers",
        "tests/test_utils.py",
    )
    tool = _graph_tool(graph, tmp_path)

    candidates = tool.rank_query_candidates("How does normalize_headers work?")

    assert candidates[0]["file"] == "src/utils.py"


def test_candidate_tie_breaking_is_stable(tmp_path):
    graph = nx.MultiDiGraph()
    for name in ("alpha_handler", "beta_handler"):
        _add_symbol(
            graph,
            f"src/handlers.py::{name}",
            NodeType.FUNCTION,
            name,
            "src/handlers.py",
        )
    tool = _graph_tool(graph, tmp_path)

    first = tool.rank_query_candidates("handler")
    second = tool.rank_query_candidates("handler")

    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_embedding_failure_returns_lexical_fallback(tmp_path):
    class _BrokenRanker:
        calls = 0

        def build_index(self, graph):
            self.calls += 1
            raise ConnectionError("ollama unavailable")

    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/utils.py::normalize_headers",
        NodeType.FUNCTION,
        "normalize_headers",
        "src/utils.py",
    )
    tool = _graph_tool(graph, tmp_path)
    tool.embedding_ranker = _BrokenRanker()

    result = tool.retrieve_query_candidates(
        "normalize response headers",
        use_embeddings=True,
    )

    assert result.candidates[0].id == "src/utils.py::normalize_headers"
    assert result.status == "fallback"
    assert "embedding" in result.stages_attempted
    assert any("ollama unavailable" in item for item in result.diagnostics)

    tool.retrieve_query_candidates(
        "normalize response headers",
        use_embeddings=True,
    )
    assert tool.embedding_ranker.calls == 1


def test_candidate_merge_preserves_retrieval_sources(tmp_path):
    class _Ranker:
        def build_index(self, graph):
            return None

        def rank(self, query, limit=12):
            return [{
                "id": "src/utils.py::normalize_headers",
                "name": "normalize_headers",
                "type": "FUNCTION",
                "file": "src/utils.py",
                "score": 0.8,
            }]

    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/utils.py::normalize_headers",
        NodeType.FUNCTION,
        "normalize_headers",
        "src/utils.py",
    )
    tool = _graph_tool(graph, tmp_path)
    tool.embedding_ranker = _Ranker()

    result = tool.retrieve_query_candidates(
        "normalize response headers",
        use_embeddings=True,
    )

    assert {"lexical", "embedding"} <= set(result.candidates[0].sources)


def test_fuzzy_fallback_handles_misspelled_symbol(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/utils.py::get_environ_proxies",
        NodeType.FUNCTION,
        "get_environ_proxies",
        "src/utils.py",
    )
    tool = _graph_tool(graph, tmp_path)

    result = tool.retrieve_query_candidates(
        "get envron proxys",
        limit=3,
        use_embeddings=False,
    )

    assert result.candidates[0].id == "src/utils.py::get_environ_proxies"
    assert "fuzzy" in result.candidates[0].sources


def test_path_and_decorator_fields_are_searchable(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/layout.py::LayoutRule::_get_indexes",
        NodeType.METHOD,
        "_get_indexes",
        "src/layout.py",
        extra={"decorators": ["staticmethod"]},
    )
    tool = _graph_tool(graph, tmp_path)

    candidates = tool.rank_query_candidates(
        "static method in src layout",
    )

    assert candidates[0]["name"] == "_get_indexes"
    assert {"file", "decorator"} <= set(candidates[0]["matched_fields"])


class _FinalModel:
    def __init__(self):
        self.last_messages = None

    def query(self, messages):
        self.last_messages = messages
        return {
            "content": "FINAL: evidence collected",
            "raw_content": "FINAL: evidence collected",
            "tool_calls": [],
        }

    def generate(self, messages):
        return "answer"


def test_agent_returns_result_when_embedding_is_unavailable(tmp_path):
    class _BrokenRanker:
        def build_index(self, graph):
            raise ConnectionError("ollama unavailable")

    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/utils.py::normalize_headers",
        NodeType.FUNCTION,
        "normalize_headers",
        "src/utils.py",
    )
    tool = _graph_tool(graph, tmp_path)
    tool.enable_embeddings = True
    tool.embedding_ranker = _BrokenRanker()
    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run("How are response headers normalized?")

    assert result.answer == "evidence collected"
    assert result.retrieval.status == "fallback"
    assert result.anchor_candidates


@pytest.mark.skip(reason="P2: 自动锚点预取不在本次 P0/P1 范围")
def test_agent_prefetches_query_anchors_before_model_selection(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "formatters.py").write_text(
        "def format_header():\n"
        "    return 'header'\n"
        "\n"
        "class OutputFormatter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/formatters.py::format_header",
        NodeType.FUNCTION,
        "format_header",
        "src/formatters.py",
        docstring="Format a header.",
        signature="def format_header()",
    )
    _add_symbol(
        graph,
        "src/formatters.py::OutputFormatter",
        NodeType.CLASS,
        "OutputFormatter",
        "src/formatters.py",
        docstring="Manage formatted output.",
    )
    tool = _graph_tool(graph, tmp_path)
    model = _FinalModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run(
        "Compare the header formatting function and output formatter class."
    )

    assert result.answer == "answer"
    assert len(agent._evidence) == 2
    assert {item["name"] for item in agent.last_query_plan["anchors"]} == {
        "format_header",
        "OutputFormatter",
    }
    user_message = model.last_messages[1]["content"]
    assert "[自动验证锚点的证据]" in user_message
    assert "format_header" in user_message
    assert "OutputFormatter" in user_message


class _RepeatedFinalModel:
    def __init__(self):
        self.query_messages = []
        self.generate_messages = None

    def query(self, messages):
        self.query_messages.append(copy.deepcopy(messages))
        return {
            "content": "证据已经足够，无需继续查询",
            "raw_content": (
                "THOUGHT: 证据已经足够，无需继续查询\n"
                "FINAL: header 和 formatter 只是职责分离"
            ),
            "tool_calls": [],
        }

    def generate(self, messages):
        self.generate_messages = messages
        return "verified answer"


@pytest.mark.skip(reason="P3: 关系充分性门控不在本次 P0/P1 范围")
def test_relation_gate_expands_shared_caller_before_accepting_final(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "formatters.py").write_text(
        "def format_linting_result_header():\n"
        "    return '==== readout ===='\n"
        "\n"
        "class OutputStreamFormatter:\n"
        "    def _dispatch(self, message):\n"
        "        self._output_stream.write(message)\n",
        encoding="utf-8",
    )
    (source_dir / "outputstream.py").write_text(
        "def make_output_stream(config):\n"
        "    return OutputStream(config)\n",
        encoding="utf-8",
    )
    caller_lines = [
        "def lint(config):",
        "    output_stream = make_output_stream(config)",
        "    formatter = OutputStreamFormatter(output_stream)",
    ]
    caller_lines.extend(f"    filler_{index} = {index}" for index in range(70))
    caller_lines.append("    click.echo(format_linting_result_header())")
    (source_dir / "commands.py").write_text(
        "\n".join(caller_lines) + "\n",
        encoding="utf-8",
    )

    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/formatters.py::format_linting_result_header",
        NodeType.FUNCTION,
        "format_linting_result_header",
        "src/formatters.py",
        docstring="Format the linting result header.",
    )
    _add_symbol(
        graph,
        "src/formatters.py::OutputStreamFormatter",
        NodeType.CLASS,
        "OutputStreamFormatter",
        "src/formatters.py",
        docstring="Manage formatted output streams.",
    )
    _add_symbol(
        graph,
        "src/outputstream.py::make_output_stream",
        NodeType.FUNCTION,
        "make_output_stream",
        "src/outputstream.py",
        docstring="Construct the managed output stream.",
    )
    _add_symbol(
        graph,
        "src/commands.py::lint",
        NodeType.FUNCTION,
        "lint",
        "src/commands.py",
        docstring="Lint files and emit results.",
    )
    graph.nodes["src/formatters.py::OutputStreamFormatter"]["end_line"] = 6
    graph.nodes["src/formatters.py::OutputStreamFormatter"]["start_line"] = 4
    graph.nodes["src/commands.py::lint"]["end_line"] = len(caller_lines)
    for target in (
        "src/formatters.py::format_linting_result_header",
        "src/outputstream.py::make_output_stream",
    ):
        graph.add_edge(
            "src/commands.py::lint",
            target,
            edge_type=EdgeType.CALLS,
            confidence=0.95,
            strategy="test",
        )

    tool = _graph_tool(graph, tmp_path)
    model = _RepeatedFinalModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run(
        "What is the relationship between the standalone header formatting "
        "function and the output stream formatter class?"
    )

    assert result.answer == "verified answer"
    assert len(model.query_messages) == 2
    expansions = agent.last_query_plan["relation_expansions"]
    assert [item["id"] for item in expansions] == ["src/commands.py::lint"]
    retry_message = model.query_messages[1][-1]["content"]
    assert "[证据充分性检查未通过]" in retry_message
    assert "click.echo(format_linting_result_header())" in retry_message
    synthesis_prompt = model.generate_messages[0]["content"]
    assert "click.echo(format_linting_result_header())" in synthesis_prompt
    assert "header 和 formatter 只是职责分离" in synthesis_prompt


@pytest.mark.skip(reason="P4: 完整证据账本与请求审计不在本次 P0/P1 范围")
def test_synthesis_sends_and_audits_full_untrimmed_evidence(tmp_path, capsys):
    model = _FinalModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
    )
    late_marker = "late-marker-" + ("x" * 12000)
    agent._evidence = [
        "first evidence",
        "long evidence\n" + late_marker,
    ]

    result = agent._synthesize("Explain the complete relationship.", candidates=[])

    assert result.answer == "answer"
    request = agent.last_model_requests[-1]
    assert request["stage"] == "answer_synthesis"
    assert late_marker in request["messages"][0]["content"]
    printed = capsys.readouterr().out
    assert late_marker in printed
    assert "[...省略低相关源码...]" not in printed


@pytest.mark.skip(reason="P4: 完整模型请求审计不在本次 P0/P1 范围")
def test_model_request_audit_is_complete_and_human_readable(tmp_path, capsys):
    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
    )
    messages = [
        {"role": "system", "content": "SYSTEM-CONTENT"},
        {"role": "user", "content": "USER-CONTENT"},
        {
            "role": "assistant",
            "content": "需要查询调用者",
            "tool_calls": [{
                "id": "call_full",
                "type": "function",
                "function": {
                    "name": "query_graph",
                    "arguments": json.dumps({
                        "action": "contextualize",
                        "name": "src/example.py::target",
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_full",
            "content": "FULL-TOOL-RESULT",
        },
    ]

    agent._audit_model_request("exploration_step_2", messages)

    printed = capsys.readouterr().out
    assert "发给大模型的完整内容 | 探索阶段 · 第 2 轮" in printed
    assert "消息 1/4 | SYSTEM" in printed
    assert "SYSTEM-CONTENT" in printed
    assert "消息 2/4 | USER" in printed
    assert "USER-CONTENT" in printed
    assert "工具名称: query_graph" in printed
    assert '"name": "src/example.py::target"' in printed
    assert "关联工具调用: call_full" in printed
    assert "FULL-TOOL-RESULT" in printed
    assert agent.last_model_requests[-1]["messages"] == messages
