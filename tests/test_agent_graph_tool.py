# -*- coding: utf-8 -*-
"""query 候选排序与自动预取测试。"""

import copy
import json

import networkx as nx
from accg.models import EdgeType, NodeType
from accg.query import GraphQuery

from mini_agent.agent import Agent
from mini_agent.environment import EnvConfig, Environment
from mini_agent.evidence import EvidenceItem
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

        def build_index(self, graph, summaries=None):
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
        def build_index(self, graph, summaries=None):
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


def test_class_hierarchy_returns_resolved_node_ids_and_correct_direction(tmp_path):
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/base.py::Base",
        NodeType.CLASS,
        "Base",
        "src/base.py",
    )
    _add_symbol(
        graph,
        "src/current.py::Current",
        NodeType.CLASS,
        "Current",
        "src/current.py",
    )
    _add_symbol(
        graph,
        "src/child.py::Child",
        NodeType.CLASS,
        "Child",
        "src/child.py",
    )
    graph.add_edge(
        "src/current.py::Current",
        "src/base.py::Base",
        edge_type=EdgeType.INHERITS,
    )
    graph.add_edge(
        "src/child.py::Child",
        "src/current.py::Current",
        edge_type=EdgeType.INHERITS,
    )
    tool = _graph_tool(graph, tmp_path)

    raw = tool.execute_full(
        "class_hierarchy",
        class_name="src/current.py::Current",
    )
    items = EvidenceItem.from_tool_result(
        "query_graph",
        {
            "action": "class_hierarchy",
            "class_name": "src/current.py::Current",
        },
        raw,
        step=1,
    )

    assert {
        (item.source_node_id, item.target_node_id)
        for item in items
    } == {
        ("src/current.py::Current", "src/base.py::Base"),
        ("src/child.py::Child", "src/current.py::Current"),
    }


def test_instantiation_evidence_preserves_via_class():
    raw = json.dumps({
        "query": "src/base.py::BaseFormatter",
        "exact": True,
        "results": [{
            "id": "src/base.py::BaseFormatter",
            "name": "BaseFormatter",
            "type": "CLASS",
            "file": "src/base.py",
            "start_line": 1,
            "end_line": 3,
            "source_context": "class BaseFormatter:\n    pass",
            "instantiated_by": [{
                "id": "src/factory.py::build_formatter",
                "name": "build_formatter",
                "file": "src/factory.py",
                "confidence": 0.9,
                "strategy": "constructor",
                "via_class": "src/child.py::ChildFormatter",
            }],
        }],
    })

    items = EvidenceItem.from_tool_result(
        "query_graph",
        {
            "action": "contextualize",
            "name": "src/base.py::BaseFormatter",
        },
        raw,
        step=1,
    )
    relation = next(
        item
        for item in items
        if item.edge_type == "INSTANTIATED_BY"
    )

    assert relation.payload["via_class"] == (
        "src/child.py::ChildFormatter"
    )


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
        def build_index(self, graph, summaries=None):
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

    # P4 门控：FINAL 无证据时返回错误而非直接输出草稿
    assert result.error is not None
    assert "证据不足" in result.error
    assert result.retrieval.status == "fallback"
    assert result.anchor_candidates


def test_agent_retrieves_deep_candidate_pool_but_displays_top_eight(tmp_path):
    class _Retrieval:
        def __init__(self):
            self.candidates = [
                type(
                    "_Candidate",
                    (),
                    {
                        "to_dict": lambda self, index=index: {
                            "id": f"node-{index}",
                            "name": f"candidate_{index}",
                            "type": "FUNCTION",
                            "file": f"src/{index}.py",
                            "score": float(100 - index),
                            "sources": ["lexical"],
                            "matched_terms": [],
                            "matched_fields": [],
                        },
                    },
                )()
                for index in range(24)
            ]
            self.diagnostics = []
            self.duration_ms = 0.0

    class _PoolGraphTool:
        is_ready = True
        enable_embeddings = False

        def __init__(self):
            self.requested_limit = None

        def ensure_built(self):
            return "ready"

        def search(self, query, limit=12, use_embeddings=None):
            self.requested_limit = limit
            return _Retrieval()

        def select_query_anchors(self, query, candidates, max_anchors=3):
            return []

    tool = _PoolGraphTool()
    model = _FinalModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    agent.run("Explain candidate retrieval.")

    assert tool.requested_limit >= 24
    user_message = model.last_messages[1]["content"]
    assert "candidate_7" in user_message
    assert "candidate_8" not in user_message


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
    assert result.rounds == 1
    assert result.explorations == 0
    assert len(result.evidence) == 2
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


class _StopOnlyModel:
    def __init__(self):
        self.query_messages = []
        self.generate_messages = None

    def query(self, messages):
        self.query_messages.append(copy.deepcopy(messages))
        return {
            "content": "There is no caller in the bounded search scope.",
            "raw_content": "THOUGHT: There is no caller in the bounded search scope.",
            "tool_calls": [],
            "finish_reason": "stop",
        }

    def generate(self, messages):
        self.generate_messages = copy.deepcopy(messages)
        return "bounded negative answer"


def test_finish_reason_stop_cannot_bypass_relation_gate(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "isolated.py").write_text(
        "def isolated_function():\n"
        "    return 1\n"
        "# end\n",
        encoding="utf-8",
    )
    graph = nx.MultiDiGraph()
    _add_symbol(
        graph,
        "src/isolated.py::isolated_function",
        NodeType.FUNCTION,
        "isolated_function",
        "src/isolated.py",
    )
    tool = _graph_tool(graph, tmp_path)
    model = _StopOnlyModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=tool,
    )

    result = agent.run("What calls isolated_function?")

    assert result.answer == "bounded negative answer"
    assert len(model.query_messages) == 2
    assert not any(
        message["role"] == "assistant"
        for message in model.query_messages[1]
    )
    expansion = agent.last_query_plan["relation_expansions"][0]
    assert expansion["action"] == "transitive_callers"
    assert expansion["status"] == "completed"
    assert expansion["result_count"] == 0
    synthesis_prompt = model.generate_messages[0]["content"]
    assert "isolated_function" in synthesis_prompt
    assert "What calls isolated_function?" in synthesis_prompt


def test_auto_expansion_error_is_not_recorded_as_completed(tmp_path):
    from mini_agent.query_plan import QueryPlan
    from mini_agent.sufficiency import ExpansionRequest

    class _ErrorGraphTool:
        def execute_full(self, action, **kwargs):
            return json.dumps({
                "error": "graph query failed",
                "action": action,
            })

    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=_ErrorGraphTool(),
    )
    query_plan = QueryPlan(query="What calls target?")

    agent._execute_expansions(
        [ExpansionRequest(
            action="transitive_callers",
            symbol="src/a.py::target",
            edge_types=["CALLS"],
        )],
        query_plan,
    )

    expansion = query_plan.relation_expansions[0]
    assert expansion["status"] == "failed"
    assert "graph query failed" in expansion["error"]
    assert query_plan.diagnostics


def test_call_paths_expansion_contextualizes_path_nodes(tmp_path):
    from mini_agent.query_plan import QueryPlan
    from mini_agent.sufficiency import ExpansionRequest

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for name in ("start", "middle", "target"):
        (source_dir / f"{name}.py").write_text(
            f"def {name}():\n"
            f"    return '{name}'\n",
            encoding="utf-8",
        )

    graph = nx.MultiDiGraph()
    for name in ("start", "middle", "target"):
        _add_symbol(
            graph,
            f"src/{name}.py::{name}",
            NodeType.FUNCTION,
            name,
            f"src/{name}.py",
        )
        graph.nodes[f"src/{name}.py::{name}"]["end_line"] = 2
    graph.add_edge(
        "src/start.py::start",
        "src/middle.py::middle",
        edge_type=EdgeType.CALLS,
        confidence=0.9,
        strategy="test",
    )
    graph.add_edge(
        "src/middle.py::middle",
        "src/target.py::target",
        edge_type=EdgeType.CALLS,
        confidence=0.9,
        strategy="test",
    )

    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=_graph_tool(graph, tmp_path),
    )
    query_plan = QueryPlan(query="How does start reach target?")

    observed = agent._execute_expansions(
        [ExpansionRequest(
            action="call_paths",
            symbol="src/start.py::start",
            target="src/target.py::target",
            edge_types=["CALLS"],
        )],
        query_plan,
    )

    expansion = query_plan.relation_expansions[0]
    assert expansion["expanded_node_ids"] == [
        "src/start.py::start",
        "src/middle.py::middle",
        "src/target.py::target",
    ]
    assert expansion["source_evidence_ids"]
    assert any(
        item.kind == "source"
        and item.node_id == "src/middle.py::middle"
        for item in observed
    )


def test_shared_callers_zero_result_has_no_source_evidence(tmp_path):
    from mini_agent.query_plan import QueryPlan
    from mini_agent.sufficiency import ExpansionRequest

    graph = nx.MultiDiGraph()
    for name in ("first", "second"):
        _add_symbol(
            graph,
            f"src/{name}.py::{name}",
            NodeType.FUNCTION,
            name,
            f"src/{name}.py",
        )
    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=_graph_tool(graph, tmp_path),
    )
    query_plan = QueryPlan(query="How are first and second related?")

    agent._execute_expansions(
        [ExpansionRequest(
            action="shared_callers",
            symbol="src/first.py::first",
            target="src/second.py::second",
            edge_types=["CALLS"],
        )],
        query_plan,
    )

    expansion = query_plan.relation_expansions[0]
    assert expansion["status"] == "completed"
    assert expansion["result_count"] == 0
    assert expansion["expanded_node_ids"] == []
    assert expansion["source_evidence_ids"] == []


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
    (source_dir / "upstream.py").write_text(
        "def second_hop():\n"
        "    return lint(None)\n"
        "\n"
        "def third_hop():\n"
        "    return second_hop()\n",
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
    _add_symbol(
        graph,
        "src/upstream.py::second_hop",
        NodeType.FUNCTION,
        "second_hop",
        "src/upstream.py",
    )
    _add_symbol(
        graph,
        "src/upstream.py::third_hop",
        NodeType.FUNCTION,
        "third_hop",
        "src/upstream.py",
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
    graph.add_edge(
        "src/upstream.py::second_hop",
        "src/commands.py::lint",
        edge_type=EdgeType.CALLS,
        confidence=0.95,
        strategy="test",
    )
    graph.add_edge(
        "src/upstream.py::third_hop",
        "src/upstream.py::second_hop",
        edge_type=EdgeType.CALLS,
        confidence=0.95,
        strategy="test",
    )

    tool = _graph_tool(graph, tmp_path)
    expansion_calls = []
    original_execute_full = tool.execute_full

    def _record_execute_full(action, **kwargs):
        expansion_calls.append((action, copy.deepcopy(kwargs)))
        return original_execute_full(action, **kwargs)

    tool.execute_full = _record_execute_full
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
    assert result.finish_draft == "header 和 formatter 只是职责分离"
    assert len(model.query_messages) == 2
    expansions = agent.last_query_plan["relation_expansions"]
    assert expansions
    assert "src/commands.py::lint" in expansions[0]["expanded_node_ids"]
    assert "src/upstream.py::second_hop" in expansions[0]["expanded_node_ids"]
    assert "src/upstream.py::third_hop" not in expansions[0]["expanded_node_ids"]
    transitive_call = next(
        call
        for call in expansion_calls
        if call[0] == "transitive_callers"
    )
    assert transitive_call[1]["max_depth"] == 2
    assert transitive_call[1]["min_confidence"] == 0.45
    assert expansions[0]["max_depth"] == 2
    assert expansions[0]["min_confidence"] == 0.45
    assert expansions[0]["status"] == "completed"
    assert expansions[0]["source_evidence_ids"]
    retry_message = model.query_messages[1][-1]["content"]
    assert "[证据充分性检查未通过]" in retry_message
    assert "click.echo(format_linting_result_header())" in retry_message
    synthesis_prompt = model.generate_messages[0]["content"]
    assert "click.echo(format_linting_result_header())" in synthesis_prompt
    assert "header 和 formatter 只是职责分离" in synthesis_prompt


def test_synthesis_sends_and_audits_full_untrimmed_evidence(tmp_path, capsys):
    """完整长证据进入合成请求，不在中间截断。"""
    from mini_agent.evidence import EvidenceItem

    model = _FinalModel()
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
    )
    late_marker = "late-marker-" + ("x" * 8000)
    agent._ledger.add(EvidenceItem(
        evidence_id="ev-1", kind="source", source="query_graph",
        node_id="src/a.py::first_func", file="src/a.py",
        start_line=1, end_line=50,
        payload={"name": "first_func", "type": "FUNCTION",
                 "source_context": "first evidence", "signature": "def first_func()"},
        tool_name="query_graph", tool_args={}, step=1,
    ))
    agent._ledger.add(EvidenceItem(
        evidence_id="ev-2", kind="source", source="read_file",
        node_id="src/b.py::long_func", file="src/b.py",
        start_line=10, end_line=200,
        payload={"name": "long_func", "type": "FUNCTION",
                 "source_context": late_marker, "signature": "def long_func()"},
        tool_name="read_file", tool_args={"path": "src/b.py"}, step=2,
    ))

    # 临时放大预算确保完整长证据被选中
    agent._ledger.SYNTHESIS_CHAR_BUDGET = 50000
    result = agent._synthesize("Explain the complete relationship.", candidates=[])

    assert result.answer == "answer"
    request = agent.last_model_requests[-1]
    assert request["stage"] == "answer_synthesis"
    assert late_marker in request["messages"][0]["content"]


def test_model_request_audit_is_complete_and_human_readable(tmp_path, capsys):
    """审计保存完整 messages，数据不受 LLM 展示预算影响。"""
    audit_output = []
    agent = Agent(
        _FinalModel(),
        Environment(EnvConfig(cwd=str(tmp_path))),
        on_audit=audit_output.append,
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

    # 审计数据保存完整（深拷贝，不受预算影响）
    assert len(agent.last_model_requests) == 1
    saved = agent.last_model_requests[0]
    assert saved["stage"] == "exploration_step_2"
    assert saved["messages"] == messages
    assert saved["messages"] is not messages  # 深拷贝
    # 验证内容完整性
    assert saved["messages"][0]["content"] == "SYSTEM-CONTENT"
    assert saved["messages"][3]["content"] == "FULL-TOOL-RESULT"
    assert len(audit_output) == 1
    assert "消息 1/4 | SYSTEM" in audit_output[0]
    assert "工具名称: query_graph" in audit_output[0]
    assert "调用 ID: call_full" in audit_output[0]
    assert "FULL-TOOL-RESULT" in audit_output[0]


def test_graph_tool_full_result_channel_bypasses_display_trimming(tmp_path):
    tool = GraphTool(str(tmp_path))
    tool._built = True
    marker = "late-marker-" + ("x" * 4000)
    tool._dispatch = lambda action, args: {
        "results": [{"id": f"node-{i}", "source_context": marker} for i in range(12)],
    }

    display_result = tool.execute("contextualize", name="target")
    full_result = tool.execute_full("contextualize", name="target")

    assert len(json.loads(display_result)["results"]) == 10
    assert marker in full_result
    assert len(json.loads(full_result)["results"]) == 12


def test_agent_keeps_full_tool_result_while_budgeting_observation(tmp_path):
    marker = "FULL_LEDGER_MARKER_" + ("x" * 9000)

    class _Retrieval:
        candidates = []

    class _FullGraphTool:
        is_ready = True
        enable_embeddings = False

        def __init__(self):
            self.full_calls = 0

        def ensure_built(self):
            return "ready"

        def retrieve_query_candidates(self, *args, **kwargs):
            return _Retrieval()

        def execute(self, **kwargs):
            raise AssertionError("Agent 不应使用裁剪通道")

        def execute_full(self, **kwargs):
            self.full_calls += 1
            return json.dumps({
                "query": "src/a.py::large",
                "exact": True,
                "results": [{
                    "id": "src/a.py::large",
                    "name": "large",
                    "type": "FUNCTION",
                    "file": "src/a.py",
                    "start_line": 1,
                    "end_line": 300,
                    "source_context": marker,
                    "calls": [],
                    "called_by": [],
                }],
            })

    class _ToolThenFinalModel:
        def __init__(self):
            self.query_messages = []
            self.generate_messages = []

        def query(self, messages):
            self.query_messages.append(copy.deepcopy(messages))
            if len(self.query_messages) == 1:
                return {
                    "content": "读取目标",
                    "raw_content": "THOUGHT: 读取目标",
                    "tool_calls": [{
                        "id": "call_full_result",
                        "type": "function",
                        "function": {
                            "name": "query_graph",
                            "arguments": json.dumps({
                                "action": "contextualize",
                                "name": "src/a.py::large",
                            }),
                        },
                    }],
                }
            return {
                "content": "证据足够",
                "raw_content": "FINAL: 初步结论",
                "tool_calls": [],
            }

        def generate(self, messages):
            self.generate_messages = copy.deepcopy(messages)
            return "最终答案"

    graph_tool = _FullGraphTool()
    model = _ToolThenFinalModel()
    audit_output = []
    agent = Agent(
        model,
        Environment(EnvConfig(cwd=str(tmp_path))),
        graph_tool=graph_tool,
        on_audit=audit_output.append,
    )

    result = agent.run("解释 large")

    assert graph_tool.full_calls == 1
    # 预算提升后完整源码能放入观察消息
    assert marker in model.query_messages[1][-1]["content"]
    assert marker in result.evidence[0].payload["source_context"]
    assert marker in model.generate_messages[0]["content"]
    assert "Agent 结束草稿（仅供参考，不属于证据）" in result.synthesis.prompt
    assert "初步结论" in result.synthesis.prompt
    assert marker in result.model_requests[-1]["full_tool_results"]["call_full_result"]
    assert marker in audit_output[-1]


class TestExactSymbolAnchor:
    """exact_symbol :: 限定名加分 + fuzzy 阈值回归测试。"""

    def test_qualified_name_gets_bonus(self, tmp_path):
        """含 :: 的限定名应比纯简单名分数高。"""
        graph = nx.MultiDiGraph()
        _add_symbol(
            graph,
            "src/a.py::Foo::spam",
            NodeType.METHOD,
            "spam",
            "src/a.py",
            parent_id="src/a.py::Foo",
        )
        _add_symbol(
            graph,
            "src/b.py::spam",
            NodeType.FUNCTION,
            "spam",
            "src/b.py",
        )
        tool = _graph_tool(graph, tmp_path)
        candidates = tool.rank_query_candidates("What does Foo.spam do?")
        # 限定名 Foo::spam 应该排在同名简单函数前面
        assert "Foo" in candidates[0]["id"], (
            f"qualified name should rank first, got {candidates[0]['id']}"
        )

    def test_lexical_multi_field_match(self, tmp_path):
        """多字段命中的候选应比单字段命中分数高。"""
        graph = nx.MultiDiGraph()
        _add_symbol(
            graph,
            "src/formatters.py::format_linting_result_header",
            NodeType.FUNCTION,
            "format_linting_result_header",
            "src/formatters.py",
            docstring="Format the linting result header using StringIO",
            signature="def format_linting_result_header(io)",
        )
        _add_symbol(
            graph,
            "src/other.py::other_func",
            NodeType.FUNCTION,
            "other_func",
            "src/other.py",
            docstring="does something else",
            signature="def other_func()",
        )
        tool = _graph_tool(graph, tmp_path)
        candidates = tool.rank_query_candidates(
            "format linting result header function"
        )
        assert candidates[0]["name"] == "format_linting_result_header"
        assert candidates[0]["score"] > candidates[1]["score"]

    def test_fuzzy_threshold_filters_noise(self, tmp_path):
        """fuzzy_min_similarity=0.35 应过滤掉低相似度候选。"""
        from mini_agent.retrieval import RetrievalConfig
        config = RetrievalConfig()
        assert config.fuzzy_min_similarity == 0.35, (
            f"fuzzy threshold should be 0.35, got {config.fuzzy_min_similarity}"
        )
