# -*- coding: utf-8 -*-
"""多实体并行检索编排器测试。"""

import json

import networkx as nx
import pytest

from accg.models import NodeType
from mini_agent.multi_entity import Entity
from mini_agent.evidence import EvidenceLedger
from mini_agent.multi_entity import MultiEntityOrchestrator, MultiEntityPrelude
from mini_agent.retrieval import Candidate, RetrievalResult


def _build_graph_with_two_symbols(tmp_path):
    """构建包含两个符号的图。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text(
        "def format_header():\n    return 'header'\n", encoding="utf-8"
    )
    (src / "b.py").write_text(
        "class OutputStreamFormatter:\n    pass\n", encoding="utf-8"
    )
    graph = nx.MultiDiGraph()
    graph.add_node(
        "src/a.py::format_header",
        node_type=NodeType.FUNCTION,
        name="format_header",
        file_path="src/a.py",
        start_line=1,
        end_line=2,
        docstring="Format a header.",
        signature="def format_header()",
        parent_id=None,
        extra={},
    )
    graph.add_node(
        "src/b.py::OutputStreamFormatter",
        node_type=NodeType.CLASS,
        name="OutputStreamFormatter",
        file_path="src/b.py",
        start_line=1,
        end_line=2,
        docstring="Output stream formatter.",
        signature="",
        parent_id=None,
        extra={},
    )
    return graph


class _StubGraphTool:
    """桩 GraphTool：按 entity.query 返回预设候选。"""

    def __init__(self, graph, candidates_by_query=None):
        self._graph = graph
        self._built = True
        self._candidates_by_query = candidates_by_query or {}
        self._default_candidates = [
            Candidate(
                id="src/a.py::format_header",
                name="format_header",
                type="FUNCTION",
                file="src/a.py",
                score=200.0,
                sources=["lexical"],
            ),
        ]

    def ensure_built(self):
        return "ready"

    @property
    def is_ready(self):
        return True

    def search(self, query, limit=24, use_embeddings=None):
        candidates = self._candidates_by_query.get(
            query, self._default_candidates
        )
        return RetrievalResult(
            candidates=candidates,
            stages_attempted=["exact_id", "exact_symbol", "lexical"],
            stages_succeeded=["lexical"],
            diagnostics=[],
        )

    def retrieve_query_candidates(self, text, limit=12, use_embeddings=True):
        return self.search(text, limit=limit, use_embeddings=use_embeddings)

    def select_query_anchors(self, task, candidates, max_anchors=3,
                             preferred_types=None, required_types=None,
                             prefer_term_coverage=False):
        result = []
        for c in candidates[:max_anchors]:
            d = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            d.setdefault("matched_terms", [])
            d.setdefault("matched_fields", ["name"])
            d.setdefault("covered_terms", [])
            d.setdefault("selection_reason", "score_top1")
            d.setdefault("candidate_sources", d.get("sources", []))
            result.append(d)
        return result

    def validate_query_anchor(self, anchor):
        nid = anchor.get("id", "")
        return {
            "valid": nid in self._graph.nodes,
            "reason": "exact_contextualize_result" if nid in self._graph.nodes else "node_id_not_found",
            "message": "",
            "suggestions": [],
        }

    def inspect(self, node_id):
        node = self._graph.nodes.get(node_id, {})
        return {
            "results": [{
                "id": node_id,
                "name": node.get("name", node_id),
                "type": str(node.get("node_type", "FUNCTION")),
                "file": node.get("file_path", ""),
                "start_line": node.get("start_line", 1),
                "end_line": node.get("end_line", 2),
                "signature": node.get("signature", ""),
                "docstring": node.get("docstring", ""),
                "source_context": f"1| {node.get('name', '')}",
                "calls": [],
                "called_by": [],
                "methods": [],
                "inherits": [],
                "instantiated_by": [],
            }],
        }


def test_per_entity_retrieval_finds_both_symbols(tmp_path):
    graph = _build_graph_with_two_symbols(tmp_path)
    stub = _StubGraphTool(graph, {
        "format_header": [
            Candidate(
                id="src/a.py::format_header",
                name="format_header",
                type="FUNCTION",
                file="src/a.py",
                score=200.0,
                sources=["lexical"],
            ),
        ],
        "OutputStreamFormatter": [
            Candidate(
                id="src/b.py::OutputStreamFormatter",
                name="OutputStreamFormatter",
                type="CLASS",
                file="src/b.py",
                score=190.0,
                sources=["lexical"],
            ),
        ],
    })
    orch = MultiEntityOrchestrator(stub)
    ledger = EvidenceLedger()
    entities = [
        Entity(name="format_header", query="format_header",
               description="格式化函数", type_hint="FUNCTION"),
        Entity(name="OutputStreamFormatter", query="OutputStreamFormatter",
               description="格式化器类", type_hint="CLASS"),
    ]

    prelude = orch.run(
        entities=entities,
        task="Compare format_header and OutputStreamFormatter",
        ledger=ledger,
        recommended_count=2,
    )

    # 两个实体都有锚点
    anchor_ids = {a["id"] for e in prelude.entity_anchors.values()
                  for a in e}
    assert "src/a.py::format_header" in anchor_ids
    assert "src/b.py::OutputStreamFormatter" in anchor_ids
    # 账本有两条 source 证据
    assert len(ledger.source_items) == 2


def test_prelude_text_includes_both_entities(tmp_path):
    graph = _build_graph_with_two_symbols(tmp_path)
    stub = _StubGraphTool(graph, {
        "format_header": [
            Candidate(
                id="src/a.py::format_header",
                name="format_header",
                type="FUNCTION",
                file="src/a.py",
                score=200.0,
                sources=["lexical"],
            ),
        ],
        "OutputStreamFormatter": [
            Candidate(
                id="src/b.py::OutputStreamFormatter",
                name="OutputStreamFormatter",
                type="CLASS",
                file="src/b.py",
                score=190.0,
                sources=["lexical"],
            ),
        ],
    })
    orch = MultiEntityOrchestrator(stub)
    ledger = EvidenceLedger()
    entities = [
        Entity(name="format_header", query="format_header",
               description="格式化函数", type_hint="FUNCTION"),
        Entity(name="OutputStreamFormatter", query="OutputStreamFormatter",
               description="格式化器类", type_hint="CLASS"),
    ]

    prelude = orch.run(
        entities=entities,
        task="Compare format_header and OutputStreamFormatter",
        ledger=ledger,
        recommended_count=2,
    )

    assert "format_header" in prelude.text
    assert "OutputStreamFormatter" in prelude.text
    # 实体标签出现在 prelude 中
    assert "格式化函数" in prelude.text
    assert "格式化器类" in prelude.text


def test_single_entity_produces_one_anchor(tmp_path):
    graph = _build_graph_with_two_symbols(tmp_path)
    stub = _StubGraphTool(graph)
    orch = MultiEntityOrchestrator(stub)
    ledger = EvidenceLedger()
    entities = [
        Entity(name="format_header", query="format_header",
               description="格式化函数", type_hint="FUNCTION"),
    ]

    prelude = orch.run(
        entities=entities,
        task="What does format_header do?",
        ledger=ledger,
        recommended_count=1,
    )

    assert len(prelude.entity_anchors) == 1
    assert len(prelude.entity_anchors["format_header"]) == 1


def test_invalid_anchor_is_rejected(tmp_path):
    graph = _build_graph_with_two_symbols(tmp_path)
    stub = _StubGraphTool(graph, {
        "missing_func": [
            Candidate(
                id="src/nonexistent.py::missing_func",
                name="missing_func",
                type="FUNCTION",
                file="src/nonexistent.py",
                score=500.0,
                sources=["exact_symbol"],
            ),
        ],
        "format_header": [
            Candidate(
                id="src/a.py::format_header",
                name="format_header",
                type="FUNCTION",
                file="src/a.py",
                score=100.0,
                sources=["lexical"],
            ),
        ],
    })
    orch = MultiEntityOrchestrator(stub)
    ledger = EvidenceLedger()
    entities = [
        Entity(name="missing_func", query="missing_func",
               description="不存在的函数", type_hint="FUNCTION"),
        Entity(name="format_header", query="format_header",
               description="格式化函数", type_hint="FUNCTION"),
    ]

    prelude = orch.run(
        entities=entities,
        task="Compare missing_func and format_header",
        ledger=ledger,
        recommended_count=2,
    )

    # missing_func 被拒绝
    rejected = prelude.rejected_anchors
    missing = [r for r in rejected if "missing" in str(r.get("candidate", {}).get("id", ""))]
    assert len(missing) >= 1
    # format_header 被接受
    assert len(prelude.entity_anchors.get("format_header", [])) >= 1


def test_entity_anchors_record_entity_name(tmp_path):
    """每个锚点对应到相同的实体名"""
    graph = _build_graph_with_two_symbols(tmp_path)
    stub = _StubGraphTool(graph, {
        "format_header": [
            Candidate(
                id="src/a.py::format_header",
                name="format_header",
                type="FUNCTION",
                file="src/a.py",
                score=200.0,
                sources=["lexical"],
            ),
        ],
    })
    orch = MultiEntityOrchestrator(stub)
    ledger = EvidenceLedger()
    entities = [
        Entity(name="format_header", query="format_header",
               description="格式化函数", type_hint="FUNCTION"),
    ]

    prelude = orch.run(
        entities=entities,
        task="test",
        ledger=ledger,
        recommended_count=1,
    )

    assert prelude.entity_anchors["format_header"][0]["id"] == "src/a.py::format_header"


class TestMultiEntityCandidateRecording:
    """验证多实体编排保存候选并生成可评估的检索指标。"""

    def test_two_entities_hit_two_symbols(self, tmp_path):
        """两个实体分别命中两个符号，prelude.candidates 包含两者。"""
        graph = _build_graph_with_two_symbols(tmp_path)
        stub = _StubGraphTool(graph, {
            "calculate_tax": [
                Candidate(
                    id="src/a.py::calculate_tax",
                    name="calculate_tax", type="FUNCTION",
                    file="src/a.py", score=200.0, sources=["lexical"],
                ),
            ],
            "format_header": [
                Candidate(
                    id="src/b.py::format_header",
                    name="format_header", type="FUNCTION",
                    file="src/b.py", score=150.0, sources=["lexical"],
                ),
            ],
        })
        orch = MultiEntityOrchestrator(stub)
        ledger = EvidenceLedger()
        entities = [
            Entity(name="calculate_tax", query="calculate_tax",
                   description="计算税率", type_hint="FUNCTION"),
            Entity(name="format_header", query="format_header",
                   description="格式化头部", type_hint="FUNCTION"),
        ]

        prelude = orch.run(
            entities=entities,
            task="Compare calculate_tax and format_header",
            ledger=ledger,
            recommended_count=2,
        )

        candidate_ids = {c.id for c in prelude.candidates}
        assert "src/a.py::calculate_tax" in candidate_ids
        assert "src/b.py::format_header" in candidate_ids
        assert len(prelude.candidates) >= 2

    def test_duplicate_candidate_keeps_highest_score(self, tmp_path):
        """同一 id 的候选被两个实体命中时保留最高分。"""
        graph = _build_graph_with_two_symbols(tmp_path)
        candidate = Candidate(
            id="src/a.py::shared_func",
            name="shared_func", type="FUNCTION",
            file="src/a.py", score=100.0, sources=["lexical"],
        )
        stronger = Candidate(
            id="src/a.py::shared_func",
            name="shared_func", type="FUNCTION",
            file="src/a.py", score=300.0, sources=["exact_symbol"],
        )
        stub = _StubGraphTool(graph, {
            "entity_a": [candidate],
            "entity_b": [stronger],
        })
        orch = MultiEntityOrchestrator(stub)
        ledger = EvidenceLedger()
        entities = [
            Entity(name="A", query="entity_a", description="a", type_hint="FUNCTION"),
            Entity(name="B", query="entity_b", description="b", type_hint="FUNCTION"),
        ]

        prelude = orch.run(
            entities=entities,
            task="test",
            ledger=ledger,
            recommended_count=2,
        )

        shared = [c for c in prelude.candidates if c.id == "src/a.py::shared_func"]
        assert len(shared) == 1
        assert shared[0].score == 300.0

    def test_retrieval_result_includes_candidates(self, tmp_path):
        """MultiEntityOrchestrator.run 返回的 prelude.candidates 包含正确的 Candidate 列表。"""
        graph = _build_graph_with_two_symbols(tmp_path)
        stub = _StubGraphTool(graph, {
            "format_header": [
                Candidate(
                    id="src/b.py::format_header",
                    name="format_header", type="FUNCTION",
                    file="src/b.py", score=200.0, sources=["lexical"],
                ),
            ],
        })
        orch = MultiEntityOrchestrator(stub)
        ledger = EvidenceLedger()

        prelude = orch.run(
            entities=[Entity(name="fmt", query="format_header",
                            description="format", type_hint="FUNCTION")],
            task="What does format_header do?",
            ledger=ledger,
            recommended_count=1,
        )

        assert len(prelude.candidates) == 1
        c = prelude.candidates[0]
        assert isinstance(c, Candidate)
        assert c.id == "src/b.py::format_header"
        assert "lexical" in prelude.stages_attempted
        assert "lexical" in prelude.stages_succeeded
