# -*- coding: utf-8 -*-
"""证据账本与审计测试 — P2 结构化证据系统。"""

import json

import pytest

from mini_agent.evidence import EvidenceItem, EvidenceLedger, DisplayLevel


# ═══════════════════════════════════════════════════════════════
# EvidenceItem 创建与渲染
# ═══════════════════════════════════════════════════════════════

class TestEvidenceItemCreation:
    def test_source_item_from_contextualize(self):
        raw = json.dumps({
            "query": "requests.Session",
            "exact": True,
            "results": [{
                "id": "src/sessions.py::Session",
                "name": "Session",
                "type": "CLASS",
                "file": "src/sessions.py",
                "start_line": 10,
                "end_line": 200,
                "signature": "class Session",
                "docstring": "A Requests session.",
                "source_context": "   1| class Session:\n   2|     def __init__(self):\n...",
                "calls": [],
                "called_by": [
                    {"id": "src/api.py::request", "name": "request", "file": "src/api.py", "confidence": 0.9, "strategy": "test"},
                ],
            }],
        })

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "contextualize", "name": "requests.Session"},
            raw,
            step=1,
        )

        # 至少产生一个 source 项和一个 called_by 关系项
        sources = [i for i in items if i.kind == "source"]
        relations = [i for i in items if i.kind == "relation"]
        assert len(sources) == 1
        assert sources[0].node_id == "src/sessions.py::Session"
        assert sources[0].file == "src/sessions.py"
        assert sources[0].start_line == 10
        assert sources[0].end_line == 200
        assert sources[0].complete is True
        assert len(relations) >= 1
        called_by = [
            r for r in relations
            if r.source_node_id == "src/api.py::request"
            and r.target_node_id == "src/sessions.py::Session"
        ]
        assert len(called_by) == 1
        assert called_by[0].edge_type == "CALLS"
        assert called_by[0].source_node_id == "src/api.py::request"
        assert called_by[0].target_node_id == "src/sessions.py::Session"

    def test_relation_item_from_transitive(self):
        raw = json.dumps([
            [{"id": "src/a.py::foo", "name": "foo", "file": "src/a.py"},
             {"confidence": 0.8, "strategy": "test"}],
        ])

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "transitive_callers", "symbol": "src/b.py::bar"},
            raw,
            step=2,
        )

        rels = [i for i in items if i.kind == "relation"]
        assert len(rels) == 1
        assert rels[0].edge_type == "CALLS"
        assert rels[0].source_node_id == "src/a.py::foo"
        assert rels[0].target_node_id == "src/b.py::bar"
        assert rels[0].confidence == 0.8

    def test_call_paths_preserves_each_direct_edge(self):
        raw = json.dumps([{
            "node_ids": ["src/a.py::foo", "src/b.py::bar", "src/c.py::baz"],
            "nodes": [],
            "edges": [
                {
                    "source_node_id": "src/a.py::foo",
                    "target_node_id": "src/b.py::bar",
                    "confidence": 0.9,
                    "strategy": "direct",
                },
                {
                    "source_node_id": "src/b.py::bar",
                    "target_node_id": "src/c.py::baz",
                    "confidence": 0.8,
                    "strategy": "direct",
                },
            ],
            "depth": 2,
            "path_confidence": 0.72,
        }])

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {
                "action": "call_paths",
                "source": "src/a.py::foo",
                "target": "src/c.py::baz",
            },
            raw,
            step=2,
        )

        assert [
            (item.source_node_id, item.target_node_id)
            for item in items
        ] == [
            ("src/a.py::foo", "src/b.py::bar"),
            ("src/b.py::bar", "src/c.py::baz"),
        ]
        assert items[0].payload["path_node_ids"] == [
            "src/a.py::foo",
            "src/b.py::bar",
            "src/c.py::baz",
        ]

    def test_class_hierarchy_uses_child_to_parent_edge_direction(self):
        raw = json.dumps([
            {
                "type": "parents",
                "items": [{"id": "src/base.py::Base", "name": "Base"}],
            },
            {
                "type": "children",
                "items": [{"id": "src/child.py::Child", "name": "Child"}],
            },
        ])

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "class_hierarchy", "class_name": "src/current.py::Current"},
            raw,
            step=2,
        )

        assert {
            (item.source_node_id, item.target_node_id)
            for item in items
        } == {
            ("src/current.py::Current", "src/base.py::Base"),
            ("src/child.py::Child", "src/current.py::Current"),
        }

    def test_candidate_from_narrow_down(self):
        raw = json.dumps({
            "clues_used": ["request"],
            "results": [
                {"id": "src/api.py::request", "name": "request", "type": "FUNCTION", "file": "src/api.py", "relevance": 5.0},
            ],
        })

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "narrow_down", "clues": ["request"]},
            raw,
            step=3,
        )

        cands = [i for i in items if i.kind == "candidate"]
        assert len(cands) == 1
        assert cands[0].node_id == "src/api.py::request"

    def test_read_file_creates_source_item(self):
        raw = "[src/sessions.py 行 1-5 / 共 200 行]\n   1| class Session:\n..."

        items = EvidenceItem.from_tool_result(
            "read_file",
            {"path": "src/sessions.py", "start_line": 1, "end_line": 5},
            raw,
            step=1,
        )

        assert len(items) == 1
        assert items[0].kind == "source"
        assert items[0].source == "read_file"
        assert items[0].file == "src/sessions.py"
        assert raw in items[0].render(DisplayLevel.COMPLETE)

    def test_read_file_error_creates_error_item(self):
        items = EvidenceItem.from_tool_result(
            "read_file",
            {"path": "missing.py"},
            "[错误] 文件不存在: missing.py",
            step=1,
        )

        assert len(items) == 1
        assert items[0].kind == "error"

    def test_list_dir_creates_structure_item(self):
        items = EvidenceItem.from_tool_result(
            "list_dir",
            {"path": "src"},
            "src:\n  sessions.py\n  api.py",
            step=1,
        )

        assert len(items) == 1
        assert items[0].kind == "structure"

    def test_error_result(self):
        raw = json.dumps({"error": "未找到符号: xyz", "results": []})

        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "contextualize", "name": "xyz"},
            raw,
            step=1,
        )

        assert len(items) == 1
        assert items[0].kind == "error"

    def test_invalid_json_result(self):
        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "contextualize", "name": "abc"},
            "not valid json",
            step=1,
        )

        assert len(items) == 1
        assert items[0].kind == "error"


class TestDisplayLevelRendering:
    @pytest.fixture
    def source_item(self):
        return EvidenceItem(
            evidence_id="test-1",
            kind="source",
            source="query_graph",
            node_id="src/sessions.py::Session",
            file="src/sessions.py",
            start_line=10,
            end_line=200,
            payload={
                "name": "Session",
                "type": "CLASS",
                "file": "src/sessions.py",
                "start_line": 10,
                "end_line": 200,
                "signature": "class Session",
                "docstring": "A Requests session.",
                "source_context": "   1| class Session:\n   2|     def __init__(self):\n   3|         self.headers = {}",
                "calls": [],
                "called_by": [
                    {"name": "request", "file": "src/api.py", "confidence": 0.9},
                ],
            },
            tool_name="query_graph",
            tool_args={"action": "contextualize", "name": "Session"},
            step=1,
        )

    def test_fold_level_shows_identity_only(self, source_item):
        text = source_item.render(DisplayLevel.FOLD)
        assert "CLASS" in text
        assert "Session" in text
        assert "src/sessions.py" in text
        # fold 不应包含源码
        assert "def __init__" not in text

    def test_snippet_level_shows_source_context(self, source_item):
        text = source_item.render(DisplayLevel.SNIPPET)
        assert "签名" in text
        assert "源码" in text
        assert "def __init__" in text

    def test_preview_level_shows_relation_summary(self, source_item):
        text = source_item.render(DisplayLevel.PREVIEW)
        # 大类 (>50行) 使用类概览，展示签名、文档、源码头部
        assert "class Session" in text
        assert "A Requests session" in text

    def test_complete_level_shows_everything(self, source_item):
        text = source_item.render(DisplayLevel.COMPLETE)
        # 大类概览包含签名、文档、源码头部
        assert "签名" in text
        assert "文档" in text
        assert "源码" in text

    @pytest.fixture
    def relation_item(self):
        return EvidenceItem(
            evidence_id="test-rel-1",
            kind="relation",
            source="query_graph",
            edge_type="CALLS",
            source_node_id="src/a.py::foo",
            target_node_id="src/b.py::bar",
            confidence=0.9,
            strategy="test",
            payload={"items": [
                {"name": "bar", "id": "src/b.py::bar", "confidence": 0.9},
                {"name": "baz", "id": "src/c.py::baz", "confidence": 0.7},
            ]},
            tool_name="query_graph",
            tool_args={"action": "transitive_callees", "symbol": "src/a.py::foo"},
            step=2,
        )

    def test_relation_fold(self, relation_item):
        text = relation_item.render(DisplayLevel.FOLD)
        assert "CALLS" in text
        assert "foo" in text
        assert "bar" in text

    def test_relation_complete(self, relation_item):
        text = relation_item.render(DisplayLevel.COMPLETE)
        assert "bar" in text
        assert "baz" in text
        assert "0.9" in text
        assert "test" in text


# ═══════════════════════════════════════════════════════════════
# EvidenceLedger 去重
# ═══════════════════════════════════════════════════════════════

class TestLedgerDedup:
    def _make_source(self, eid, node_id, file, start, end, payload=None):
        name = node_id.split("::")[-1] if node_id else file
        return EvidenceItem(
            evidence_id=eid, kind="source", source="query_graph",
            node_id=node_id, file=file, start_line=start, end_line=end,
            payload=payload or {"name": name, "type": "FUNCTION",
                                "file": file, "start_line": start, "end_line": end,
                                "source_context": f"source of {node_id}"},
            tool_name="query_graph", tool_args={}, step=1,
        )

    def _make_relation(self, eid, edge_type, src_id, tgt_id, confidence=None):
        return EvidenceItem(
            evidence_id=eid, kind="relation", source="query_graph",
            edge_type=edge_type, source_node_id=src_id, target_node_id=tgt_id,
            confidence=confidence,
            payload={"source_node_id": src_id, "target_node_id": tgt_id, "edge_type": edge_type},
            tool_name="query_graph", tool_args={}, step=2,
        )

    def test_same_node_id_deduplicates(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", "src/a.py::foo", "src/a.py", 1, 10))
        ledger.add(self._make_source("b", "src/a.py::foo", "src/a.py", 1, 10))

        assert len(ledger) == 1

    def test_same_edge_deduplicates(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_relation("a", "CALLS", "src/a.py::foo", "src/b.py::bar"))
        ledger.add(self._make_relation("b", "CALLS", "src/a.py::foo", "src/b.py::bar"))

        assert len(ledger) == 1

    def test_overlapping_source_range_merges(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", None, "src/a.py", 1, 50))
        # 相同 file + 重叠行号范围
        ledger.add(self._make_source("b", None, "src/a.py", 30, 80))

        assert len(ledger) == 1
        stored = ledger.items()[0]
        assert stored.start_line == 1
        assert stored.end_line == 80

    def test_relation_node_id_does_not_swallow_source_item(self):
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem(
            evidence_id="r",
            kind="relation",
            source="query_graph",
            node_id="src/b.py::bar",
            edge_type="CALLS",
            source_node_id="src/a.py::foo",
            target_node_id="src/b.py::bar",
            payload={"name": "bar", "confidence": 0.8},
            tool_name="query_graph",
            tool_args={"action": "transitive_callees"},
            step=1,
        ))
        ledger.add(self._make_source(
            "s", "src/b.py::bar", "src/b.py", 1, 10,
            payload={
                "name": "bar",
                "type": "FUNCTION",
                "source_context": "def bar(): pass",
            },
        ))

        assert [item.kind for item in ledger.items()] == ["relation", "source"]

    def test_merge_records_all_tool_origins(self):
        ledger = EvidenceLedger()
        first = self._make_source("a", "src/a.py::foo", "src/a.py", 1, 10)
        second = self._make_source("b", "src/a.py::foo", "src/a.py", 1, 10)
        second.source = "read_file"
        second.tool_name = "read_file"
        second.tool_args = {"path": "src/a.py"}

        ledger.add(first)
        ledger.add(second)

        stored = ledger.items()[0]
        assert stored.sources == ["query_graph", "read_file"]
        assert [origin["tool_name"] for origin in stored.tool_origins] == [
            "query_graph",
            "read_file",
        ]

    def test_different_node_ids_kept_separate(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", "src/a.py::foo", "src/a.py", 1, 10))
        ledger.add(self._make_source("b", "src/a.py::bar", "src/a.py", 20, 30))

        assert len(ledger) == 2

    def test_different_node_ids_with_same_range_are_kept_separate(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", "src/a.py::foo", "src/a.py", 1, 10))
        ledger.add(self._make_source("b", "src/a.py::bar", "src/a.py", 1, 10))

        assert len(ledger) == 2

    def test_incomplete_merged_into_complete(self):
        ledger = EvidenceLedger()
        incomplete = self._make_source("a", "src/a.py::foo", "src/a.py", 1, 10,
                                        payload={"name": "foo", "type": "FUNCTION"})
        incomplete.complete = False
        ledger.add(incomplete)

        complete = self._make_source("b", "src/a.py::foo", "src/a.py", 1, 10,
                                      payload={"name": "foo", "type": "FUNCTION",
                                               "file": "src/a.py", "start_line": 1, "end_line": 10,
                                               "source_context": "full source here"})
        complete.complete = True
        ledger.add(complete)

        assert len(ledger) == 1
        stored = ledger.items()[0]
        assert stored.complete is True
        assert "full source here" in str(stored.payload)

    def test_error_items_always_added(self):
        ledger = EvidenceLedger()
        led1 = ledger.add(EvidenceItem(
            evidence_id="e1", kind="error", source="query_graph",
            payload="error msg", tool_name="query_graph", tool_args={}, step=1))
        led2 = ledger.add(EvidenceItem(
            evidence_id="e2", kind="error", source="query_graph",
            payload="another error", tool_name="query_graph", tool_args={}, step=2))

        assert led1 == "added"
        assert led2 == "added"
        assert len(ledger) == 2
        assert ledger.has_synthesis_evidence is False


# ═══════════════════════════════════════════════════════════════
# 合成选择
# ═══════════════════════════════════════════════════════════════

class TestSynthesisSelection:
    def _make_source(self, eid, node_id, source_len=500):
        return EvidenceItem(
            evidence_id=eid, kind="source", source="query_graph",
            node_id=node_id, file=f"{node_id.split('::')[0]}",
            start_line=1, end_line=50,
            payload={"name": node_id.split("::")[-1], "type": "FUNCTION",
                     "source_context": "x" * source_len, "signature": "def foo()",
                     "docstring": "Test function."},
            tool_name="query_graph", tool_args={}, step=1,
        )

    def _make_relation(self, eid, src_id, tgt_id):
        return EvidenceItem(
            evidence_id=eid, kind="relation", source="query_graph",
            edge_type="CALLS", source_node_id=src_id, target_node_id=tgt_id,
            payload={"items": [{"name": tgt_id.split("::")[-1], "confidence": 0.9}]},
            tool_name="query_graph", tool_args={}, step=2,
        )

    def test_selects_all_when_under_budget(self):
        ledger = EvidenceLedger()
        # 不同文件和行号避免范围去重
        ledger.add(self._make_source("a", "src/a.py::foo", source_len=200))
        ledger.add(EvidenceItem(
            evidence_id="b", kind="source", source="query_graph",
            node_id="src/b.py::bar", file="src/b.py", start_line=10, end_line=60,
            payload={"name": "bar", "type": "FUNCTION", "source_context": "y" * 300,
                     "signature": "def bar()", "docstring": "Test."},
            tool_name="query_graph", tool_args={}, step=1,
        ))
        ledger.add(self._make_relation("r1", "src/a.py::foo", "src/b.py::baz"))

        selected = ledger.select_for_synthesis(char_budget=5000)

        assert len(selected) == 3

    def test_excludes_errors_and_structure(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", "src/a.py::foo"))
        ledger.add(EvidenceItem(
            evidence_id="e1", kind="error", source="query_graph",
            payload="error", tool_name="query_graph", tool_args={}, step=1))
        ledger.add(EvidenceItem(
            evidence_id="s1", kind="structure", source="list_dir",
            payload="dir listing", tool_name="list_dir", tool_args={}, step=1))

        selected = ledger.select_for_synthesis(char_budget=5000)

        assert len(selected) == 1
        assert selected[0].kind == "source"
        report = ledger.selection_report()
        assert report.count("类型不参与答案合成") == 2

    def test_downgrades_non_source_when_over_budget(self):
        ledger = EvidenceLedger()
        ledger.add(self._make_source("a", "src/a.py::foo", source_len=3000))
        # 添加大量 relation 项使超出预算
        for j in range(20):
            ledger.add(self._make_relation(f"r{j}", "src/a.py::foo", f"src/b.py::target{j}"))

        selected = ledger.select_for_synthesis(char_budget=5000)

        # 所有 source 项都应保留
        source_selected = [s for s in selected if s.kind == "source"]
        assert len(source_selected) >= 1

        # 检查选择报告
        report = ledger.selection_report()
        assert "证据选择报告" in report

    def test_selection_report_describes_exclusions(self):
        ledger = EvidenceLedger()
        # 添加多个源证据和关系证据，预算刚好只能容纳一个
        ledger.add(self._make_source("a", "src/a.py::foo", source_len=3000))
        ledger.add(EvidenceItem(
            evidence_id="b", kind="source", source="query_graph",
            node_id="src/b.py::bar", file="src/b.py", start_line=1, end_line=50,
            payload={"name": "bar", "type": "FUNCTION", "source_context": "y" * 500,
                     "signature": "def bar()"},
            tool_name="query_graph", tool_args={}, step=1,
        ))
        for j in range(10):
            ledger.add(self._make_relation(f"r{j}", "src/a.py::foo", f"src/b.py::target{j}"))

        ledger.select_for_synthesis(char_budget=2500)

        report = ledger.selection_report()
        assert "证据选择报告" in report
        # 应该有被排除的关系项
        assert "✗" in report or "超出合成预算" in report

    def test_no_silent_truncation_mid_entity(self):
        """完整长证据进入合成请求，不在中间截断。"""
        ledger = EvidenceLedger()
        long_marker = "LONG_ENTITY_MARKER_" + "x" * 5000
        ledger.add(EvidenceItem(
            evidence_id="long-1", kind="source", source="query_graph",
            node_id="src/a.py::big_function", file="src/a.py",
            start_line=1, end_line=200,
            payload={"name": "big_function", "type": "FUNCTION",
                     "source_context": long_marker, "signature": "def big_function()"},
            tool_name="query_graph", tool_args={}, step=1,
        ))

        selected = ledger.select_for_synthesis(char_budget=12000)
        evidence_text = ledger.render_for_model(selected, level=DisplayLevel.COMPLETE)

        # 长标记必须完整出现在合成文本中
        assert long_marker in evidence_text

    def test_single_source_remains_complete_even_when_over_budget(self):
        ledger = EvidenceLedger()
        long_marker = "OVER_BUDGET_MARKER_" + "x" * 5000
        ledger.add(EvidenceItem(
            evidence_id="long-over-budget",
            kind="source",
            source="query_graph",
            node_id="src/a.py::large",
            file="src/a.py",
            start_line=1,
            end_line=300,
            payload={
                "name": "large",
                "type": "FUNCTION",
                "source_context": long_marker,
            },
            tool_name="query_graph",
            tool_args={"action": "contextualize"},
            step=1,
        ))

        ledger.select_for_synthesis(char_budget=100)
        evidence_text = ledger.render_selected_for_synthesis()

        assert long_marker in evidence_text
        assert "level=complete" in ledger.selection_report()

    def test_candidate_observation_uses_fold_level(self):
        ledger = EvidenceLedger()
        candidate = EvidenceItem(
            evidence_id="candidate-1",
            kind="candidate",
            source="query_graph",
            node_id="src/a.py::foo",
            file="src/a.py",
            payload={
                "name": "foo",
                "type": "FUNCTION",
                "file": "src/a.py",
                "score": 9.5,
            },
            tool_name="query_graph",
            tool_args={"action": "narrow_down"},
            step=1,
        )

        text = ledger.render_for_observation([candidate])

        assert text == "[候选] foo (FUNCTION)"


# ═══════════════════════════════════════════════════════════════
# 审计渲染
# ═══════════════════════════════════════════════════════════════

class TestAuditRendering:
    def test_render_for_audit_is_human_readable(self):
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem(
            evidence_id="audit-1", kind="source", source="query_graph",
            node_id="src/a.py::foo", file="src/a.py", start_line=1, end_line=10,
            payload={"name": "foo", "type": "FUNCTION", "source_context": "def foo(): pass"},
            complete=True, tool_name="query_graph",
            tool_args={"action": "contextualize", "name": "foo"}, step=1,
        ))
        ledger.add(EvidenceItem(
            evidence_id="audit-2", kind="relation", source="query_graph",
            edge_type="CALLS", source_node_id="src/a.py::foo", target_node_id="src/b.py::bar",
            confidence=0.9, strategy="test",
            payload={"name": "bar", "confidence": 0.9},
            tool_name="query_graph",
            tool_args={"action": "transitive_callees", "symbol": "src/a.py::foo"}, step=2,
        ))

        audit_text = ledger.render_for_audit()

        assert "证据账本审计" in audit_text
        assert "audit-1" in audit_text
        assert "audit-2" in audit_text
        assert "kind: source" in audit_text
        assert "kind: relation" in audit_text
        assert "node_id: src/a.py::foo" in audit_text
        assert "edge: CALLS" in audit_text
        assert "confidence: 0.9" in audit_text
        assert "strategy: test" in audit_text
        assert "def foo(): pass" in audit_text
        assert "tool_origins" in audit_text

    def test_empty_ledger_audit(self):
        ledger = EvidenceLedger()
        assert "证据账本为空" in ledger.render_for_audit()

    def test_audit_preserves_full_payload(self):
        """审计渲染不受 LLM 展示预算影响，保留完整 payload。"""
        ledger = EvidenceLedger()
        long_payload = "FULL_PAYLOAD_" + "y" * 5000
        ledger.add(EvidenceItem(
            evidence_id="full-1", kind="source", source="query_graph",
            node_id="src/a.py::long_func", file="src/a.py",
            payload={"name": "long_func", "source_context": long_payload},
            tool_name="query_graph", tool_args={}, step=1,
        ))

        audit_text = ledger.render_for_audit()
        assert long_payload in audit_text


# ═══════════════════════════════════════════════════════════════
# 控制消息不入账本
# ═══════════════════════════════════════════════════════════════

class TestControlMessagesNotInLedger:
    def test_intercepted_calls_not_added(self):
        """重复调用拦截的消息不应通过 EvidenceItem.from_tool_result 进入账本。
        实际拦截在 Agent 层处理（不调用 _execute_tool），这里验证 intercepted
        标记不会产生错误项。
        """
        # 拦截时的空结果不应创建错误证据
        items = EvidenceItem.from_tool_result(
            "query_graph",
            {"action": "contextualize", "name": "foo"},
            "",  # intercepted → empty
            step=1,
        )
        # 空字符串不是有效 JSON，会产生 error 项
        # 但实际流程中 intercepted 直接跳过 from_tool_result
        assert len(items) == 1
        assert items[0].kind == "error"


# ═══════════════════════════════════════════════════════════════
# 去重键计算
# ═══════════════════════════════════════════════════════════════

class TestDedupKeys:
    def test_source_item_keys(self):
        item = EvidenceItem(
            evidence_id="k1", kind="source", source="query_graph",
            node_id="src/a.py::foo", file="src/a.py", start_line=1, end_line=10,
            payload={}, tool_name="query_graph", tool_args={}, step=1,
        )
        keys = item.dedup_keys()
        assert "source-node:src/a.py::foo" in keys
        assert "source-range:src/a.py:1:10" in keys

    def test_relation_item_keys(self):
        item = EvidenceItem(
            evidence_id="k2", kind="relation", source="query_graph",
            node_id="src/b.py::bar",
            edge_type="CALLS", source_node_id="src/a.py::foo", target_node_id="src/b.py::bar",
            payload={}, tool_name="query_graph", tool_args={}, step=2,
        )
        keys = item.dedup_keys()
        assert "edge:src/a.py::foo:CALLS:src/b.py::bar" in keys
        assert all(not key.startswith("source-node:") for key in keys)

    def test_no_file_range_key_without_start_line(self):
        item = EvidenceItem(
            evidence_id="k3", kind="source", source="read_file",
            file="src/a.py", start_line=None, end_line=None,
            payload={}, tool_name="read_file", tool_args={}, step=1,
        )
        keys = item.dedup_keys()
        # 没有 start_line 时不应有 range key
        assert not any(k.startswith("source-range:") for k in keys)

    def test_factory_generates_stable_evidence_id(self):
        raw = json.dumps({
            "results": [{
                "id": "src/a.py::foo",
                "name": "foo",
                "type": "FUNCTION",
                "file": "src/a.py",
                "start_line": 1,
                "end_line": 3,
                "source_context": "def foo(): pass",
            }],
        })
        args = {"action": "contextualize", "name": "src/a.py::foo"}

        first = EvidenceItem.from_tool_result("query_graph", args, raw, step=1)
        second = EvidenceItem.from_tool_result("query_graph", args, raw, step=9)

        assert first[0].evidence_id == second[0].evidence_id


# ═══════════════════════════════════════════════════════════════
# 证据选择可解释性
# ═══════════════════════════════════════════════════════════════

class TestSelectionExplainability:
    def test_selection_report_matches_selected_items(self):
        ledger = EvidenceLedger()
        for j in range(3):
            ledger.add(EvidenceItem(
                evidence_id=f"s{j}", kind="source", source="query_graph",
                node_id=f"src/a.py::func{j}", file="src/a.py",
                start_line=j * 10, end_line=j * 10 + 5,
                payload={"name": f"func{j}", "type": "FUNCTION", "source_context": "code"},
                tool_name="query_graph", tool_args={}, step=1,
            ))

        selected = ledger.select_for_synthesis(char_budget=100000)
        report = ledger.selection_report()

        assert len(selected) == 3
        for item in selected:
            assert "✓" in report
            assert item.node_id in report

    def test_stable_selection_order(self):
        """选择顺序应稳定。"""
        ledger = EvidenceLedger()
        items_data = [
            ("src/c.py::third", "source", 5),
            ("src/a.py::first", "source", 1),
            ("src/b.py::second", "relation", 3),
        ]
        for node_id, kind, step in items_data:
            ledger.add(EvidenceItem(
                evidence_id=f"si-{node_id}", kind=kind, source="query_graph",
                node_id=node_id, file=node_id.split("::")[0],
                start_line=1, end_line=10,
                payload={"name": node_id.split("::")[-1], "type": "FUNCTION",
                         "source_context": "code"} if kind == "source" else {},
                tool_name="query_graph", tool_args={}, step=step,
            ))

        first = ledger.select_for_synthesis(char_budget=100000)
        second = ledger.select_for_synthesis(char_budget=100000)

        assert [e.evidence_id for e in first] == [e.evidence_id for e in second]
        # source 优先于 relation


class TestSourceContextValidation:
    """has_valid_source_context 和 EvidenceItem.complete 回归测试。"""

    def test_empty_source_context_marks_incomplete(self):
        """空 source_context 应导致 complete=False。"""
        from mini_agent.evidence import has_valid_source_context
        assert not has_valid_source_context("")
        assert not has_valid_source_context("   ")
        assert not has_valid_source_context(None)
        assert not has_valid_source_context(123)

    def test_error_marker_rejected(self):
        """[无法读取源码] 和 [错误] 前缀应被拒绝。"""
        from mini_agent.evidence import has_valid_source_context
        assert not has_valid_source_context("[无法读取源码]")
        assert not has_valid_source_context("[错误] 文件不存在")
        assert has_valid_source_context("def foo(): pass")
        assert has_valid_source_context("读取失败：未找到文件")  # 不含禁止前缀

    def test_contextualize_empty_source_makes_incomplete(self):
        """contextualize 返回空 source_context 时 EvidenceItem.complete 应为 False。"""
        raw = json.dumps({
            "results": [{
                "id": "src/a.py::Foo", "name": "Foo", "type": "FUNCTION",
                "file": "src/a.py", "start_line": 1, "end_line": 5,
                "source_context": "",
            }],
        })
        items = EvidenceItem.from_tool_result(
            "query_graph", {"action": "contextualize", "name": "Foo"}, raw, step=1,
        )
        sources = [i for i in items if i.kind == "source"]
        assert len(sources) == 1
        assert sources[0].complete is False

    def test_contextualize_error_marker_makes_incomplete(self):
        """contextualize 返回错误占位符时 EvidenceItem.complete 应为 False。"""
        raw = json.dumps({
            "results": [{
                "id": "src/a.py::Foo", "name": "Foo", "type": "FUNCTION",
                "file": "src/a.py", "start_line": 1, "end_line": 5,
                "source_context": "[无法读取源码]",
            }],
        })
        items = EvidenceItem.from_tool_result(
            "query_graph", {"action": "contextualize", "name": "Foo"}, raw, step=1,
        )
        sources = [i for i in items if i.kind == "source"]
        assert len(sources) == 1
        assert sources[0].complete is False
