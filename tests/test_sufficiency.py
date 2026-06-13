# -*- coding: utf-8 -*-
"""证据充分性门控测试 — P4 FinishAction + SufficiencyGate。"""

from mini_agent.sufficiency import (
    FinishAction,
    SufficiencyGate,
    GateDecision,
    ExpansionRequest,
)


class TestFinishAction:
    def test_parses_final_with_colon(self):
        fa = FinishAction.from_content("FINAL: 这是答案草案")
        assert fa is not None
        assert fa.draft == "这是答案草案"

    def test_parses_final_with_newline(self):
        fa = FinishAction.from_content("THOUGHT: 分析完毕\nFINAL: 最终答案在这里")
        assert fa is not None
        assert fa.draft == "最终答案在这里"

    def test_parses_final_without_colon(self):
        fa = FinishAction.from_content("FINAL 无冒号的答案")
        assert fa is not None
        assert fa.draft == "无冒号的答案"

    def test_no_final_returns_none(self):
        fa = FinishAction.from_content("THOUGHT: 还需要更多信息")
        assert fa is None

    def test_final_before_action_limits_scope(self):
        fa = FinishAction.from_content(
            "FINAL: 答案\nACTION: 不应该解析这个"
        )
        assert fa is not None
        assert "ACTION" not in fa.draft

    def test_multiline_draft(self):
        fa = FinishAction.from_content(
            "FINAL: 第一行\n第二行\n第三行"
        )
        assert fa is not None
        assert "第一行" in fa.draft
        assert "第二行" in fa.draft


class TestSufficiencyGateBasic:
    def test_no_evidence_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释一下这个函数怎么工作",
        )
        assert not decision.passed
        assert any("源码" in m for m in decision.missing_requirements)

    def test_single_source_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 get_environ_proxies 函数",
            evidence_items=[
                _src("src/a.py::get_environ_proxies"),
            ],
        )
        assert decision.passed

    def test_comparison_with_two_sources_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Compare the format_header function and OutputFormatter class.",
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::OutputFormatter"),
            ],
        )
        assert decision.passed

    def test_comparison_with_one_source_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Compare the format_header function and OutputFormatter class.",
            evidence_items=[
                _src("src/a.py::format_header"),
            ],
        )
        assert not decision.passed
        assert any("2 个" in m for m in decision.missing_requirements)

    def test_relationship_without_connection_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question=(
                "What is the relationship between the header formatting "
                "function and the output stream formatter class?"
            ),
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::OutputStreamFormatter"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header", atype="FUNCTION"),
                    _anchor("src/a.py::OutputStreamFormatter", atype="CLASS"),
                ],
            },
        )
        assert not decision.passed
        assert decision.expansion_requests

    def test_relationship_with_connecting_edges_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question=(
                "What is the relationship between the header formatting "
                "function and the output stream formatter class?"
            ),
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::OutputStreamFormatter"),
                _rel("CALLS", "src/b.py::caller", "src/a.py::format_header", 0.9),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputStreamFormatter"),
                    _anchor("src/b.py::caller"),
                ],
            },
        )
        assert decision.passed

    def test_low_confidence_edge_does_not_count(self):
        gate = SufficiencyGate()
        gate.DEFAULT_MIN_CONFIDENCE = 0.8
        decision = gate.evaluate(
            question="What calls the format function?",
            evidence_items=[
                _src("src/a.py::format"),
                _rel("CALLS", "src/b.py::caller", "src/a.py::format", 0.3),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format"),
                ],
            },
        )
        assert not decision.passed

    def test_expansion_respects_max_count(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls the format function?",
            evidence_items=[_src("src/a.py::format")],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format", atype="FUNCTION"),
                ],
            },
            expansion_count=2,  # 已达上限
        )
        assert not decision.passed
        assert not decision.expansion_requests

    def test_inheritance_with_hierarchy_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the inheritance hierarchy of BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
                _rel("INHERITS", "src/a.py::ChildFormatter", "src/a.py::BaseFormatter", 0.9),
            ],
        )
        assert decision.passed

    def test_inheritance_without_hierarchy_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the inheritance hierarchy of BaseFormatter?",
            evidence_items=[_src("src/a.py::BaseFormatter")],
        )
        assert not decision.passed

    def test_negative_conclusion_needs_search_record(self):
        """否定关系结论应记录查询方向、边类型、深度和置信度阈值。"""
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls the isolated_function?",
            evidence_items=[
                _src("src/a.py::isolated_function"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::isolated_function", atype="FUNCTION"),
                ],
            },
            draft="There is no caller for this function.",
        )
        assert not decision.passed
        # 否定结论需要搜索记录
        assert any(
            "否定" in m or "关系" in m
            for m in decision.missing_requirements
        )

    def test_gate_passed_reasons_are_recorded(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 get_environ_proxies",
            evidence_items=[_src("src/a.py::get_environ_proxies")],
        )
        assert decision.passed
        assert decision.reasons

    def test_gate_failure_missing_are_recorded(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls isolated_function?",
            evidence_items=[_src("src/a.py::isolated_function")],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::isolated_function", atype="FUNCTION"),
                ],
            },
        )
        assert not decision.passed
        assert decision.missing_requirements


# ── 辅助 ──────────────────────────────────────────────────────

class _FakeItem:
    def __init__(self, kind, node_id=None, complete=True, confidence=None,
                 edge_type=None, source_node_id=None, target_node_id=None,
                 payload=None):
        self.kind = kind
        self.node_id = node_id
        self.complete = complete
        self.confidence = confidence
        self.edge_type = edge_type
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.payload = payload or {}


def _src(node_id):
    return _FakeItem("source", node_id=node_id)


def _rel(edge_type, src_id, tgt_id, confidence):
    return _FakeItem(
        "relation",
        edge_type=edge_type,
        source_node_id=src_id,
        target_node_id=tgt_id,
        confidence=confidence,
    )


def _anchor(aid, atype="FUNCTION"):
    return {"id": aid, "type": atype, "name": aid.split("::")[-1],
            "validation": {"valid": True}}
