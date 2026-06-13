# -*- coding: utf-8 -*-
"""证据充分性门控测试 — P4 FinishAction + SufficiencyGate。"""

from mini_agent.sufficiency import (
    FinishAction,
    SufficiencyGate,
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

    def test_final_text_inside_thought_does_not_finish(self):
        fa = FinishAction.from_content(
            "THOUGHT: 现在不要输出 FINAL: 还需要继续查询"
        )
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
            query_plan={
                "anchors": [
                    _anchor("src/a.py::get_environ_proxies"),
                ],
            },
        )
        assert decision.passed

    def test_single_source_without_validated_anchor_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 get_environ_proxies 函数",
            evidence_items=[
                _src("src/a.py::get_environ_proxies"),
            ],
        )

        assert not decision.passed
        assert any(
            "已验证锚点" in item
            for item in decision.missing_requirements
        )

    def test_comparison_with_two_sources_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Compare the format_header function and OutputFormatter class.",
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::OutputFormatter"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputFormatter", atype="CLASS"),
                ],
            },
        )
        assert decision.passed

    def test_comparison_with_one_source_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Compare the format_header function and OutputFormatter class.",
            evidence_items=[
                _src("src/a.py::format_header"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputFormatter", atype="CLASS"),
                ],
            },
        )
        assert not decision.passed
        assert any("2 个" in m for m in decision.missing_requirements)

    def test_comparison_requires_independent_sources_for_each_entity(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Compare format_header and OutputFormatter.",
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::format_header"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputFormatter", atype="CLASS"),
                ],
            },
        )

        assert not decision.passed
        assert any(
            "每个主要实体" in item
            for item in decision.missing_requirements
        )

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

    def test_two_callable_entities_expand_shared_callers_first(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the relationship between parse and render?",
            evidence_items=[
                _src("src/a.py::parse"),
                _src("src/a.py::render"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::parse"),
                    _anchor("src/a.py::render"),
                ],
            },
        )

        assert not decision.passed
        assert len(decision.expansion_requests) == 1
        request = decision.expansion_requests[0]
        assert request.action == "shared_callers"
        assert request.symbol == "src/a.py::parse"
        assert request.target == "src/a.py::render"

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
                _rel(
                    "CALLS",
                    "src/a.py::format_header",
                    "src/a.py::OutputStreamFormatter",
                    0.9,
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputStreamFormatter"),
                ],
            },
        )
        assert decision.passed

    def test_relationship_edge_to_third_anchor_does_not_connect_primary_entities(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the relationship between format_header and OutputFormatter?",
            evidence_items=[
                _src("src/a.py::format_header"),
                _src("src/a.py::OutputFormatter"),
                _src("src/b.py::caller"),
                _rel(
                    "CALLS",
                    "src/b.py::caller",
                    "src/a.py::format_header",
                    0.9,
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                    _anchor("src/a.py::OutputFormatter"),
                    _anchor("src/b.py::caller"),
                ],
            },
        )

        assert not decision.passed
        assert any(
            "主要实体" in item
            for item in decision.missing_requirements
        )

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

    def test_relation_without_strategy_does_not_count(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls the format function?",
            evidence_items=[
                _src("src/a.py::format"),
                _rel(
                    "CALLS",
                    "src/b.py::caller",
                    "src/a.py::format",
                    0.9,
                    strategy=None,
                ),
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

    def test_instantiation_question_uses_bounded_contextualize_search(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Who instantiates BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
            },
        )

        assert not decision.passed
        assert len(decision.expansion_requests) == 1
        request = decision.expansion_requests[0]
        assert request.action == "contextualize"
        assert request.edge_types == ["INSTANTIATED_BY"]

    def test_negative_instantiation_passes_after_direct_search_record(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Who instantiates BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
                "relation_expansions": [{
                    "action": "contextualize",
                    "symbol": "src/a.py::BaseFormatter",
                    "target": None,
                    "max_depth": 1,
                    "min_confidence": 0.45,
                    "edge_types": ["INSTANTIATED_BY"],
                    "status": "completed",
                    "result_count": 1,
                    "relation_result_count": 0,
                }],
            },
            draft="No code instantiates BaseFormatter in the bounded scope.",
            expansion_count=1,
            expanded_relations={
                "contextualize:src/a.py::BaseFormatter",
            },
        )

        assert decision.passed

    def test_chinese_relation_question_uses_relation_gate(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="parse 和 render 的调用关系是什么？",
            evidence_items=[
                _src("src/a.py::parse"),
                _src("src/a.py::render"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::parse"),
                    _anchor("src/a.py::render"),
                ],
            },
        )

        assert not decision.passed
        assert decision.expansion_requests

    def test_inheritance_with_hierarchy_passes(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the inheritance hierarchy of BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
                _rel(
                    "INHERITS",
                    "src/a.py::ChildFormatter",
                    "src/a.py::BaseFormatter",
                    0.9,
                    payload={
                        "hierarchy_item": {
                            "id": "src/a.py::ChildFormatter",
                            "file": "src/a.py",
                        },
                    },
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
            },
        )
        assert decision.passed

    def test_inheritance_without_hierarchy_fails(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the inheritance hierarchy of BaseFormatter?",
            evidence_items=[_src("src/a.py::BaseFormatter")],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
            },
        )
        assert not decision.passed

    def test_inheritance_relation_must_touch_primary_class(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What is the inheritance hierarchy of BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
                _rel(
                    "INHERITS",
                    "src/other.py::Child",
                    "src/other.py::Base",
                    0.9,
                    payload={
                        "hierarchy_item": {
                            "id": "src/other.py::Child",
                        },
                    },
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
            },
        )

        assert not decision.passed

    def test_negative_inheritance_passes_after_hierarchy_search(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What subclasses inherit from BaseFormatter?",
            evidence_items=[
                _src("src/a.py::BaseFormatter"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::BaseFormatter", atype="CLASS"),
                ],
                "relation_expansions": [{
                    "action": "class_hierarchy",
                    "symbol": "src/a.py::BaseFormatter",
                    "target": None,
                    "max_depth": 2,
                    "min_confidence": 0.45,
                    "edge_types": ["INHERITS"],
                    "status": "completed",
                    "result_count": 0,
                    "relation_result_count": 0,
                }],
            },
            draft="No subclasses were found in the bounded hierarchy search.",
            expansion_count=1,
            expanded_relations={
                "class_hierarchy:src/a.py::BaseFormatter",
            },
        )

        assert decision.passed

    def test_draft_code_entity_must_be_locatable_in_evidence(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 get_environ_proxies 函数",
            evidence_items=[
                _src("src/a.py::get_environ_proxies"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::get_environ_proxies"),
                ],
            },
            draft="The result is delegated to `UnknownProxyService`.",
        )

        assert not decision.passed
        assert any(
            "UnknownProxyService" in item
            for item in decision.missing_requirements
        )

    def test_relation_draft_mentions_choose_later_validated_anchors(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question=(
                "What is the propagation path through the indentation "
                "checking workflow?"
            ),
            evidence_items=[
                _src("src/respace.py::_construct_alignment_whitespace"),
                _src("src/rule.py::Rule::_get_indentation"),
                _src("src/reindent.py::construct_single_indent"),
                _rel(
                    "CALLS",
                    "src/rule.py::Rule::_get_indentation",
                    "src/reindent.py::construct_single_indent",
                    0.9,
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor(
                        "src/respace.py::_construct_alignment_whitespace"
                    ),
                    _anchor("src/rule.py::Rule::_get_indentation"),
                    _anchor("src/reindent.py::construct_single_indent"),
                ],
            },
            draft=(
                "`_get_indentation` obtains indentation through "
                "`construct_single_indent`."
            ),
        )

        assert decision.passed

    def test_draft_identifier_present_in_source_is_locatable(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 format_header 函数",
            evidence_items=[
                _src(
                    "src/a.py::format_header",
                    source_context=(
                        "from io import StringIO\n"
                        "def format_header():\n"
                        "    buffer = StringIO()\n"
                    ),
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::format_header"),
                ],
            },
            draft="The function builds its result with `StringIO`.",
        )

        assert decision.passed

    def test_draft_parent_classes_are_locatable_in_nested_node_ids(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释两个方法如何协调",
            evidence_items=[
                _src(
                    "src/config.py::FluffConfig::process_raw_file_for_config"
                ),
                _src("src/noqa.py::IgnoreMask::from_source"),
            ],
            query_plan={
                "anchors": [
                    _anchor(
                        "src/config.py::FluffConfig::process_raw_file_for_config"
                    ),
                    _anchor("src/noqa.py::IgnoreMask::from_source"),
                ],
            },
            draft="`FluffConfig` coordinates with `IgnoreMask`.",
        )

        assert decision.passed

    def test_draft_literals_expressions_and_sql_phrases_are_not_entities(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="Where does StringLexer.search return a tuple or None?",
            evidence_items=[
                _src("src/lexer.py::StringLexer::search"),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/lexer.py::StringLexer::search"),
                ],
            },
            draft=(
                "`forward_string.find(self.template)` returns a position or "
                "`None`; the separate grammar phrase "
                "`WHEN NOT MATCHED BY SOURCE` is syntax, not a code entity."
            ),
        )

        assert decision.passed

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

    def test_none_and_sql_not_do_not_create_negative_relation_claim(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls parser?",
            evidence_items=[
                _src("src/a.py::parser"),
                _rel(
                    "CALLS",
                    "src/a.py::caller",
                    "src/a.py::parser",
                    0.9,
                ),
            ],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::parser"),
                ],
            },
            draft=(
                "The caller handles `None` while parsing "
                "`WHEN NOT MATCHED BY SOURCE`."
            ),
        )

        assert decision.passed
        assert not any(
            "否定关系结论" in item
            for item in decision.missing_requirements
        )

    def test_negative_conclusion_passes_after_bounded_search_is_recorded(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="What calls the isolated_function?",
            evidence_items=[
                _src("src/a.py::isolated_function"),
            ],
            query_plan={
                "anchors": [
                    _anchor(
                        "src/a.py::isolated_function",
                        atype="FUNCTION",
                    ),
                ],
                "relation_expansions": [{
                    "action": "transitive_callers",
                    "symbol": "src/a.py::isolated_function",
                    "target": None,
                    "max_depth": 2,
                    "min_confidence": 0.45,
                    "edge_types": ["CALLS"],
                    "status": "completed",
                    "result_count": 0,
                }],
            },
            draft="There is no caller in the bounded search scope.",
            expansion_count=1,
            expanded_relations={
                "transitive_callers:src/a.py::isolated_function",
            },
        )

        assert decision.passed
        assert any(
            "限定搜索" in reason
            for reason in decision.reasons
        )

    def test_gate_passed_reasons_are_recorded(self):
        gate = SufficiencyGate()
        decision = gate.evaluate(
            question="解释 get_environ_proxies",
            evidence_items=[_src("src/a.py::get_environ_proxies")],
            query_plan={
                "anchors": [
                    _anchor("src/a.py::get_environ_proxies"),
                ],
            },
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
                 payload=None, file=None, start_line=None, end_line=None,
                 strategy="test"):
        self.kind = kind
        self.node_id = node_id
        self.complete = complete
        self.confidence = confidence
        self.edge_type = edge_type
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.payload = payload or {}
        self.file = file
        self.start_line = start_line
        self.end_line = end_line
        self.strategy = strategy


def _src(node_id, source_context="pass"):
    file_path = node_id.split("::", 1)[0]
    return _FakeItem(
        "source",
        node_id=node_id,
        file=file_path,
        start_line=1,
        end_line=3,
        payload={
            "id": node_id,
            "name": node_id.split("::")[-1],
            "type": "FUNCTION",
            "file": file_path,
            "start_line": 1,
            "end_line": 3,
            "source_context": source_context,
        },
    )


def _rel(
    edge_type,
    src_id,
    tgt_id,
    confidence,
    payload=None,
    strategy="test",
):
    return _FakeItem(
        "relation",
        edge_type=edge_type,
        source_node_id=src_id,
        target_node_id=tgt_id,
        confidence=confidence,
        payload=payload,
        strategy=strategy,
    )


def _anchor(aid, atype="FUNCTION"):
    return {"id": aid, "type": atype, "name": aid.split("::")[-1],
            "validation": {"valid": True}}
