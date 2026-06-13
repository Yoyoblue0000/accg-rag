# -*- coding: utf-8 -*-
"""证据充分性门控 — 确定性规则判断证据是否足够进入答案合成。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FinishAction:
    """模型显式提出的完成请求。"""
    draft: str = ""
    raw_content: str = ""

    @classmethod
    def from_content(cls, content: str) -> "FinishAction | None":
        """从 LLM 原始输出解析 FINAL 文本为 FinishAction。"""
        m = _FINAL_PATTERN.search(content)
        if not m:
            return None
        return cls(draft=m.group(1).strip(), raw_content=content)


_FINAL_PATTERN = re.compile(
    r"FINAL[:\s]\s*(.+?)(?=\n(?:ACTION|THOUGHT):|\Z)", re.DOTALL
)


@dataclass
class ExpansionRequest:
    """门控失败后触发的受控关系扩展请求。"""
    action: str               # traverse action: "transitive_callers" | "call_paths" | "class_hierarchy"
    symbol: str               # 扩展的源符号
    target: str | None = None # call_paths 需要的目标符号
    max_depth: int = 2
    reason: str = ""
    edge_types: list[str] = field(default_factory=list)


@dataclass
class GateDecision:
    """证据充分性门控的判定结果。"""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    expansion_requests: list[ExpansionRequest] = field(default_factory=list)


# ── 问题类型检测 ──────────────────────────────────────────────

_COMPARISON_KEYWORDS = {
    "compare", "comparison", "between", "relationship",
    "difference", "versus", "vs", "differ", "contrast",
}
_RELATION_KEYWORDS = {
    "call", "calls", "calling", "caller", "callee",
    "invoke", "invocation", "invokes",
    "inherit", "inherits", "inheritance", "subclass", "parent", "child",
    "instantiate", "instantiation", "instance",
    "relationship", "relation", "connect", "connection",
    "flow", "dataflow", "path",
}
_INHERITANCE_KEYWORDS = {
    "inherit", "inherits", "inheritance", "subclass", "superclass",
    "parent", "child", "base class", "derived",
    "hierarchy", "extends", "polymorphic", "polymorphism",
}


def _has_keywords(text: str, keywords: set[str]) -> bool:
    return bool(set(re.findall(r"[a-z]+", text.lower())) & keywords)


def _count_source_items(evidence_items: list) -> int:
    return sum(1 for e in evidence_items if getattr(e, "kind", "") == "source")


def _count_relation_items(evidence_items: list) -> int:
    return sum(1 for e in evidence_items if getattr(e, "kind", "") == "relation")


def _get_validated_anchor_ids(query_plan: dict) -> set[str]:
    anchors = query_plan.get("anchors", [])
    return {
        a.get("id", "")
        for a in anchors
        if a.get("validation", {}).get("valid")
    }


def _get_source_node_ids(evidence_items: list) -> set[str]:
    ids = set()
    for e in evidence_items:
        if getattr(e, "kind", "") == "source" and getattr(e, "node_id", None):
            ids.add(e.node_id)
        if getattr(e, "kind", "") == "source" and hasattr(e, "payload"):
            p = e.payload if isinstance(e.payload, dict) else {}
            nid = p.get("id", "")
            if nid:
                ids.add(nid)
    return ids


def _get_high_conf_relations(evidence_items: list, min_conf: float = 0.45) -> list:
    return [
        e for e in evidence_items
        if getattr(e, "kind", "") == "relation"
        and (getattr(e, "confidence", None) or 0) >= min_conf
    ]


class SufficiencyGate:
    """确定性证据充分性门控。"""

    DEFAULT_MIN_CONFIDENCE = 0.45
    EXPANSION_MAX_DEPTH = 2
    MAX_AUTO_EXPANSIONS = 2

    def evaluate(
        self,
        question: str,
        query_plan: dict | None = None,
        evidence_items: list | None = None,
        draft: str = "",
        expansion_count: int = 0,
        expanded_relations: set[str] | None = None,
    ) -> GateDecision:
        """评估当前证据是否足够回答用户问题。"""
        if query_plan is None:
            query_plan = {}
        if evidence_items is None:
            evidence_items = []
        if expanded_relations is None:
            expanded_relations = set()

        reasons: list[str] = []
        missing: list[str] = []
        expansions: list[ExpansionRequest] = []

        source_items = [e for e in evidence_items if getattr(e, "kind", "") == "source"]
        relation_items = [e for e in evidence_items if getattr(e, "kind", "") == "relation"]
        validated_anchors = _get_validated_anchor_ids(query_plan)
        source_ids = _get_source_node_ids(evidence_items)
        anchors = query_plan.get("anchors", [])

        # ── 最低要求：至少一个 source 证据 ──
        if not source_items:
            return GateDecision(
                passed=False,
                reasons=["无任何源码证据"],
                missing_requirements=["至少需要一个源码证据才能合成答案"],
            )

        is_comparison = _has_keywords(question, _COMPARISON_KEYWORDS)
        is_relation = _has_keywords(question, _RELATION_KEYWORDS)
        is_inheritance = _has_keywords(question, _INHERITANCE_KEYWORDS)

        # anchor 实体在 draft 中可定位（信息性，非阻断）
        if draft:
            draft_entities = sum(
                1 for a in anchors
                if a.get("name", "") and a["name"].lower() in draft.lower()
            )
            if draft_entities > 0:
                reasons.append(f"草稿中引用了 {draft_entities} 个锚点实体")

        # ── 单实体解释 ──
        if not is_comparison and not is_relation and not is_inheritance:
            if validated_anchors:
                reasons.append(f"{len(validated_anchors)} 个已验证锚点")
                complete_sources = [
                    e for e in source_items
                    if getattr(e, "complete", False)
                    and getattr(e, "node_id", None) in validated_anchors
                ]
                if complete_sources:
                    reasons.append(f"{len(complete_sources)} 个锚点有完整源码")
                else:
                    missing.append("锚点缺少完整源码")
            elif source_items:
                reasons.append(f"{len(source_items)} 个源码证据（无预取锚点）")
            else:
                missing.append("缺少源码证据")

            if not missing:
                return GateDecision(passed=True, reasons=reasons)

        # ── 多实体比较 ──
        if is_comparison:
            source_count = len(source_items)
            entity_names_in_sources = source_ids | {
                getattr(e, "node_id", "") for e in source_items
            }
            if source_count >= 2:
                reasons.append(f"比较问题有 {source_count} 个实体的源码证据")
            elif source_count == 1:
                missing.append("比较问题需要至少 2 个实体的源码证据")
                # 请求扩展：inspect 第二个候选锚点
                second_anchor = None
                for a in anchors:
                    if a.get("id") not in source_ids:
                        second_anchor = a
                        break
                if second_anchor and expansion_count < self.MAX_AUTO_EXPANSIONS:
                    ctx_key = f"contextualize:{second_anchor.get('id', '')}"
                    if ctx_key not in expanded_relations:
                        expansions.append(ExpansionRequest(
                            action="contextualize",
                            symbol=second_anchor.get("id", ""),
                            reason="比较问题缺少第二个实体的源码",
                        ))
            else:
                missing.append("比较问题需要至少 2 个实体的源码证据")

        # ── 继承关系 ──
        inherits_rels: list = []
        if is_inheritance:
            inherits_rels = [
                e for e in relation_items
                if getattr(e, "edge_type", "") == "INHERITS"
            ]
            if inherits_rels:
                reasons.append(f"{len(inherits_rels)} 条继承关系证据")
            else:
                missing.append("继承问题缺少类层次证据")
                for a in anchors:
                    if a.get("type") == "CLASS" and expansion_count < self.MAX_AUTO_EXPANSIONS:
                        ch_key = f"class_hierarchy:{a.get('id', '')}"
                        if ch_key not in expanded_relations:
                            expansions.append(ExpansionRequest(
                                action="class_hierarchy",
                                symbol=a.get("id", ""),
                                reason=f"继承问题缺少类层次，扩展 {a.get('id','')} 的继承关系",
                                edge_types=["INHERITS"],
                            ))
                            break

        # ── 关系要求（继承证据充足时可跳过） ──
        if is_relation and not (is_inheritance and inherits_rels):
            high_conf_rels = _get_high_conf_relations(
                evidence_items, self.DEFAULT_MIN_CONFIDENCE
            )
            # 检查关系是否连接至少两个不同锚点/源实体
            anchor_ids = {a.get("id", "") for a in anchors} | source_ids
            connected_entities: set[str] = set()
            for rel in high_conf_rels:
                src = getattr(rel, "source_node_id", "") or ""
                tgt = getattr(rel, "target_node_id", "") or ""
                if src in anchor_ids:
                    connected_entities.add(src)
                if tgt in anchor_ids:
                    connected_entities.add(tgt)

            if high_conf_rels and len(connected_entities) >= 2:
                reasons.append(
                    f"{len(high_conf_rels)} 条高置信度关系证据，"
                    f"连接 {len(connected_entities)} 个锚点实体"
                )
            else:
                if not high_conf_rels:
                    missing.append(
                        "关系问题缺少满足置信度阈值的关系证据"
                    )
                else:
                    missing.append(
                        "关系证据未连接多个锚点实体，"
                        "需要共享调用者或跨实体路径"
                    )
                # 请求扩展：call_paths 优先（共享调用者），再回退到单锚点 transitive_callers
                if anchors and expansion_count < self.MAX_AUTO_EXPANSIONS:
                    func_anchors = [
                        a for a in anchors
                        if a.get("type") in ("FUNCTION", "METHOD")
                    ]
                    # 优先：两个锚点间的 call_paths
                    if len(func_anchors) >= 2:
                        src = func_anchors[0].get("id", "")
                        tgt = func_anchors[1].get("id", "")
                        cp_key = f"call_paths:{src}:{tgt}"
                        if cp_key not in expanded_relations:
                            expansions.append(ExpansionRequest(
                                action="call_paths",
                                symbol=src,
                                target=tgt,
                                max_depth=self.EXPANSION_MAX_DEPTH,
                                reason=f"关系问题缺少跨实体关系，扩展 {src} ↔ {tgt} 调用路径",
                                edge_types=["CALLS"],
                            ))
                    if not expansions:
                        # 回退：单锚点传递调用者
                        for a in func_anchors:
                            tc_key = f"transitive_callers:{a.get('id', '')}"
                            if tc_key not in expanded_relations:
                                expansions.append(ExpansionRequest(
                                    action="transitive_callers",
                                    symbol=a.get("id", ""),
                                    max_depth=self.EXPANSION_MAX_DEPTH,
                                    reason=f"关系问题缺少跨实体关系证据，扩展 {a.get('id','')} 的传递调用者",
                                    edge_types=["CALLS"],
                                ))
                                break

        # ── 否定结论检测 ──
        if draft and _has_keywords(draft, {"no", "not", "none", "doesn't", "don't", "isn't"}):
            if not relation_items and is_relation:
                scope_parts = []
                if anchors:
                    scope_parts.append(f"查询目标: {', '.join(a.get('id','') for a in anchors[:2])}")
                scope_parts.append(f"边类型: CALLS, INHERITS, INSTANTIATED_BY")
                scope_parts.append(f"搜索深度: ≤{self.EXPANSION_MAX_DEPTH}")
                scope_parts.append(f"置信度阈值: ≥{self.DEFAULT_MIN_CONFIDENCE}")
                if expanded_relations:
                    scope_parts.append(
                        f"已尝试: {', '.join(sorted(expanded_relations)[:5])}"
                    )
                missing.append(
                    "否定关系结论需要搜索范围记录: " + "; ".join(scope_parts)
                )

        passed = len(missing) == 0
        return GateDecision(
            passed=passed,
            reasons=reasons,
            missing_requirements=missing,
            expansion_requests=expansions if not passed else [],
        )
