# -*- coding: utf-8 -*-
"""证据充分性门控 — 确定性规则判断证据是否足够进入答案合成。"""

from __future__ import annotations

import re
from collections import deque
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
    r"^[ \t]*FINAL[:\s]\s*(.+?)(?=^[ \t]*(?:ACTION|THOUGHT):|\Z)",
    re.DOTALL | re.MULTILINE,
)


@dataclass
class ExpansionRequest:
    """门控失败后触发的受控关系扩展请求。"""
    action: str               # traverse action: "transitive_callers" | "call_paths" | "class_hierarchy"
    symbol: str               # 扩展的源符号
    target: str | None = None # call_paths 需要的目标符号
    max_depth: int = 2
    min_confidence: float = 0.45
    reason: str = ""
    edge_types: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        parts = [self.action, self.symbol]
        if self.target:
            parts.append(self.target)
        return ":".join(parts)


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
    "比较", "对比", "区别", "差异", "关系",
}
_RELATION_KEYWORDS = {
    "call", "calls", "calling", "caller", "callee",
    "invoke", "invocation", "invokes",
    "inherit", "inherits", "inheritance", "subclass", "parent", "child",
    "instantiate", "instantiation", "instance",
    "relationship", "relation", "connect", "connection",
    "flow", "dataflow", "path",
    "调用", "被调用", "调用者", "被调用者", "关系", "连接",
    "数据流", "路径", "继承", "实例化",
}
_INHERITANCE_KEYWORDS = {
    "inherit", "inherits", "inheritance", "subclass", "superclass",
    "parent", "child", "base class", "derived",
    "hierarchy", "extends", "polymorphic", "polymorphism",
    "继承", "子类", "父类", "基类", "派生类", "类层次", "多态",
}
_INSTANTIATION_KEYWORDS = {
    "instantiate", "instantiates", "instantiated", "instantiating",
    "instantiation", "instance", "constructor", "construct",
    "实例化", "创建实例", "构造",
}
_CODE_LITERALS = {
    "none", "true", "false", "null", "nil",
    "self", "cls",
}
_PROTOCOL_TOOL_NAMES = {
    "query_graph",
    "contextualize",
    "narrow_down",
    "extract_clues",
    "transitive_callers",
    "transitive_callees",
    "call_paths",
    "class_hierarchy",
    "module_tree",
    "module_structure",
    "read_file",
    "list_dir",
}
_NEGATIVE_RELATION_PATTERNS = (
    re.compile(
        r"\bno\s+(?:(?:direct|indirect|bounded)\s+)?"
        r"(?:caller|callers|callee|callees|call|calls|path|paths|"
        r"relation|relationship|connection|subclass|subclasses|"
        r"parent|parents|child|children|instance|instances|"
        r"instantiation|instantiations)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+code\s+"
        r"(?:calls?|invokes?|inherits?|instantiates?|connects?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does\s+not|doesn't|do\s+not|don't|is\s+not|isn't|"
        r"was\s+not|wasn't|cannot|can't)\s+"
        r"(?:directly\s+|indirectly\s+)?"
        r"(?:call|called|invoke|invoked|inherit|inherited|instantiate|"
        r"instantiated|connect|connected)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:没有|不存在|未发现|找不到).{0,12}"
        r"(?:调用|调用者|被调用者|路径|关系|连接|继承|子类|父类|"
        r"实例化|实例)"
    ),
    re.compile(r"无(?:调用|路径|关系|连接|继承|实例化)"),
)


def _has_keywords(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    for keyword in keywords:
        normalized = keyword.lower()
        if re.fullmatch(r"[a-z]+", normalized):
            if normalized in words:
                return True
        elif normalized in lowered:
            return True
    return False


def _is_negative_relation_conclusion(draft: str) -> bool:
    return any(pattern.search(draft) for pattern in _NEGATIVE_RELATION_PATTERNS)


def _get_validated_anchor_ids(query_plan: dict) -> set[str]:
    anchors = query_plan.get("anchors", [])
    return {
        a.get("id", "")
        for a in anchors
        if a.get("validation", {}).get("valid")
    }


def _get_validated_anchors(query_plan: dict) -> list[dict]:
    return [
        anchor
        for anchor in query_plan.get("anchors", [])
        if anchor.get("validation", {}).get("valid")
    ]


def _get_primary_anchors(
    question: str,
    anchors: list[dict],
    limit: int,
    draft: str = "",
) -> list[dict]:
    if len(anchors) <= limit:
        return anchors

    def mentions(text: str, anchor: dict) -> bool:
        lowered = text.lower()
        node_id = anchor.get("id", "").lower()
        name = anchor.get("name", "").lower()
        if node_id and node_id in lowered:
            return True
        if not name:
            return False
        return bool(re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            lowered,
        ))

    ranked = sorted(
        enumerate(anchors),
        key=lambda pair: (
            -int(mentions(question, pair[1])),
            -int(bool(draft) and mentions(draft, pair[1])),
            pair[0],
        ),
    )
    return [anchor for _, anchor in ranked[:limit]]


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


def _is_complete_source(item) -> bool:
    if getattr(item, "kind", "") != "source":
        return False
    if not getattr(item, "complete", False):
        return False
    payload = getattr(item, "payload", {})
    if not isinstance(payload, dict):
        return False
    return bool(
        getattr(item, "node_id", None)
        and (getattr(item, "file", None) or payload.get("file"))
        and (
            getattr(item, "start_line", None) is not None
            or payload.get("start_line") is not None
        )
        and (
            getattr(item, "end_line", None) is not None
            or payload.get("end_line") is not None
        )
        and payload.get("type")
    )


def _complete_sources_by_node(evidence_items: list) -> dict[str, object]:
    return {
        getattr(item, "node_id", ""): item
        for item in evidence_items
        if _is_complete_source(item)
    }


def _unresolved_draft_entities(
    draft: str,
    evidence_items: list,
) -> list[str]:
    if not draft:
        return []
    explicit_entities = set(
        match.strip()
        for match in re.findall(r"`([^`\n]+)`", draft)
        if match.strip()
    )
    explicit_entities.update(
        re.findall(
            r"[A-Za-z0-9_./\\-]+(?:::[A-Za-z0-9_]+)+",
            draft,
        )
    )
    if not explicit_entities:
        return []

    known = set()
    source_contexts = []
    for item in evidence_items:
        node_id = getattr(item, "node_id", "") or ""
        if node_id:
            known.add(node_id.lower())
            node_parts = node_id.split("::")
            known.update(part.lower() for part in node_parts[1:] if part)
            known.update(
                "::".join(node_parts[index:]).lower()
                for index in range(1, len(node_parts))
            )
        for endpoint in (
            getattr(item, "source_node_id", "") or "",
            getattr(item, "target_node_id", "") or "",
        ):
            if endpoint:
                known.add(endpoint.lower())
                endpoint_parts = endpoint.split("::")
                known.update(
                    part.lower()
                    for part in endpoint_parts[1:]
                    if part
                )
        payload = getattr(item, "payload", {})
        file_path = getattr(item, "file", "") or ""
        if isinstance(payload, dict):
            if payload.get("name"):
                known.add(str(payload["name"]).lower())
            file_path = file_path or str(payload.get("file", ""))
            source_context = payload.get("source_context")
            if source_context:
                source_contexts.append(str(source_context).lower())
        elif isinstance(payload, str):
            source_contexts.append(payload.lower())
        if file_path:
            normalized_file = file_path.replace("\\", "/").lower()
            known.add(normalized_file)
            known.add(normalized_file.rsplit("/", 1)[-1])

    def is_entity_reference(value: str) -> bool:
        stripped = value.strip()
        lowered = stripped.lower()
        if (
            not stripped
            or lowered in _CODE_LITERALS
            or lowered in _PROTOCOL_TOOL_NAMES
        ):
            return False
        if "::" in stripped:
            return True
        if any(char.isspace() for char in stripped):
            return False
        if re.search(r"[()[\]{}=,+*/%<>\"']", stripped):
            return False
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", stripped))

    def appears_in_source(value: str) -> bool:
        lowered = value.lower()
        if "::" in value:
            parts = [
                part.lower()
                for part in value.split("::")
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part)
            ]
            if not parts:
                return False
            patterns = [
                re.compile(rf"\b{re.escape(part)}\b")
                for part in parts
            ]
            return any(
                all(pattern.search(context) for pattern in patterns)
                for context in source_contexts
            )
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            pattern = re.compile(rf"\b{re.escape(lowered)}\b")
            return any(pattern.search(context) for context in source_contexts)
        return any(lowered in context for context in source_contexts)

    return sorted(
        entity
        for entity in explicit_entities
        if (
            is_entity_reference(entity)
            and entity.lower() not in known
            and entity.split("::")[-1].lower() not in known
            and not appears_in_source(entity)
        )
    )


def _get_high_conf_relations(evidence_items: list, min_conf: float = 0.45) -> list:
    return [
        e for e in evidence_items
        if getattr(e, "kind", "") == "relation"
        and (getattr(e, "confidence", None) or 0) >= min_conf
        and bool(getattr(e, "strategy", None))
    ]


def _relations_connect(
    relations: list,
    source_id: str,
    target_id: str,
) -> bool:
    if not source_id or not target_id:
        return False
    adjacency: dict[str, set[str]] = {}
    for relation in relations:
        source = getattr(relation, "source_node_id", "") or ""
        target = getattr(relation, "target_node_id", "") or ""
        if not source or not target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    queue = deque([source_id])
    visited = {source_id}
    while queue:
        current = queue.popleft()
        if current == target_id:
            return True
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _shared_source_verifies_entities(
    source_items: list,
    relations: list,
    primary_anchors: list[dict],
) -> bool:
    if len(primary_anchors) < 2:
        return False
    entity_names = [
        anchor.get("name", "").lower()
        for anchor in primary_anchors
        if anchor.get("name")
    ]
    primary_ids = {
        anchor.get("id", "")
        for anchor in primary_anchors
    }
    if len(entity_names) < 2:
        return False
    for item in source_items:
        node_id = getattr(item, "node_id", "") or ""
        if not node_id or node_id in primary_ids:
            continue
        payload = getattr(item, "payload", {})
        if not isinstance(payload, dict):
            continue
        source_context = str(payload.get("source_context", "")).lower()
        if not all(name in source_context for name in entity_names):
            continue
        if any(
            node_id in {
                getattr(relation, "source_node_id", "") or "",
                getattr(relation, "target_node_id", "") or "",
            }
            and (
                getattr(relation, "source_node_id", "") in primary_ids
                or getattr(relation, "target_node_id", "") in primary_ids
            )
            for relation in relations
        ):
            return True
    return False


def _has_completed_negative_search(
    query_plan: dict,
    primary_anchor_ids: set[str],
    max_depth: int,
    min_confidence: float,
) -> bool:
    for record in query_plan.get("relation_expansions", []):
        if record.get("status") != "completed":
            continue
        if record.get(
            "relation_result_count",
            record.get("result_count"),
        ) != 0:
            continue
        if record.get("action") not in {
            "call_paths",
            "transitive_callers",
            "transitive_callees",
            "class_hierarchy",
            "contextualize",
        }:
            continue
        searched_ids = {
            record.get("symbol", ""),
            record.get("target", ""),
        }
        if primary_anchor_ids and not (
            primary_anchor_ids & searched_ids
        ):
            continue
        if int(record.get("max_depth", max_depth)) > max_depth:
            continue
        if float(record.get("min_confidence", 0.0)) < min_confidence:
            continue
        if not record.get("edge_types"):
            continue
        return True
    return False


class SufficiencyGate:
    """确定性证据充分性门控。"""

    DEFAULT_MIN_CONFIDENCE = 0.45
    EXPANSION_MAX_DEPTH = 2
    MAX_AUTO_EXPANSIONS = 2

    def recommended_anchor_count(self, question: str) -> int:
        """返回首次预取应覆盖的主要实体数量。"""
        if _has_keywords(question, _COMPARISON_KEYWORDS):
            return 2
        return 1

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
        validated_anchor_items = _get_validated_anchors(query_plan)
        source_ids = _get_source_node_ids(evidence_items)
        complete_sources = _complete_sources_by_node(evidence_items)
        anchors = query_plan.get("anchors", [])

        # ── 最低要求：至少一个 source 证据 ──
        if not source_items:
            return GateDecision(
                passed=False,
                reasons=["无任何源码证据"],
                missing_requirements=["至少需要一个源码证据才能合成答案"],
            )

        is_comparison = _has_keywords(question, _COMPARISON_KEYWORDS)
        is_inheritance = _has_keywords(question, _INHERITANCE_KEYWORDS)
        is_instantiation = _has_keywords(
            question,
            _INSTANTIATION_KEYWORDS,
        )
        is_relation = (
            _has_keywords(question, _RELATION_KEYWORDS)
            or is_instantiation
        )
        is_negative_draft = bool(
            draft and _is_negative_relation_conclusion(draft)
        )

        # anchor 实体在 draft 中可定位（信息性，非阻断）
        if draft:
            draft_entities = sum(
                1 for a in anchors
                if a.get("name", "") and a["name"].lower() in draft.lower()
            )
            if draft_entities > 0:
                reasons.append(f"草稿中引用了 {draft_entities} 个锚点实体")
            unresolved_entities = _unresolved_draft_entities(
                draft,
                evidence_items,
            )
            if unresolved_entities:
                missing.append(
                    "草稿中的代码实体无法在证据中定位: "
                    + ", ".join(unresolved_entities)
                )

        # ── 单实体解释 ──
        if not is_comparison and not is_relation and not is_inheritance:
            if validated_anchors:
                reasons.append(f"{len(validated_anchors)} 个已验证锚点")
                primary_anchor = _get_primary_anchors(
                    question,
                    validated_anchor_items,
                    limit=1,
                )[0]
                if primary_anchor.get("id") in complete_sources:
                    reasons.append("主要锚点有完整源码")
                else:
                    missing.append("锚点缺少完整源码")
            else:
                missing.append("单实体问题至少需要一个已验证锚点")

            if not missing:
                return GateDecision(passed=True, reasons=reasons)

        # ── 多实体比较 ──
        if is_comparison:
            primary_anchors = _get_primary_anchors(
                question,
                validated_anchor_items,
                limit=2,
            )
            primary_ids = [
                anchor.get("id", "")
                for anchor in primary_anchors
                if anchor.get("id")
            ]
            covered_primary_ids = {
                node_id
                for node_id in primary_ids
                if node_id in complete_sources
            }
            if (
                len(primary_ids) >= 2
                and len(covered_primary_ids) == len(set(primary_ids))
            ):
                reasons.append(
                    f"比较问题的 {len(covered_primary_ids)} 个主要实体"
                    "均有独立完整源码"
                )
            else:
                missing.append(
                    "比较问题需要至少 2 个已验证主要实体，"
                    "且每个主要实体都有独立完整源码"
                )
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

        # ── 继承关系 ──
        inherits_rels: list = []
        if is_inheritance:
            primary_classes = [
                anchor
                for anchor in _get_primary_anchors(
                    question,
                    validated_anchor_items,
                    limit=1,
                )
                if anchor.get("type") == "CLASS"
            ]
            primary_class_id = (
                primary_classes[0].get("id", "")
                if primary_classes
                else ""
            )
            bounded_inheritance_search = (
                is_negative_draft
                and _has_completed_negative_search(
                    query_plan,
                    {primary_class_id} if primary_class_id else set(),
                    self.EXPANSION_MAX_DEPTH,
                    self.DEFAULT_MIN_CONFIDENCE,
                )
            )
            inherits_rels = [
                e for e in relation_items
                if getattr(e, "edge_type", "") == "INHERITS"
                and primary_class_id in {
                    getattr(e, "source_node_id", "") or "",
                    getattr(e, "target_node_id", "") or "",
                }
                and isinstance(getattr(e, "payload", {}), dict)
                and any(
                    key in getattr(e, "payload", {})
                    for key in (
                        "hierarchy_item",
                        "hierarchy_entry",
                        "id",
                    )
                )
            ]
            if not primary_class_id:
                missing.append("继承问题缺少已验证的主要类锚点")
            elif primary_class_id not in complete_sources:
                missing.append("继承问题的主要类缺少完整源码")
            elif inherits_rels:
                reasons.append(f"{len(inherits_rels)} 条继承关系证据")
            elif bounded_inheritance_search:
                reasons.append("已完成限定范围的类层次搜索")
            else:
                missing.append("继承问题缺少类层次证据")
                for a in validated_anchor_items:
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
            primary_anchors = _get_primary_anchors(
                question,
                validated_anchor_items or anchors,
                limit=2,
                draft=draft if not is_comparison else "",
            )
            primary_ids = [
                anchor.get("id", "")
                for anchor in primary_anchors
                if anchor.get("id")
            ]
            bounded_negative_search = (
                is_negative_draft
                and _has_completed_negative_search(
                    query_plan,
                    set(primary_ids),
                    self.EXPANSION_MAX_DEPTH,
                    self.DEFAULT_MIN_CONFIDENCE,
                )
            )
            if len(primary_ids) >= 2:
                relation_satisfied = _relations_connect(
                    high_conf_rels,
                    primary_ids[0],
                    primary_ids[1],
                ) or _shared_source_verifies_entities(
                    source_items,
                    high_conf_rels,
                    primary_anchors,
                )
            elif len(primary_ids) == 1:
                relation_satisfied = any(
                    primary_ids[0] in {
                        getattr(relation, "source_node_id", "") or "",
                        getattr(relation, "target_node_id", "") or "",
                    }
                    for relation in high_conf_rels
                )
            else:
                relation_satisfied = False

            if relation_satisfied:
                reasons.append(
                    f"{len(high_conf_rels)} 条高置信度关系证据连接主要实体"
                )
            elif bounded_negative_search:
                reasons.append(
                    "已完成带方向、边类型、深度和置信度阈值的限定搜索"
                )
            else:
                if not high_conf_rels:
                    missing.append(
                        "关系问题缺少满足置信度阈值的关系证据"
                    )
                else:
                    missing.append(
                        "关系证据未连接问题中的主要实体，"
                        "需要共享调用者或跨实体路径"
                    )
                # 请求扩展：共享调用者优先，再回退到调用路径或单锚点调用者
                if primary_anchors and expansion_count < self.MAX_AUTO_EXPANSIONS:
                    func_anchors = [
                        a for a in primary_anchors
                        if a.get("type") in ("FUNCTION", "METHOD")
                    ]
                    if len(func_anchors) >= 2:
                        src = func_anchors[0].get("id", "")
                        tgt = func_anchors[1].get("id", "")
                        shared_key = f"shared_callers:{src}:{tgt}"
                        if shared_key not in expanded_relations:
                            expansions.append(ExpansionRequest(
                                action="shared_callers",
                                symbol=src,
                                target=tgt,
                                max_depth=self.EXPANSION_MAX_DEPTH,
                                reason=(
                                    "关系问题缺少跨实体关系，"
                                    f"查找 {src} 与 {tgt} 的共享调用者"
                                ),
                                edge_types=["CALLS"],
                            ))
                        else:
                            cp_key = f"call_paths:{src}:{tgt}"
                            if cp_key not in expanded_relations:
                                expansions.append(ExpansionRequest(
                                    action="call_paths",
                                    symbol=src,
                                    target=tgt,
                                    max_depth=self.EXPANSION_MAX_DEPTH,
                                    reason=(
                                        "未找到共享调用者，"
                                        f"回退到 {src} → {tgt} 调用路径"
                                    ),
                                    edge_types=["CALLS"],
                                ))
                    if not expansions:
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
                    if not expansions and is_instantiation:
                        class_anchors = [
                            anchor
                            for anchor in validated_anchor_items
                            if anchor.get("type") == "CLASS"
                        ]
                        if class_anchors:
                            class_id = class_anchors[0].get("id", "")
                            context_key = f"contextualize:{class_id}"
                            if context_key not in expanded_relations:
                                expansions.append(ExpansionRequest(
                                    action="contextualize",
                                    symbol=class_id,
                                    max_depth=1,
                                    reason=(
                                        "实例化问题缺少直接关系证据，"
                                        f"重新检查 {class_id} 的实例化来源"
                                    ),
                                    edge_types=["INSTANTIATED_BY"],
                                ))

        # ── 否定结论检测 ──
        if is_negative_draft and is_relation:
            primary_anchor_ids = {
                anchor.get("id", "")
                for anchor in _get_primary_anchors(
                    question,
                    validated_anchor_items or anchors,
                    limit=2,
                    draft=draft if not is_comparison else "",
                )
                if anchor.get("id")
            }
            if not _has_completed_negative_search(
                query_plan,
                primary_anchor_ids,
                self.EXPANSION_MAX_DEPTH,
                self.DEFAULT_MIN_CONFIDENCE,
            ):
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
