# -*- coding: utf-8 -*-
"""多实体独立检索编排 —— 实体提取 + 每实体独立检索+选锚点+预取。

重排策略：合并候选后统一重排。每实体独立检索不做重排。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .evidence import EvidenceItem, EvidenceLedger
from .retrieval import Candidate, STOP_WORDS


@dataclass
class MultiEntityConfig:
    """多实体检索配置。"""
    max_entities: int = 4
    candidate_limit: int = 12
    candidate_display_limit: int = 6


_EXTRACTION_PROMPT = """\
你是一个代码实体分解器。给定一个关于代码库的问题，识别出需要定位的独立代码符号（函数、类、方法、模块）。

对每个实体输出：
- name: 代码中实际存在的符号名（如 Linter、parse、BaseGrammar），不要编造新名称
- query: 空格分隔的搜索关键词（使用符号名、方法名、关键标识符，不用自然语言句子）
- description: 这个实体是什么/做什么（一句话）
- type_hint: FUNCTION, CLASS, METHOD, MODULE, 或 CONCEPT

如果无法确定符号名，用 CONCEPT 类型并填描述性名称。

输出纯 JSON 数组，不要其他文字。

问题: __QUESTION__"""

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def clean_entity_query(text: str) -> str:
    """保留合法标识符 token，并删除检索停用词。"""
    return " ".join(
        token
        for token in _IDENTIFIER_RE.findall(text)
        if token.casefold() not in STOP_WORDS
    )


@dataclass
class Entity:
    """问题中需要定位的代码实体。"""

    name: str
    query: str = ""
    description: str = ""
    type_hint: str = "CONCEPT"

    def __post_init__(self):
        if not self.query:
            self.query = self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "query": self.query,
            "description": self.description,
            "type_hint": self.type_hint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            name=str(d.get("name", "")),
            query=str(d.get("query", "")),
            description=str(d.get("description", "")),
            type_hint=str(d.get("type_hint", "CONCEPT")),
        )


class EntityExtractor:
    """用轻量 LLM 调用将问题分解为独立实体，检索前使用。"""

    def __init__(self, model=None):
        self._model = model

    def extract(self, question: str, max_entities: int = 4) -> list[Entity]:
        if self._model is None:
            return [Entity(name="primary", query=question)]

        prompt = _EXTRACTION_PROMPT.replace("__QUESTION__", question)
        try:
            raw = self._model.generate([{"role": "user", "content": prompt}])
        except Exception:
            return [Entity(name="primary", query=question)]

        entities = self._parse(raw)
        if not entities:
            return [Entity(name="primary", query=question)]

        # 删除空名称、按 name.casefold() 去重、限制数量
        seen: set[str] = set()
        filtered: list[Entity] = []
        for e in entities:
            if not e.name.strip():
                continue
            cleaned_query = clean_entity_query(e.query)
            if not cleaned_query:
                cleaned_query = clean_entity_query(e.name)
            if not cleaned_query:
                continue
            key = e.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            e.query = cleaned_query
            filtered.append(e)
            if len(filtered) >= max_entities:
                break
        if not filtered:
            return [Entity(name="primary", query=question)]
        return filtered

    @staticmethod
    def _parse(raw: str) -> list[Entity]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list) or len(data) == 0:
            return []

        entities = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entities.append(Entity.from_dict(item))
        return entities


def _prefetch_anchor(
    graph_tool,
    anchor_data: dict,
    ledger: EvidenceLedger,
    step: int = 0,
) -> tuple[dict | None, dict | None, list[str]]:
    """验证并预取单个锚点。"""
    validation = graph_tool.validate_query_anchor(anchor_data)
    if not validation.get("valid"):
        return None, {
            "candidate": anchor_data,
            "reason": validation.get("reason", "invalid"),
            "message": validation.get("message", ""),
            "suggestions": validation.get("suggestions", []),
        }, []

    raw = json.dumps(
        graph_tool.inspect(anchor_data["id"]),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    evidence_items = EvidenceItem.from_tool_result(
        "query_graph",
        {"action": "contextualize", "name": anchor_data["id"]},
        raw,
        step=step,
    )
    source_items = [
        item for item in evidence_items
        if item.kind == "source" and item.node_id == anchor_data["id"]
    ]
    if not source_items:
        return None, {
            "candidate": anchor_data,
            "reason": "prefetch_without_source",
            "message": "inspect 未返回锚点源码",
            "suggestions": [],
        }, []

    for item in evidence_items:
        ledger.add(item)

    prefetch_ids = [item.evidence_id for item in source_items]
    anchor_data["evidence_ids"] = list(prefetch_ids)
    return anchor_data, None, prefetch_ids


def _prefetch_anchors(graph_tool, ordered_anchors: list[dict],
                      max_anchors: int, ledger: EvidenceLedger, step: int = 0):
    """锚点验证+预取+证据写入。返回 (accepted, rejected, prefetch_evidence_ids)。"""
    accepted = []
    rejected = []
    prefetch_ids = []

    for anchor_data in ordered_anchors:
        if len(accepted) >= max_anchors:
            rejected.append({
                "candidate": anchor_data,
                "reason": "max_anchor_limit",
                "message": f"已达锚点上限 ({max_anchors})",
                "suggestions": [],
            })
            continue

        anchor, rejection, anchor_prefetch_ids = _prefetch_anchor(
            graph_tool,
            anchor_data,
            ledger,
            step=step,
        )
        if rejection is not None:
            rejected.append(rejection)
            continue
        accepted.append(anchor)
        prefetch_ids.extend(anchor_prefetch_ids)

    return accepted, rejected, prefetch_ids


@dataclass
class MultiEntityPrelude:
    """多实体编排结果。"""

    text: str = ""
    entity_anchors: dict[str, list[dict]] = field(default_factory=dict)
    entity_candidates: dict[str, list[dict]] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    candidate_entities: dict[str, list[str]] = field(default_factory=dict)
    global_anchors: list[dict] = field(default_factory=list)
    stages_attempted: list[str] = field(default_factory=list)
    stages_succeeded: list[str] = field(default_factory=list)
    rejected_anchors: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    prefetch_evidence_ids: list[str] = field(default_factory=list)

    @property
    def anchor_count(self) -> int:
        return len(self.global_anchors)


class MultiEntityOrchestrator:
    """对每个实体独立检索，再合并候选并统一选择锚点。"""

    def __init__(self, graph_tool, config: MultiEntityConfig | None = None):
        self.graph_tool = graph_tool
        self._config = config or MultiEntityConfig()

    def run(
        self,
        entities: list[Entity],
        task: str,
        ledger: EvidenceLedger,
        recommended_count: int = 1,
    ) -> MultiEntityPrelude:
        prelude = MultiEntityPrelude()
        all_attempted: set[str] = set()
        all_succeeded: set[str] = set()
        ordered_by_entity: dict[str, list[dict]] = {}

        for entity in entities:
            entity_section = self._retrieve_entity(entity)
            for stage in entity_section["stages_attempted"]:
                all_attempted.add(stage)
            for stage in entity_section["stages_succeeded"]:
                all_succeeded.add(stage)
            prelude.entity_candidates[entity.name] = entity_section["candidates"]
            prelude.entity_anchors[entity.name] = []
            prelude.diagnostics.extend(entity_section["diagnostics"])
            if entity_section["prelude_text"]:
                if prelude.text:
                    prelude.text += "\n"
                prelude.text += entity_section["prelude_text"]

            ordered_by_entity[entity.name] = [
                {
                    **candidate,
                    "selection_reason": f"覆盖实体 {entity.name}",
                    "covered_terms": list(
                        candidate.get("matched_terms", [])
                    ),
                    "candidate_sources": list(
                        candidate.get("sources", [])
                    ),
                }
                for candidate in entity_section["candidates"]
            ]

        merged = self._merge_candidates(entities, prelude.entity_candidates)
        prelude.candidate_entities = {
            candidate_id: list(candidate["entity_names"])
            for candidate_id, candidate in merged.items()
        }
        prelude.candidates = [
            self._candidate_from_dict(candidate)
            for candidate in sorted(
                merged.values(),
                key=lambda item: (-float(item.get("score", 0.0)), item["id"]),
            )
        ]
        prelude.stages_attempted = sorted(all_attempted)
        prelude.stages_succeeded = sorted(all_succeeded)

        self._select_global_anchors(
            entities=entities,
            ordered_by_entity=ordered_by_entity,
            merged=merged,
            max_anchors=max(0, recommended_count),
            ledger=ledger,
            prelude=prelude,
        )
        self._render_prefetch_evidence(ledger, prelude)

        if prelude.anchor_count == 0:
            prelude.diagnostics.append("所有实体均未通过锚点验证")

        return prelude

    def _retrieve_entity(
        self,
        entity: Entity,
    ) -> dict:
        """执行单实体检索并生成候选展示。"""
        result: dict = {
            "candidates": [],
            "prelude_text": "",
            "diagnostics": [],
            "stages_attempted": [],
            "stages_succeeded": [],
        }

        # 1. 检索
        try:
            retrieval = self.graph_tool.search(
                entity.query,
                limit=self._config.candidate_limit,
                use_embeddings=(
                    getattr(self.graph_tool, "enable_embeddings", False)
                ),
            )
        except Exception as e:
            result["diagnostics"].append(
                f"[{entity.name}] 检索失败: {e}"
            )
            return result

        candidates = [
            c.to_dict() if hasattr(c, "to_dict") else dict(c)
            for c in retrieval.candidates
        ]
        result["stages_attempted"] = list(retrieval.stages_attempted)
        result["stages_succeeded"] = list(retrieval.stages_succeeded)
        if not candidates:
            result["diagnostics"].append(
                f"[{entity.name}] 未检索到候选"
            )
            return result

        result["candidates"] = candidates

        display_items = []
        for c in candidates[:self._config.candidate_display_limit]:
            sources = ",".join(c.get("sources", []))
            display_items.append(
                f"  - {c['name']} ({c['type']}) {c['id']} "
                f"[score={c['score']:.2f}; sources={sources}]"
            )
        prelude_parts = [
            f"\n═══ 实体: \"{entity.name}\"",
            f"     描述: {entity.description}",
            f"     类型: {entity.type_hint} ═══",
            "",
            "与问题最相关的候选:",
            *display_items,
        ]
        result["prelude_text"] = "\n".join(prelude_parts)
        return result

    @staticmethod
    def _merge_candidates(
        entities: list[Entity],
        entity_candidates: dict[str, list[dict]],
    ) -> dict[str, dict]:
        """按 ID 合并候选，保留最高分并合并来源元数据。"""
        merged: dict[str, dict] = {}
        for entity in entities:
            for candidate in entity_candidates.get(entity.name, []):
                candidate_id = str(candidate.get("id", ""))
                if not candidate_id:
                    continue
                current = merged.get(candidate_id)
                if current is None:
                    current = {
                        **candidate,
                        "sources": list(candidate.get("sources", [])),
                        "matched_terms": list(
                            candidate.get("matched_terms", [])
                        ),
                        "matched_fields": list(
                            candidate.get("matched_fields", [])
                        ),
                        "entity_names": [],
                    }
                    merged[candidate_id] = current
                elif float(candidate.get("score", 0.0)) > float(
                    current.get("score", 0.0)
                ):
                    for field_name in ("name", "type", "file", "score"):
                        current[field_name] = candidate.get(
                            field_name,
                            current.get(field_name),
                        )

                current["sources"] = MultiEntityOrchestrator._ordered_union(
                    current.get("sources", []),
                    candidate.get("sources", []),
                )
                current["matched_terms"] = sorted(
                    set(current.get("matched_terms", []))
                    | set(candidate.get("matched_terms", []))
                )
                current["matched_fields"] = sorted(
                    set(current.get("matched_fields", []))
                    | set(candidate.get("matched_fields", []))
                )
                if entity.name not in current["entity_names"]:
                    current["entity_names"].append(entity.name)
        return merged

    @staticmethod
    def _ordered_union(first: list[str], second: list[str]) -> list[str]:
        result = list(first)
        for item in second:
            if item not in result:
                result.append(item)
        return result

    @staticmethod
    def _candidate_from_dict(candidate: dict) -> Candidate:
        return Candidate(
            id=candidate["id"],
            name=candidate.get("name", ""),
            type=candidate.get("type", ""),
            file=candidate.get("file", ""),
            score=float(candidate.get("score", 0.0)),
            sources=list(candidate.get("sources", [])),
            matched_terms=list(candidate.get("matched_terms", [])),
            matched_fields=list(candidate.get("matched_fields", [])),
            entity_names=list(candidate.get("entity_names", [])),
        )

    def _select_global_anchors(
        self,
        entities: list[Entity],
        ordered_by_entity: dict[str, list[dict]],
        merged: dict[str, dict],
        max_anchors: int,
        ledger: EvidenceLedger,
        prelude: MultiEntityPrelude,
    ) -> None:
        """优先覆盖每个实体，再按全局分数补足锚点。"""
        if max_anchors <= 0:
            return

        accepted_by_id: dict[str, dict] = {}
        rejected_ids: set[str] = set()

        def accept_candidate(candidate: dict) -> bool:
            candidate_id = candidate["id"]
            if candidate_id in accepted_by_id:
                return True
            if candidate_id in rejected_ids:
                return False

            anchor_data = {
                **candidate,
                "entity_names": list(
                    merged[candidate_id].get("entity_names", [])
                ),
            }
            anchor, rejection, prefetch_ids = _prefetch_anchor(
                self.graph_tool,
                anchor_data,
                ledger,
                step=0,
            )
            if rejection is not None:
                prelude.rejected_anchors.append(rejection)
                rejected_ids.add(candidate_id)
                return False

            accepted_by_id[candidate_id] = anchor
            prelude.global_anchors.append(anchor)
            prelude.prefetch_evidence_ids.extend(prefetch_ids)
            for entity_name in anchor.get("entity_names", []):
                entity_anchors = prelude.entity_anchors.setdefault(
                    entity_name,
                    [],
                )
                if not any(
                    item["id"] == candidate_id for item in entity_anchors
                ):
                    entity_anchors.append(anchor)
            return True

        covered_entities: set[str] = set()
        for entity in entities:
            if len(accepted_by_id) >= max_anchors:
                break
            if entity.name in covered_entities:
                continue
            for ordered_candidate in ordered_by_entity.get(entity.name, []):
                candidate_id = ordered_candidate.get("id", "")
                if candidate_id not in merged:
                    continue
                candidate = {
                    **merged[candidate_id],
                    **{
                        key: value
                        for key, value in ordered_candidate.items()
                        if key in {
                            "selection_reason",
                            "covered_terms",
                            "candidate_sources",
                        }
                    },
                }
                if accept_candidate(candidate):
                    covered_entities.update(
                        merged[candidate_id].get("entity_names", [])
                    )
                    break

        if len(accepted_by_id) < max_anchors:
            for candidate in prelude.candidates:
                if len(accepted_by_id) >= max_anchors:
                    break
                candidate_id = candidate.id
                if candidate_id in merged:
                    accept_candidate({
                        **merged[candidate_id],
                        "selection_reason": "按全局 final_score 补足锚点",
                        "covered_terms": list(candidate.matched_terms),
                        "candidate_sources": list(candidate.sources),
                    })

        for candidate in prelude.candidates:
            if (
                candidate.id in accepted_by_id
                or candidate.id in rejected_ids
            ):
                continue
            prelude.rejected_anchors.append({
                "candidate": candidate.to_dict(),
                "reason": "max_anchor_limit",
                "message": f"已达锚点上限 ({max_anchors})",
                "suggestions": [],
            })

    @staticmethod
    def _render_prefetch_evidence(
        ledger: EvidenceLedger,
        prelude: MultiEntityPrelude,
    ) -> None:
        if not prelude.prefetch_evidence_ids:
            return

        prefetched = [
            item for item in ledger.source_items
            if item.evidence_id in set(prelude.prefetch_evidence_ids)
        ]
        evidence_text, display_reports = ledger.render_prefetch_evidence(
            prefetched
        )
        prelude.text += "\n\n[自动验证锚点的证据]\n" + evidence_text
        for report in display_reports:
            for anchor in prelude.global_anchors:
                if report["evidence_id"] in anchor.get("evidence_ids", []):
                    anchor["display_level"] = report["display_level"]
                    anchor["omitted_reason"] = report["omitted_reason"]
