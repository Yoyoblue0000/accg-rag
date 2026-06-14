# -*- coding: utf-8 -*-
"""多实体独立检索编排 —— 实体提取 + 每实体独立检索+选锚点+预取。

重排策略：合并候选后统一重排。每实体独立检索不做重排。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .evidence import EvidenceItem, EvidenceLedger
from .retrieval import Candidate


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
            key = e.name.casefold()
            if key in seen:
                continue
            seen.add(key)
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

        validation = graph_tool.validate_query_anchor(anchor_data)
        if not validation.get("valid"):
            rejected.append({
                "candidate": anchor_data,
                "reason": validation.get("reason", "invalid"),
                "message": validation.get("message", ""),
                "suggestions": validation.get("suggestions", []),
            })
            continue

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
            rejected.append({
                "candidate": anchor_data,
                "reason": "prefetch_without_source",
                "message": "inspect 未返回锚点源码",
                "suggestions": [],
            })
            continue

        for item in evidence_items:
            ledger.add(item)

        anchor_data["evidence_ids"] = [
            item.evidence_id for item in source_items
        ]
        accepted.append(anchor_data)
        prefetch_ids.extend(item.evidence_id for item in source_items)

    return accepted, rejected, prefetch_ids


@dataclass
class MultiEntityPrelude:
    """多实体编排结果。"""

    text: str = ""
    entity_anchors: dict[str, list[dict]] = field(default_factory=dict)
    entity_candidates: dict[str, list[dict]] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    stages_attempted: list[str] = field(default_factory=list)
    stages_succeeded: list[str] = field(default_factory=list)
    rejected_anchors: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    prefetch_evidence_ids: list[str] = field(default_factory=list)

    @property
    def anchor_count(self) -> int:
        return sum(len(v) for v in self.entity_anchors.values())


class MultiEntityOrchestrator:
    """对每个实体独立检索、选锚点、预取，合并为一个 prelude。

    重排策略：各实体独立检索不做重排；合并候选后，由上层 Agent 统一执行在线重排。
    """

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

        per_entity_budget = max(
            1, recommended_count // max(len(entities), 1)
        )

        all_attempted: set[str] = set()
        all_succeeded: set[str] = set()
        for entity in entities:
            entity_section = self._process_entity(
                entity=entity,
                task=task,
                ledger=ledger,
                max_anchors=max(1, per_entity_budget),
            )
            for stage in entity_section.get("stages_attempted", []):
                all_attempted.add(stage)
            for stage in entity_section.get("stages_succeeded", []):
                all_succeeded.add(stage)
            self._merge_entity_result(entity, entity_section, prelude)

        # 按 id 合并各实体候选，重复候选保留最高分
        merged: dict[str, dict] = {}
        for candidates in prelude.entity_candidates.values():
            for c in candidates:
                cid = c.get("id", "")
                if cid not in merged or c.get("score", 0) > merged[cid].get("score", 0):
                    merged[cid] = c
        prelude.candidates = [
            Candidate(
                id=c["id"], name=c.get("name", ""), type=c.get("type", ""),
                file=c.get("file", ""), score=c.get("score", 0.0),
                sources=list(c.get("sources", [])),
                matched_terms=list(c.get("matched_terms", [])),
                matched_fields=list(c.get("matched_fields", [])),
            )
            for c in merged.values()
        ]
        prelude.stages_attempted = sorted(all_attempted)
        prelude.stages_succeeded = sorted(all_succeeded)

        if prelude.anchor_count == 0:
            prelude.diagnostics.append("所有实体均未通过锚点验证")

        return prelude

    def _process_entity(
        self,
        entity: Entity,
        task: str,
        ledger: EvidenceLedger,
        max_anchors: int,
    ) -> dict:
        """单实体完整检索流程。"""
        result: dict = {
            "candidates": [],
            "anchors": [],
            "rejected": [],
            "prefetch_ids": [],
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
        if not candidates:
            result["diagnostics"].append(
                f"[{entity.name}] 未检索到候选"
            )
            return result

        result["candidates"] = candidates
        result["stages_attempted"] = list(retrieval.stages_attempted)
        result["stages_succeeded"] = list(retrieval.stages_succeeded)

        # 2. 候选展示
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

        # 3. 锚点选择 + 验证 + 预取
        ordered = self.graph_tool.select_query_anchors(
            entity.query,
            candidates,
            max_anchors=len(candidates),
        )
        accepted, rejected, prefetch_ids = _prefetch_anchors(
            self.graph_tool, ordered, max_anchors, ledger, step=0,
        )
        result["anchors"] = accepted
        result["rejected"].extend(rejected)
        result["prefetch_ids"] = prefetch_ids

        # 4. 渲染预取证据
        if result["prefetch_ids"]:
            prefetched = [
                item for item in ledger.source_items
                if item.evidence_id in set(result["prefetch_ids"])
            ]
            evidence_text, display_reports = (
                ledger.render_prefetch_evidence(prefetched)
            )
            prelude_parts.append("\n[自动验证锚点的证据]")
            prelude_parts.append(evidence_text)
            for report in display_reports:
                for anchor in result["anchors"]:
                    if report["evidence_id"] in anchor.get("evidence_ids", []):
                        anchor["display_level"] = report["display_level"]
                        anchor["omitted_reason"] = report["omitted_reason"]

        result["prelude_text"] = "\n".join(prelude_parts)
        return result

    @staticmethod
    def _merge_entity_result(
        entity: Entity,
        result: dict,
        prelude: MultiEntityPrelude,
    ) -> None:
        prelude.entity_anchors[entity.name] = result["anchors"]
        prelude.entity_candidates[entity.name] = result["candidates"]
        prelude.rejected_anchors.extend(result["rejected"])
        prelude.prefetch_evidence_ids.extend(result["prefetch_ids"])
        prelude.diagnostics.extend(result["diagnostics"])
        if result["prelude_text"]:
            if prelude.text:
                prelude.text += "\n"
            prelude.text += result["prelude_text"]
