# -*- coding: utf-8 -*-
"""确定性的查询计划与锚点审计结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .retrieval import Candidate


@dataclass
class Anchor:
    id: str
    name: str
    type: str
    file: str
    score: float
    sources: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    selection_reason: str = ""
    covered_terms: list[str] = field(default_factory=list)
    candidate_sources: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    prefetch_action: dict = field(default_factory=dict)
    display_level: str = ""
    omitted_reason: str = ""

    @classmethod
    def from_dict(cls, item: dict) -> "Anchor":
        return cls(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            type=str(item.get("type", "")),
            file=str(item.get("file", "")),
            score=float(item.get("score", 0.0)),
            sources=list(item.get("sources", [])),
            matched_terms=list(item.get("matched_terms", [])),
            matched_fields=list(item.get("matched_fields", [])),
            selection_reason=str(item.get("selection_reason", "")),
            covered_terms=list(item.get("covered_terms", [])),
            candidate_sources=list(
                item.get("candidate_sources", item.get("sources", []))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryPlan:
    query: str
    candidates: list[Candidate] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    rejected_anchors: list[dict] = field(default_factory=list)
    prefetch_evidence_ids: list[str] = field(default_factory=list)
    relation_expansions: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    rerank: dict | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "rejected_anchors": list(self.rejected_anchors),
            "prefetch_evidence_ids": list(self.prefetch_evidence_ids),
            "relation_expansions": list(self.relation_expansions),
            "diagnostics": list(self.diagnostics),
            "rerank": self.rerank,
        }
