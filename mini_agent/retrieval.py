# -*- coding: utf-8 -*-
"""确定性的候选符号检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

_STAGE_ORDER = {
    "exact_id": 0,
    "exact_symbol": 1,
    "lexical": 2,
    "embedding": 3,
    "fuzzy": 4,
}

@dataclass
class RetrievalConfig:
    """候选检索的可配置参数。"""

    recall_pool_limit: int = 200
    w_lexical: float = 0.35
    w_embedding: float = 0.35
    w_exact: float = 0.20
    w_fuzzy: float = 0.10

    # 类别降权系数
    test_category_multiplier: float = 0.55
    docs_category_multiplier: float = 0.7

    # 模糊匹配最低阈值
    fuzzy_min_similarity: float = 0.35

    # BM25 字段权重
    field_boosts: dict[str, float] = field(default_factory=lambda: {
        "id": 3.0,
        "name": 4.0,
        "qualified_name": 3.0,
        "type": 1.2,
        "file": 1.4,
        "signature": 2.0,
        "docstring": 1.0,
        "decorator": 2.5,
        "summary": 5.0,
    })


# 保留模块级默认实例，供外部模块直接引用（向后兼容）
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "work", "works", "responsibility", "relationship",
})


@dataclass
class Candidate:
    id: str
    name: str
    type: str
    file: str
    score: float
    sources: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    stages_attempted: list[str]
    stages_succeeded: list[str]
    diagnostics: list[str]
    status: str = "ok"
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "stages_attempted": list(self.stages_attempted),
            "stages_succeeded": list(self.stages_succeeded),
            "diagnostics": list(self.diagnostics),
            "candidates": [
                {"rank": rank, **candidate.to_dict()}
                for rank, candidate in enumerate(self.candidates, 1)
            ],
        }


@dataclass
class _Entry:
    id: str
    name: str
    type: str
    file: str
    category: str
    fields: dict[str, list[str]]


@dataclass
class _CandidateSignals:
    entry: _Entry
    norm_lexical: float = 0.0
    norm_embedding: float = 0.0
    exact_bonus: float = 0.0
    fuzzy_bonus: float = 0.0
    sources: set[str] = field(default_factory=set)
    matched_terms: set[str] = field(default_factory=set)
    matched_fields: set[str] = field(default_factory=set)


def _stem_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes")) and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and token[-3] in {"s", "x", "z"}:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    for suffix in ("ing", "ed", "ers", "er"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stem = token[:-len(suffix)]
            if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
                stem = stem[:-1]
            return stem
    return token


# CJK 统一汉字范围（CJK Unified Ideographs）
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


def tokenize(text: str) -> list[str]:
    """拆分路径、snake_case、CamelCase 与 CJK 字符，并做轻量词形归一化。"""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", str(text))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    lowered = text.lower()
    raw_tokens = re.findall(r"[A-Za-z0-9]+", lowered)
    # CJK 字符按单字拆分，确保中文查询词参与检索
    for cjk_run in _CJK_RE.findall(text):
        raw_tokens.extend(list(cjk_run))
    return [
        _stem_token(token)
        for token in raw_tokens
        if token not in STOP_WORDS and (len(token) > 1 or _is_cjk(token))
    ]


def _category_for_path(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    basename = parts[-1] if parts else normalized
    if (
        "tests" in parts
        or "test" in parts
        or basename.startswith("test_")
        or basename.endswith("_test.py")
    ):
        return "test"
    if "docs" in parts or "doc" in parts or normalized.endswith(".rst"):
        return "docs"
    return "source"


def _query_targets_category(query: str, category: str) -> bool:
    terms = set(tokenize(query))
    if category == "test":
        return bool(terms & {"test", "pytest", "fixture"})
    if category == "docs":
        return bool(terms & {"doc", "documentation", "readme"})
    return True


def _category_multiplier(
    entry: _Entry,
    query: str,
    config: RetrievalConfig | None = None,
) -> float:
    cfg = config or DEFAULT_RETRIEVAL_CONFIG
    if entry.category == "source" or _query_targets_category(query, entry.category):
        return 1.0
    return (
        cfg.test_category_multiplier
        if entry.category == "test"
        else cfg.docs_category_multiplier
    )


def build_entries(graph, summaries: dict[str, str] | None = None) -> list[_Entry]:
    """从图中构建检索条目，可选注入离线摘要。"""
    if summaries is None:
        summaries = {}
    entries: list[_Entry] = []
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        type_name = getattr(node_type, "name", str(node_type or ""))
        if type_name not in {"FUNCTION", "METHOD", "CLASS"}:
            continue

        name = str(data.get("name", ""))
        if not name:
            continue

        file_path = str(data.get("file_path", ""))
        parent_id = str(data.get("parent_id") or "")
        id_parts = str(node_id).split("::")
        parent_name = ""
        if parent_id:
            parent_name = parent_id.rsplit("::", 1)[-1]
        elif len(id_parts) >= 3:
            parent_name = id_parts[-2]

        extra = data.get("extra") or {}
        decorators = data.get("decorators") or extra.get("decorators") or []
        if isinstance(decorators, str):
            decorators = [decorators]

        qualified_name = "::".join(
            part for part in (parent_name, name) if part
        )
        decorator_text = " ".join(str(item) for item in decorators)
        decorator_text = re.sub(
            r"\b(static|class)method\b",
            r"\1 method",
            decorator_text,
            flags=re.IGNORECASE,
        )
        field_values = {
            "id": str(node_id),
            "name": name,
            "qualified_name": qualified_name,
            "type": type_name,
            "file": file_path,
            "signature": str(data.get("signature", "")),
            "docstring": str(data.get("docstring", "")),
            "decorator": decorator_text,
            "summary": summaries.get(str(node_id), ""),
        }
        entries.append(_Entry(
            id=str(node_id),
            name=name,
            type=type_name,
            file=file_path,
            category=_category_for_path(file_path),
            fields={
                field_name: tokenize(value)
                for field_name, value in field_values.items()
                if value
            },
        ))
    return sorted(entries, key=lambda item: item.id)


class CandidateRetriever:
    """在图符号元数据上执行可观察、可降级的检索级联。"""

    def __init__(
        self,
        entries: list[_Entry],
        config: RetrievalConfig | None = None,
    ):
        self._config = config or DEFAULT_RETRIEVAL_CONFIG
        self.entries = entries
        self._by_id = {entry.id: entry for entry in entries}
        self._by_id_folded = {
            entry.id.casefold(): entry
            for entry in entries
        }
        self._by_name: dict[str, list[_Entry]] = {}
        self._by_qualified_name: dict[str, list[_Entry]] = {}
        self._phrase_entries: dict[str, list[_Entry]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._weighted_documents: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, float] = {}
        for entry in entries:
            self._by_name.setdefault(entry.name.casefold(), []).append(entry)
            qualified_name = "::".join(entry.id.split("::")[1:])
            if qualified_name:
                self._by_qualified_name.setdefault(
                    qualified_name.casefold(),
                    [],
                ).append(entry)
            for field_name in ("name", "qualified_name"):
                tokens = entry.fields.get(field_name, [])
                if len(tokens) >= 2:
                    self._phrase_entries.setdefault(
                        " ".join(tokens),
                        [],
                    ).append(entry)

            weighted = Counter()
            seen = set()
            for field_name, tokens in entry.fields.items():
                boost = self._config.field_boosts[field_name]
                for token in tokens:
                    weighted[token] += boost
                    seen.add(token)
            self._weighted_documents[entry.id] = weighted
            self._document_lengths[entry.id] = sum(weighted.values())
            self._document_frequency.update(seen)
        self._average_document_length = (
            sum(self._document_lengths.values()) / len(self._document_lengths)
            if self._document_lengths else 1.0
        )

    def retrieve(
        self,
        query: str,
        limit: int = 12,
        embedding_candidates: list[dict] | None = None,
        embedding_attempted: bool = False,
        embedding_error: str | None = None,
    ) -> RetrievalResult:
        """四阶段级联：召回→精确→细化→加权排序。"""
        attempted: list[str] = []
        succeeded: list[str] = []
        diagnostics: list[str] = []
        cfg = self._config

        # Stage 1: lexical 与 embedding 各自归一化，合并后截断召回池。
        recall_signals: dict[str, _CandidateSignals] = {}
        attempted.append("lexical")
        lexical = self._lexical_rank(query)[:cfg.recall_pool_limit]
        if lexical:
            succeeded.append("lexical")
            _lex_scores = [score for _, score, _, _ in lexical if score > 0]
            lexical_max = max(_lex_scores) if _lex_scores else 1.0
            for entry, score, terms, fields in lexical:
                signals = recall_signals.setdefault(
                    entry.id,
                    _CandidateSignals(entry=entry),
                )
                signals.norm_lexical = score / lexical_max
                signals.sources.add("lexical")
                signals.matched_terms.update(terms)
                signals.matched_fields.update(fields)

        if embedding_attempted:
            attempted.append("embedding")
            if embedding_error:
                diagnostics.append(f"embedding 失败: {embedding_error}")
            elif embedding_candidates:
                valid_embedding = []
                for item in embedding_candidates[:cfg.recall_pool_limit]:
                    entry = self._by_id.get(str(item.get("id", "")))
                    if entry is None:
                        continue
                    score = max(0.0, float(item.get("score", 0.0)))
                    if score > 0:
                        valid_embedding.append((entry, score))
                if valid_embedding:
                    succeeded.append("embedding")
                    _emb_scores = [score for _, score in valid_embedding]
                    embedding_max = max(_emb_scores) if _emb_scores else 1.0
                    for entry, score in valid_embedding:
                        signals = recall_signals.setdefault(
                            entry.id,
                            _CandidateSignals(entry=entry),
                        )
                        signals.norm_embedding = score / embedding_max
                        signals.sources.add("embedding")
                else:
                    diagnostics.append("embedding 未返回有效候选")
            else:
                diagnostics.append("embedding 未返回候选")

        ranked_recall = sorted(
            recall_signals.values(),
            key=lambda signals: (
                -self._recall_score(signals),
                self._category_rank(signals.entry.file, query),
                signals.entry.id,
            ),
        )
        pool = {
            signals.entry.id: signals
            for signals in ranked_recall[:cfg.recall_pool_limit]
        }

        # Stage 2: exact 只能提升召回池内候选。
        attempted.extend(["exact_id", "exact_symbol"])
        exact_id, exact_symbol = self._exact_matches(query)
        exact_id_in_pool = [entry for entry in exact_id if entry.id in pool]
        if exact_id_in_pool:
            succeeded.append("exact_id")
            for entry in exact_id_in_pool:
                signals = pool[entry.id]
                signals.exact_bonus = 1.0
                signals.sources.add("exact_id")
                signals.matched_terms.add(entry.id)
                signals.matched_fields.add("id")

        exact_symbol_in_pool = [
            entry for entry in exact_symbol if entry.id in pool
        ]
        if exact_symbol_in_pool:
            succeeded.append("exact_symbol")
            for entry in exact_symbol_in_pool:
                signals = pool[entry.id]
                signals.exact_bonus = 1.0
                signals.sources.add("exact_symbol")
                signals.matched_terms.update(tokenize(entry.name))
                signals.matched_fields.add("name")

        # Stage 3: fuzzy 仅扫描召回池。
        attempted.append("fuzzy")
        fuzzy = self._fuzzy_rank_for_pool(query, pool)
        if fuzzy:
            succeeded.append("fuzzy")
            for entry, score, terms, fields in fuzzy:
                signals = pool[entry.id]
                signals.fuzzy_bonus = score
                signals.sources.add("fuzzy")
                signals.matched_terms.update(terms)
                signals.matched_fields.update(fields)

        # Stage 4: 严格按四项权重计算最终相关度。
        scored_candidates = [
            self._to_candidate(signals)
            for signals in pool.values()
        ]
        candidates = sorted(
            scored_candidates,
            key=lambda item: (
                -item.score,
                self._category_rank(item.file, query),
                item.id,
            ),
        )[:limit]
        embedding_degraded = (
            embedding_attempted and "embedding" not in succeeded
        )
        status = "fallback" if embedding_degraded and candidates else "ok"
        if not candidates:
            status = "failed"
            diagnostics.append("所有检索阶段均未返回候选")
        return RetrievalResult(
            candidates=candidates,
            stages_attempted=attempted,
            stages_succeeded=succeeded,
            diagnostics=diagnostics,
            status=status,
        )

    def _recall_score(self, signals: _CandidateSignals) -> float:
        return (
            self._config.w_lexical * signals.norm_lexical
            + self._config.w_embedding * signals.norm_embedding
        )

    def _to_candidate(self, signals: _CandidateSignals) -> Candidate:
        score = (
            self._recall_score(signals)
            + self._config.w_exact * signals.exact_bonus
            + self._config.w_fuzzy * signals.fuzzy_bonus
        )
        return Candidate(
            id=signals.entry.id,
            name=signals.entry.name,
            type=signals.entry.type,
            file=signals.entry.file,
            score=round(score, 6),
            sources=sorted(
                signals.sources,
                key=lambda item: _STAGE_ORDER[item],
            ),
            matched_terms=sorted(signals.matched_terms),
            matched_fields=sorted(signals.matched_fields),
        )

    def _exact_matches(self, query: str) -> tuple[list[_Entry], list[_Entry]]:
        stripped = query.strip().strip("`")
        exact_entry = self._by_id_folded.get(stripped.casefold())
        exact_id = [exact_entry] if exact_entry is not None else []
        if exact_id:
            return exact_id, []

        mentions = {
            match.group(1).strip().casefold()
            for match in re.finditer(r"`([^`]+)`", query)
        }
        explicit_identifiers = {
            match.group(0).rstrip("(").casefold()
            for match in re.finditer(
                r"\b(?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*"
                r"|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"
                r"|\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b"
                r"|\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
                query,
            )
        }
        query_words = " ".join(tokenize(query))
        exact_symbols: dict[str, _Entry] = {}
        for identifier in mentions | explicit_identifiers:
            entry = self._by_id_folded.get(identifier)
            if entry is not None:
                exact_symbols[entry.id] = entry
            for entry in self._by_name.get(identifier, []):
                exact_symbols[entry.id] = entry
            for entry in self._by_qualified_name.get(identifier, []):
                exact_symbols[entry.id] = entry

        padded_query = f" {query_words} "
        for phrase, entries in self._phrase_entries.items():
            if f" {phrase} " not in padded_query:
                continue
            for entry in entries:
                exact_symbols[entry.id] = entry
        return [], sorted(exact_symbols.values(), key=lambda entry: entry.id)

    def _lexical_rank(
        self,
        query: str,
    ) -> list[tuple[_Entry, float, list[str], list[str]]]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not self.entries:
            return []

        total_documents = len(self.entries)
        ranked = []
        for entry in self.entries:
            weighted = self._weighted_documents[entry.id]
            length = self._document_lengths[entry.id]
            score = 0.0
            matched_terms = []
            matched_fields = set()
            for term in query_terms:
                frequency = weighted.get(term, 0.0)
                if not frequency:
                    continue
                matched_terms.append(term)
                for field_name, tokens in entry.fields.items():
                    if term in tokens:
                        matched_fields.add(field_name)
                idf = math.log(
                    1.0 + (
                        total_documents - self._document_frequency[term] + 0.5
                    ) / (self._document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1.0 - 0.75
                    + 0.75 * length / max(
                        self._average_document_length,
                        1.0,
                    )
                )
                score += idf * frequency * 2.5 / denominator

            if score:
                score *= _category_multiplier(entry, query, self._config)
                ranked.append((
                    entry,
                    round(score, 6),
                    sorted(set(matched_terms)),
                    sorted(matched_fields),
                ))
        return sorted(ranked, key=lambda item: (-item[1], item[0].id))

    def _fuzzy_rank(
        self,
        query: str,
    ) -> list[tuple[_Entry, float, list[str], list[str]]]:
        return self._fuzzy_rank_entries(query, self.entries)

    def _fuzzy_rank_for_pool(
        self,
        query: str,
        pool: dict[str, _CandidateSignals],
    ) -> list[tuple[_Entry, float, list[str], list[str]]]:
        entries = [
            self._by_id[candidate_id]
            for candidate_id in pool
            if candidate_id in self._by_id
        ]
        return self._fuzzy_rank_entries(query, entries)

    def _fuzzy_rank_entries(
        self,
        query: str,
        entries: list[_Entry],
    ) -> list[tuple[_Entry, float, list[str], list[str]]]:
        query_text = " ".join(tokenize(query))
        query_terms = set(query_text.split())
        if not query_text or not entries:
            return []

        ranked = []
        for entry in entries:
            best_score = 0.0
            best_field = ""
            matched_terms = set()
            for field_name in ("name", "qualified_name", "file", "signature"):
                field_tokens = entry.fields.get(field_name, [])
                if not field_tokens:
                    continue
                field_text = " ".join(field_tokens)
                overlap = query_terms & set(field_tokens)
                coverage = len(overlap) / max(len(query_terms), 1)
                similarity = SequenceMatcher(
                    None, query_text, field_text
                ).ratio()
                score = max(similarity, coverage)
                if score > best_score:
                    best_score = score
                    best_field = field_name
                    matched_terms = overlap
            if best_score >= self._config.fuzzy_min_similarity:
                ranked.append((
                    entry,
                    round(best_score, 6),
                    sorted(matched_terms),
                    [best_field] if best_field else [],
                ))
        return sorted(ranked, key=lambda item: (-item[1], item[0].id))

    @staticmethod
    def _category_rank(file_path: str, query: str) -> int:
        category = _category_for_path(file_path)
        if category == "source" or _query_targets_category(query, category):
            return 0
        return 1 if category == "docs" else 2


def select_query_anchors(
    candidates: list[Candidate],
    max_anchors: int = 3,
    preferred_types: list[str] | None = None,
    required_types: list[str] | None = None,
    prefer_term_coverage: bool = False,
    query_terms: set[str] | None = None,
) -> list[Candidate]:
    """优先覆盖函数与类，按名去重，考虑候选名与 query 的 token 重叠。"""
    if max_anchors <= 0:
        return []

    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    selected_names: set[str] = set()  # 同名去重
    query_terms = query_terms or set()

    def _add(candidate: Candidate) -> bool:
        if candidate.id in selected_ids:
            return False
        if candidate.name.casefold() in selected_names:
            return False
        selected.append(candidate)
        selected_ids.add(candidate.id)
        selected_names.add(candidate.name.casefold())
        return len(selected) >= max_anchors

    def _name_relevance(candidate: Candidate) -> float:
        """候选名 token 与 query token 的重叠度，:: 限定名加分。"""
        name_tokens = set(tokenize(candidate.name))
        if not name_tokens or not query_terms:
            return 0.0
        overlap = name_tokens & query_terms
        score = len(overlap) / len(name_tokens)
        if "::" in candidate.id:
            score *= 1.5
        return score

    for candidate in candidates:
        if "exact_id" in candidate.sources and _add(candidate):
            return selected

    required = required_types or []
    if required:
        for wanted_type in required:
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.type == wanted_type
                    and "exact_symbol" in candidate.sources
                    and candidate.id not in selected_ids
                ),
                None,
            )
            if match is not None and _add(match):
                return selected

        for wanted_type in required:
            if any(
                candidate.type == wanted_type
                for candidate in selected
            ):
                continue
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.type == wanted_type
                    and candidate.id not in selected_ids
                ),
                None,
            )
            if match is not None and _add(match):
                return selected

    if not required and candidates and not selected:
        if _add(candidates[0]):
            return selected

    for candidate in candidates:
        if (
            "exact_symbol" in candidate.sources
            and _add(candidate)
        ):
            return selected

    if prefer_term_coverage:
        remaining = [
            candidate
            for candidate in candidates
            if candidate.id not in selected_ids
        ]
        if remaining and not selected:
            first = remaining.pop(0)
            _add(first)
        covered_terms = {
            term
            for candidate in selected
            for term in candidate.matched_terms
        }
        while remaining and len(selected) < max_anchors:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    len(
                        set(remaining[index].matched_terms)
                        - covered_terms
                    ),
                    -index,
                ),
            )
            candidate = remaining.pop(best_index)
            _add(candidate)
            covered_terms.update(candidate.matched_terms)
        if len(selected) >= max_anchors:
            return selected

    # 剩余候补：分数 + query 重叠度 + 类型平局打破
    type_rank = (
        {t: i for i, t in enumerate(preferred_types)}
        if preferred_types
        else {}
    )
    remaining = [
        candidate
        for candidate in candidates
        if candidate.id not in selected_ids
        and candidate.name.casefold() not in selected_names
    ]
    remaining.sort(
        key=lambda c: (
            -c.score,
            -_name_relevance(c),
            type_rank.get(c.type, len(type_rank)),
            candidates.index(c),
        )
    )
    for candidate in remaining:
        if _add(candidate):
            break
    return selected
