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
    """候选检索的可配置参数。各阶段分数为叠加值，层级差距确保排序稳定。"""

    # 阶段基础分
    exact_id_score: float = 1000.0
    exact_symbol_score: float = 300.0
    lexical_base: float = 100.0
    embedding_base: float = 80.0
    embedding_scale: float = 80.0
    fuzzy_base: float = 10.0
    fuzzy_scale: float = 20.0

    # 类别降权系数
    test_category_multiplier: float = 0.55
    docs_category_multiplier: float = 0.7

    # 模糊匹配最低阈值
    fuzzy_min_similarity: float = 0.35

    # exact_symbol 同名命中数降权阈值
    # 同名符号数超过此值时，分数 = base / log2(hit_count)
    exact_symbol_hit_count_threshold: int = 3

    # 词法阶段缩放倍数
    lexical_scale: float = 1.0

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
        "source": 3.0,
    })


# 保留模块级默认实例，供外部模块直接引用（向后兼容）
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "work", "works", "responsibility", "relationship",
}


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
        if token not in _STOP_WORDS and (len(token) > 1 or _is_cjk(token))
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


def build_entries(graph, summaries: dict[str, str] | None = None,
                  project_path: str | None = None) -> list[_Entry]:
    """从图中构建检索条目，可选注入离线摘要和源码片段。"""
    if summaries is None:
        summaries = {}
    from pathlib import Path as _Path
    root = _Path(project_path).resolve() if project_path else None
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
        # 读取源码片段（前 2000 字符）
        source_text = ""
        start_line = int(data.get("start_line", 0))
        end_line = int(data.get("end_line", 0))
        if file_path and start_line > 0 and root is not None:
            sf = root / file_path
            if sf.is_file():
                try:
                    lines = sf.read_text(encoding="utf-8", errors="replace").splitlines()
                    source_text = "\n".join(
                        lines[max(0, start_line - 1):max(start_line - 1, end_line)]
                    )[:2000]
                except Exception:
                    pass
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
            "source": source_text,
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
        self._term_frequency: dict[str, Counter[str]] = {}
        self._term_boost: dict[str, Counter[str]] = {}
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

            tf = Counter()
            boost_acc = Counter()
            seen = set()
            for field_name, tokens in entry.fields.items():
                boost = self._config.field_boosts[field_name]
                for token in tokens:
                    tf[token] += 1
                    boost_acc[token] += boost
                    seen.add(token)
            self._term_frequency[entry.id] = tf
            self._term_boost[entry.id] = boost_acc
            self._document_lengths[entry.id] = sum(tf.values())
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
        """级联检索：召回层(lexical+embedding)→精确层(exact)→细化层(fuzzy)。"""
        attempted: list[str] = []
        succeeded: list[str] = []
        diagnostics: list[str] = []
        cfg = self._config

        # ════════════════════════════════════════════════════
        # 第一层：召回 —— lexical + embedding 并行构建候选池
        # 限制召回池大小，后续阶段只在池内操作
        # ════════════════════════════════════════════════════
        pool: dict[str, Candidate] = {}
        _RECALL_LIMIT = max(limit * 12, 150)

        attempted.append("lexical")
        lexical = self._lexical_rank(query)
        if lexical:
            succeeded.append("lexical")
            for entry, score, terms, fields in lexical[:_RECALL_LIMIT]:
                self._merge(
                    pool, entry,
                    cfg.lexical_base + score * cfg.lexical_scale,
                    "lexical", terms, fields,
                )

        if embedding_attempted:
            attempted.append("embedding")
            if embedding_error:
                diagnostics.append(f"embedding 失败: {embedding_error}")
            elif embedding_candidates:
                succeeded.append("embedding")
                for item in embedding_candidates:
                    entry = self._by_id.get(str(item.get("id", "")))
                    if entry is None:
                        continue
                    score = max(0.0, float(item.get("score", 0.0)))
                    self._merge(
                        pool, entry,
                        cfg.embedding_base + score * cfg.embedding_scale,
                        "embedding", [], [],
                    )
            else:
                diagnostics.append("embedding 未返回候选")

        # ════════════════════════════════════════════════════
        # 第二层：精确 —— exact_id/exact_symbol 对池内候选 boost
        # 池外候选也加入，但只有精确分（无 lexical/embedding 支撑）
        # ════════════════════════════════════════════════════
        attempted.extend(["exact_id", "exact_symbol"])
        exact_id, exact_symbol = self._exact_matches(query)

        if exact_id:
            succeeded.append("exact_id")
            for entry in exact_id:
                in_pool = entry.id in pool
                self._merge(
                    pool, entry, cfg.exact_id_score, "exact_id",
                    matched_terms=[entry.id],
                    matched_fields=["id"],
                )
                if in_pool:
                    diagnostics.append(f"exact_id 命中并提升: {entry.id}")

        if exact_symbol:
            succeeded.append("exact_symbol")
            for entry in exact_symbol:
                multiplier = 1.0
                if "::" in entry.id:
                    multiplier = 1.5
                bonus = cfg.exact_symbol_score * multiplier * _category_multiplier(entry, query, cfg)
                self._merge(
                    pool, entry, bonus, "exact_symbol",
                    matched_terms=tokenize(entry.name),
                    matched_fields=["name"],
                )

        # ════════════════════════════════════════════════════
        # 第三层：细化 —— fuzzy 对池内候选做文本对齐打分
        # ════════════════════════════════════════════════════
        attempted.append("fuzzy")
        fuzzy = self._fuzzy_rank_for_pool(query, pool)
        if fuzzy:
            succeeded.append("fuzzy")
            for entry, score, terms, fields in fuzzy:
                self._merge(
                    pool, entry,
                    cfg.fuzzy_base + score * cfg.fuzzy_scale,
                    "fuzzy", terms, fields,
                )

        candidates = sorted(
            pool.values(),
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
            tf_map = self._term_frequency[entry.id]
            boost_map = self._term_boost[entry.id]
            doc_len = self._document_lengths[entry.id]
            score = 0.0
            matched_terms = []
            matched_fields = set()
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if not tf:
                    continue
                matched_terms.append(term)
                for field_name, tokens in entry.fields.items():
                    if term in tokens:
                        matched_fields.add(field_name)
                boost_acc = boost_map.get(term, 0)
                boost_factor = boost_acc / max(tf, 1)
                idf = math.log(
                    1.0 + (
                        total_documents - self._document_frequency[term] + 0.5
                    ) / (self._document_frequency[term] + 0.5)
                )
                denominator = tf + 1.5 * (
                    1.0 - 0.75
                    + 0.75 * doc_len / max(
                        self._average_document_length,
                        1.0,
                    )
                )
                score += idf * tf * 2.5 * boost_factor / denominator

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
        query_text = " ".join(tokenize(query))
        query_terms = set(query_text.split())
        if not query_text:
            return []

        ranked = []
        for entry in self.entries:
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
                best_score *= _category_multiplier(entry, query, self._config)
                ranked.append((
                    entry,
                    round(best_score, 6),
                    sorted(matched_terms),
                    [best_field] if best_field else [],
                ))
        return sorted(ranked, key=lambda item: (-item[1], item[0].id))

    def _fuzzy_rank_for_pool(
        self,
        query: str,
        pool: dict[str, Candidate],
    ) -> list[tuple[_Entry, float, list[str], list[str]]]:
        """对指定候选池做 fuzzy 文本对齐，避免全图扫描。"""
        query_text = " ".join(tokenize(query))
        query_terms = set(query_text.split())
        if not query_text or not pool:
            return []

        ranked = []
        for candidate_id in pool:
            entry = self._by_id.get(candidate_id)
            if entry is None:
                continue
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
                best_score *= _category_multiplier(entry, query, self._config)
                ranked.append((
                    entry,
                    round(best_score, 6),
                    sorted(matched_terms),
                    [best_field] if best_field else [],
                ))
        return sorted(ranked, key=lambda item: (-item[1], item[0].id))

    @staticmethod
    def _merge(
        merged: dict[str, Candidate],
        entry: _Entry,
        score: float,
        source: str,
        matched_terms: list[str],
        matched_fields: list[str],
    ) -> None:
        candidate = merged.get(entry.id)
        if candidate is None:
            candidate = Candidate(
                id=entry.id,
                name=entry.name,
                type=entry.type,
                file=entry.file,
                score=0.0,
            )
            merged[entry.id] = candidate
        candidate.score = round(candidate.score + score, 6)
        candidate.sources = sorted(
            set(candidate.sources) | {source},
            key=lambda item: _STAGE_ORDER[item],
        )
        candidate.matched_terms = sorted(
            set(candidate.matched_terms) | set(matched_terms)
        )
        candidate.matched_fields = sorted(
            set(candidate.matched_fields) | set(matched_fields)
        )

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
