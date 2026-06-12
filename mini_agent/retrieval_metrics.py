# -*- coding: utf-8 -*-
"""候选检索的临时 gold 抽取与确定性指标。"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field


_SYMBOL_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$"
)
_QUALIFIED_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)+\b"
)
_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.?/)?[A-Za-z0-9_.-]+"
    r"(?:/[A-Za-z0-9_.-]+)*\.py\b"
)


@dataclass
class GoldLocations:
    paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.paths) + len(self.symbols)

    def to_dict(self) -> dict:
        return {
            "paths": list(self.paths),
            "symbols": list(self.symbols),
            "source": "reference_answer_heuristic",
        }


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./").lower()


def _normalize_symbol(symbol: str) -> str:
    return symbol.rstrip("()").strip().casefold()


def extract_provisional_gold(answer: str) -> GoldLocations:
    """从参考答案中提取反引号路径、符号和限定名。"""
    paths: set[str] = set()
    symbols: set[str] = set()

    for path_match in _PATH_PATTERN.finditer(answer):
        paths.add(_normalize_path(path_match.group(0)))

    for quoted in re.findall(r"`([^`\n]+)`", answer):
        value = quoted.strip()
        if ".py" in value:
            for path_match in _PATH_PATTERN.finditer(value):
                paths.add(_normalize_path(path_match.group(0)))
            continue
        if _SYMBOL_PATTERN.fullmatch(value):
            symbols.add(_normalize_symbol(value))

    for match in _QUALIFIED_PATTERN.finditer(answer):
        value = match.group(0)
        if ".py" not in value:
            symbols.add(_normalize_symbol(value))

    return GoldLocations(
        paths=sorted(paths),
        symbols=sorted(symbols),
    )


def _candidate_value(candidate, key: str, default=""):
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _matched_gold(candidate, gold: GoldLocations) -> set[str]:
    matches = set()
    file_path = _normalize_path(str(_candidate_value(candidate, "file", "")))
    candidate_id = _normalize_symbol(str(_candidate_value(candidate, "id", "")))
    name = _normalize_symbol(str(_candidate_value(candidate, "name", "")))

    for path in gold.paths:
        if file_path == path or file_path.endswith("/" + path):
            matches.add("path:" + path)

    for symbol in gold.symbols:
        terminal = re.split(r"::|\.", symbol)[-1]
        if (
            name == terminal
            or candidate_id.endswith("::" + symbol)
            or candidate_id.endswith("::" + terminal)
        ):
            matches.add("symbol:" + symbol)
    return matches


def evaluate_candidates(
    candidates: list,
    gold: GoldLocations,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """计算基于临时 gold 的 Recall、MRR 与二元相关 NDCG。"""
    matched_by_rank = [_matched_gold(candidate, gold) for candidate in candidates]
    relevant = [bool(matches) for matches in matched_by_rank]
    metrics = {
        "evaluable": gold.count > 0,
        "gold_count": gold.count,
        "binary_relevance": True,
        "matched_gold": sorted(set().union(*matched_by_rank))
        if matched_by_rank else [],
    }

    for k in k_values:
        found = set().union(*matched_by_rank[:k]) if matched_by_rank[:k] else set()
        recall = len(found) / gold.count if gold.count else 0.0
        metrics[f"recall_at_{k}"] = round(recall, 6)

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, is_relevant in enumerate(relevant[:k], 1)
            if is_relevant
        )
        ideal_count = min(sum(relevant), k)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        metrics[f"ndcg_at_{k}"] = round(dcg / idcg, 6) if idcg else 0.0

    first_relevant = next(
        (rank for rank, is_relevant in enumerate(relevant, 1) if is_relevant),
        None,
    )
    metrics["mrr"] = (
        round(1.0 / first_relevant, 6)
        if first_relevant is not None else 0.0
    )
    return metrics


def aggregate_retrieval_metrics(records: list[dict]) -> dict:
    """聚合 QA 记录中的检索指标与降级计数。"""
    evaluable = [
        record for record in records
        if record.get("retrieval_metrics", {}).get("evaluable")
    ]
    metric_names = [
        *(f"recall_at_{k}" for k in (1, 3, 5, 10)),
        "mrr",
        *(f"ndcg_at_{k}" for k in (1, 3, 5, 10)),
    ]
    summary = {
        "questions": len(records),
        "evaluable_questions": len(evaluable),
        "retrieval_failures": sum(
            record.get("retrieval", {}).get("status") == "failed"
            for record in records
        ),
        "fallbacks": sum(
            record.get("retrieval", {}).get("status") == "fallback"
            for record in records
        ),
        "relevance": "binary_from_reference_answer_heuristic",
    }
    latencies = [
        float(record.get("retrieval", {}).get("duration_ms", 0.0))
        for record in records
        if record.get("retrieval")
    ]
    sorted_latencies = sorted(latencies)
    summary["latency_ms_mean"] = (
        round(statistics.fmean(latencies), 3) if latencies else 0.0
    )
    summary["latency_ms_p50"] = (
        round(statistics.median(latencies), 3) if latencies else 0.0
    )
    summary["latency_ms_p95"] = (
        round(
            sorted_latencies[
                max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
            ],
            3,
        )
        if sorted_latencies else 0.0
    )
    for metric_name in metric_names:
        values = [
            record["retrieval_metrics"][metric_name]
            for record in evaluable
        ]
        summary[metric_name] = (
            round(sum(values) / len(values), 6) if values else 0.0
        )
    return summary
