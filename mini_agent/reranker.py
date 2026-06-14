# -*- coding: utf-8 -*-
"""在线重排序 — 用小模型对候选列表做语义相关性判断，滤除检索噪声。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .model import Model

_RERANK_PROMPT = """You are a code search relevance evaluator. Given a user question, evaluate each candidate symbol below and select the 5 most relevant ones.

## Question
{question}

## Candidates
{candidates_text}

## Output format
Output a single JSON object, nothing else:
{{"relevant": [1st_rank_index, 2nd_rank_index, ...], "reasoning": "one sentence explaining the ranking rationale"}}

The index is the [N] number before each candidate. Output at most 5 indices."""

_CANDIDATE_DIGEST = """[{index}] {name} ({type})
  file: {file}
  signature: {signature}{docstring}{source_excerpt}"""


def _read_source_snippet(root: Path, file_path: str, start: int, end: int, max_lines: int = 25) -> str:
    """读取函数源码的前若干行作为摘要信号。"""
    full_path = root / file_path
    if not full_path.is_file() or start <= 0:
        return ""
    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(0, start - 1)
        e = min(s + max_lines, end, len(lines))
        snippet = "\n".join(lines[s:e])
        if len(snippet) > 800:
            snippet = snippet[:800] + "\n..."
        return snippet
    except Exception:
        return ""


def _build_candidate_digest(index: int, c: dict, root: Path) -> str:
    """为单个候选构建简洁的信号摘要。"""
    name = c.get("name", "?")
    ctype = c.get("type", "?")
    c.get("id", "?")
    file_path = str(c.get("file", ""))
    signature = str(c.get("signature", ""))[:200]

    docstring = ""
    source_excerpt = ""
    start = int(c.get("start_line", 0))
    end = int(c.get("end_line", 0))
    if file_path and start > 0:
        source_excerpt = _read_source_snippet(root, file_path, start, end)
        if source_excerpt:
            source_excerpt = "\n  source:\n```\n" + source_excerpt + "\n```"

    if c.get("docstring"):
        doc = str(c.get("docstring", ""))[:200]
        if doc:
            docstring = f"\n  docstring: {doc}"

    return _CANDIDATE_DIGEST.format(
        index=index,
        name=name,
        type=ctype,
        file=file_path,
        signature=signature,
        docstring=docstring,
        source_excerpt=source_excerpt,
    )


@dataclass
class RerankResult:
    """重排序结果。"""
    ranked_ids: list[str]           # 按相关性降序排列的候选 ID
    relevant_indices: list[int]      # 模型返回的相关候序号（0-based）
    reasoning: str = ""
    raw_response: str = ""
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return len(self.ranked_ids) > 0 and self.error is None


class Reranker:
    """用小模型对候选列表做在线语义重排。"""

    MAX_CANDIDATES = 24      # 送入重排的最大候选数
    MAX_OUTPUT = 5            # 输出最多 5 个
    MODEL_TEMPERATURE = 0.1   # 排序任务需要确定性

    def __init__(self, model: Model, project_root: str = ""):
        self.model = model
        self.project_root = Path(project_root) if project_root else Path()

    def rerank(self, question: str, candidates: list[dict]) -> RerankResult:
        """对候选列表进行语义重排，返回过滤后的相关候选 ID 列表。"""
        if not candidates:
            return RerankResult(
                ranked_ids=[], relevant_indices=[], error="empty candidate list",
            )

        limit = min(len(candidates), self.MAX_CANDIDATES)
        batch = candidates[:limit]

        # 构建每个候选的摘要
        digests = []
        for i, c in enumerate(batch):
            digests.append(_build_candidate_digest(i, c, self.project_root))

        candidates_text = "\n".join(digests)
        prompt = _RERANK_PROMPT.format(
            question=question,
            candidates_text=candidates_text,
        )

        t0 = time.perf_counter()
        raw = self.model.generate(
            [{"role": "user", "content": prompt}],
        )
        elapsed = (time.perf_counter() - t0) * 1000

        indices, reasoning = self._parse_response(raw)

        ranked_ids = []
        for idx in indices:
            if 0 <= idx < len(batch):
                ranked_ids.append(batch[idx].get("id", ""))

        if not ranked_ids:
            return RerankResult(
                ranked_ids=[], relevant_indices=[],
                reasoning=reasoning, raw_response=raw,
                elapsed_ms=elapsed,
                error="model returned no valid relevant indices",
            )

        return RerankResult(
            ranked_ids=ranked_ids,
            relevant_indices=indices,
            reasoning=reasoning,
            raw_response=raw,
            elapsed_ms=elapsed,
        )

    def _parse_response(self, raw: str) -> tuple[list[int], str]:
        """从模型输出中提取相关候序号列表。"""
        reasoning = ""
        indices: list[int] = []

        # 尝试直接 JSON 解析
        json_match = re.search(r'\{[^{}]*"relevant"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                raw_indices = data.get("relevant", [])
                if isinstance(raw_indices, list):
                    indices = [int(i) for i in raw_indices if isinstance(i, (int, float))]
                    reasoning = str(data.get("reasoning", ""))
                    return indices, reasoning
            except (json.JSONDecodeError, ValueError):
                pass

        # 回退：从文本中提取数字列表
        list_match = re.search(r'\[([0-9,\s]+)\]', raw)
        if list_match:
            try:
                indices = [
                    int(x.strip())
                    for x in list_match.group(1).split(",")
                    if x.strip().isdigit()
                ]
            except ValueError:
                pass

        # 回退：搜索 RANK: 前缀
        rank_match = re.search(r'RANK[:\s]\s*([0-9,\s]+)', raw)
        if rank_match and not indices:
            try:
                indices = [
                    int(x.strip())
                    for x in rank_match.group(1).split(",")
                    if x.strip().isdigit()
                ]
            except ValueError:
                pass

        return indices, reasoning

    def apply(self, question: str, candidates: list[dict],
              rerank_result: RerankResult | None = None) -> list[dict]:
        """便捷方法：重排并返回重排后的候选列表。

        保持原始候选结构，将相关候选提到前面，截断不相关的。
        可传入预计算的 rerank_result 避免重复推理。
        """
        result = rerank_result if rerank_result is not None else self.rerank(question, candidates)
        if not result.passed:
            return list(candidates)[:self.MAX_OUTPUT]

        # 构建重排列表：先放选中的（按模型顺序），再补足未选中但靠前的
        ranked_set = set(result.ranked_ids)
        reordered = []
        for cid in result.ranked_ids:
            match = next((c for c in candidates if c.get("id") == cid), None)
            if match:
                reordered.append(dict(match))

        # 补足：未选中的候选排在后面（保留原始顺序）
        for c in candidates:
            if c.get("id") not in ranked_set:
                reordered.append(dict(c))

        return reordered
