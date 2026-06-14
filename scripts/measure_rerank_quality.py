# -*- coding: utf-8 -*-
"""直接测重排排序质量：gold 实体在重排前后排名变化。"""

import json, statistics
from pathlib import Path

from mini_agent.graph_tool import GraphTool
from mini_agent.model import Model, ModelConfig
from mini_agent.reranker import Reranker
from mini_agent.retrieval_metrics import (
    _matched_gold, _normalize_symbol, extract_provisional_gold,
)

qa = json.loads(Path("/home/amd-jk6kg8k/program/sqlfluff_qa.json").read_text())
gt = GraphTool("/home/amd-jk6kg8k/program/sqlfluff_repo", enable_embeddings=False)
gt.ensure_built()
reranker_model = Model(ModelConfig(
    base_url="http://localhost:11434/v1", api_key="ollama",
    model_name="qwen2.5:7b", quiet=True,
))
reranker = Reranker(reranker_model, project_root="/home/amd-jk6kg8k/program/sqlfluff_repo")

rank_deltas = []  # 重排后排名 - 重排前排名（负值=提升）
gold_in_candidates = 0
gold_not_in_candidates = 0
per_question: list[dict] = []

for i, q in enumerate(qa):
    question = q["question"]
    gold = extract_provisional_gold(q.get("answer", ""))

    retrieval = gt.retrieve_query_candidates(question, limit=24, use_embeddings=False)
    candidates = [c.to_dict() for c in retrieval.candidates]

    # 记录每个 gold 实体在原始候选中的排名
    gold_ranks_before = {}
    for sym in gold.symbols:
        for rank, c in enumerate(candidates, 1):
            c_id = _normalize_symbol(str(c.get("id", "")))
            s = _normalize_symbol(sym)
            terminal = s.split("::")[-1] if "::" in s else s
            if c_id.endswith("::" + s) or c_id.endswith("::" + terminal):
                gold_ranks_before[sym] = rank
                break
            elif _normalize_symbol(str(c.get("name", ""))) == terminal:
                gold_ranks_before[sym] = rank
                break

    # 重排
    rr = reranker.rerank(question, candidates)
    reranked = reranker.apply(question, candidates, rerank_result=rr) if rr.passed else candidates

    # 记录每个 gold 实体在重排后的排名
    gold_ranks_after = {}
    for sym in gold.symbols:
        for rank, c in enumerate(reranked, 1):
            c_id = _normalize_symbol(str(c.get("id", "")))
            s = _normalize_symbol(sym)
            terminal = s.split("::")[-1] if "::" in s else s
            if c_id.endswith("::" + s) or c_id.endswith("::" + terminal):
                gold_ranks_after[sym] = rank
                break
            elif _normalize_symbol(str(c.get("name", ""))) == terminal:
                gold_ranks_after[sym] = rank
                break

    # 统计
    for sym in gold.symbols:
        before = gold_ranks_before.get(sym)
        after = gold_ranks_after.get(sym)
        if before is None:
            gold_not_in_candidates += 1
        else:
            gold_in_candidates += 1
            delta = after - before if after else 0
            rank_deltas.append({
                "q": i + 1, "symbol": sym, "before": before, "after": after, "delta": delta,
            })

    improved = sum(1 for sym in gold.symbols
                   if gold_ranks_before.get(sym) and gold_ranks_after.get(sym)
                   and gold_ranks_after[sym] < gold_ranks_before[sym])
    degraded = sum(1 for sym in gold.symbols
                   if gold_ranks_before.get(sym) and gold_ranks_after.get(sym)
                   and gold_ranks_after[sym] > gold_ranks_before[sym])
    per_question.append({"q": i + 1, "gold_total": gold.count, "in_candidates": len(gold_ranks_before), "improved": improved, "degraded": degraded})

# ── 汇总 ──
deltas = [d["delta"] for d in rank_deltas if d["delta"] != 0]
improved = [d for d in rank_deltas if d["delta"] < 0]
degraded = [d for d in rank_deltas if d["delta"] > 0]
unchanged = [d for d in rank_deltas if d["delta"] == 0]

print(f"Gold 实体在候选列表内: {gold_in_candidates} / {gold_in_candidates + gold_not_in_candidates}")
print(f"平均排名变化: {statistics.fmean(deltas):+.1f}" if deltas else "无排名变化")
print(f"排名中位数变化: {statistics.median(deltas):+.0f}" if deltas else "")
print(f"提升: {len(improved)}  退化: {len(degraded)}  不变: {len(unchanged)}")
if improved:
    avg_improve = statistics.fmean([-d["delta"] for d in improved])
    print(f"提升幅度: 平均上升 {avg_improve:.1f} 位")

# Top-5 命中率对比
before_top5 = sum(1 for d in rank_deltas if d["before"] and d["before"] <= 5)
after_top5 = sum(1 for d in rank_deltas if d["after"] and d["after"] <= 5)
print(f"\nGold 进入 Top-5: 重排前={before_top5} → 重排后={after_top5}  ({after_top5 - before_top5:+d})")

# 按题目汇总
questions_improved = sum(1 for pq in per_question if pq["improved"] > pq["degraded"])
questions_degraded = sum(1 for pq in per_question if pq["degraded"] > pq["improved"])
questions_same = len(per_question) - questions_improved - questions_degraded
print(f"\n题目级别: 提升={questions_improved} 退化={questions_degraded} 持平={questions_same}")

Path("/tmp/rerank_quality.json").write_text(
    json.dumps({
        "summary": {
            "gold_in_candidates": gold_in_candidates,
            "gold_not_in_candidates": gold_not_in_candidates,
            "avg_rank_delta": statistics.fmean(deltas) if deltas else 0,
            "median_rank_delta": statistics.median(deltas) if deltas else 0,
            "improved": len(improved),
            "degraded": len(degraded),
            "unchanged": len(unchanged),
            "top5_before": before_top5,
            "top5_after": after_top5,
        },
        "rank_deltas": rank_deltas,
        "per_question": per_question,
    }, ensure_ascii=False, indent=2)
)
print("Done → /tmp/rerank_quality.json")
