# -*- coding: utf-8 -*-
"""重排 vs 无重排 锚点命中对比测试。"""

import json
from pathlib import Path

from mini_agent.graph_tool import GraphTool
from mini_agent.model import Model, ModelConfig
from mini_agent.reranker import Reranker
from mini_agent.retrieval_metrics import _matched_gold, extract_provisional_gold

qa = json.loads(Path("/home/amd-jk6kg8k/program/sqlfluff_qa.json").read_text())
gt = GraphTool("/home/amd-jk6kg8k/program/sqlfluff_repo", enable_embeddings=False)
gt.ensure_built()
reranker_model = Model(ModelConfig(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model_name="qwen2.5:7b",
    quiet=True,
))
reranker = Reranker(reranker_model, project_root="/home/amd-jk6kg8k/program/sqlfluff_repo")

results = []
for i, q in enumerate(qa):
    question = q["question"]
    expected = q.get("answer", "")
    gold = extract_provisional_gold(expected)

    retrieval = gt.retrieve_query_candidates(question, limit=24, use_embeddings=False)
    candidates = [c.to_dict() for c in retrieval.candidates]

    # 无重排锚点
    anchors_no = gt.select_query_anchors(question, candidates, max_anchors=3)
    no_matches = sum(1 for a in anchors_no if _matched_gold(a, gold))

    # 有重排锚点
    rerank_result = reranker.rerank(question, candidates)
    if rerank_result.passed:
        reranked = reranker.apply(question, candidates)
        anchors_rerank = gt.select_query_anchors(question, reranked, max_anchors=3)
        rerank_matches = sum(1 for a in anchors_rerank if _matched_gold(a, gold))
    else:
        anchors_rerank = anchors_no
        rerank_matches = no_matches

    results.append({
        "index": i + 1,
        "gold_count": gold.count,
        "no_rerank_matches": no_matches,
        "rerank_matches": rerank_matches,
        "rerank_top5": rerank_result.ranked_ids[:5],
        "rerank_reasoning": rerank_result.reasoning,
        "rerank_ms": round(rerank_result.elapsed_ms, 0),
    })

    delta = rerank_matches - no_matches
    mark = "+" if delta > 0 else ("-" if delta < 0 else "=")
    print(f"[{i+1:>2}] {mark} gold={gold.count} before={no_matches} after={rerank_matches} | {rerank_result.elapsed_ms:.0f}ms | {rerank_result.reasoning[:80]}")

# 汇总
total = len(results)
improved = sum(1 for r in results if r["rerank_matches"] > r["no_rerank_matches"])
degraded = sum(1 for r in results if r["rerank_matches"] < r["no_rerank_matches"])
unchanged = total - improved - degraded
no_rerank_total = sum(r["no_rerank_matches"] for r in results)
rerank_total = sum(r["rerank_matches"] for r in results)
avg_ms = sum(r["rerank_ms"] for r in results) / max(total, 1)
print(f"\n{'='*60}")
print(f"汇总: 提升={improved} 退化={degraded} 不变={unchanged}")
print(f"锚点命中: {no_rerank_total} → {rerank_total}  ({rerank_total - no_rerank_total:+d})")
print(f"平均重排耗时: {avg_ms:.0f}ms")
Path("/tmp/rerank_comparison.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)
print("结果已保存到 /tmp/rerank_comparison.json")
