# -*- coding: utf-8 -*-
"""候选检索诊断脚本 —— 逐阶段展示级联检索全流程"""

import json
import os
import re
import sys
from pathlib import Path

from accg.builder import GraphBuilder

from mini_agent.retrieval import (
    Candidate,
    CandidateRetriever,
    build_entries,
    select_query_anchors,
    tokenize,
)


def load_questions(qa_path: str, limit: int | None = None) -> list[dict]:
    data = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    selected = data if limit is None else data[:limit]
    return [
        {"index": i, "question": q["question"], "answer": q.get("answer", "")}
        for i, q in enumerate(selected)
    ]


def strip_query_for_exact_matches(retriever: CandidateRetriever, query: str):
    """模拟 exact_id 和 exact_symbol 匹配逻辑并打印。"""
    print("\n── 阶段1: exact_id（×1000）──")
    stripped = query.strip().strip("`")
    exact_entry = retriever._by_id_folded.get(stripped.casefold())
    if exact_entry:
        print(f"  ✓ 命中: {exact_entry.id} → 直接返回，跳过后续所有阶段")
        return True
    print("  无精确 Node ID 匹配")

    print("\n── 阶段2: exact_symbol（×900）──")
    mentions = {
        m.group(1).strip().casefold()
        for m in re.finditer(r"`([^`]+)`", query)
    }
    identifiers = {
        m.group(0).rstrip("(").casefold()
        for m in re.finditer(
            r"\b(?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*"
            r"|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"
            r"|\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b"
            r"|\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
            query,
        )
    }
    all_identifiers = mentions | identifiers
    if all_identifiers:
        print(f"  提取标识符: {sorted(all_identifiers)[:8]}")
    else:
        print("  未提取到语法明确标识符")

    hit_count = 0
    for ident in all_identifiers:
        hits = retriever._by_name.get(ident, [])
        for h in hits:
            hit_count += 1
            if hit_count <= 5:
                print(f"    命中: {h.id} (name={h.name})")
        q_hits = retriever._by_qualified_name.get(ident, [])
        for h in q_hits:
            hit_count += 1
            if hit_count <= 5:
                print(f"    命中: {h.id} (qualified)")
    if hit_count > 5:
        print(f"    ...共 {hit_count} 条")
    if hit_count == 0:
        print("  未命中任何符号")
    return False


def print_lexical_stage(retriever: CandidateRetriever, query: str):
    """运行词法阶段并打印详情。"""
    print("\n── 阶段3: lexical（BM25F 字段加权）──")
    query_terms = list(dict.fromkeys(tokenize(query)))
    print(f"  Query 分词: {query_terms}")
    print(f"  候选总数: {len(retriever.entries)} 条")
    print(f"  平均文档长度: {retriever._average_document_length:.1f}")
    print("  字段权重: id×3.0 name×4.0 qname×3.0 type×1.2 file×1.4 sig×2.0 doc×1.0 decor×2.5 summary×5.0")

    results = retriever._lexical_rank(query)
    print(f"  BM25 命中: {len(results)} 条")
    for entry, score, terms, fields in results[:8]:
        types = {
            "id", "name", "qualified_name", "type", "file",
            "signature", "docstring", "decorator", "summary",
        }
        hit_fields = [f for f in fields if f in types]
        print(
            f"    {entry.id}"
            f"  score={score:.2f}"
            f"  terms={terms[:4]}"
            f"  fields={hit_fields[:4]}"
        )
    if len(results) > 8:
        print(f"    ...共 {len(results)} 条")


def print_fuzzy_stage(retriever: CandidateRetriever, query: str):
    """运行模糊阶段并打印详情。"""
    print("\n── 阶段5: fuzzy（SequenceMatcher + 词覆盖）──")
    results = retriever._fuzzy_rank(query)
    print(f"  最低相似度阈值: {retriever._config.fuzzy_min_similarity}")
    print(f"  模糊命中: {len(results)} 条")
    for entry, score, terms, fields in results[:5]:
        print(f"    {entry.id}  score={score:.4f}  terms={terms[:3]}  field={fields}")
    if len(results) > 5:
        print(f"    ...共 {len(results)} 条")
    if not results:
        print("  无模糊匹配结果")


def print_anchors(graph_tool, candidates: list[dict], query: str):
    """展示锚点选择。"""
    print("\n── 锚点选择 ──")
    typed_candidates = []
    for item in candidates:
        typed_candidates.append(Candidate(
            id=item.get("id", ""),
            name=item.get("name", ""),
            type=item.get("type", ""),
            file=item.get("file", ""),
            score=float(item.get("score", 0.0)),
            sources=list(item.get("sources", [])),
            matched_terms=list(item.get("matched_terms", [])),
            matched_fields=list(item.get("matched_fields", [])),
        ))
    anchors = select_query_anchors(typed_candidates, max_anchors=3)
    for i, a in enumerate(anchors, 1):
        print(f"  #{i} {a.id}  type={a.type}  score={a.score:.2f}  sources={a.sources}")


def main():
    proj_path = os.environ.get("PROJECT_PATH") or os.getenv("PROJECT_PATH")
    qa_path = os.environ.get("QA_PATH") or os.getenv("QA_PATH")
    if not proj_path or not qa_path:
        # 服务器路径回退
        proj_path = os.path.expanduser("~/program/sqlfluff_repo")
        qa_path = os.path.expanduser("~/program/sqlfluff_qa.json")

    print(f"项目: {proj_path}")
    print(f"QA: {qa_path}")

    questions = load_questions(qa_path)
    print(f"共 {len(questions)} 题\n")

    # 构建图和检索器（只执行一次）
    sys.stdout.write("构建代码图...")
    sys.stdout.flush()
    builder = GraphBuilder()
    graph = builder.build(proj_path)
    print(f" {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

    sys.stdout.write("初始化候选检索器...")
    sys.stdout.flush()
    summaries = {}
    summ_path = Path(proj_path) / ".accg" / "summary_index.json"
    if summ_path.is_file():
        summaries = json.loads(summ_path.read_text(encoding="utf-8"))
        print(f" 加载 {len(summaries)} 条离线摘要")
    else:
        print(" 无离线摘要")
    entries = build_entries(graph, summaries)
    retriever = CandidateRetriever(entries)
    print(f" 索引 {len(entries)} 条候选")

    for qa in questions:
        q = qa["question"]
        q_index = qa["index"]
        print(f"\n{'=' * 80}")
        print(f"QA #{q_index+1}: {q[:120]}...")
        print(f"{'=' * 80}")

        # 阶段1+2
        exact = strip_query_for_exact_matches(retriever, q)
        if exact:
            continue

        # 阶段3
        print_lexical_stage(retriever, q)

        # 阶段4: embedding
        print("\n── 阶段4: embedding（语义向量）──")
        print("  (未启用 embedding，跳过)")

        # 阶段5
        print_fuzzy_stage(retriever, q)

        # 完整检索结果汇总
        print("\n── 阶段合并 ──")
        result = retriever.retrieve(
            q,
            limit=10,
            embedding_candidates=None,
            embedding_attempted=False,
            embedding_error="未启用",
        )
        print(f"  stages_attempted: {result.stages_attempted}")
        print(f"  stages_succeeded: {result.stages_succeeded}")
        print(f"  status: {result.status}")
        print(f"  diagnostics: {result.diagnostics}")

        print("\n── 最终候选 Top-10 ──")
        for i, c in enumerate(result.candidates, 1):
            print(
                f"  [{i:2d}] {c.id}"
                f"  type={c.type}"
                f"  score={c.score:.2f}"
                f"  sources={c.sources}"
                f"  terms={c.matched_terms[:3]}"
            )

        # Anchors
        print_anchors(None, [c.to_dict() for c in result.candidates], q)

        # 非交互模式单题展示
        if "--pause" in sys.argv:
            input("\n按回车继续下一题...")

    print(f"\n{'=' * 80}")
    print(f"全部 {len(questions)} 题检索诊断完成")


if __name__ == "__main__":
    main()
