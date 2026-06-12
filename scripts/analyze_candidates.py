# -*- coding: utf-8 -*-
"""分析 embedding 候选与 QA 问题的相关性 —— 前 10 题"""

import json
import sys
from pathlib import Path

from mini_agent.graph_tool import GraphTool, EmbeddingRanker, _split_camel, _build_embed_text


def analyze(project_path: str, qa_path: str, limit: int = 10):
    qa_data = json.loads(Path(qa_path).read_text(encoding="utf-8"))

    gt = GraphTool(project_path)
    gt.embedding_ranker = EmbeddingRanker()
    print("构建图中...", flush=True)
    print(gt.ensure_built(), flush=True)

    print(f"\n{'='*80}")
    print(f"Embedding 候选与 QA 问题相关性分析 (前 {limit} 题)")
    print(f"模型: nomic-embed-text | 仓库: sqlfluff | 索引符号数: {len(gt.embedding_ranker._embeddings) if gt.embedding_ranker._embeddings else 'N/A'}")
    print(f"{'='*80}")

    for idx, qa in enumerate(qa_data[:limit]):
        q = qa["question"]
        expected = qa.get("answer", "")

        print(f"\n{'─'*80}")
        print(f"## QA {idx+1} / {limit}")
        print(f"问题: {q}")
        print()

        # 提取答案中提到的关键符号
        import re
        # 匹配答案中的函数/类/方法名和文件路径
        answer_symbols = set()
        for m in re.finditer(r'`([^`]+)`', expected):
            name = m.group(1)
            # 过滤掉纯路径、行号等
            if '/' not in name and '\\' not in name and 'line' not in name.lower():
                answer_symbols.add(name)
        # 也提取文件路径
        answer_files = set()
        for m in re.finditer(r'`([^`]*\.py[^`]*)`', expected):
            answer_files.add(m.group(1))
        # 提取裸写的函数调用模式
        for m in re.finditer(r'\b([a-z_]+\(\))', expected):
            answer_symbols.add(m.group(1).rstrip('()'))

        print(f"答案中提到的符号: {sorted(answer_symbols)[:15] if answer_symbols else '(未提取到)'}")
        if answer_files:
            print(f"答案中提到的文件: {sorted(answer_files)[:10]}")

        # 运行 embedding 排名
        candidates = gt.embedding_ranker.rank(q, limit=10)

        print(f"\nTop-10 Embedding 候选:")
        print(f"{'排名':<5} {'相似度':<8} {'类型':<10} {'名称':<45} {'文件':<50}")
        print(f"{'─'*120}")

        hit_count = 0
        for rank, c in enumerate(candidates, 1):
            name = c["name"]
            ctype = c["type"]
            cfile = c.get("file", "")
            score = c["score"]

            # 检查是否命中答案符号
            hit_mark = ""
            if answer_symbols:
                name_lower = name.lower()
                for sym in answer_symbols:
                    sym_clean = sym.lower().rstrip('()')
                    if sym_clean in name_lower or name_lower in sym_clean:
                        hit_mark = " ←★ 命中"
                        hit_count += 1
                        break

            print(f"{rank:<5} {score:<8.4f} {ctype:<10} {name:<45} {cfile:<50}{hit_mark}")

        if hit_count == 0 and answer_symbols:
            print(f"\n  ⚠ 前10候选均未命中答案中提到的关键符号: {sorted(answer_symbols)[:10]}")

        # 简要分析
        top1 = candidates[0]
        print(f"\n  简要评估:")
        print(f"    Top-1: {top1['name']} ({top1['type']}, sim={top1['score']:.4f})")

        # 从问题中提取关键短语，看看 top1 名字是否相关
        q_words = set(re.findall(r'[A-Z][a-z]+|[a-z]+', q.lower()))
        top1_name_parts = set(_split_camel(top1['name']).split())
        overlap = q_words & top1_name_parts
        print(f"    问题词 ∩ Top-1名 = {overlap if overlap else '(无直接词重叠)'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--qa-path", required=True)
    args = parser.parse_args()
    analyze(args.project_path, args.qa_path)
