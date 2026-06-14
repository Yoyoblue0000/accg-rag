# -*- coding: utf-8 -*-
"""评估用小模型对全量函数/方法生成摘要的耗时。

用法（服务器）:
  .venv/bin/python scripts/benchmark_summarize.py \
    --project-path ~/program/sqlfluff_repo \
    --model qwen2.5:7b \
    --sample 50
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

# 不依赖 accg 也能跑基本统计
try:
    from mini_agent.model import Model, ModelConfig
except ImportError:
    print("请先安装 mini_agent: uv pip install -e .")
    sys.exit(1)

SUMMARY_PROMPT = """You are a code analysis expert. Summarize the purpose of the following function in one sentence (max 50 words). Only output the summary, no explanation, no code.

Signature: {signature}
Source:
```
{source}
```

One-sentence summary:"""


def _read_source_lines(root: Path, file_path: str, start: int, end: int) -> str:
    """从磁盘文件读取指定行范围的源码。"""
    full_path = root / file_path
    if not full_path.is_file():
        return ""
    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        # start_line / end_line 是 1-based
        s = max(0, start - 1)
        e = max(s, min(end, len(lines)))
        return "\n".join(lines[s:e])
    except Exception:
        return ""


def extract_functions(project_path: str) -> list[dict]:
    """从 ACCG 图中提取所有函数/方法的签名和源码。"""
    from mini_agent.graph_tool import GraphTool

    gt = GraphTool(project_path, enable_embeddings=False)
    gt.ensure_built()
    graph = gt._graph
    root = Path(project_path).resolve()

    funcs = []
    source_reads = 0
    source_skipped = 0

    for node_id, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        type_name = getattr(node_type, "name", str(node_type or ""))
        if type_name not in ("FUNCTION", "METHOD"):
            continue

        name = data.get("name", "")
        file_path = str(data.get("file_path", ""))
        signature = data.get("signature", "") or name
        start_line = int(data.get("start_line", 0))
        end_line = int(data.get("end_line", 0))

        if not file_path or start_line <= 0:
            source_skipped += 1
            continue

        source_code = _read_source_lines(root, file_path, start_line, end_line)
        if not source_code.strip():
            source_skipped += 1
            continue

        source_reads += 1

        # 限制长度：最长 2000 字符（太长的函数摘要价值低）
        snippet = source_code[:2000]

        funcs.append({
            "id": str(node_id),
            "name": name,
            "type": type_name,
            "file": str(file_path),
            "signature": str(signature)[:300],
            "source": snippet,
            "start_line": int(start_line),
            "end_line": int(end_line),
            "source_len": len(source_code),
        })

    print(f"  [extract] 找到 {len(funcs)} 个函数（读到 {source_reads}，跳过 {source_skipped}）")
    return funcs


def summarize_one(model: Model, func: dict) -> dict:
    """对单个函数生成摘要，返回耗时和结果。"""
    prompt = SUMMARY_PROMPT.format(
        signature=func.get("signature", func["name"]),
        source=func["source"],
    )
    messages = [{"role": "user", "content": prompt}]

    t0 = time.perf_counter()
    result = model.generate(messages)
    elapsed = time.perf_counter() - t0
    summary = result if isinstance(result, str) else result.get("content", "")

    return {
        "id": func["id"],
        "name": func["name"],
        "summary": summary.strip()[:200],
        "elapsed_ms": round(elapsed * 1000, 1),
        "source_len": func["source_len"],
    }


def main():
    parser = argparse.ArgumentParser(description="评估函数摘要生成耗时")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--output", default="/tmp/summary_benchmark.json")
    parser.add_argument("--full-run", action="store_true",
                        help="不做采样，跑全量")
    args = parser.parse_args()

    print("提取函数列表...")
    all_funcs = extract_functions(args.project_path)
    print(f"共 {len(all_funcs)} 个函数/方法")

    # 统计
    total_source_chars = sum(f["source_len"] for f in all_funcs)
    avg_source_len = total_source_chars / max(len(all_funcs), 1)
    print(f"平均源码长度: {avg_source_len:.0f} 字符")
    print(f"总源码量: {total_source_chars / 1e6:.2f} MB")

    # 采样
    if args.full_run:
        sample = all_funcs
    else:
        random.seed(42)
        sample = random.sample(all_funcs, min(args.sample, len(all_funcs)))

    print(f"\n模型: {args.model}")
    print(f"运行 {len(sample)} 次摘要请求...")

    model = Model(ModelConfig(
        base_url=args.base_url,
        api_key="ollama",
        model_name=args.model,
        quiet=True,
    ))

    results = []
    times_ms = []
    total_chars = 0

    for i, func in enumerate(sample):
        print(f"  [{i+1:>3}/{len(sample)}] {func['name'][:50]}...", end=" ", flush=True)
        r = summarize_one(model, func)
        results.append(r)
        times_ms.append(r["elapsed_ms"])
        total_chars += func["source_len"]
        print(f"{r['elapsed_ms']:>6.0f}ms  →  {r['summary'][:60]}")

    # 统计
    avg_ms = statistics.fmean(times_ms) if times_ms else 0
    p50_ms = statistics.median(times_ms) if times_ms else 0
    p95_ms = sorted(times_ms)[int(len(times_ms) * 0.95)] if times_ms else 0
    throughput = 1000 / avg_ms if avg_ms > 0 else 0  # func/s

    # 估算全量
    total_funcs = len(all_funcs)
    total_est_ms = avg_ms * total_funcs
    total_est_min = total_est_ms / 1000 / 60
    total_est_h = total_est_min / 60

    print(f"\n{'='*60}")
    print(f"耗时统计（{len(times_ms)} 个样本）")
    print(f"{'='*60}")
    print(f"  平均: {avg_ms:.0f}ms/函数")
    print(f"  P50:  {p50_ms:.0f}ms")
    print(f"  P95:  {p95_ms:.0f}ms")
    print(f"  吞吐: {throughput:.1f} 函数/秒 = {throughput * 60:.0f} 函数/分钟")
    print(f"  平均输入长度: {total_chars / max(len(sample), 1):.0f} 字符")
    print(f"\n  全量估算（{total_funcs} 个函数）:")
    print(f"    总耗时: {total_est_min:.1f} 分钟 = {total_est_h:.2f} 小时")
    print(f"    并发 4 路: {total_est_min / 4:.1f} 分钟 = {total_est_h / 4:.2f} 小时")

    # 保存结果
    output = {
        "model": args.model,
        "total_functions": total_funcs,
        "sample_size": len(sample),
        "avg_source_len": round(avg_source_len, 0),
        "total_source_mb": round(total_source_chars / 1e6, 2),
        "avg_ms": round(avg_ms, 1),
        "p50_ms": round(p50_ms, 1),
        "p95_ms": round(p95_ms, 1),
        "throughput_func_per_sec": round(throughput, 2),
        "total_est_min": round(total_est_min, 1),
        "total_est_hours": round(total_est_h, 2),
        "concurrent4_est_min": round(total_est_min / 4, 1),
        "results": [
            {
                "id": r["id"],
                "name": r["name"],
                "summary": r["summary"],
                "elapsed_ms": r["elapsed_ms"],
                "source_len": r["source_len"],
            }
            for r in results
        ],
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
