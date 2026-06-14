# -*- coding: utf-8 -*-
"""对比不同大小模型生成摘要的质量与速度。"""

import json
import random
import statistics
import time
from pathlib import Path

from mini_agent.graph_tool import GraphTool
from mini_agent.model import Model, ModelConfig

SUMMARY_PROMPT = """You are a code analysis expert. Summarize the purpose of the following function in one sentence (max 50 words). Only output the summary, no explanation, no code.

Signature: {signature}
Source:
```
{source}
```

One-sentence summary:"""


def collect_functions(project_path: str, n: int) -> list[dict]:
    """随机采样 n 个函数。"""
    gt = GraphTool(project_path, enable_embeddings=False)
    gt.ensure_built()
    root = Path(project_path).resolve()
    funcs = []
    for nid, data in gt._graph.nodes(data=True):
        nt = data.get("node_type")
        tn = getattr(nt, "name", str(nt or ""))
        if tn not in ("FUNCTION", "METHOD"):
            continue
        fp = str(data.get("file_path", ""))
        sl = int(data.get("start_line", 0))
        el = int(data.get("end_line", 0))
        if not fp or sl <= 0:
            continue
        sf = root / fp
        if not sf.is_file():
            continue
        try:
            lines = sf.read_text(encoding="utf-8", errors="replace").splitlines()
            src = "\n".join(lines[max(0, sl - 1):max(sl - 1, el)])
            if not src.strip():
                continue
        except Exception:
            continue
        funcs.append({
            "id": str(nid), "name": data.get("name", ""),
            "type": tn, "file": fp,
            "signature": str(data.get("signature", ""))[:300],
            "source": src[:2000], "source_len": len(src),
        })
    random.seed(42)
    return random.sample(funcs, min(n, len(funcs)))


def run_model(model_name: str, funcs: list[dict], base_url: str) -> list[dict]:
    """用指定模型对一批函数生成摘要。"""
    model = Model(ModelConfig(
        base_url=base_url, api_key="ollama",
        model_name=model_name, quiet=True,
    ))
    results = []
    for f in funcs:
        prompt = SUMMARY_PROMPT.format(
            signature=f.get("signature", f["name"]),
            source=f["source"],
        )
        t0 = time.perf_counter()
        raw = model.generate([{"role": "user", "content": prompt}])
        elapsed = (time.perf_counter() - t0) * 1000
        summary = raw if isinstance(raw, str) else raw.get("content", raw)
        results.append({
            "id": f["id"], "name": f["name"], "source_len": f["source_len"],
            "summary": str(summary).strip()[:200],
            "elapsed_ms": round(elapsed, 1),
        })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", default="/home/amd-jk6kg8k/program/sqlfluff_repo")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--models", default="qwen2.5:0.5b,qwen2.5:7b,qwen2.5:14b,qwen3:4b")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--output", default="/tmp/summary_cmp.json")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    print(f"采样 {args.sample} 个函数...")
    funcs = collect_functions(args.project_path, args.sample)
    print(f"共 {len(funcs)} 个函数\n")

    all_results = {}
    for model_name in model_list:
        print(f"── 模型: {model_name} ──")
        results = run_model(model_name, funcs, args.base_url)
        times = [r["elapsed_ms"] for r in results]
        failures = sum(1 for r in results if not r["summary"].strip())
        print(f"  平均: {statistics.fmean(times):.0f}ms  P50: {statistics.median(times):.0f}ms  P95: {sorted(times)[int(len(times)*0.95)]:.0f}ms")
        print(f"  失败(空输出): {failures}/{len(results)}")
        for r in results:
            print(f"    [{r['elapsed_ms']:>6.0f}ms] {r['name'][:40]:<40} → {r['summary'][:80]}")
        print()
        all_results[model_name] = {
            "avg_ms": round(statistics.fmean(times), 1),
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 1),
            "failures": failures,
            "results": results,
        }

    # 全量估算
    total_funcs = 2208
    print(f"\n{'='*60}")
    print(f"全量估算 ({total_funcs} 个函数)")
    print(f"{'='*60}")
    for mn, data in all_results.items():
        if data["failures"] > len(funcs) * 0.5:
            print(f"  {mn}: 不可用（失败率 {data['failures']}/{len(funcs)}）")
            continue
        total_min = data["avg_ms"] * total_funcs / 1000 / 60
        print(f"  {mn}: 串行 {total_min:.1f}min, 4路并发 {total_min/4:.1f}min  (avg={data['avg_ms']:.0f}ms)")

    out = {"models": all_results, "functions": funcs}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果保存到 {args.output}")


if __name__ == "__main__":
    main()
