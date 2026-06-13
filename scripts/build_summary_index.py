# -*- coding: utf-8 -*-
"""离线摘要索引构建 — 用 7B 模型对全量函数生成摘要并持久化到 .accg/summary_index.json。

用法:
  .venv/bin/python scripts/build_summary_index.py \
    --project-path ~/program/sqlfluff_repo \
    --model qwen2.5:7b \
    --workers 4
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SUMMARY_PROMPT = """You are a code analysis expert. Summarize the purpose of the following function in one sentence (max 50 words). Only output the summary, no explanation, no code.

Signature: {signature}
Source:
```
{source}
```

One-sentence summary:"""


def _collect_nodes(project_path: str) -> list[dict]:
    """从图中收集所有 FUNCTION/METHOD 节点及其源码。"""
    from mini_agent.graph_tool import GraphTool
    gt = GraphTool(project_path, enable_embeddings=False)
    gt.ensure_built()
    root = Path(project_path).resolve()
    nodes = []
    skipped = 0
    for nid, data in gt._graph.nodes(data=True):
        nt = data.get("node_type")
        tn = getattr(nt, "name", str(nt or ""))
        if tn not in ("FUNCTION", "METHOD"):
            continue
        fp = str(data.get("file_path", ""))
        sl = int(data.get("start_line", 0))
        el = int(data.get("end_line", 0))
        if not fp or sl <= 0:
            skipped += 1
            continue
        sf = root / fp
        if not sf.is_file():
            skipped += 1
            continue
        try:
            lines = sf.read_text(encoding="utf-8", errors="replace").splitlines()
            src = "\n".join(lines[max(0, sl - 1):max(sl - 1, el)])
            if not src.strip():
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        nodes.append({
            "id": str(nid), "name": data.get("name", ""),
            "type": tn, "file": fp,
            "signature": str(data.get("signature", ""))[:300],
            "source": src[:2000], "source_len": len(src),
        })
    print(f"收集到 {len(nodes)} 个函数/方法（跳过 {skipped}）")
    return nodes


def _summarize_one(node: dict, model_cfg: dict) -> tuple[str, str, float]:
    """对单个节点生成摘要，返回 (node_id, summary, elapsed_sec)。"""
    from mini_agent.model import Model, ModelConfig
    model = Model(ModelConfig(**model_cfg))
    prompt = SUMMARY_PROMPT.format(
        signature=node.get("signature", node["name"]),
        source=node["source"],
    )
    t0 = time.perf_counter()
    raw = model.generate([{"role": "user", "content": prompt}])
    elapsed = time.perf_counter() - t0
    summary = raw if isinstance(raw, str) else raw.get("content", str(raw))
    return node["id"], str(summary).strip(), elapsed


def main():
    parser = argparse.ArgumentParser(description="构建离线摘要索引")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    proj = Path(args.project_path).resolve()
    output_path = args.output or str(proj / ".accg" / "summary_index.json")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有进度
    existing: dict[str, str] = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            print(f"加载已有摘要: {len(existing)} 条")
        except Exception:
            pass

    nodes = _collect_nodes(str(proj))
    pending = [n for n in nodes if n["id"] not in existing]
    print(f"待生成: {len(pending)} / {len(nodes)}")

    if not pending:
        print("全部已完成，无需生成")
        return

    model_cfg = {
        "base_url": args.base_url, "api_key": "ollama",
        "model_name": args.model, "quiet": True,
    }

    done = 0
    total = len(pending)
    total_sec = 0.0
    start_time = time.perf_counter()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_summarize_one, n, model_cfg): n for n in pending}
            for f in as_completed(futures):
                nid, summary, elapsed = f.result()
                existing[nid] = summary
                done += 1
                total_sec += elapsed
                eta = (total_sec / done) * (total - done) if done > 0 else 0
                print(f"[{done:>4}/{total}] {elapsed:.1f}s ETA={eta:.0f}s | {summary[:60]}")
                # 每 100 条写一次
                if done % 100 == 0:
                    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        from mini_agent.model import Model, ModelConfig
        model = Model(ModelConfig(**model_cfg))
        for node in pending:
            t0 = time.perf_counter()
            prompt = SUMMARY_PROMPT.format(
                signature=node.get("signature", node["name"]),
                source=node["source"],
            )
            raw = model.generate([{"role": "user", "content": prompt}])
            elapsed = time.perf_counter() - t0
            summary = raw if isinstance(raw, str) else raw.get("content", str(raw))
            existing[node["id"]] = str(summary).strip()
            done += 1
            total_sec += elapsed
            eta = (total_sec / done) * (total - done) if done > 0 else 0
            print(f"[{done:>4}/{total}] {elapsed:.1f}s ETA={eta:.0f}s | {str(summary).strip()[:60]}")
            if done % 100 == 0:
                out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed_total = time.perf_counter() - start_time
    print(f"\n完成: {done} 条新增摘要, 总计 {len(existing)} 条, 耗时 {elapsed_total:.0f}s")
    print(f"已保存到 {out}")


if __name__ == "__main__":
    main()
