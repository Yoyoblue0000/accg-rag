# -*- coding: utf-8 -*-
"""QA 评估脚本 — 使用图增强 Agent 回答 sweqa_requests 问题"""

import argparse
import json
import os
import sys
from pathlib import Path

from mini_agent.model import Model, ModelConfig
from mini_agent.environment import Environment, EnvConfig
from mini_agent.agent import Agent, RunResult, MsgRecord
from mini_agent.graph_tool import GraphTool
from mini_agent.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_candidates,
    extract_provisional_gold,
)


def _fmt_candidates(anchors: list[dict]) -> str:
    """格式化 Top-2 候选锚点"""
    if not anchors:
        return "无"
    parts = []
    for a in anchors[:2]:
        label = a.get("label", a.get("name", "?"))
        score = a.get("score", a.get("similarity", 0))
        parts.append(f"{label}({score:.2f})")
    return " · ".join(parts)


def _status_mark(result: RunResult) -> str:
    """✓ 正常 / ⚠ 零探索 / ✗ 出错"""
    if result.error:
        return "✗"
    if result.rounds > 0 and result.explorations == 0:
        return "⚠"
    return "✓"


def _print_silent(run_index: int, total: int, qa: dict, result: RunResult):
    """静默层：每题一行"""
    q = qa["question"][:80]
    anchors = _fmt_candidates(result.anchor_candidates)
    ans_len = len(result.answer) if result.answer else 0
    mark = _status_mark(result)
    print(f"[{run_index+1:>2}/{total}] {mark} "
          f"{result.rounds}轮{result.explorations}探 | "
          f"锚:{anchors} | "
          f"{ans_len}字")


def _print_summary(results: list[tuple[int, dict, RunResult]]):
    """跑完后汇总表"""
    total = len(results)
    total_rounds = sum(r.rounds for _, _, r in results)
    total_explor = sum(r.explorations for _, _, r in results)
    zero_explore = sum(1 for _, _, r in results if r.rounds > 0 and r.explorations == 0)
    errors = sum(1 for _, _, r in results if r.error)

    print(f"\n{'='*60}")
    print(f"QA 汇总 ({total}题)")
    print(f"{'='*60}")
    print(f"{'#':>3} {'状态':<3} {'轮':>3} {'探':>3} {'字数':>5}  {'首锚点'}")
    print(f"{'─'*60}")
    for idx, qa, r in results:
        mark = _status_mark(r)
        top1 = "—"
        if r.anchor_candidates:
            top1 = r.anchor_candidates[0].get("label", r.anchor_candidates[0].get("name", "?"))
        ans_len = len(r.answer) if r.answer else 0
        print(f"{idx:>3} {mark:<3} {r.rounds:>3} {r.explorations:>3} {ans_len:>5}  {top1}")
    print(f"{'─'*60}")
    print(f"总轮: {total_rounds} · 总探: {total_explor} · 零探索: {zero_explore}题 · 错误: {errors}题")


def _print_verbose_step(m: MsgRecord) -> None:
    """-v：单条消息即时输出"""

    if m.role == "system":
        return

    elif m.role == "user":
        content = m.content.replace("[候选符号] 以下与问题语义最相关:", "\n  [候选锚点]")
        print(f"\n{'─'*60}")
        print(f"[Init] 用户问题 + 候选锚点")
        print(f"{'─'*60}")
        print(content)

    elif m.role == "assistant":
        print(f"\n{'─'*60}")
        print(f"[Step {m.step}] LLM 返回")
        print(f"{'─'*60}")
        print(m.content)

    elif m.role == "tool":
        tag = " (拦截)" if m.intercepted else ""
        print(f"\n[Step {m.step}] {m.tool_name}{tag} {json.dumps(m.tool_args or {}, ensure_ascii=False)}")
        print(m.content)


def _print_verbose2_step(m: MsgRecord) -> None:
    """-vv：单条消息 + 原始 JSON"""

    if m.role == "system":
        return

    elif m.role == "user":
        print(f"\n{'─'*60}")
        print(f"[Init] user")
        print(f"{'─'*60}")
        print(m.content)

    elif m.role == "assistant":
        print(f"\n{'─'*60}")
        print(f"[Step {m.step}] LLM 原话")
        print(f"{'─'*60}")
        print(m.content)

    elif m.role == "tool":
        tag = " (拦截)" if m.intercepted else ""
        print(f"\n{'─'*60}")
        print(f"[Step {m.step}] {m.tool_name}{tag}")
        print(f"{'─'*60}")
        if m.raw_json:
            print(f"[原始JSON]\n{m.raw_json}")
        else:
            print(m.content)


def _print_synthesis(result: RunResult, verbosity: int) -> None:
    """合成阶段输出（Agent 返回后）"""
    if not result.synthesis:
        return
    if verbosity >= 2:
        print(f"\n{'─'*60}")
        print("[合成 Prompt]")
        print(f"{'─'*60}")
        print(result.synthesis.prompt)
    print(f"\n{'─'*60}")
    print("[合成答案]")
    print(f"{'─'*60}")
    print(result.synthesis.answer)


def main():
    _qa_default = os.environ.get("QA_PATH") or os.path.expanduser("~/program/sqlfluff_qa.json")
    _proj_default = os.environ.get("PROJECT_PATH") or os.path.expanduser("~/program/sqlfluff_repo")

    parser = argparse.ArgumentParser(description="QA 评估脚本")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v 摘要 / -vv 详细（含原始 JSON）")
    parser.add_argument("--limit", type=int, default=3, help="只跑前 N 条问题")
    parser.add_argument("--id", type=int, nargs="+", help="指定跑第几条问题（从 1 开始）")
    parser.add_argument("--qa-path", default=_qa_default, help="QA 数据 JSON 路径")
    parser.add_argument("--project-path", default=_proj_default, help="待分析项目路径")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OLLAMA_MODEL",
            "qwen2.5-coder:14b-instruct",
        ),
        help="模型名",
    )
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"), help="Ollama API 地址")
    parser.add_argument("--output", default="/tmp/qa_results.json", help="结果输出路径")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只运行候选检索与确定性指标，不调用回答模型",
    )
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--embedding",
        action="store_true",
        help="启用 embedding 候选增强；默认仅运行确定性检索",
    )
    embedding_group.add_argument(
        "--no-embedding",
        action="store_true",
        help="禁用 embedding（兼容旧命令，当前已是默认行为）",
    )
    args = parser.parse_args()

    verbosity = min(args.verbose, 2)

    model = None
    if not args.retrieval_only:
        model = Model(ModelConfig(
            base_url=args.base_url,
            api_key="ollama",
            model_name=args.model,
            quiet=True,
        ))
    env = Environment(EnvConfig(cwd=args.project_path))
    graph_tool = GraphTool(
        args.project_path,
        enable_embeddings=args.embedding and not args.no_embedding,
    )

    qa_path = Path(args.qa_path)
    if not qa_path.is_file():
        print("QA_PATH 未设置或文件不存在", file=sys.stderr)
        return
    questions = json.loads(qa_path.read_text(encoding="utf-8"))

    if args.id:
        selected_questions = [
            (i - 1, questions[i - 1])
            for i in args.id
            if 1 <= i <= len(questions)
        ]
    else:
        selected_questions = list(enumerate(questions[: args.limit]))

    out_path = Path(args.output)
    out_path.write_text("[]", encoding="utf-8")
    summary_path = out_path.with_name(f"{out_path.stem}.summary.json")

    results_for_summary: list[tuple[int, dict, RunResult]] = []
    artifact_records: list[dict] = []

    if args.retrieval_only:
        print(graph_tool.ensure_built(), flush=True)

    for run_index, (source_index, qa) in enumerate(selected_questions):
        question = qa["question"]

        # 构建回调：非静默模式下每步即时输出
        def _on_step(record: MsgRecord) -> None:
            if verbosity == 0:
                return
            elif verbosity == 1:
                _print_verbose_step(record)
            else:
                _print_verbose2_step(record)

        if verbosity >= 1:
            print(f"\n{'='*60}")
            print(f"[QA {run_index+1}/{len(selected_questions)}] {question[:100]}...")
            print(f"{'='*60}")

        if args.retrieval_only:
            retrieval = graph_tool.retrieve_query_candidates(
                question,
                limit=10,
                use_embeddings=args.embedding and not args.no_embedding,
            )
            result = RunResult(
                answer="",
                anchor_candidates=[
                    candidate.to_dict()
                    for candidate in retrieval.candidates
                ],
                retrieval=retrieval,
            )
        else:
            agent = Agent(
                model,
                env,
                graph_tool=graph_tool,
                max_steps=12,
                on_step=_on_step,
                on_audit=print if verbosity >= 2 else None,
            )
            result = agent.run(question)

        if verbosity >= 1:
            _print_synthesis(result, verbosity)

        expected_answer = qa.get("answer", "")
        gold = extract_provisional_gold(expected_answer)
        retrieval_payload = (
            result.retrieval.to_dict()
            if result.retrieval is not None
            else {
                "status": "failed",
                "stages_attempted": [],
                "stages_succeeded": [],
                "diagnostics": ["Agent 未返回检索记录"],
                "candidates": [],
            }
        )
        retrieval_metrics = evaluate_candidates(
            retrieval_payload["candidates"],
            gold,
        )

        # 每答完一题立即写入
        artifact_records.append({
            "index": source_index,
            "question": question,
            "expected_answer": expected_answer,
            "expected_snippet": expected_answer[:200],
            "agent_answer": result.answer,
            "rounds": result.rounds,
            "explorations": result.explorations,
            "error": result.error,
            "retrieval": retrieval_payload,
            "provisional_gold": gold.to_dict(),
            "retrieval_metrics": retrieval_metrics,
        })
        out_path.write_text(
            json.dumps(artifact_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        results_for_summary.append((source_index, qa, result))

        if verbosity == 0:
            _print_silent(run_index, len(selected_questions), qa, result)

    summary_path.write_text(
        json.dumps(
            aggregate_retrieval_metrics(artifact_records),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if verbosity == 0:
        _print_summary(results_for_summary)
        retrieval_summary = aggregate_retrieval_metrics(artifact_records)
        print(
            "检索指标: "
            f"R@1={retrieval_summary['recall_at_1']:.3f} "
            f"R@3={retrieval_summary['recall_at_3']:.3f} "
            f"R@5={retrieval_summary['recall_at_5']:.3f} "
            f"R@10={retrieval_summary['recall_at_10']:.3f} "
            f"MRR={retrieval_summary['mrr']:.3f} "
            f"P50={retrieval_summary['latency_ms_p50']:.1f}ms "
            f"P95={retrieval_summary['latency_ms_p95']:.1f}ms "
            f"fallback={retrieval_summary['fallbacks']} "
            f"failed={retrieval_summary['retrieval_failures']}"
        )

    print(f"\n结果已保存到 {out_path}")
    print(f"检索汇总已保存到 {summary_path}")


if __name__ == "__main__":
    main()
