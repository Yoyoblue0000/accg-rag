# -*- coding: utf-8 -*-
"""QA 评估脚本 — 使用图增强 Agent 回答 sweqa_requests 问题，含发布门禁。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from mini_agent.agent import Agent, MsgRecord, RunResult
from mini_agent.environment import EnvConfig, Environment
from mini_agent.graph_tool import GraphTool
from mini_agent.model import Model, ModelConfig
from mini_agent.multi_entity import EntityExtractor
from mini_agent.reranker import Reranker
from mini_agent.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_anchors,
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


def _status_mark(result: RunResult, *, retrieval_only: bool = False) -> str:
    """✓ 正常 / ⚠ 零探索 / ✗ 出错或未完成"""
    if retrieval_only:
        if result.retrieval is None or result.retrieval.status == "failed":
            return "✗"
        return "✓"
    if result.error:
        return "✗"
    if not result.answer or result.answer.startswith("[达到最大步数]"):
        return "✗"
    if result.rounds > 0 and result.explorations == 0:
        return "⚠"
    return "✓"


_JUDGE_PROMPT = """\
评估以下 Agent 答案与参考答案的一致性。仅输出一个 JSON 对象，无其他文字。

## 问题
{question}

## 参考答案
{expected}

## Agent 答案
{actual}

## 输出格式
{{"score": 0.0-1.0, "label": "正确|部分正确|无关|错误", "reason": "一句话理由"}}

评分标准:
- 1.0: 核心结论与参考答案一致，关键事实正确
- 0.7-0.9: 大部分正确，但缺少细节或有小错误
- 0.4-0.6: 部分正确，但缺少关键信息或有主要错误
- 0.1-0.3: 基本不相关或大部分错误
- 0.0: 完全错误或无关"""


def _evaluate_answer_quality(judge_model, question: str, expected: str, actual: str) -> dict:
    """用独立 LLM judge 评估答案质量，校验 score 范围。"""
    if not judge_model or not expected or not actual:
        return {"score": None, "label": "未评估", "reason": "缺少评估模型、参考答案或 Agent 答案"}
    if actual.startswith("[达到最大步数]") or actual.startswith("[错误]"):
        return {"score": 0.0, "label": "错误", "reason": "Agent 未完成运行"}
    prompt = _JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    try:
        raw = judge_model.generate([{"role": "user", "content": prompt}])
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            score = result.get("score", -1)
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                return {"score": None, "label": "解析失败", "reason": f"score 超出 [0,1]: {score}"}
            return {
                "score": float(score),
                "label": str(result.get("label", "未知")),
                "reason": str(result.get("reason", "")),
            }
    except Exception:
        pass
    return {"score": None, "label": "解析失败", "reason": "LLM judge 输出无法解析"}


def _print_silent(run_index: int, total: int, qa: dict, result: RunResult,
                  *, retrieval_only: bool = False):
    """静默层：每题一行"""
    qa["question"][:80]
    selected_anchors = (
        result.query_plan.get("anchors", [])
        if result.query_plan
        else result.anchor_candidates
    )
    anchors = _fmt_candidates(selected_anchors)
    ans_len = len(result.answer) if result.answer else 0
    mark = _status_mark(result, retrieval_only=retrieval_only)
    print(f"[{run_index+1:>2}/{total}] {mark} "
          f"{result.rounds}轮{result.explorations}探 | "
          f"锚:{anchors} | "
          f"{ans_len}字")


def _print_summary(results: list[tuple[int, dict, RunResult]], *,
                   retrieval_only: bool = False):
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
        mark = _status_mark(r, retrieval_only=retrieval_only)
        top1 = "—"
        selected_anchors = (
            r.query_plan.get("anchors", [])
            if r.query_plan
            else r.anchor_candidates
        )
        if selected_anchors:
            top1 = selected_anchors[0].get(
                "label",
                selected_anchors[0].get("name", "?"),
            )
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
        print("[Init] 用户问题 + 候选锚点")
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
        print("[Init] user")
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


def _print_trace(result: RunResult) -> None:
    """输出全流程详细信息"""
    print(f"\n{'━'*60}")
    print("[全流程 Trace]")
    print(f"{'━'*60}")

    # 1. 实体提取
    entities = result.entities or []
    print(f"\n[1] 实体提取 ({len(entities)} 个)")
    for i, entity in enumerate(entities, 1):
        name = entity.get("name", "?")
        query = entity.get("query", "?")
        desc = entity.get("description", "")
        hint = entity.get("type_hint", "?")
        print(f"  {i}. {name} (type={hint})")
        print(f"     query: {query}")
        if desc:
            print(f"     desc: {desc}")

    # 2. 检索阶段
    retrieval = result.retrieval
    if retrieval:
        print(f"\n[2] 检索阶段")
        print(f"  状态: {retrieval.status}")
        print(f"  尝试阶段: {', '.join(retrieval.stages_attempted)}")
        print(f"  成功阶段: {', '.join(retrieval.stages_succeeded)}")
        print(f"  候选数量: {len(retrieval.candidates)}")
        if retrieval.diagnostics:
            print(f"  诊断信息:")
            for diag in retrieval.diagnostics:
                print(f"    - {diag}")

    # 3. 候选列表
    if retrieval and retrieval.candidates:
        print(f"\n[3] 候选列表 (top 10)")
        for i, c in enumerate(retrieval.candidates[:10], 1):
            print(f"  {i}. {c.name} ({c.type}) score={c.score:.3f}")
            print(f"     id: {c.id}")
            print(f"     file: {c.file}")
            print(f"     sources: {', '.join(c.sources)}")

    # 4. 锚点选择
    query_plan = result.query_plan
    if query_plan:
        anchors = query_plan.get("anchors", [])
        rejected = query_plan.get("rejected_anchors", [])
        print(f"\n[4] 锚点选择 ({len(anchors)} 个有效, {len(rejected)} 个拒绝)")
        for i, anchor in enumerate(anchors, 1):
            print(f"  {i}. {anchor.get('name', '?')} ({anchor.get('type', '?')})")
            print(f"     id: {anchor.get('id', '?')}")
            print(f"     score: {anchor.get('score', 0):.3f}")
            print(f"     reason: {anchor.get('selection_reason', '?')}")
            print(f"     display_level: {anchor.get('display_level', '?')}")
            if anchor.get('omitted_reason'):
                print(f"     omitted_reason: {anchor.get('omitted_reason')}")

        if rejected:
            print(f"\n  拒绝的锚点:")
            for i, rej in enumerate(rejected[:5], 1):
                candidate = rej.get("candidate", {})
                reason = rej.get("reason", "?")
                print(f"  {i}. {candidate.get('name', '?')} - {reason}")

    # 5. 证据账本
    evidence = result.evidence or []
    if evidence:
        print(f"\n[5] 证据账本 ({len(evidence)} 条)")
        for i, item in enumerate(evidence, 1):
            kind = item.kind
            node_id = item.node_id or "?"
            file = item.file or "?"
            print(f"  {i}. [{kind}] {node_id}")
            if file != "?":
                print(f"     file: {file}")

    # 6. 证据充分性门控
    if result.error and "证据不足" in result.error:
        print(f"\n[6] 证据充分性门控: 未通过")
        print(f"  错误: {result.error}")
    else:
        print(f"\n[6] 证据充分性门控: 通过")

    # 7. 运行统计
    print(f"\n[7] 运行统计")
    print(f"  轮次: {result.rounds}")
    print(f"  探索: {result.explorations}")
    print(f"  答案长度: {len(result.answer) if result.answer else 0}")

    # 8. 诊断信息
    if query_plan:
        diagnostics = query_plan.get("diagnostics", [])
        if diagnostics:
            print(f"\n[8] 诊断信息")
            for diag in diagnostics:
                print(f"  - {diag}")


def _is_judge_passed(judge_result: dict, threshold: float) -> bool:
    """Judge 评估通过：无解析失败且分数 >= 阈值。"""
    score = judge_result.get("score")
    return isinstance(score, (int, float)) and score >= threshold


def _build_run_metadata(args) -> dict:
    """收集运行环境元数据，记录 git SHA、模型版本等。"""
    meta: dict = {"dirty": None, "model": args.model}
    repo_root = str(Path(__file__).resolve().parent.parent)
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        if r.returncode == 0:
            meta["git_sha"] = r.stdout.strip()

        dirty = False
        for check_cmd, desc in [
            (["git", "diff", "--quiet"], "unstaged"),
            (["git", "diff", "--cached", "--quiet"], "staged"),
        ]:
            result = subprocess.run(check_cmd, capture_output=True, cwd=repo_root)
            if result.returncode != 0:
                meta.setdefault("dirty_details", []).append(desc)
                dirty = True
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=repo_root,
        )
        if untracked.stdout.strip():
            meta.setdefault("dirty_details", []).append("untracked")
            dirty = True
        meta["dirty"] = dirty
    except Exception:
        meta["git_sha"] = "unknown"

    meta["embedding_model"] = getattr(args, "embedding_model", "")
    try:
        from importlib.metadata import version
        meta["accg_version"] = version("accg")
    except Exception:
        meta["accg_version"] = "unknown"

    qa_path = Path(args.qa_path)
    if qa_path.is_file():
        meta["dataset_sha256"] = hashlib.sha256(
            qa_path.read_bytes()
        ).hexdigest()
    else:
        meta["dataset_sha256"] = ""

    meta["config"] = {
        "max_steps": 12,
        "embedding": args.embedding and not args.no_embedding,
        "reranker_model": args.reranker_model or None,
        "judge_model": args.judge_model or None,
        "judge_threshold": args.judge_threshold,
    }
    return meta


def main():
    _qa_default = os.environ.get("QA_PATH") or os.path.expanduser("~/program/sqlfluff_qa.json")
    _proj_default = os.environ.get("PROJECT_PATH") or os.path.expanduser("~/program/sqlfluff_repo")

    parser = argparse.ArgumentParser(description="QA 评估脚本，含发布门禁")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v 摘要 / -vv 详细（含原始 JSON）")
    parser.add_argument("--trace", action="store_true",
                        help="输出全流程详细信息（实体提取、检索、锚点选择、证据、门控）")
    parser.add_argument("--limit", type=int, default=3, help="只跑前 N 条问题")
    parser.add_argument("--id", type=int, nargs="+", help="指定跑第几条问题（从 1 开始）")
    parser.add_argument("--qa-path", default=_qa_default, help="QA 数据 JSON 路径")
    parser.add_argument("--project-path", default=_proj_default, help="待分析项目路径")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b-instruct"),
        help="模型名",
    )
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
                        help="Ollama API 地址")
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
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("RERANKER_MODEL", ""),
        help="重排小模型名（如 qwen2.5:7b），留空则不启用重排",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large"),
        help="embedding 模型名（默认 mxbai-embed-large）",
    )
    # ── 门禁参数 ──
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", ""),
        help="LLM-as-judge 专用模型（与回答模型分离）。留空则不启用 judge",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=float(os.environ.get("JUDGE_THRESHOLD", "0.7")),
        help="Judge 通过的最低分数阈值，默认 0.7",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=os.environ.get("QA_FAIL_ON_ERROR", "") in {"1", "true", "yes"},
        help="任何错误/空答案/judge 失败均触发非零退出码",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有输出文件恢复，跳过已完成的题目",
    )
    parser.add_argument(
        "--prohibit-dirty",
        action="store_true",
        help="工作区有未提交修改时拒绝运行",
    )
    args = parser.parse_args()

    # ── 运行元数据与 dirty 检查 ──
    run_meta = _build_run_metadata(args)
    if args.prohibit_dirty and run_meta.get("dirty"):
        print("[错误] 工作区存在未提交修改，禁止 --prohibit-dirty 模式运行", file=sys.stderr)
        sys.exit(3)

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
        embedding_model=args.embedding_model,
    )

    reranker = None
    if args.reranker_model:
        reranker_model = Model(ModelConfig(
            base_url=args.base_url,
            api_key="ollama",
            model_name=args.reranker_model,
            quiet=True,
        ))
        reranker = Reranker(reranker_model, project_root=args.project_path)

    # 独立 Judge 模型
    judge_model = None
    if args.judge_model:
        judge_model = Model(ModelConfig(
            base_url=args.base_url,
            api_key="ollama",
            model_name=args.judge_model,
            quiet=True,
        ))

    qa_path = Path(args.qa_path)
    if not qa_path.is_file():
        print(f"QA 文件不存在: {qa_path}", file=sys.stderr)
        sys.exit(2)
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
    summary_path = out_path.with_name(f"{out_path.stem}.summary.json")

    # ── 恢复模式 ──
    artifact_records: list[dict] = []
    completed_indices: set[int] = set()
    if args.resume and out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for record in existing:
                if not isinstance(record, dict):
                    continue
                artifact_records.append(record)
                # 仅将成功完成的记录标记为已完成，失败的允许重新运行
                error = record.get("error")
                answer = record.get("agent_answer")
                retrieval = record.get("retrieval") or {}
                if (
                    not error
                    and answer is not None
                    and retrieval.get("status") != "failed"
                ):
                    completed_indices.add(record.get("index", -1))
        except json.JSONDecodeError:
            pass
    else:
        out_path.write_text("[]", encoding="utf-8")

    results_for_summary: list[tuple[int, dict, RunResult]] = []
    current_run_indices: set[int] = set()

    if args.retrieval_only:
        print(graph_tool.ensure_built(), flush=True)

    # ── 门禁统计 ──
    exit_code = 0
    completed_count = 0
    failed_count = 0

    for run_index, (source_index, qa) in enumerate(selected_questions):
        question = qa["question"]

        # 跳过已完成的题目
        if args.resume and source_index in completed_indices:
            if verbosity >= 1:
                print(f"\n[QA {run_index+1}/{len(selected_questions)}] 跳过已完成 #{source_index}: {question[:60]}...")
            continue

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

        # ── 单题 try/except 保护 ──
        try:
            result = RunResult(answer="", error="未执行")
            if args.retrieval_only:
                retrieval = graph_tool.retrieve_query_candidates(
                    question,
                    limit=10,
                    use_embeddings=args.embedding and not args.no_embedding,
                )
                candidate_dicts = [
                    candidate.to_dict()
                    for candidate in retrieval.candidates
                ]
                anchors = graph_tool.select_query_anchors(
                    question,
                    candidate_dicts,
                    max_anchors=3,
                )
                result = RunResult(
                    answer="",
                    anchor_candidates=candidate_dicts,
                    retrieval=retrieval,
                    query_plan={
                        "query": question,
                        "candidates": candidate_dicts,
                        "anchors": anchors,
                        "rejected_anchors": [],
                        "prefetch_evidence_ids": [],
                        "relation_expansions": [],
                        "diagnostics": list(retrieval.diagnostics),
                    },
                )
            else:
                entity_extractor = EntityExtractor(model)
                agent = Agent(
                    model,
                    env,
                    graph_tool=graph_tool,
                    max_steps=12,
                    on_step=_on_step,
                    on_audit=print if verbosity >= 2 else None,
                    reranker=reranker,
                    entity_extractor=entity_extractor,
                )
                result = agent.run(question)
        except Exception as exc:
            result = RunResult(answer="", error=f"未捕获异常: {exc}")
            if verbosity >= 1:
                print(f"[错误] QA 题目 {source_index} 异常: {exc}")

        if verbosity >= 1:
            _print_synthesis(result, verbosity)

        # 输出全流程 trace
        if args.trace:
            _print_trace(result)

        expected_answer = qa.get("answer", "")
        gold = extract_provisional_gold(expected_answer)

        # LLM Judge 答案质量评估（独立模型）
        judge_result = _evaluate_answer_quality(judge_model, question, expected_answer, result.answer)

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
        query_plan_payload = result.query_plan or {
            "query": question,
            "candidates": retrieval_payload["candidates"],
            "anchors": [],
            "rejected_anchors": [],
            "prefetch_evidence_ids": [],
            "relation_expansions": [],
            "diagnostics": ["Agent 未返回查询计划"],
        }
        anchor_metrics = evaluate_anchors(
            query_plan_payload["anchors"],
            gold,
            question,
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
            "query_plan": query_plan_payload,
            "anchor_metrics": anchor_metrics,
            "prefetch_evidence_ids": query_plan_payload["prefetch_evidence_ids"],
            "first_request_display_levels": [
                {
                    "id": anchor.get("id", ""),
                    "display_level": anchor.get("display_level", ""),
                    "omitted_reason": anchor.get("omitted_reason", ""),
                }
                for anchor in query_plan_payload["anchors"]
            ],
            "rerank": query_plan_payload.get("rerank"),
            "answer_judge": judge_result,
        })
        completed_indices.add(source_index)
        current_run_indices.add(source_index)
        out_path.write_text(
            json.dumps(artifact_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        results_for_summary.append((source_index, qa, result))

        if verbosity == 0:
            _print_silent(run_index, len(selected_questions), qa, result,
                          retrieval_only=args.retrieval_only)

        # ── 单题失败计数 ──
        is_failed = (
            bool(result.error)
            or (not args.retrieval_only and (not result.answer or result.answer.startswith("[达到最大步数]")))
            or (args.retrieval_only and (result.retrieval is None or result.retrieval.status == "failed"))
        )
        if is_failed:
            failed_count += 1
        else:
            completed_count += 1

    # ── 门禁汇总 ──
    total_run = completed_count + failed_count
    current_records = [
        r for r in artifact_records
        if r.get("index", -1) in current_run_indices
    ]
    judge_scores = []
    judge_evaluable = 0
    judge_passed = 0
    for record in current_records:
        jr = record.get("answer_judge", {})
        score = jr.get("score")
        if isinstance(score, (int, float)):
            judge_scores.append(score)
            judge_evaluable += 1
            if score >= args.judge_threshold:
                judge_passed += 1

    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    judge_pass_rate = judge_passed / judge_evaluable if judge_evaluable > 0 else 0.0

    # ── 保存汇总 ──
    retrieval_summary = aggregate_retrieval_metrics(current_records)
    summary_data = {
        "run_metadata": run_meta,
        "completed": completed_count,
        "failed": failed_count,
        "total": total_run,
        "file_total_artifacts": len(artifact_records),
        "judge_evaluable": judge_evaluable,
        "judge_mean": round(judge_mean, 3),
        "judge_pass_rate": round(judge_pass_rate, 3),
        "judge_threshold": args.judge_threshold,
        "retrieval": retrieval_summary,
    }
    summary_path.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbosity == 0:
        _print_summary(results_for_summary, retrieval_only=args.retrieval_only)
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
            f"failed={retrieval_summary['retrieval_failures']} "
            f"anchor_P={retrieval_summary['anchor_precision']:.3f} "
            f"anchor_R={retrieval_summary['anchor_recall']:.3f} "
            f"type_cov={retrieval_summary['type_coverage']:.3f}"
        )

    # ── 门禁判定 ──
    print(f"\n{'='*60}")
    print("发布门禁")
    print(f"{'='*60}")
    print(f"完成: {completed_count} 题  失败: {failed_count} 题  (共 {total_run} 题)")
    if judge_evaluable > 0:
        print(f"Judge 可评估: {judge_evaluable} 题  "
              f"平均分: {judge_mean:.3f}  "
              f"通过率 (≥{args.judge_threshold}): {judge_pass_rate:.1%}")

    gate_reasons = []
    if args.fail_on_error:
        if failed_count > 0:
            gate_reasons.append(f"{failed_count} 题失败")
        if not args.retrieval_only and judge_model:
            if judge_evaluable > 0:
                if judge_passed < judge_evaluable:
                    gate_reasons.append(
                        f"Judge 通过率 {judge_pass_rate:.1%} < 100% "
                        f"({judge_evaluable - judge_passed} 题低于阈值 {args.judge_threshold})"
                    )
                parse_failures = sum(
                    1 for record in current_records
                    if record.get("answer_judge", {}).get("label") == "解析失败"
                )
                if parse_failures > 0:
                    gate_reasons.append(f"{parse_failures} 题 Judge 解析失败")
            elif len(current_records) > 0:
                gate_reasons.append(
                    f"Judge 全部 {len(current_records)} 题解析失败，无法评估答案质量"
                )

    if gate_reasons:
        print(f"\n[门禁未通过] {'; '.join(gate_reasons)}")
        exit_code = 1
    else:
        print("\n[门禁通过]")

    print(f"\n结果已保存到 {out_path}")
    print(f"汇总已保存到 {summary_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
