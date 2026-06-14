# -*- coding: utf-8 -*-
"""锚点选择深度分析 — 逐题对比候选→锚点→gold，诊断选择错误根因。"""

import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mini_agent.graph_tool import GraphTool
from mini_agent.retrieval_metrics import (
    GoldLocations,
    _matched_gold,
    _normalize_path,
    _normalize_symbol,
    extract_provisional_gold,
)


@dataclass
class AnchorDiagnosis:
    """单题的锚点诊断报告。"""

    index: int
    question: str
    expected_snippet: str  # 参考答案前 200 字符
    gold: GoldLocations
    candidates: list[dict]
    selected_anchors: list[dict]
    rejected_anchors: list[dict]

    # 逐候选分析
    candidate_details: list[dict] = field(default_factory=list)

    # 错失的 gold 实体
    missed_gold_paths: list[str] = field(default_factory=list)
    missed_gold_symbols: list[str] = field(default_factory=list)

    # 根因分类
    root_cause: str = ""  # "检索层遗漏" | "选择层误排" | "正确"
    diagnosis: str = ""


def _candidate_full_id(c: dict) -> str:
    return c.get("id", "?")


def _candidate_label(c: dict) -> str:
    name = c.get("name", "?")
    ftype = c.get("type", "?")
    fid = c.get("id", "?")
    score = c.get("score", 0)
    sources = ",".join(c.get("sources", []))
    return f"{name}({ftype}) score={score:.1f} src={sources}  {fid}"


def _find_gold_in_candidates(
    gold_item: str, candidates: list[dict], item_type: str
) -> list[int]:
    """在候选列表中查找匹配 gold 项的排名位置（1-based）。"""
    ranks = []
    gold_normalized = _normalize_symbol(gold_item) if item_type == "symbol" else _normalize_path(gold_item)
    for rank, c in enumerate(candidates, 1):
        if item_type == "symbol":
            c_id = _normalize_symbol(str(c.get("id", "")))
            c_name = _normalize_symbol(str(c.get("name", "")))
            terminal = re.split(r"::|\.", gold_normalized)[-1]
            if c_name == terminal or c_id.endswith("::" + gold_normalized) or c_id.endswith("::" + terminal):
                ranks.append(rank)
        else:
            c_file = _normalize_path(str(c.get("file", "")))
            if c_file == gold_normalized or c_file.endswith("/" + gold_normalized):
                ranks.append(rank)
    return ranks


def diagnose_single(
    qa_index: int,
    question: str,
    expected_answer: str,
    candidates: list[dict],
    selected_anchors: list[dict],
    rejected_anchors: list[dict],
) -> AnchorDiagnosis:
    """对单题进行完整的锚点选择诊断。"""
    gold = extract_provisional_gold(expected_answer)
    diag = AnchorDiagnosis(
        index=qa_index,
        question=question[:200],
        expected_snippet=expected_answer[:200],
        gold=gold,
        candidates=list(candidates),
        selected_anchors=list(selected_anchors),
        rejected_anchors=list(rejected_anchors),
    )

    anchor_ids = {a.get("id", "") for a in selected_anchors}

    # ── 逐候选详情 ──
    for rank, c in enumerate(candidates, 1):
        cid = c.get("id", "")
        detail = {
            "rank": rank,
            "id": cid,
            "name": c.get("name", ""),
            "type": c.get("type", ""),
            "file": c.get("file", ""),
            "score": c.get("score", 0),
            "sources": c.get("sources", []),
            "matched_terms": c.get("matched_terms", []),
            "matched_fields": c.get("matched_fields", []),
            "is_anchor": cid in anchor_ids,
            "matches_gold": list(_matched_gold(c, gold)),
        }
        diag.candidate_details.append(detail)

    # ── 错失的 gold ──
    selected_and_candidates = selected_anchors + candidates
    for path in gold.paths:
        found = _find_gold_in_candidates(path, selected_and_candidates, "path")
        if not found:
            diag.missed_gold_paths.append(path)

    for sym in gold.symbols:
        found = _find_gold_in_candidates(sym, selected_and_candidates, "symbol")
        if not found:
            diag.missed_gold_symbols.append(sym)

    # ── 根因分类 ──
    anchor_matches = sum(
        1 for a in selected_anchors if _matched_gold(a, gold)
    )
    total_gold = gold.count

    if total_gold == 0:
        diag.root_cause = "无_gold"
        diag.diagnosis = "参考答案无法提取临时 gold（可能太简略或纯自然语言描述）"
    elif anchor_matches == len(selected_anchors) and len(diag.missed_gold_paths) == 0 and len(diag.missed_gold_symbols) == 0:
        diag.root_cause = "正确"
        diag.diagnosis = f"所有 {len(selected_anchors)} 个锚点命中 gold，未遗漏"
    elif anchor_matches > 0:
        # 部分命中
        missed_in_candidates = 0
        wrong_selection = 0
        for sym in diag.missed_gold_symbols:
            ranks = _find_gold_in_candidates(sym, candidates, "symbol")
            if ranks:
                wrong_selection += 1
            else:
                missed_in_candidates += 1
        parts = []
        if wrong_selection:
            parts.append(f"{wrong_selection} 个 gold 实体在候选列表中但未被选为锚点（选择层问题）")
        if missed_in_candidates:
            parts.append(f"{missed_in_candidates} 个 gold 实体未进入候选列表（检索层问题）")
        diag.root_cause = "部分命中"
        diag.diagnosis = "; ".join(parts)
    else:
        # 完全未命中
        in_candidates = 0
        not_in_candidates = 0
        for sym in diag.missed_gold_symbols:
            ranks = _find_gold_in_candidates(sym, candidates, "symbol")
            if ranks:
                in_candidates += 1
            else:
                not_in_candidates += 1
        for path in diag.missed_gold_paths:
            ranks = _find_gold_in_candidates(path, candidates, "path")
            if ranks:
                in_candidates += 1
            else:
                not_in_candidates += 1

        if in_candidates > 0 and not_in_candidates == 0:
            diag.root_cause = "选择层误排"
            diag.diagnosis = f"所有 {total_gold} 个 gold 实体均在候选列表中（共 {len(candidates)} 候选），但未被选为锚点"
        elif not_in_candidates > 0 and in_candidates == 0:
            diag.root_cause = "检索层遗漏"
            diag.diagnosis = f"所有 {total_gold} 个 gold 实体均未进入候选列表"
        else:
            diag.root_cause = "检索+选择混合"
            diag.diagnosis = (
                f"{in_candidates} 个在候选但未选中（选择层），"
                f"{not_in_candidates} 个未进入候选（检索层）"
            )

    return diag


def print_diagnosis(diag: AnchorDiagnosis, verbosity: int = 0) -> None:
    """打印单题诊断结果。"""
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"[QA #{diag.index}] {diag.root_cause}")
    print(f"{sep}")
    print(f"问题: {diag.question}...")

    if diag.gold.count > 0:
        print("\n── Gold（从参考答案提取）──")
        if diag.gold.paths:
            print(f"  路径: {diag.gold.paths}")
        if diag.gold.symbols:
            print(f"  符号: {diag.gold.symbols}")
    else:
        print("\n── Gold ── 无（参考答案无法提取结构化实体）")

    print(f"\n── 选中锚点 ({len(diag.selected_anchors)}) ──")
    for a in diag.selected_anchors:
        matches = _matched_gold(a, diag.gold)
        hit_mark = " ✓命中" if matches else " ✗未命中"
        reason = a.get("selection_reason", "?")
        print(f"  [{reason}]{hit_mark}")
        print(f"  {_candidate_label(a)}")

    if diag.rejected_anchors:
        print(f"\n── 被拒锚点 ({len(diag.rejected_anchors)}) ──")
        for r in diag.rejected_anchors:
            c = r.get("candidate", {})
            reason = r.get("reason", "?")
            msg = r.get("message", "")
            print(f"  [{reason}] {msg}")
            if c:
                print(f"  {_candidate_label(c)}")

    if diag.missed_gold_paths or diag.missed_gold_symbols:
        print("\n── 错失的 Gold 实体 ──")
        for p in diag.missed_gold_paths:
            ranks = _find_gold_in_candidates(p, diag.candidates, "path")
            print(f"  路径: {p}  → 候选排名: {ranks if ranks else '未进入候选列表'}")
        for s in diag.missed_gold_symbols:
            ranks = _find_gold_in_candidates(s, diag.candidates, "symbol")
            print(f"  符号: {s}  → 候选排名: {ranks if ranks else '未进入候选列表'}")

    print(f"\n── 诊断: {diag.root_cause} ──")
    print(f"  {diag.diagnosis}")

    if verbosity >= 1:
        # 完整候选列表
        print(f"\n── 完整候选列表 ({len(diag.candidates)}) ──")
        for detail in diag.candidate_details:
            anchor_tag = " [锚点]" if detail["is_anchor"] else ""
            gold_tag = f" [gold:{detail['matches_gold']}]" if detail["matches_gold"] else ""
            print(
                f"  #{detail['rank']:>2}{anchor_tag}{gold_tag} "
                f"{detail['name']}({detail['type']}) "
                f"score={detail['score']:.1f} "
                f"src={','.join(detail['sources'])} "
                f"terms={detail['matched_terms']} "
                f"fields={detail['matched_fields']}"
            )


def print_summary(diagnoses: list[AnchorDiagnosis]) -> dict:
    """打印汇总统计并返回聚合字典。"""
    total = len(diagnoses)
    evaluable = [d for d in diagnoses if d.gold.count > 0]

    cause_counter = Counter(d.root_cause for d in diagnoses)
    print(f"\n{'='*70}")
    print(f"锚点选择分析汇总（{total} 题）")
    print(f"{'='*70}")

    print("\n── 根因分布 ──")
    for cause, count in cause_counter.most_common():
        pct = count / total * 100
        print(f"  {cause}: {count} ({pct:.0f}%)")

    if evaluable:
        # 锚点精确率/召回率
        precisions = []
        recalls = []
        for d in evaluable:
            matched = sum(1 for a in d.selected_anchors if _matched_gold(a, d.gold))
            precisions.append(matched / len(d.selected_anchors) if d.selected_anchors else 0)
            all_matched = set()
            for a in d.selected_anchors:
                all_matched.update(_matched_gold(a, d.gold))
            recalls.append(len(all_matched) / d.gold.count if d.gold.count else 0)

        avg_p = statistics.fmean(precisions) if precisions else 0
        avg_r = statistics.fmean(recalls) if recalls else 0
        print(f"\n── 锚点指标（{len(evaluable)} 道可评估题）──")
        print(f"  平均精确率: {avg_p:.3f}")
        print(f"  平均召回率: {avg_r:.3f}")

        # Gold 覆盖率细粒度
        gold_in_candidates = 0
        gold_total = 0
        gold_in_anchors = 0
        for d in evaluable:
            gold_total += d.gold.count
            for sym in d.gold.symbols:
                if _find_gold_in_candidates(sym, d.candidates, "symbol"):
                    gold_in_candidates += 1
                if _find_gold_in_candidates(sym, d.selected_anchors, "symbol"):
                    gold_in_anchors += 1
            for path in d.gold.paths:
                if _find_gold_in_candidates(path, d.candidates, "path"):
                    gold_in_candidates += 1
                if _find_gold_in_candidates(path, d.selected_anchors, "path"):
                    gold_in_anchors += 1

        print(f"  候选覆盖率: {gold_in_candidates}/{gold_total} ({gold_in_candidates/gold_total:.1%})")
        print(f"  锚点覆盖率: {gold_in_anchors}/{gold_total} ({gold_in_anchors/gold_total:.1%})")

    # 检索层级联贡献
    stage_contributions = Counter()
    for d in diagnoses:
        for a in d.selected_anchors:
            for src in a.get("sources", []):
                stage_contributions[src] += 1
    print("\n── 锚点来源阶段分布 ──")
    for stage in ["exact_id", "exact_symbol", "lexical", "embedding", "fuzzy"]:
        count = stage_contributions.get(stage, 0)
        pct = count / max(sum(stage_contributions.values()), 1) * 100
        print(f"  {stage}: {count} ({pct:.0f}%)")

    return {
        "total": total,
        "evaluable": len(evaluable),
        "cause_distribution": dict(cause_counter),
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "candidate_gold_coverage": gold_in_candidates / max(gold_total, 1),
        "anchor_gold_coverage": gold_in_anchors / max(gold_total, 1),
        "stage_contributions": dict(stage_contributions),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="锚点选择深度分析")
    parser.add_argument(
        "--project-path",
        default="~/program/sqlfluff_repo",
        help="项目路径",
    )
    parser.add_argument(
        "--qa-path",
        default="~/program/sqlfluff_qa.json",
        help="QA JSON 路径",
    )
    parser.add_argument("--limit", type=int, default=0, help="分析前 N 题（0=全部）")
    parser.add_argument("--embedding", action="store_true", help="启用 embedding 语义检索")
    parser.add_argument("--id", type=int, nargs="+", help="只分析指定题号")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v 显示完整候选列表")
    parser.add_argument(
        "--output",
        default="/tmp/anchor_analysis.json",
        help="分析结果输出路径",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa_path).expanduser()
    if not qa_path.is_file():
        print(f"QA 文件不存在: {qa_path}", file=sys.stderr)
        sys.exit(1)

    questions = json.loads(qa_path.read_text(encoding="utf-8"))
    total_available = len(questions)

    if args.id:
        indices = [i - 1 for i in args.id if 1 <= i <= total_available]
    elif args.limit > 0:
        indices = list(range(min(args.limit, total_available)))
    else:
        indices = list(range(total_available))

    proj_path = Path(args.project_path).expanduser().resolve()
    print(f"项目: {proj_path}")
    print(f"QA: {qa_path} ({len(indices)}/{total_available} 题)")
    print("初始化图工具...")

    gt = GraphTool(str(proj_path), enable_embeddings=args.embedding)
    print(gt.ensure_built())

    diagnoses = []

    for idx in indices:
        qa = questions[idx]
        question = qa["question"]
        expected = qa.get("answer", "")

        # 检索
        retrieval = gt.retrieve_query_candidates(
            question, limit=24, use_embeddings=args.embedding
        )
        candidate_dicts = [c.to_dict() for c in retrieval.candidates]

        # 锚点选择
        anchors = gt.select_query_anchors(question, candidate_dicts, max_anchors=3)

        # 搜集被拒锚点（验证失败的 + 超出上限的）
        rejected = []
        accepted_ids = {a.get("id") for a in anchors}

        # 验证通过的锚点才算有效
        validated_anchors = []
        for a in anchors:
            validation = gt.validate_query_anchor(a)
            if validation.get("valid"):
                validated_anchors.append(a)
            else:
                rejected.append({
                    "candidate": a,
                    "reason": validation.get("reason", "invalid"),
                    "message": validation.get("message", ""),
                    "suggestions": validation.get("suggestions", []),
                })

        for c in candidate_dicts:
            cid = c.get("id")
            if cid in accepted_ids:
                continue
            # 检查是否已在 rejected 中
            already_rejected = any(
                r.get("candidate", {}).get("id") == cid for r in rejected
            )
            if already_rejected:
                continue
            rejected.append({
                "candidate": c,
                "reason": "max_anchor_limit",
                "message": "候选排名靠后，未达到锚点选择阈值",
                "suggestions": [],
            })

        diag = diagnose_single(
            qa_index=idx + 1,
            question=question,
            expected_answer=expected,
            candidates=candidate_dicts,
            selected_anchors=validated_anchors,
            rejected_anchors=rejected,
        )
        diagnoses.append(diag)
        print_diagnosis(diag, verbosity=args.verbose)

    summary = print_summary(diagnoses)

    # 保存完整分析结果
    output_path = Path(args.output).expanduser()
    output_data = {
        "summary": summary,
        "per_question": [
            {
                "index": d.index,
                "question": d.question,
                "expected_snippet": d.expected_snippet,
                "gold_paths": d.gold.paths,
                "gold_symbols": d.gold.symbols,
                "root_cause": d.root_cause,
                "diagnosis": d.diagnosis,
                "anchor_count": len(d.selected_anchors),
                "anchors": [
                    {
                        "id": a.get("id", ""),
                        "name": a.get("name", ""),
                        "type": a.get("type", ""),
                        "file": a.get("file", ""),
                        "score": a.get("score", 0),
                        "sources": a.get("sources", []),
                        "selection_reason": a.get("selection_reason", ""),
                        "matches_gold": list(_matched_gold(a, d.gold)),
                    }
                    for a in d.selected_anchors
                ],
                "missed_gold_paths": d.missed_gold_paths,
                "missed_gold_symbols": d.missed_gold_symbols,
                "candidates_with_gold": [
                    detail for detail in d.candidate_details
                    if detail["matches_gold"]
                ],
            }
            for d in diagnoses
        ],
    }
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
