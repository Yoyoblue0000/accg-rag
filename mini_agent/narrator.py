# -*- coding: utf-8 -*-
"""图查询结果 → 紧凑自然语言文本。"""

import json

from accg.models import NodeId


def narrate(action: str, data: dict, args: dict) -> str:
    """将 JSON 工具结果转为紧凑自然语言，未知 action 返回原始 JSON 文本。"""
    return _NARRATORS.get(action, lambda d, a: json.dumps(d, ensure_ascii=False, indent=2))(data, args)


def narrate_contextualize(data: dict, args: dict) -> str:
    lines = []
    query = data.get("query", "")
    exact = data.get("exact", False)
    results = data.get("results", [])
    lines.append(f"[contextualize] {query} -- {'精确匹配' if exact else f'找到{len(results)}个结果'}")

    for i, r in enumerate(results[:3]):
        nid = r.get("id", "")
        nt = r.get("type", "")
        f = r.get("file", "")
        sl = r.get("start_line", 0)
        el = r.get("end_line", 0)
        sig = r.get("signature", "")
        doc = r.get("docstring", "")
        sc = r.get("source_context", "")

        lines.append(f"\n  {r.get('name','')} ({nt}, {f}:{sl}-{el})")
        if sig:
            lines.append(f"  签名: {sig}")
        if doc:
            lines.append(f"  文档: {doc.strip()}")
        if sc and sc != "[无法读取源码]":
            lines.append(f"  源码:\n{sc}")

        calls = r.get("calls", [])
        if calls:
            by_file = {}
            for c in calls:
                f = c.get("file") or (NodeId.file_of(c.get("id", "")) if NodeId.has_symbol(c.get("id", "")) else "?")
                by_file.setdefault(f, []).append(c)
            lines.append("  调用了 (CALLS -- 实现依赖):")
            for f, items in by_file.items():
                names = ", ".join(f"{c['name']}({c.get('id','')}, conf={c.get('confidence','?')})" for c in items)
                lines.append(f"    {f} -- {names}")

        called_by = r.get("called_by", [])
        if called_by:
            src_calls = [c for c in called_by if not c.get("file", "").startswith("tests/")]
            test_calls = [c for c in called_by if c.get("file", "").startswith("tests/")]
            lines.append("  被调用 (CALLED_BY -- 使用场景):")
            if src_calls:
                by_file = {}
                for c in src_calls:
                    f = c.get("file", "?")
                    by_file.setdefault(f, []).append(c)
                lines.append("    源码:")
                for f, items in by_file.items():
                    names = ", ".join(f"{c['name']}({c.get('id','')}, conf={c.get('confidence','?')})" for c in items)
                    lines.append(f"      {f} -- {names}")
            if test_calls:
                by_file = {}
                for c in test_calls:
                    f = c.get("file", "?")
                    by_file.setdefault(f, []).append(c)
                lines.append("    测试:")
                for f, items in by_file.items():
                    names = ", ".join(f"{c['name']}({c.get('id','')})" for c in items[:4])
                    if len(items) > 4:
                        names += f" 等{len(items)}个"
                    lines.append(f"      {f} -- {names}")

        methods = r.get("methods", [])
        if methods:
            names = ", ".join(m["name"] for m in methods)
            lines.append(f"  方法 (CONTAINS): {names}")

        inst = r.get("instantiated_by", [])
        if inst:
            direct = [c for c in inst if not c.get("via_class")]
            indirect = [c for c in inst if c.get("via_class")]
            lines.append("  实例化 (CALLS -- 谁创建了实例):")
            for c in direct[:8]:
                cid = c.get("id", "")
                lines.append(f"    {c['name']} ({cid}, conf={c.get('confidence','?')})")
            if indirect:
                by_via: dict[str, list] = {}
                for c in indirect:
                    via = c.get("via_class", "?")
                    by_via.setdefault(via, []).append(c)
                lines.append("    通过子类:")
                for via, items in sorted(by_via.items()):
                    parts = []
                    for c in items[:5]:
                        cid = c.get("id", "")
                        parts.append(f"{c['name']}({cid}, conf={c.get('confidence','?')})")
                    lines.append(f"      → {via}: {', '.join(parts)}")

        inherits = r.get("inherits", [])
        if inherits:
            for h in inherits:
                if h.get("type") == "parents":
                    parents = h.get("items", [])
                    if parents:
                        names = [p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in parents]
                        lines.append(f"  父类 (INHERITS): {', '.join(names)}")
                elif h.get("type") == "children":
                    children = h.get("items", [])
                    if children:
                        names = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in children]
                        lines.append(f"  子类 (INHERITS): {', '.join(names)}")

    return "\n".join(lines)


def narrate_narrow_down(data, args) -> str:
    clues = data.get("clues_used", args.get("clues", []))
    results = data.get("results", [])
    lines = [f"[narrow_down] clues={clues} → 匹配{data.get('matched', 0)}/{data.get('total_candidates', 0)}个"]
    for r in results:
        lines.append(f"  {r.get('name','')} ({r.get('type','')}, {r.get('file','')}) relevance={r.get('relevance','?')}")
    return "\n".join(lines)


def narrate_calls(data, args) -> str:
    action = args.get("action", "calls")
    direction = "调用了" if action in ("calls_from", "transitive_callees") else "被调用"
    symbol = args.get("symbol", "")
    items = data if isinstance(data, list) else data.get("items", data.get("results", []))
    lines = [f"[{action}] {symbol} -- {len(items)}个{direction}:"]

    by_file = {}
    for c in items:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            info, edge = c[0], c[1]
            f = info.get("file", "?")
            by_file.setdefault(f, []).append((info, edge))
        elif isinstance(c, dict):
            f = c.get("file", "?")
            by_file.setdefault(f, []).append(c)

    for f, entries in by_file.items():
        if isinstance(entries[0], tuple):
            names = ", ".join(f"{e[0].get('name','')}(conf={e[1].get('confidence','?')})" for e in entries[:6])
        else:
            names = ", ".join(f"{e.get('name','')}(conf={e.get('confidence','?')})" for e in entries[:6])
        lines.append(f"  {f}: {names}")

    if len(items) > 6:
        lines.append(f"  ...共{len(items)}个")
    return "\n".join(lines)


def narrate_class_hierarchy(data, args) -> str:
    parents = [d for d in data if isinstance(d, dict) and d.get("type") == "parents"]
    children = [d for d in data if isinstance(d, dict) and d.get("type") == "children"]
    lines = [f"[class_hierarchy] {args.get('class_name','')}"]
    for p in parents:
        items = p.get("items", [])
        if items:
            names = [i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items]
            lines.append(f"  父类: {', '.join(names)}")
    for c in children:
        items = c.get("items", [])
        if items:
            names = [i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items]
            lines.append(f"  子类: {', '.join(names)}")
    return "\n".join(lines)


def narrate_extract_clues(data, args) -> str:
    clues = data.get("clues", [])
    lines = [f"[extract_clues] 提取到{len(clues)}个可定位符号:"]
    for c in clues[:8]:
        lines.append(f"  {c.get('name','')} ({c.get('type','')}, {c.get('file','')})")
    if len(clues) > 8:
        lines.append(f"  ...共{len(clues)}个")
    return "\n".join(lines)


def narrate_module(data, args) -> str:
    prefix = args.get("prefix", "")
    items = data if isinstance(data, list) else data.get("items", [])
    total = data.get("total", len(items)) if isinstance(data, dict) else len(items)
    lines = [f"[module_structure] {prefix or '(根)'} -- {total}个条目"]
    for m in items[:15]:
        lines.append(f"  {m.get('name','')} ({m.get('type','')})")
    if total > 15:
        lines.append(f"  ...已截断")
    return "\n".join(lines)


_NARRATORS = {
    "contextualize": narrate_contextualize,
    "narrow_down": narrate_narrow_down,
    "transitive_callers": narrate_calls,
    "transitive_callees": narrate_calls,
    "call_paths": narrate_calls,
    "class_hierarchy": narrate_class_hierarchy,
    "extract_clues": narrate_extract_clues,
    "module_structure": narrate_module,
    "module_tree": narrate_module,
}
