# -*- coding: utf-8 -*-
"""结构化证据账本 — 替换 list[str] 证据流，支持去重、分级展示、完整审计。"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from dataclasses import dataclass, field


class DisplayLevel(enum.Enum):
    """证据展示等级（渲染策略，不修改原始证据）。"""
    COMPLETE = "complete"   # 完整源码或完整结构化关系
    PREVIEW = "preview"     # 签名、文档、结构、有限源码
    SNIPPET = "snippet"     # 明确文件和行号的局部源码
    FOLD = "fold"           # 仅实体身份、类型和来源


# 源码实体的关键字段
_SOURCE_KEY_FIELDS = {
    "id", "name", "type", "file", "start_line", "end_line",
    "signature", "docstring", "source_context",
}
# 图关系的关键字段
_RELATION_KEY_FIELDS = {
    "id", "name", "file", "confidence", "strategy",
    "source_node_id", "target_node_id", "edge_type",
}


@dataclass
class EvidenceItem:
    """受控更新的证据项，包含完整 payload 和结构化元数据。"""

    evidence_id: str
    kind: str                     # "source" | "relation" | "candidate" | "structure" | "error"
    source: str                   # 工具名
    node_id: str | None = None
    file: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    edge_type: str | None = None
    source_node_id: str | None = None
    target_node_id: str | None = None
    confidence: float | None = None
    strategy: str | None = None
    payload: dict | str = field(default_factory=dict)
    complete: bool = True         # payload 是否完整（未被裁剪）
    retrieval_stage: str | None = None
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    step: int = 0
    sources: list[str] = field(default_factory=list)
    tool_origins: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.retrieval_stage is None:
            self.retrieval_stage = self.tool_args.get("action") or self.tool_name or None
        if self.source and self.source not in set(self.sources):
            self.sources.append(self.source)
        self._record_tool_origin(self.tool_name, self.tool_args, self.step)

    def _record_tool_origin(self, tool_name: str, tool_args: dict, step: int) -> None:
        if not tool_name:
            return
        origin = {
            "tool_name": tool_name,
            "tool_args": dict(tool_args),
            "step": step,
        }
        known = {
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in self.tool_origins
        }
        key = json.dumps(origin, ensure_ascii=False, sort_keys=True, default=str)
        if key not in known:
            self.tool_origins.append(origin)

    # ── 去重键 ──────────────────────────────────────────────

    def dedup_keys(self) -> list[str]:
        """返回去重键列表，同一 item 可匹配多个键。"""
        keys = []
        if self.kind == "source" and self.node_id:
            keys.append(f"source-node:{self.node_id}")
        elif self.kind == "candidate" and self.node_id:
            keys.append(f"candidate-node:{self.node_id}")
        if self.kind == "relation" and self.source_node_id and self.target_node_id and self.edge_type:
            keys.append(f"edge:{self.source_node_id}:{self.edge_type}:{self.target_node_id}")
        if self.kind == "source" and self.file and self.start_line:
            keys.append(
                f"source-range:{self.file}:{self.start_line}:"
                f"{self.end_line or self.start_line}"
            )
        return keys

    # ── 渲染 ────────────────────────────────────────────────

    def render(self, level: DisplayLevel) -> str:
        """按展示等级渲染证据文本。"""
        if self.kind == "source":
            return self._render_source(level)
        if self.kind == "relation":
            return self._render_relation(level)
        if self.kind == "candidate":
            return self._render_candidate(level)
        if self.kind == "structure":
            return self._render_structure(level)
        if self.kind == "error":
            return self._render_error()
        return self._render_fold()

    def _render_source(self, level: DisplayLevel) -> str:
        p = self.payload if isinstance(self.payload, dict) else {}
        name = p.get("name", self.node_id or "?")
        nt = p.get("type", "?")
        f = self.file or p.get("file", "?")
        sl = self.start_line or p.get("start_line", 0)
        el = self.end_line or p.get("end_line", 0)
        sig = p.get("signature", "")
        doc = p.get("docstring", "")
        sc = p.get("source_context", "")
        if not sc and isinstance(self.payload, str):
            sc = self.payload

        line_count = (el - sl + 1) if sl and el else 0
        large_class = nt == "CLASS" and line_count > 50

        if level == DisplayLevel.FOLD:
            return f"[{nt}] {name} ({f}:{sl}-{el})"

        if level == DisplayLevel.SNIPPET:
            if large_class:
                return self._render_class_overview(p, name, nt, f, sl, el, sig, doc, sc, max_source=20)
            lines = [f"[{nt}] {name} ({f}:{sl}-{el})"]
            if sig:
                lines.append(f"  签名: {sig}")
            if sc:
                sc_lines = sc.split("\n")
                if len(sc_lines) > 30:
                    sc = "\n".join(sc_lines[:30]) + "\n[...已截断]"
                lines.append(f"  源码:\n{sc}")
            return "\n".join(lines)

        if level == DisplayLevel.PREVIEW:
            if large_class:
                return self._render_class_overview(p, name, nt, f, sl, el, sig, doc, sc, max_source=30)
            lines = [f"[{nt}] {name} ({f}:{sl}-{el})"]
            if sig:
                lines.append(f"  签名: {sig}")
            if doc:
                lines.append(f"  文档: {doc.strip()}")
            if sc:
                sc_lines = sc.split("\n")
                if len(sc_lines) > 50:
                    sc = "\n".join(sc_lines[:50]) + "\n[...可升级为 COMPLETE]"
                lines.append(f"  源码:\n{sc}")
            # 关系摘要
            rel_lines = self._relation_summary(p)
            if rel_lines:
                lines.extend(rel_lines)
            return "\n".join(lines)

        # COMPLETE
        if large_class:
            return self._render_class_overview(p, name, nt, f, sl, el, sig, doc, sc, max_source=40)
        lines = [f"[{nt}] {name} ({f}:{sl}-{el})"]
        lines.append(self._provenance_line())
        if sig:
            lines.append(f"  签名: {sig}")
        if doc:
            lines.append(f"  文档: {doc.strip()}")
        if sc:
            lines.append(f"  源码:\n{sc}")
        return "\n".join(lines)

    @staticmethod
    def _render_class_overview(
        p: dict,
        name: str, nt: str, f: str,
        sl: int, el: int,
        sig: str, doc: str, sc: str,
        max_source: int = 30,
    ) -> str:
        """大类结构化视图：方法清单含摘要 + 关系，不展示完整源码。"""
        lines = [f"[{nt}] {name} ({f}:{sl}-{el})"]
        if sig:
            lines.append(f"  签名: {sig}")
        if doc:
            lines.append(f"  文档: {doc.strip()}")

        methods = p.get("methods", [])
        if methods:
            lines.append(f"\n  方法 ({len(methods)} 个):")
            for m in methods[:20]:
                m_name = m.get("name", "?")
                m_summary = m.get("summary", "")
                if m_summary:
                    lines.append(f"    {m_name} — {m_summary}")
                else:
                    lines.append(f"    {m_name}")
            if len(methods) > 20:
                lines.append(f"    ... 共 {len(methods)} 个方法，用 contextualize 查看具体方法")

        # 继承和实例化关系
        inherits = p.get("inherits", [])
        if inherits:
            for h in inherits:
                htype = h.get("type", "?")
                items = h.get("items", [])
                if not items:
                    continue
                item_names = [
                    hi.get("name", str(hi)) if isinstance(hi, dict) else str(hi)
                    for hi in items[:8]
                ]
                if len(items) > 8:
                    item_names.append(f"...共 {len(items)} 个")
                label = {"parents": "父类", "children": "子类"}.get(htype, htype)
                lines.append(f"\n  继承 - {label}: {', '.join(item_names)}")

        instantiated_by = p.get("instantiated_by", [])
        if instantiated_by:
            caller_names = []
            for c in instantiated_by[:8]:
                via = c.get("via_class", "")
                cname = c.get("name", "?")
                if via:
                    caller_names.append(f"{cname} (via {via})")
                else:
                    caller_names.append(cname)
            if len(instantiated_by) > 8:
                caller_names.append(f"...共 {len(instantiated_by)} 个")
            lines.append(f"\n  被实例化: {', '.join(caller_names)}")

        if methods:
            lines.append("\n  提示: 用 contextualize 查看具体方法的完整源码")

        if sc:
            sc_lines = sc.split("\n")
            if len(sc_lines) > max_source:
                sc = "\n".join(sc_lines[:max_source]) + "\n[...类定义头部，完整方法用 contextualize 查看]"
            lines.append(f"\n  源码 (头 {min(len(sc_lines), max_source)} 行):\n{sc}")

        return "\n".join(lines)

    def _render_relation(self, level: DisplayLevel) -> str:
        p = self.payload if isinstance(self.payload, dict) else {}
        et = self.edge_type or p.get("edge_type", "?")
        src = self.source_node_id or p.get("source_node_id", "?")
        tgt = self.target_node_id or p.get("target_node_id", "?")

        items = p.get("items", p.get("results", p.get("edges", [])))
        if isinstance(p, list):
            items = p
        if not items:
            lines = [f"[{et}] {src} → {tgt}"]
            if level == DisplayLevel.FOLD:
                return lines[0]
            if self.confidence is not None:
                lines.append(f"  confidence: {self.confidence}")
            if self.strategy:
                lines.append(f"  strategy: {self.strategy}")
            if level == DisplayLevel.COMPLETE:
                lines.append(self._provenance_line())
                payload_text = (
                    self.payload
                    if isinstance(self.payload, str)
                    else json.dumps(self.payload, ensure_ascii=False, indent=2, default=str)
                )
                if payload_text:
                    lines.append(f"  完整关系数据:\n{payload_text}")
            return "\n".join(lines)

        if level == DisplayLevel.FOLD:
            return f"[{et}] {src} → {tgt}"

        total = p.get("total", len(items)) if isinstance(p, dict) else len(items)
        lines = [f"[{et}] {src} → {tgt}（{total} 条）"]

        if level == DisplayLevel.SNIPPET:
            for item in (items if isinstance(items, list) else [])[:3]:
                lines.append(f"  - {self._edge_str(item)}")
            return "\n".join(lines)

        if level == DisplayLevel.PREVIEW:
            for item in (items if isinstance(items, list) else [])[:8]:
                lines.append(f"  - {self._edge_str(item)}")
            if total > 8:
                lines.append(f"  ...共 {total} 条，可升级为 COMPLETE")
            return "\n".join(lines)

        # COMPLETE
        lines.append(self._provenance_line())
        if self.confidence is not None:
            lines.append(f"  confidence: {self.confidence}")
        if self.strategy:
            lines.append(f"  strategy: {self.strategy}")
        for item in (items if isinstance(items, list) else []):
            lines.append(f"  - {self._edge_str(item)}")
        return "\n".join(lines)

    def _provenance_line(self) -> str:
        sources = ",".join(self.sources) if self.sources else self.source
        return f"  来源: {sources}; evidence_id={self.evidence_id}; step={self.step}"

    def _render_candidate(self, level: DisplayLevel) -> str:
        p = self.payload if isinstance(self.payload, dict) else {}
        name = p.get("name", "?")
        nt = p.get("type", "?")
        f = p.get("file", "?")
        if level == DisplayLevel.FOLD:
            return f"[候选] {name} ({nt})"
        score = p.get("score", p.get("relevance", "?"))
        sources = p.get("sources", [])
        src_str = ",".join(sources) if sources else "?"
        text = f"[候选] {name} ({nt}, {f}) score={score} sources={src_str}"
        if level == DisplayLevel.COMPLETE:
            text += "\n" + self._provenance_line()
        return text

    def _render_structure(self, level: DisplayLevel) -> str:
        if level == DisplayLevel.FOLD:
            return f"[结构] source={self.source}"
        return str(self.payload) if isinstance(self.payload, str) else json.dumps(
            self.payload, ensure_ascii=False, indent=2, default=str)

    def _render_error(self) -> str:
        return str(self.payload) if isinstance(self.payload, str) else json.dumps(
            self.payload, ensure_ascii=False, default=str)

    def _render_fold(self) -> str:
        name = "?"
        if isinstance(self.payload, dict):
            name = self.payload.get("name", "?")
        return f"[{self.kind}] {name}"

    @staticmethod
    def _edge_str(item) -> str:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            info, edge = item[0], item[1]
            name = info.get("name", "?") if isinstance(info, dict) else str(info)
            conf = edge.get("confidence", "?") if isinstance(edge, dict) else "?"
            strategy = edge.get("strategy", "") if isinstance(edge, dict) else ""
            extra = f", strategy={strategy}" if strategy else ""
            return f"{name} (conf={conf}{extra})"
        if isinstance(item, dict):
            name = item.get("name", "?")
            conf = item.get("confidence", "?")
            fid = item.get("file", item.get("id", ""))
            strategy = item.get("strategy", "")
            extra = f", strategy={strategy}" if strategy else ""
            return f"{name} ({fid}, conf={conf}{extra})"
        return str(item)

    @staticmethod
    def _relation_summary(p: dict) -> list[str]:
        """从 payload 提取关系摘要（calls/called_by/inherits/instantiated_by）。"""
        lines = []
        calls = p.get("calls", [])
        if calls:
            lines.append(f"  调用了 ({len(calls)} 个): " + ", ".join(
                f"{c.get('name','?')}(conf={c.get('confidence','?')})" for c in calls[:5]))
        called_by = p.get("called_by", [])
        if called_by:
            lines.append(f"  被调用 ({len(called_by)} 个): " + ", ".join(
                f"{c.get('name','?')}({c.get('file','?')}, conf={c.get('confidence','?')})" for c in called_by[:5]))
        inherits = p.get("inherits", [])
        if inherits:
            for h in inherits:
                htype = h.get("type", "?")
                items = h.get("items", [])
                names = [i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items[:5]]
                lines.append(f"  {htype}: {', '.join(names)}")
        inst = p.get("instantiated_by", [])
        if inst:
            lines.append(f"  实例化来源 ({len(inst)} 个): " + ", ".join(
                f"{c.get('name','?')}(conf={c.get('confidence','?')})" for c in inst[:5]))
        methods = p.get("methods", [])
        if methods:
            lines.append(f"  方法 ({len(methods)} 个): " + ", ".join(m["name"] for m in methods[:8]))
        return lines

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def from_tool_result(cls, tool_name: str, args: dict, raw_result: str, step: int) -> list[EvidenceItem]:
        """从工具执行结果创建 EvidenceItem 列表。"""
        eid_prefix = "pending"
        if tool_name == "query_graph":
            items = cls._from_graph_result(tool_name, args, raw_result, step, eid_prefix)
        elif tool_name == "read_file":
            items = cls._from_read_file(args, raw_result, step, eid_prefix)
        elif tool_name == "list_dir":
            items = cls._from_list_dir(args, raw_result, step, eid_prefix)
        else:
            items = [cls(
                evidence_id=f"{eid_prefix}-0",
                kind="error", source=tool_name,
                payload=raw_result, tool_name=tool_name, tool_args=args, step=step,
            )]
        for item in items:
            item.evidence_id = cls._stable_evidence_id(item)
        return items

    @staticmethod
    def _stable_evidence_id(item: EvidenceItem) -> str:
        keys = item.dedup_keys()
        preferred_key = None
        for prefix in (
            "source-node:",
            "edge:",
            "candidate-node:",
            "source-range:",
        ):
            preferred_key = next(
                (key for key in keys if key.startswith(prefix)),
                None,
            )
            if preferred_key:
                break
        identity = {"kind": item.kind, "key": preferred_key}
        if preferred_key is None:
            identity.update({
                "source": item.source,
                "file": item.file,
                "tool_name": item.tool_name,
                "tool_args": item.tool_args,
                "payload": item.payload,
            })
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
        return f"ev-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"

    @classmethod
    def _from_graph_result(cls, tool_name: str, args: dict, raw_result: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从图查询结果创建证据项。"""
        try:
            data = json.loads(raw_result)
        except json.JSONDecodeError:
            return [cls(
                evidence_id=f"{eid_prefix}-0",
                kind="error", source=tool_name,
                payload=raw_result, tool_name=tool_name, tool_args=args, step=step,
            )]

        if (
            isinstance(data, dict)
            and "error" in data
            and not isinstance(data.get("error"), dict)
        ):
            return [cls(
                evidence_id=f"{eid_prefix}-0",
                kind="error", source=tool_name,
                payload=data, tool_name=tool_name, tool_args=args, step=step,
            )]

        action = args.get("action", "")

        if action == "contextualize":
            return cls._from_contextualize(data, args, step, eid_prefix)
        elif action in ("transitive_callers", "transitive_callees", "call_paths"):
            return cls._from_transitive(data, args, action, step, eid_prefix)
        elif action == "class_hierarchy":
            return cls._from_class_hierarchy(data, args, step, eid_prefix)
        elif action in ("narrow_down", "extract_clues"):
            return cls._from_candidates(data, args, action, step, eid_prefix)
        elif action in ("module_structure", "module_tree"):
            return cls._from_structure(data, args, action, step, eid_prefix)

        return [cls(
            evidence_id=f"{eid_prefix}-0",
            kind="structure", source=tool_name,
            payload=data, tool_name=tool_name, tool_args=args, step=step,
        )]

    @classmethod
    def _from_contextualize(cls, data: dict, args: dict, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从 contextualize 结果拆分：每个结果项创建 source 证据。"""
        items = []
        results = data.get("results", [])
        for i, r in enumerate(results):
            nid = r.get("id", "")
            nt = r.get("type", "")
            f = r.get("file", "")
            sl = r.get("start_line", 0)
            el = r.get("end_line", 0)

            # 源码证据
            items.append(cls(
                evidence_id=f"{eid_prefix}-src-{i}",
                kind="source", source="query_graph",
                node_id=nid, file=f, start_line=sl, end_line=el,
                payload=r, complete=True,
                tool_name="query_graph", tool_args=args, step=step,
            ))

            # 调用关系证据（每个被调用者单独一条）
            calls = r.get("calls", [])
            for j, c in enumerate(calls):
                items.append(cls(
                    evidence_id=f"{eid_prefix}-calls-{i}-{j}",
                    kind="relation", source="query_graph",
                    node_id=c.get("id", ""), file=c.get("file", ""),
                    edge_type="CALLS", source_node_id=nid, target_node_id=c.get("id", ""),
                    confidence=c.get("confidence"), strategy=c.get("strategy"),
                    payload=c, complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))

            called_by = r.get("called_by", [])
            for j, c in enumerate(called_by):
                items.append(cls(
                    evidence_id=f"{eid_prefix}-calledby-{i}-{j}",
                    kind="relation", source="query_graph",
                    node_id=c.get("id", ""), file=c.get("file", ""),
                    edge_type="CALLS", source_node_id=c.get("id", ""), target_node_id=nid,
                    confidence=c.get("confidence"), strategy=c.get("strategy"),
                    payload=c, complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))

            # 继承关系
            inherits = r.get("inherits", [])
            for j, h in enumerate(inherits):
                htype = h.get("type", "?")
                h_items = h.get("items", [])
                for k, hi in enumerate(h_items):
                    hi_id = hi.get("id", hi.get("name", "")) if isinstance(hi, dict) else str(hi)
                    hi_name = hi.get("name", str(hi)) if isinstance(hi, dict) else str(hi)
                    rel_payload = {
                        "type": htype,
                        "name": hi_name,
                        "id": hi_id,
                        "hierarchy_item": hi,
                    }
                    if htype == "parents":
                        src_id, tgt_id = nid, hi_id
                    else:
                        src_id, tgt_id = hi_id, nid
                    items.append(cls(
                        evidence_id=f"{eid_prefix}-inherits-{i}-{j}-{k}",
                        kind="relation", source="query_graph",
                        node_id=hi_id,
                        edge_type="INHERITS", source_node_id=src_id, target_node_id=tgt_id,
                        payload=rel_payload, complete=True,
                        tool_name="query_graph", tool_args=args, step=step,
                    ))

            # 实例化关系
            inst = r.get("instantiated_by", [])
            for j, c in enumerate(inst):
                items.append(cls(
                    evidence_id=f"{eid_prefix}-inst-{i}-{j}",
                    kind="relation", source="query_graph",
                    node_id=c.get("id", ""), file=c.get("file", ""),
                    edge_type="INSTANTIATED_BY", source_node_id=c.get("id", ""), target_node_id=nid,
                    confidence=c.get("confidence"), strategy=c.get("strategy"),
                    payload=c, complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))

        return items

    @classmethod
    def _from_transitive(cls, data, args: dict, action: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从传递调用结果创建关系证据。"""
        items_list = data if isinstance(data, list) else data.get("items", data.get("results", []))
        symbol = args.get("symbol", "")
        edge_type = "CALLS"

        items = []
        for i, entry in enumerate(items_list):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                info, edge = entry[0], entry[1]
                nid = info.get("id", "")
                items.append(cls(
                    evidence_id=f"{eid_prefix}-rel-{i}",
                    kind="relation", source="query_graph",
                    node_id=nid, file=info.get("file", ""),
                    edge_type=edge_type,
                    source_node_id=nid if action == "transitive_callers" else symbol,
                    target_node_id=symbol if action == "transitive_callers" else nid,
                    confidence=edge.get("confidence"), strategy=edge.get("strategy"),
                    payload={"info": info, "edge": edge}, complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))
            elif isinstance(entry, dict):
                path_edges = entry.get("edges", [])
                if path_edges:
                    path_node_ids = entry.get("node_ids", [])
                    for j, edge in enumerate(path_edges):
                        if not isinstance(edge, dict):
                            continue
                        src_id = edge.get("source_node_id", "")
                        tgt_id = edge.get("target_node_id", "")
                        payload = dict(edge)
                        payload.update({
                            "path_node_ids": list(path_node_ids),
                            "path_confidence": entry.get("path_confidence"),
                            "path_depth": entry.get("depth"),
                            "endpoint_node_id": entry.get("endpoint_node_id"),
                            "anchor_node_id": entry.get("anchor_node_id"),
                            "path_record": entry,
                        })
                        items.append(cls(
                            evidence_id=f"{eid_prefix}-path-{i}-{j}",
                            kind="relation", source="query_graph",
                            edge_type="CALLS",
                            source_node_id=src_id,
                            target_node_id=tgt_id,
                            confidence=edge.get("confidence"),
                            strategy=edge.get("strategy"),
                            payload=payload,
                            complete=True,
                            tool_name="query_graph",
                            tool_args=args,
                            step=step,
                        ))
                    continue

                nid = entry.get("id", entry.get("endpoint_node_id", ""))
                anchor = entry.get("anchor_node_id", symbol)
                src_id = nid if action == "transitive_callers" else anchor
                tgt_id = anchor if action == "transitive_callers" else nid
                if action == "call_paths":
                    src_id = args.get("source", "")
                    tgt_id = args.get("target", "")
                items.append(cls(
                    evidence_id=f"{eid_prefix}-rel-{i}",
                    kind="relation", source="query_graph",
                    file=entry.get("file", ""),
                    edge_type=edge_type,
                    source_node_id=src_id,
                    target_node_id=tgt_id,
                    confidence=entry.get(
                        "path_confidence",
                        entry.get("confidence"),
                    ),
                    strategy=entry.get("strategy"),
                    payload=entry, complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))
        return items

    @classmethod
    def _from_class_hierarchy(cls, data, args: dict, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从类继承层次创建关系证据。"""
        items = []
        requested_class = args.get("class_name", "")
        for i, entry in enumerate(data if isinstance(data, list) else [data]):
            if not isinstance(entry, dict):
                continue
            class_node_id = entry.get("class_node_id", requested_class)
            htype = entry.get("type", "?")
            h_items = entry.get("items", [])
            for j, hi in enumerate(h_items):
                hi_id = hi.get("id", hi.get("name", "")) if isinstance(hi, dict) else str(hi)
                hi_name = hi.get("name", str(hi)) if isinstance(hi, dict) else str(hi)
                items.append(cls(
                    evidence_id=f"{eid_prefix}-hier-{i}-{j}",
                    kind="relation", source="query_graph",
                    node_id=hi_id,
                    edge_type="INHERITS",
                    source_node_id=class_node_id if htype == "parents" else hi_id,
                    target_node_id=hi_id if htype == "parents" else class_node_id,
                    payload={
                        "type": htype,
                        "name": hi_name,
                        "id": hi_id,
                        "hierarchy_entry": entry,
                    },
                    complete=True,
                    tool_name="query_graph", tool_args=args, step=step,
                ))
        return items

    @classmethod
    def _from_candidates(cls, data, args: dict, action: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从 narrow_down / extract_clues 创建候选证据。"""
        results = data.get("results", data.get("clues", []))
        items = []
        for i, r in enumerate(results):
            items.append(cls(
                evidence_id=f"{eid_prefix}-cand-{i}",
                kind="candidate", source="query_graph",
                node_id=r.get("id", ""), file=r.get("file", ""),
                payload=r, complete=True,
                tool_name="query_graph", tool_args=args, step=step,
            ))
        return items

    @classmethod
    def _from_structure(cls, data, args: dict, action: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从模块结构查询创建结构证据。"""
        return [cls(
            evidence_id=f"{eid_prefix}-struct-0",
            kind="structure", source="query_graph",
            payload=data, complete=True,
            tool_name="query_graph", tool_args=args, step=step,
        )]

    @classmethod
    def _from_read_file(cls, args: dict, raw_result: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从 read_file 结果创建源码证据。"""
        if raw_result.startswith("[错误]"):
            return [cls(
                evidence_id=f"{eid_prefix}-file-error",
                kind="error",
                source="read_file",
                payload=raw_result,
                tool_name="read_file",
                tool_args=args,
                step=step,
            )]
        path = args.get("path", "")
        sl = args.get("start_line", 0)
        el = args.get("end_line", 0)
        header = re.match(r"^\[.+ 行 (\d+)-(\d+) / 共 \d+ 行\]", raw_result)
        if header:
            sl = int(header.group(1))
            el = int(header.group(2))
        return [cls(
            evidence_id=f"{eid_prefix}-file-0",
            kind="source", source="read_file",
            file=path, start_line=sl or None, end_line=el or None,
            payload={
                "name": path,
                "type": "FILE",
                "file": path,
                "start_line": sl or None,
                "end_line": el or None,
                "source_context": raw_result,
            },
            complete=True,
            tool_name="read_file", tool_args=args, step=step,
        )]

    @classmethod
    def _from_list_dir(cls, args: dict, raw_result: str, step: int, eid_prefix: str) -> list[EvidenceItem]:
        """从 list_dir 结果创建结构证据。"""
        kind = "error" if raw_result.startswith("[错误]") else "structure"
        return [cls(
            evidence_id=f"{eid_prefix}-list-0",
            kind=kind, source="list_dir",
            payload=raw_result, complete=True,
            tool_name="list_dir", tool_args=args, step=step,
        )]


@dataclass
class _SelectionEntry:
    """合成选择记录。"""
    item: EvidenceItem
    level: DisplayLevel
    char_count: int
    selected: bool = True
    excluded_reason: str = ""


class EvidenceLedger:
    """结构化证据账本：去重、分级展示、合成选择、审计输出。"""

    SYNTHESIS_CHAR_BUDGET = 24000
    OBSERVATION_CHAR_BUDGET = 16000

    def __init__(self):
        self._items: list[EvidenceItem] = []
        self._dedup_index: dict[str, str] = {}  # dedup_key → evidence_id
        self._selections: list[_SelectionEntry] = []

    # ── 写入 ────────────────────────────────────────────────

    def add(self, item: EvidenceItem) -> str:
        """添加证据项，返回 merged（更新已有项）或 added（新增）。"""
        if item.source and item.source not in set(item.sources):
            item.sources.append(item.source)
        item._record_tool_origin(item.tool_name, item.tool_args, item.step)

        # 错误和控制消息直接放行
        if item.kind in ("error",):
            self._items.append(item)
            return "added"

        # 源码区间需要按重叠关系去重，不能只依赖精确字符串键。
        overlapping = self._find_overlapping_source(item)
        if overlapping is not None:
            self._merge(overlapping, item)
            self._register_keys(overlapping, item.dedup_keys())
            return "merged"

        # 去重检查
        keys = item.dedup_keys()
        for key in keys:
            existing_id = self._dedup_index.get(key)
            if existing_id is not None:
                existing = next((e for e in self._items if e.evidence_id == existing_id), None)
                if existing is not None:
                    if (
                        key.startswith("source-range:")
                        and existing.node_id
                        and item.node_id
                        and existing.node_id != item.node_id
                    ):
                        continue
                    self._merge(existing, item)
                    return "merged"

        # 新证据
        self._items.append(item)
        self._register_keys(item, keys)
        return "added"

    def _find_overlapping_source(self, item: EvidenceItem) -> EvidenceItem | None:
        if (
            item.kind != "source"
            or not item.file
            or item.start_line is None
            or item.end_line is None
        ):
            return None
        for existing in self._items:
            if (
                existing.kind != "source"
                or existing.file != item.file
                or existing.start_line is None
                or existing.end_line is None
            ):
                continue
            if (
                existing.node_id
                and item.node_id
                and existing.node_id != item.node_id
            ):
                continue
            if (
                item.start_line <= existing.end_line
                and existing.start_line <= item.end_line
            ):
                return existing
        return None

    def _register_keys(self, item: EvidenceItem, keys: list[str]) -> None:
        for key in keys:
            self._dedup_index[key] = item.evidence_id

    def _merge(self, existing: EvidenceItem, incoming: EvidenceItem) -> None:
        """将 incoming 的信息合并到 existing（保留更完整的 payload）。"""
        source_set = set(existing.sources)
        for source in incoming.sources or [incoming.source]:
            if source and source not in source_set:
                existing.sources.append(source)
                source_set.add(source)
        for origin in incoming.tool_origins:
            existing._record_tool_origin(
                origin.get("tool_name", ""),
                origin.get("tool_args", {}),
                origin.get("step", incoming.step),
            )

        if not existing.complete and incoming.complete:
            existing.payload = incoming.payload
            existing.complete = True
        elif existing.complete == incoming.complete:
            existing.payload = self._merge_payload(existing.payload, incoming.payload)

        existing.file = incoming.file or existing.file
        starts = [
            line for line in (existing.start_line, incoming.start_line)
            if line is not None
        ]
        ends = [
            line for line in (existing.end_line, incoming.end_line)
            if line is not None
        ]
        if starts:
            existing.start_line = min(starts)
        if ends:
            existing.end_line = max(ends)
        # 补充元数据
        if incoming.confidence is not None and existing.confidence is None:
            existing.confidence = incoming.confidence
        if incoming.strategy and not existing.strategy:
            existing.strategy = incoming.strategy
        # 将 incoming 独有的去重键注册到 existing，防止后续漏网
        for key in incoming.dedup_keys():
            if key not in self._dedup_index:
                self._dedup_index[key] = existing.evidence_id

    @classmethod
    def _merge_payload(cls, existing, incoming):
        if existing == incoming:
            return existing
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged = dict(existing)
            for key, value in incoming.items():
                if key not in merged or merged[key] in ("", None, [], {}):
                    merged[key] = value
                elif isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = cls._merge_lists(merged[key], value)
                elif key == "source_context" and isinstance(merged[key], str) and isinstance(value, str):
                    merged[key] = cls._merge_source_text(merged[key], value)
            return merged
        existing_text = str(existing)
        incoming_text = str(incoming)
        return incoming if len(incoming_text) > len(existing_text) else existing

    @staticmethod
    def _merge_lists(existing: list, incoming: list) -> list:
        merged = list(existing)
        known = {
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in existing
        }
        for item in incoming:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in known:
                merged.append(item)
                known.add(key)
        return merged

    @staticmethod
    def _merge_source_text(existing: str, incoming: str) -> str:
        line_pattern = re.compile(r"^\s*(\d+)\|")
        numbered: dict[int, str] = {}
        unnumbered = []
        unnumbered_seen = set()
        for text in (existing, incoming):
            for line in text.splitlines():
                match = line_pattern.match(line)
                if match:
                    numbered[int(match.group(1))] = line
                elif line not in unnumbered_seen:
                    unnumbered.append(line)
                    unnumbered_seen.add(line)
        if numbered:
            return "\n".join(unnumbered + [numbered[n] for n in sorted(numbered)])
        return incoming if len(incoming) > len(existing) else existing

    # ── 读取 ────────────────────────────────────────────────

    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    @property
    def source_items(self) -> list[EvidenceItem]:
        return [e for e in self._items if e.kind == "source"]

    @property
    def relation_items(self) -> list[EvidenceItem]:
        return [e for e in self._items if e.kind == "relation"]

    @property
    def has_synthesis_evidence(self) -> bool:
        return any(
            item.kind in {"source", "relation", "candidate"}
            for item in self._items
        )

    # ── 合成选择 ────────────────────────────────────────────

    def select_for_synthesis(self, char_budget: int | None = None) -> list[EvidenceItem]:
        """为答案合成选择证据，返回选中的证据项列表。

        选择策略：
        1. 排除 error 和 structure 类型
        2. 所有 source 和 relation 证据都以 COMPLETE 等级渲染
        3. 超出预算时先降级非 source 项，再减少项数
        4. 不在单个证据中间截断
        """
        budget = (
            self.SYNTHESIS_CHAR_BUDGET
            if char_budget is None
            else char_budget
        )
        self._selections = []

        eligible_kinds = {"source", "relation", "candidate"}
        eligible = [e for e in self._items if e.kind in eligible_kinds]
        ineligible = [e for e in self._items if e.kind not in eligible_kinds]
        if not eligible:
            self._selections.extend(
                _SelectionEntry(
                    item=item,
                    level=DisplayLevel.FOLD,
                    char_count=0,
                    selected=False,
                    excluded_reason="类型不参与答案合成",
                )
                for item in ineligible
            )
            return []

        # 优先级排序：source > relation > candidate
        kind_order = {"source": 0, "relation": 1, "candidate": 2}
        eligible.sort(key=lambda e: (kind_order.get(e.kind, 9), e.step))

        selected: list[EvidenceItem] = []
        total_chars = 0

        # 第一遍：全部 COMPLETE 渲染
        for item in eligible:
            text = item.render(DisplayLevel.COMPLETE)
            char_count = len(text)
            self._selections.append(_SelectionEntry(
                item=item, level=DisplayLevel.COMPLETE,
                char_count=char_count, selected=True,
            ))
            selected.append(item)
            total_chars += char_count
        self._selections.extend(
            _SelectionEntry(
                item=item,
                level=DisplayLevel.FOLD,
                char_count=0,
                selected=False,
                excluded_reason="类型不参与答案合成",
            )
            for item in ineligible
        )

        # 第二遍：超出预算时降级
        if total_chars > budget:
            # 对非 source 项降级到 PREVIEW
            for sel in self._selections:
                if not sel.selected or sel.item.kind == "source":
                    continue
                if sel.level == DisplayLevel.COMPLETE:
                    preview_text = sel.item.render(DisplayLevel.PREVIEW)
                    delta = sel.char_count - len(preview_text)
                    sel.level = DisplayLevel.PREVIEW
                    sel.char_count = len(preview_text)
                    total_chars -= delta
                    if total_chars <= budget:
                        break

        # 第三遍：仍超出预算，对非 source 项降级到 FOLD
        if total_chars > budget:
            for sel in self._selections:
                if not sel.selected or sel.item.kind == "source":
                    continue
                if sel.level != DisplayLevel.FOLD:
                    fold_text = sel.item.render(DisplayLevel.FOLD)
                    delta = sel.char_count - len(fold_text)
                    sel.level = DisplayLevel.FOLD
                    sel.char_count = len(fold_text)
                    total_chars -= delta
                    if total_chars <= budget:
                        break

        # 第四遍：仍超出，排除低优先级项（先 non-source 后 source）
        # 保留至少 1 个 source 项，不全部排除
        while total_chars > budget:
            excluded_one = False
            for sel in reversed(self._selections):
                if not sel.selected:
                    continue
                if sel.item.kind != "source":
                    sel.selected = False
                    sel.excluded_reason = "超出合成预算"
                    total_chars -= sel.char_count
                    excluded_one = True
                    break
            if not excluded_one:
                # 排除优先级最低的 source 项，但保留最后一个
                current_source = sum(1 for s in self._selections if s.selected and s.item.kind == "source")
                if current_source <= 1:
                    break
                for sel in reversed(self._selections):
                    if not sel.selected:
                        continue
                    if sel.item.kind == "source":
                        sel.selected = False
                        sel.excluded_reason = "超出合成预算"
                        total_chars -= sel.char_count
                        excluded_one = True
                        break
            if not excluded_one:
                break

        selected = [s.item for s in self._selections if s.selected]
        return selected

    def selection_report(self) -> str:
        """返回选择报告：哪些证据被选中，哪些被排除及原因。"""
        if not self._selections:
            return "[无选择记录]"
        lines = ["证据选择报告:"]
        total_chars = 0
        for i, sel in enumerate(self._selections):
            status = "✓" if sel.selected else "✗"
            name = sel.item.node_id or sel.item.file or sel.item.evidence_id
            lines.append(
                f"  {status} [{sel.item.kind}] {name} "
                f"(level={sel.level.value}, chars={sel.char_count})"
                + (f" -- {sel.excluded_reason}" if sel.excluded_reason else "")
            )
            if sel.selected:
                total_chars += sel.char_count
        estimated_tokens = (total_chars + 3) // 4
        lines.append(f"总字符数: {total_chars}")
        lines.append(f"估算 token 数: {estimated_tokens}")
        return "\n".join(lines)

    # ── 渲染 ────────────────────────────────────────────────

    def render_for_model(self, items: list[EvidenceItem] | None = None,
                         level: DisplayLevel = DisplayLevel.COMPLETE,
                         separator: str = "\n---\n") -> str:
        """将证据项渲染为 LLM 输入文本（统一展示等级）。"""
        if items is None:
            items = self._items
        texts = []
        for item in items:
            texts.append(item.render(level))
        return separator.join(texts)

    def render_for_observation(
        self,
        items: list[EvidenceItem],
        char_budget: int | None = None,
        separator: str = "\n---\n",
    ) -> str:
        """按工具观察预算渲染，不修改账本中的完整 payload。"""
        budget = (
            self.OBSERVATION_CHAR_BUDGET
            if char_budget is None
            else char_budget
        )
        rendered = []
        used = 0
        has_source_summary = any(item.kind == "source" for item in items)
        for item in items:
            if has_source_summary and item.kind == "relation":
                continue
            if item.kind == "source":
                levels = [
                    DisplayLevel.COMPLETE,
                    DisplayLevel.PREVIEW,
                    DisplayLevel.SNIPPET,
                    DisplayLevel.FOLD,
                ]
            elif item.kind == "candidate":
                levels = [DisplayLevel.FOLD]
            elif item.kind == "relation":
                levels = [DisplayLevel.PREVIEW, DisplayLevel.FOLD]
            else:
                levels = [DisplayLevel.COMPLETE, DisplayLevel.FOLD]

            text = ""
            for level in levels:
                candidate = item.render(level)
                if used + len(candidate) <= budget:
                    text = candidate
                    break
            if not text:
                continue
            rendered.append(text)
            used += len(text)
        return separator.join(rendered)

    def render_prefetch_evidence(
        self,
        items: list[EvidenceItem],
        char_budget: int | None = None,
        separator: str = "\n---\n",
    ) -> tuple[str, list[dict]]:
        """按锚点公平分配初始消息预算，并返回逐项展示审计。"""
        if not items:
            return "", []
        budget = (
            self.OBSERVATION_CHAR_BUDGET
            if char_budget is None
            else char_budget
        )
        per_anchor_budget = max(256, budget // len(items))
        rendered = []
        reports = []
        used = 0

        for item in items:
            payload = item.payload if isinstance(item.payload, dict) else {}
            node_type = str(payload.get("type", ""))
            line_count = 0
            if item.start_line is not None and item.end_line is not None:
                line_count = item.end_line - item.start_line + 1
            large_class = node_type == "CLASS" and line_count > 50

            if large_class:
                levels = [
                    DisplayLevel.COMPLETE,
                    DisplayLevel.PREVIEW,
                    DisplayLevel.SNIPPET,
                    DisplayLevel.FOLD,
                ]
                omitted_reason = ""
            else:
                levels = [
                    DisplayLevel.COMPLETE,
                    DisplayLevel.PREVIEW,
                    DisplayLevel.SNIPPET,
                    DisplayLevel.FOLD,
                ]
                omitted_reason = ""

            chosen_level = DisplayLevel.FOLD
            chosen_text = item.render(DisplayLevel.FOLD)
            available = min(
                per_anchor_budget,
                max(0, budget - used),
            )
            for level in levels:
                candidate = item.render(level)
                if len(candidate) <= available:
                    chosen_level = level
                    chosen_text = candidate
                    break

            if chosen_level != DisplayLevel.COMPLETE:
                if large_class and chosen_level == DisplayLevel.PREVIEW:
                    omitted_reason = "大型类超出 COMPLETE 预算，降级为 preview"
                elif large_class and chosen_level == DisplayLevel.SNIPPET:
                    omitted_reason = "大型类超出 preview 预算，降级为 snippet"
                elif large_class:
                    omitted_reason = "大型类超出 snippet 预算，降级为 fold"
                else:
                    omitted_reason = "单锚点展示预算不足"
            if len(chosen_text) > max(0, budget - used):
                chosen_text = item.render(DisplayLevel.FOLD)
                chosen_level = DisplayLevel.FOLD
                omitted_reason = "总展示预算不足"

            rendered.append(chosen_text)
            used += len(chosen_text)
            reports.append({
                "evidence_id": item.evidence_id,
                "node_id": item.node_id,
                "display_level": chosen_level.value,
                "char_count": len(chosen_text),
                "omitted_reason": omitted_reason,
            })

        return separator.join(rendered), reports

    def render_selected_for_synthesis(self, separator: str = "\n---\n") -> str:
        """按 select_for_synthesis() 决定的逐项展示等级渲染证据。

        必须在 select_for_synthesis() 之后调用，使用 _selections 中
        存储的逐项 level，确保预算控制真正生效。
        """
        texts = []
        for sel in self._selections:
            if sel.selected:
                texts.append(sel.item.render(sel.level))
        return separator.join(texts)

    def render_for_audit(self) -> str:
        """渲染完整审计文本（人类可读，不受展示预算影响）。"""
        if not self._items:
            return "[证据账本为空]"
        lines = [f"证据账本审计 — {len(self._items)} 条证据"]
        for i, item in enumerate(self._items):
            lines.append(f"\n{'─' * 60}")
            lines.append(f"[{i+1}] {item.evidence_id}")
            lines.append(f"  kind: {item.kind}")
            lines.append(f"  source: {item.source} (step {item.step})")
            if item.retrieval_stage:
                lines.append(f"  retrieval_stage: {item.retrieval_stage}")
            lines.append(f"  sources: {', '.join(item.sources)}")
            lines.append(
                "  tool_origins: "
                + json.dumps(
                    item.tool_origins,
                    ensure_ascii=False,
                    default=str,
                )
            )
            if item.node_id:
                lines.append(f"  node_id: {item.node_id}")
            if item.file:
                lines.append(f"  file: {item.file}:{item.start_line or '-'}-{item.end_line or '-'}")
            if item.edge_type:
                lines.append(f"  edge: {item.edge_type} {item.source_node_id} → {item.target_node_id}")
            if item.confidence is not None:
                lines.append(f"  confidence: {item.confidence}")
            if item.strategy:
                lines.append(f"  strategy: {item.strategy}")
            lines.append(f"  complete: {item.complete}")
            # 完整 payload
            payload_str = item.payload if isinstance(item.payload, str) else json.dumps(
                item.payload, ensure_ascii=False, indent=2, default=str)
            lines.append(f"  payload ({len(payload_str)} chars):")
            lines.append(payload_str)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)
