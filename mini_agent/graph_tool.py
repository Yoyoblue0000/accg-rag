# -*- coding: utf-8 -*-
"""图查询工具 - 将 ACCG GraphQuery 暴露为 Agent 工具"""
import hashlib
import json
import logging
import pickle
import re
import time
from pathlib import Path

from accg.models import NodeId, EdgeType
from .retrieval import (
    Candidate,
    CandidateRetriever,
    RetrievalResult,
    build_entries,
    select_query_anchors as _select_query_anchors,
    tokenize,
)

logger = logging.getLogger("mini_agent.graph_tool")

_TRIM_MASK = "[...已截断]"


def _split_camel(name: str) -> str:
    """CamelCase → 自然语言，用于 embedding 输入"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return s.lower().strip("_")


def _build_embed_text(node: dict) -> str:
    """构建 embedding 输入文本"""
    name = node.get("name", "")
    parts = [_split_camel(name)]
    sig = node.get("signature", "")
    if sig and len(sig) < 200:
        parts.append(_split_camel(sig))
    doc = node.get("docstring", "")
    if doc and len(doc) < 300:
        parts.append(doc)
    path = node.get("file", "")
    if path:
        parts.append(" ".join(_split_camel(p) for p in path.replace("/", " ").replace("_", " ").split()))
    parent = node.get("parent", "")
    if parent:
        parts.append(_split_camel(parent))
    decorators = node.get("decorators", [])
    if decorators:
        parts.append(" ".join(_split_camel(str(item)) for item in decorators))
    return " ".join(parts)


class EmbeddingRanker:
    """基于 Ollama embedding 的语义候选排序器，带磁盘缓存"""

    _CACHE_VERSION = 2

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        cache_dir: str | None = None,
        timeout: float = 3.0,
        model: str = "nomic-embed-text",
    ):
        self._client = None
        self._base_url = base_url
        self._model = model
        self._embeddings: list[tuple[dict, list[float]]] | None = None  # [(node, vec)]
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._timeout = timeout
        self._failed_reason: str | None = None

    def _ensure_client(self) -> bool:
        if self._failed_reason is not None:
            return False
        if self._client is not None:
            return True
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key="ollama",
                timeout=self._timeout,
                max_retries=0,
            )
            return True
        except Exception as e:
            self._failed_reason = str(e)
            logger.warning("EmbeddingRanker: 无法初始化客户端 (%s)", e)
            return False

    def _fingerprint(self, entries: list[dict], summaries: dict[str, str] | None = None) -> str:
        """基于符号列表和摘要生成指纹，代码或摘要变化则指纹变化。"""
        raw = json.dumps(
            [
                (
                    e["id"],
                    e["name"],
                    e.get("file", ""),
                    e.get("signature", ""),
                    e.get("docstring", ""),
                    e.get("parent", ""),
                    e.get("decorators", []),
                    (summaries or {}).get(e["id"], ""),
                )
                for e in entries
            ],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"embeddings_{self._model.replace('/', '_')}.pkl"

    def build_index(self, graph, summaries: dict[str, str] | None = None) -> None:
        """预计算全图节点的 embedding（批量），优先从磁盘缓存加载。

        如果有离线摘要，用摘要文本做 embedding（语义更精准）；
        否则回退到 name+sig+docstring 拼接。
        """
        if self._embeddings is not None:
            return
        if not self._ensure_client():
            raise RuntimeError(self._failed_reason or "embedding 客户端不可用")
        if summaries is None:
            summaries = {}

        # 收集符号条目
        entries = []
        texts = []
        for nid, ndata in graph.nodes(data=True):
            nt = ndata.get("node_type")
            if nt is None:
                continue
            if nt.name not in ("FUNCTION", "METHOD", "CLASS"):
                continue
            name = str(ndata.get("name", ""))
            if not name:
                continue
            file_path = str(ndata.get("file_path", ""))
            if file_path.startswith("tests/") or "test" in file_path.lower().split("/"):
                continue
            extra = ndata.get("extra") or {}
            decorators = ndata.get("decorators") or extra.get("decorators") or []
            parent_id = str(ndata.get("parent_id") or "")
            node = {
                "id": nid,
                "name": name,
                "type": nt.name,
                "file": file_path,
                "signature": str(ndata.get("signature", "")),
                "docstring": str(ndata.get("docstring", "")),
                "parent": parent_id.rsplit("::", 1)[-1] if parent_id else "",
                "decorators": decorators,
            }
            entries.append(node)
            # 摘要文本拼接到原有 embedding 文本后面（不替换）
            summary = summaries.get(str(nid), "")
            base_text = _build_embed_text(node)
            if summary:
                texts.append(f"{base_text} | {summary}")
            else:
                texts.append(base_text)

        if not entries:
            return

        # 尝试从缓存加载
        cache_path = self._cache_path()
        if cache_path is not None:
            fp = self._fingerprint(entries, summaries)
            try:
                cached = pickle.loads(cache_path.read_bytes())
                if cached.get("version") == self._CACHE_VERSION and cached.get("fingerprint") == fp:
                    self._embeddings = cached["embeddings"]
                    logger.info("EmbeddingRanker: 从缓存加载 %d 个符号 (%s)", len(self._embeddings), cache_path)
                    return
            except Exception:
                pass

        # 批量嵌入
        batch_size = 100
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = self._client.embeddings.create(model=self._model, input=batch)
            except Exception as e:
                self._failed_reason = str(e)
                raise
            for d in resp.data:
                all_vecs.append(d.embedding)

        self._embeddings = list(zip(entries, all_vecs))
        logger.info("EmbeddingRanker: 已索引 %d 个符号", len(entries))

        # 写入缓存
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pickle.dumps({
                    "version": self._CACHE_VERSION,
                    "fingerprint": fp,
                    "embeddings": self._embeddings,
                }))
                logger.info("EmbeddingRanker: 已写入缓存 %s", cache_path)
            except Exception:
                pass

    def rank(self, query: str, limit: int = 12) -> list[dict]:
        """返回按嵌入相似度排序的候选列表"""
        if not self._embeddings:
            return []
        if not self._ensure_client():
            raise RuntimeError(self._failed_reason or "embedding 客户端不可用")
        try:
            resp = self._client.embeddings.create(model=self._model, input=[query])
        except Exception as e:
            self._failed_reason = str(e)
            raise
        q_vec = resp.data[0].embedding

        scored = []
        for node, n_vec in self._embeddings:
            sim = sum(a * b for a, b in zip(q_vec, n_vec))
            scored.append({**node, "score": round(sim, 4)})

        scored.sort(key=lambda x: (-x["score"], x["id"]))
        return scored[:limit]


class GraphTool:
    """加载 ACCG 图并提供结构化查询。"""

    def __init__(self, project_path: str, enable_embeddings: bool = False,
                 embedding_model: str = "nomic-embed-text"):
        self.project_path = Path(project_path)
        self.enable_embeddings = enable_embeddings
        self._embedding_model = embedding_model
        self._graph = None
        self._query = None
        self._built = False
        self._graph_names_cache: dict[str, list[dict]] = {}
        self.embedding_ranker: EmbeddingRanker | None = None
        self._candidate_retriever: CandidateRetriever | None = None
        self._embedding_failed_reason: str | None = None

    def ensure_built(self) -> str:
        if self._built:
            return "图已就绪"
        from accg.builder import GraphBuilder
        from accg.query import GraphQuery
        builder = GraphBuilder()
        self._graph = builder.build(str(self.project_path))
        self._query = GraphQuery(self._graph)
        self._built = True
        self._candidate_retriever = None
        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()

        return f"图构建完成: {node_count} 个节点, {edge_count} 条边"

    def _load_summary_index(self) -> dict[str, str]:
        """加载离线摘要索引（.accg/summary_index.json），不存在则返回空。"""
        idx_path = self.project_path / ".accg" / "summary_index.json"
        if not idx_path.is_file():
            return {}
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {}

    @property
    def is_ready(self) -> bool:
        return self._built

    def execute(self, action: str, **kwargs) -> str:
        """执行图查询并返回适合探索消息展示的裁剪结果。"""
        return self._trim(self.execute_raw(action, **kwargs))

    def execute_full(self, action: str, **kwargs) -> str:
        """执行图查询并返回未裁剪 JSON，供证据账本和审计使用。"""
        return json.dumps(
            self.execute_raw(action, **kwargs),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    def execute_raw(self, action: str, **kwargs):
        """执行图查询并保留原始结构化结果。"""
        if not self._built:
            return {"error": "图尚未构建"}
        try:
            return self._dispatch(action, kwargs)
        except Exception as e:
            return {"error": str(e), "action": action}

    def search(
        self,
        query: str,
        limit: int = 12,
        use_embeddings: bool | None = None,
    ) -> RetrievalResult:
        """内部 search 接口：统一候选检索入口。"""
        return self.retrieve_query_candidates(
            query,
            limit=limit,
            use_embeddings=(
                self.enable_embeddings
                if use_embeddings is None
                else use_embeddings
            ),
        )

    def inspect(self, node_id: str) -> dict:
        """内部 inspect 接口：精确读取单个实体及其直接关系。"""
        return self.execute_raw("contextualize", name=node_id, limit=1)

    def traverse(self, action: str, **kwargs):
        """内部 traverse 接口：执行受控关系扩展。"""
        allowed = {
            "call_paths",
            "transitive_callers",
            "transitive_callees",
            "class_hierarchy",
        }
        if action not in allowed:
            return {"error": f"不支持的 traverse 操作: {action}"}
        return self.execute_raw(action, **kwargs)

    # ── 裁剪 ──────────────────────────────────────────────

    MAX_STR_LEN = 3000
    _PROTECTED_KEYS = {"id", "name", "type", "file", "start_line", "end_line",
                       "relevance", "confidence", "strategy", "error", "query"}

    def _trim(self, result, max_items: int = 10) -> str:
        def _trim_value(v, list_depth=0):
            if isinstance(v, str):
                if list_depth <= 1 and len(v) > self.MAX_STR_LEN:
                    return v[:self.MAX_STR_LEN] + _TRIM_MASK
                return v
            if isinstance(v, (int, float, bool, type(None))):
                return v
            if isinstance(v, dict):
                return {k: (_trim_value(vv, list_depth + 1)
                            if k not in self._PROTECTED_KEYS else vv)
                        for k, vv in v.items()}
            if isinstance(v, list):
                if len(v) > max_items:
                    return [_trim_value(x, list_depth + 1) for x in v[:max_items]]
                return [_trim_value(x, list_depth + 1) for x in v]
            return v

        trimmed = _trim_value(result)
        if isinstance(trimmed, list):
            total = len(result) if isinstance(result, list) else 0
            if total > max_items:
                text = json.dumps({"items": trimmed, "truncated": True, "total": total},
                                  ensure_ascii=False, indent=2, default=str)
            else:
                text = json.dumps(trimmed, ensure_ascii=False, indent=2, default=str)
        elif isinstance(trimmed, dict):
            text = json.dumps(trimmed, ensure_ascii=False, indent=2, default=str)
        else:
            text = json.dumps(trimmed, ensure_ascii=False, indent=2, default=str)
        return text

    # ── 路由 ──────────────────────────────────────────────

    def _dispatch(self, action: str, args: dict):
        handler = self._HANDLERS.get(action)
        if handler is None:
            return {"error": f"未知操作: {action}"}
        return handler(self, args)

    # ── 处理函数：query delegate ──────────────────────────

    def _handle_calls_to(self, args: dict):
        min_conf = args.get("min_confidence", 0.45)
        return self._query.calls_to_with_edges(args.get("symbol", ""), min_confidence=min_conf)

    def _handle_calls_from(self, args: dict):
        min_conf = args.get("min_confidence", 0.45)
        return self._query.calls_from_with_edges(args.get("symbol", ""), min_confidence=min_conf)

    def _handle_call_paths(self, args: dict):
        min_conf = args.get("min_confidence", 0.45)
        max_depth = args.get("max_depth", 3)
        return self._query.call_paths(args.get("source", ""), args.get("target", ""),
                                      max_depth=max_depth, min_confidence=min_conf)

    def _handle_transitive_callers(self, args: dict):
        min_conf = args.get("min_confidence", 0.45)
        max_depth = args.get("max_depth", 3)
        return self._query.transitive_callers(args.get("symbol", ""), max_depth=max_depth,
                                              min_confidence=min_conf)

    def _handle_transitive_callees(self, args: dict):
        min_conf = args.get("min_confidence", 0.45)
        max_depth = args.get("max_depth", 3)
        return self._query.transitive_callees(args.get("symbol", ""), max_depth=max_depth,
                                              min_confidence=min_conf)

    def _handle_class_hierarchy(self, args: dict):
        class_name = args.get("class_name", "")
        simple_name = class_name.split("::")[-1]
        class_ids = []
        if class_name in self._graph.nodes:
            class_ids.append(class_name)
        else:
            for node_id, data in self._graph.nodes(data=True):
                node_type = data.get("node_type")
                type_name = getattr(node_type, "name", str(node_type))
                if data.get("name") == simple_name and type_name == "CLASS":
                    class_ids.append(node_id)

        results = []
        for class_id in sorted(class_ids):
            parents = []
            children = []
            for _, parent_id, data in self._graph.out_edges(class_id, data=True):
                if data.get("edge_type") == EdgeType.INHERITS:
                    parents.append(self._query.node_info(parent_id))
            for child_id, _, data in self._graph.in_edges(class_id, data=True):
                if data.get("edge_type") == EdgeType.INHERITS:
                    children.append(self._query.node_info(child_id))
            results.extend([
                {
                    "class_node_id": class_id,
                    "type": "parents",
                    "items": parents,
                },
                {
                    "class_node_id": class_id,
                    "type": "children",
                    "items": children,
                },
            ])
        return results

    def _handle_module_tree(self, args: dict):
        return self._query.module_tree(args.get("prefix", ""))

    def _handle_module_structure(self, args: dict):
        return self._query.module_structure(args.get("prefix", ""))

    # ── 处理函数：已有方法 ────────────────────────────────
    # _contextualize, _narrow_down, _extract_clues_as_tool
    # 已存在，通过 _HANDLERS 映射表调用

    _HANDLERS = {
        "call_paths":         _handle_call_paths,
        "transitive_callers": _handle_transitive_callers,
        "transitive_callees": _handle_transitive_callees,
        "class_hierarchy":    _handle_class_hierarchy,
        "module_tree":        _handle_module_tree,
        "module_structure":   _handle_module_structure,
        "contextualize":      lambda self, a: self._contextualize(a),
        "narrow_down":        lambda self, a: self._narrow_down(a),
        "extract_clues":      lambda self, a: self._extract_clues_as_tool(a),
    }

    # 公开工具集（model 层引用，避免双源定义）
    ACTIONS = frozenset(_HANDLERS.keys())

    # ── 工具函数 ──────────────────────────────────────────

    @staticmethod
    def _is_test_entry(entry: dict) -> bool:
        """判断 find_symbol 返回的条目是否来自测试文件"""
        fid = entry.get("info", {}).get("file", entry.get("id", ""))
        return "/tests/" in fid or fid.startswith("tests/") or fid.startswith("test_")

    # ── find_symbol ──────────────────────────────────────

    def _find_symbol(self, args: dict) -> list:
        name = args.get("name") or args.get("symbol") or ""
        limit = args.get("limit", 5)
        results = self._query.find_symbol(name, limit=limit * 2, readable_only=True)

        src = [r for r in results if not self._is_test_entry(r)]
        tst = [r for r in results if self._is_test_entry(r)]
        results = (src + tst)[:limit]
        for r in results:
            nid = r.get("id", "")
            nt = r.get("info", {}).get("type", "")
            if nt in ("FUNCTION", "METHOD"):
                self._attach_call_edges(nid, r)
        return results

    # ── contextualize ────────────────────────────────────

    def _contextualize(self, args: dict) -> dict:
        """图定位 + 函数体源码 + 调用关系。精确命中只返回 1 条。"""
        name = args.get("name") or args.get("symbol") or ""
        limit = args.get("limit", 3)
        exact_match = None

        # 先尝试精确匹配（含 :: 或同名字段）
        if NodeId.has_symbol(name) and self._graph and name in self._graph.nodes:
            exact_match = name
        elif "::" in name and self._graph and name in self._graph.nodes:
            # 直接查图（用户传了完整 ID 但不符合 NodeId.has_symbol 的格式）
            exact_match = name
        else:
            # 全图搜同名节点（按简单名）
            simple_name = name.split("::")[-1] if "::" in name else name
            candidates = []
            for nid in self._graph.nodes:
                ndata = self._graph.nodes[nid]
                if str(ndata.get("name", "")) == simple_name:
                    candidates.append(nid)
            if len(candidates) == 1:
                exact_match = candidates[0]

        if exact_match:
            info = self._query.node_info(exact_match)
            results = [{"id": exact_match, "score": 100.0, "info": info}]
        else:
            results = self._query.find_symbol(name, limit=limit * 2, readable_only=True)

        if not results:
            return {"error": f"未找到符号: {name}", "results": []}

        # 非精确匹配时：源码优先，测试排后，取 limit 条
        if not exact_match:
            src = [r for r in results if not self._is_test_entry(r)]
            tst = [r for r in results if self._is_test_entry(r)]
            results = (src + tst)[:limit]

        output = {"query": name, "exact": bool(exact_match), "results": []}
        for r in results:
            info = r.get("info", {})
            nid = r.get("id", "")
            nt = info.get("type", "")
            file_path = info.get("file", "")
            start_line = info.get("start_line", 0)
            end_line = info.get("end_line", 0)

            entry = {
                "id": nid,
                "name": info.get("name", ""),
                "type": nt,
                "file": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "signature": info.get("signature", ""),
                "docstring": info.get("docstring", ""),
            }

            # 源码：只取函数体（start_line 到 end_line）
            if file_path and start_line:
                try:
                    full_path = self.project_path / file_path
                    if full_path.exists():
                        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        body_start = max(0, start_line - 1)
                        body_end = min(len(lines), end_line)
                        entry["source_context"] = "\n".join(
                            f"{i+1:4d}| {line}"
                            for i, line in enumerate(lines[body_start:body_end], start=body_start)
                        )
                except Exception:
                    entry["source_context"] = "[无法读取源码]"

            # 调用关系
            if nt in ("FUNCTION", "METHOD"):
                self._attach_call_edges(nid, entry)

            # 类继承
            if nt == "CLASS":
                try:
                    hierarchy = self._query.class_hierarchy(info.get("name", ""))
                    entry["inherits"] = hierarchy
                except Exception:
                    pass
                # 附加类的方法信息（CONTAINS 出边）
                self._attach_class_details(nid, entry)

            output["results"].append(entry)

        return output

    def _attach_class_details(self, nid: str, entry: dict):
        """给 CLASS 节点附加：方法列表 + 实例化关系（含通过子类的间接实例化）"""
        methods = []
        init_id = None
        for _, v, d in self._graph.out_edges(nid, data=True):
            if d.get("edge_type") == EdgeType.CONTAINS:
                m_name = self._graph.nodes[v].get("name", v)
                methods.append({"id": v, "name": m_name})
                if m_name == "__init__":
                    init_id = v

        if methods:
            summaries = self._load_summary_index()
            for m in methods:
                m["summary"] = summaries.get(m["id"], "")
            entry["methods"] = methods

        # 收集本类及所有子类的 CLASS 调用者
        from collections import deque
        children = []
        queue = deque([nid])
        visited = {nid}
        while queue:
            cur = queue.popleft()
            for u, v, d in self._graph.edges(data=True):
                if d.get("edge_type") == EdgeType.INHERITS and v == cur and u not in visited:
                    visited.add(u)
                    children.append(u)
                    queue.append(u)

        inst_set: dict[str, dict] = {}  # caller_id → entry（去重）
        # 直接实例化本类
        for info, edge in self._query.calls_to_with_edges(nid, min_confidence=0.45):
            cid = info.get("id", "")
            if cid not in inst_set:
                inst_set[cid] = {
                    "id": cid, "name": info["name"],
                    "file": info.get("file", ""),
                    "confidence": edge.get("confidence"),
                    "strategy": edge.get("strategy"),
                }
        # 通过子类的间接实例化
        for child_id in children:
            child_name = self._graph.nodes[child_id].get("name", child_id)
            for info, edge in self._query.calls_to_with_edges(child_id, min_confidence=0.45):
                cid = info.get("id", "")
                # 同 caller 可能已记录为其它子类，保留第一个或 confidence 更高的
                existing = inst_set.get(cid)
                new_conf = edge.get("confidence") or 0
                old_conf = existing.get("confidence") if existing else -1
                if existing is None or new_conf > old_conf:
                    inst_set[cid] = {
                        "id": cid, "name": info["name"],
                        "file": info.get("file", ""),
                        "confidence": new_conf,
                        "strategy": edge.get("strategy"),
                        "via_class": child_name,
                    }

        if inst_set:
            entry["instantiated_by"] = sorted(
                inst_set.values(),
                key=lambda x: (x.get("via_class") is not None, x["file"], x["name"])
            )[:12]

    def _attach_call_edges(self, nid: str, target: dict):
        """给 target dict 附加 calls 和 called_by。低置信度（>=0.3）也保留，让 LLM 知道有低质量边标注。"""
        callees = self._query.calls_from_with_edges(nid, min_confidence=0.45)
        target["calls"] = [
            {"id": callee_info.get("id", ""), "name": callee_info["name"],
             "signature": callee_info.get("signature", "")[:80],
             "confidence": callee_edge.get("confidence"),
             "strategy": callee_edge.get("strategy")}
            for callee_info, callee_edge in callees[:8]
        ]
        callers = self._query.calls_to_with_edges(nid, min_confidence=0.45)
        target["called_by"] = [
            {"id": caller_info.get("id", ""), "name": caller_info["name"],
             "file": caller_info.get("file", ""),
             "confidence": caller_edge.get("confidence"),
             "strategy": caller_edge.get("strategy")}
            for caller_info, caller_edge in callers[:8]
        ]

    # ── 候选检索 ───────────────────────────────────────

    def rank_candidates(self, text: str, limit: int = 12) -> list[dict]:
        """兼容旧接口，返回包含 embedding 增强的候选字典。"""
        result = self.retrieve_query_candidates(
            text,
            limit=limit,
            use_embeddings=self.enable_embeddings,
        )
        return [candidate.to_dict() for candidate in result.candidates]

    def rank_query_candidates(self, text: str, limit: int = 12) -> list[dict]:
        """返回确定性的精确、词法与模糊候选。"""
        result = self.retrieve_query_candidates(
            text,
            limit=limit,
            use_embeddings=False,
        )
        return [candidate.to_dict() for candidate in result.candidates]

    def retrieve_query_candidates(
        self,
        text: str,
        limit: int = 12,
        use_embeddings: bool = True,
    ) -> RetrievalResult:
        """执行候选检索；embedding 失败时保留确定性结果。"""
        started_at = time.perf_counter()
        if not self._built or self._graph is None:
            return RetrievalResult(
                candidates=[],
                stages_attempted=[],
                stages_succeeded=[],
                diagnostics=["图尚未构建"],
                status="failed",
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        if self._candidate_retriever is None:
            summaries = self._load_summary_index()
            self._candidate_retriever = CandidateRetriever(
                build_entries(self._graph, summaries)
            )

        embedding_candidates = None
        embedding_error = None
        embedding_attempted = False
        if use_embeddings:
            embedding_attempted = True
            if self._embedding_failed_reason is not None:
                embedding_error = self._embedding_failed_reason
            else:
                try:
                    self._ensure_embedding_ranker()
                    summaries = self._load_summary_index()
                    self.embedding_ranker.build_index(self._graph, summaries)
                    embedding_candidates = self.embedding_ranker.rank(
                        text,
                        limit=limit,
                    )
                except Exception as e:
                    embedding_error = str(e)
                    self._embedding_failed_reason = embedding_error
                    logger.warning(
                        "候选检索跳过 embedding，使用确定性回退: %s",
                        embedding_error,
                    )

        result = self._candidate_retriever.retrieve(
            text,
            limit=limit,
            embedding_candidates=embedding_candidates,
            embedding_attempted=embedding_attempted,
            embedding_error=embedding_error,
        )
        result.duration_ms = (time.perf_counter() - started_at) * 1000
        return result

    def select_query_anchors(
        self,
        query: str,
        candidates: list[dict],
        max_anchors: int = 3,
    ) -> list[dict]:
        """从已排序候选中确定性地选择类型多样的锚点。"""
        query_terms = set(tokenize(query))
        test_requested = bool(
            query_terms & {"test", "pytest", "fixture"}
        )
        typed_candidates = []
        for item in candidates:
            normalized_path = str(item.get("file", "")).replace("\\", "/")
            path_parts = normalized_path.lower().split("/")
            basename = path_parts[-1] if path_parts else ""
            is_test = (
                "tests" in path_parts
                or "test" in path_parts
                or basename.startswith("test_")
                or basename.endswith("_test.py")
            )
            if is_test and not test_requested:
                continue
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
        anchors = []
        explicit_type_positions = []
        for keyword, node_type in (
            ("function", "FUNCTION"),
            ("class", "CLASS"),
            ("method", "METHOD"),
        ):
            match = re.search(rf"\b{keyword}s?\b", query, re.IGNORECASE)
            if match:
                explicit_type_positions.append((match.start(), node_type))
        preferred_types = [
            node_type
            for _, node_type in sorted(explicit_type_positions)
        ]
        # 仅当 query 显式提到类型才加入，未提及时传空列表，
        # 由 select_query_anchors 按分数驱动选择，避免 FUNCTION > METHOD 等硬编码偏好
        comparison_query = bool(re.search(
            r"\b(compare|comparison|between|relationship|difference|"
            r"versus|vs)\b",
            query,
            re.IGNORECASE,
        ))
        for candidate in _select_query_anchors(
            typed_candidates,
            max_anchors=max_anchors,
            preferred_types=preferred_types,
            required_types=[
                node_type
                for _, node_type in sorted(explicit_type_positions)
            ],
            prefer_term_coverage=comparison_query,
        ):
            item = candidate.to_dict()
            if "exact_id" in candidate.sources:
                reason = "精确 Node ID 匹配"
            elif "exact_symbol" in candidate.sources:
                reason = "精确符号匹配"
            elif candidate.type.lower() in query_terms:
                reason = f"覆盖问题明确要求的 {candidate.type} 类型"
            elif candidate.matched_terms:
                reason = "覆盖新的问题关键词"
            else:
                reason = "按候选相关性补足锚点"
            item.update({
                "selection_reason": reason,
                "covered_terms": sorted(set(candidate.matched_terms)),
                "candidate_sources": list(candidate.sources),
            })
            anchors.append(item)
        return anchors

    def validate_query_anchor(self, anchor: dict) -> dict:
        """验证锚点身份、路径、行号与源码可读性。"""
        node_id = str(anchor.get("id", ""))
        if self._graph is None or node_id not in self._graph.nodes:
            name = str(anchor.get("name") or node_id.rsplit("::", 1)[-1])
            suggestions = self.rank_query_candidates(name, limit=3)
            return {
                "valid": False,
                "reason": "node_id_not_found",
                "message": f"Node ID 不存在: {node_id}",
                "suggestions": suggestions,
            }

        data = self._graph.nodes[node_id]
        node_type = data.get("node_type")
        actual_type = getattr(node_type, "name", str(node_type or ""))
        expected_type = str(anchor.get("type", ""))
        if expected_type and actual_type != expected_type:
            return {
                "valid": False,
                "reason": "type_mismatch",
                "message": (
                    f"类型不一致: 候选={expected_type}, 图中={actual_type}"
                ),
                "suggestions": [],
            }

        file_path = str(data.get("file_path") or anchor.get("file") or "")
        root = self.project_path.resolve()
        full_path = (root / file_path).resolve()
        try:
            full_path.relative_to(root)
        except ValueError:
            return {
                "valid": False,
                "reason": "path_outside_project",
                "message": f"文件位于项目根目录之外: {file_path}",
                "suggestions": [],
            }
        if not full_path.is_file():
            return {
                "valid": False,
                "reason": "file_not_found",
                "message": f"文件不存在: {file_path}",
                "suggestions": [],
            }

        try:
            lines = full_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError as exc:
            return {
                "valid": False,
                "reason": "source_unreadable",
                "message": f"源码不可读取: {exc}",
                "suggestions": [],
            }

        start_line = data.get("start_line")
        end_line = data.get("end_line")
        if (
            not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
            or end_line > len(lines)
        ):
            return {
                "valid": False,
                "reason": "invalid_line_range",
                "message": (
                    f"无效行号范围: {start_line}-{end_line}, "
                    f"文件共 {len(lines)} 行"
                ),
                "suggestions": [],
            }

        return {
            "valid": True,
            "reason": "validated",
            "message": "锚点验证通过",
            "node_id": node_id,
            "type": actual_type,
            "file": file_path,
            "start_line": start_line,
            "end_line": end_line,
        }

    def _ensure_embedding_ranker(self):
        if self.embedding_ranker is None:
            cache_dir = str(self.project_path / ".accg")
            self.embedding_ranker = EmbeddingRanker(
                cache_dir=cache_dir,
                model=self._embedding_model,
            )

    # ── extract_clues（独立工具） ────────────────────────

    def _extract_clues_as_tool(self, args: dict) -> dict:
        """从 source_context 文本中提取图上可定位符号"""
        source = args.get("source", "")
        return {"clues": self._extract_graph_clues(source)}

    def _extract_graph_clues(self, source_context: str) -> list[dict]:
        import re
        if not self._graph_names_cache:
            for nid, ndata in self._graph.nodes(data=True):
                nt = ndata.get("node_type")
                if nt and nt.name in ("FUNCTION", "METHOD", "CLASS"):
                    name = str(ndata.get("name", ""))
                    fid = str(ndata.get("file_path", ""))
                    if name and not fid.startswith("tests/"):
                        self._graph_names_cache.setdefault(name.lower(), []).append(
                            {"id": nid, "name": name, "type": nt.name, "file": fid})
        seen = set()
        clues = []
        for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\b', source_context):
            word = m.group(1)
            if word in seen or len(word) < 3:
                continue
            seen.add(word)
            if word in ("def", "class", "return", "import", "from", "self", "None",
                        "True", "False", "if", "else", "elif", "for", "while", "try",
                        "except", "with", "as", "not", "and", "or", "in", "is",
                        "assert", "raise", "pass", "break", "continue", "yield",
                        "print", "len", "range", "str", "int", "list", "dict", "set",
                        "type", "open", "enumerate", "zip", "map", "filter", "isinstance",
                        "hasattr", "getattr", "setattr", "format", "sorted", "reversed"):
                continue
            candidates = self._graph_names_cache.get(word.lower(), [])
            clues.extend(candidates)
        return clues

    # ── find_tested（独立工具） ──────────────────────────

    # _find_tested_as_tool 已废弃移除（LLM 误用 node ID 当 source 参数）

    # ── narrow_down ──────────────────────────────────────

    def _narrow_down(self, args: dict) -> dict:
        clues = args.get("clues", [])
        limit = args.get("limit", 5)
        frontier_ids = args.get("frontier_ids", [])

        if not clues:
            return {"error": "clues 参数不能为空", "results": []}

        symbol_set = set()
        for clue in clues:
            try:
                results = self._query.find_symbol(clue, limit=limit, readable_only=True)
                for r in results:
                    sid = r.get("id", "")
                    info = r.get("info", {})
                    if info.get("file", "").startswith("tests/"):
                        continue
                    if info.get("type", "") in ("FILE", "MODULE"):
                        continue
                    symbol_set.add(sid)
            except Exception:
                pass

        symbols = list(symbol_set)
        if not symbols:
            return {"error": f"未找到与线索相关的符号: {clues}", "results": []}

        clues_lower = [c.lower() for c in clues]
        scored = []
        for sid in symbols:
            info = self._query.node_info(sid)
            nt = info.get("type", "")
            info_str = json.dumps(info, ensure_ascii=False, default=str).lower()
            score = 0.0

            for clue in clues_lower:
                if clue in info_str:
                    score += 2.0
                for part in clue.split("_"):
                    if len(part) > 2 and part in info_str:
                        score += 0.5
            if sid in frontier_ids:
                score += 3.0
            if nt in ("FUNCTION", "METHOD"):
                try:
                    edges = self._query.calls_from_with_edges(sid, min_confidence=0.5)
                    score += len(edges) * 0.3
                    for _info, edge in edges:
                        if edge.get("confidence", 0) and edge["confidence"] >= 0.8:
                            score += 0.5
                except Exception:
                    pass
                try:
                    callers = self._query.calls_to_with_edges(sid, min_confidence=0.5)
                    score += len(callers) * 0.2
                    for _info, edge in callers:
                        if edge.get("confidence", 0) and edge["confidence"] >= 0.8:
                            score += 0.3
                except Exception:
                    pass
            if score > 0:
                scored.append((sid, score, info))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for sid, score, info in scored[:limit]:
            results.append({
                "id": sid, "relevance": round(score, 1),
                "name": info.get("name", ""), "type": info.get("type", ""),
                "file": info.get("file", ""), "signature": info.get("signature", ""),
                "docstring": info.get("docstring", ""),
            })
        return {
            "clues_used": clues, "total_candidates": len(symbols),
            "matched": len(results), "results": results,
        }
