# -*- coding: utf-8 -*-
"""Agent 核心 -- 极简 ReAct 循环,支持 图查询 + 只读文件 双工具"""
import copy
import json
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from .model import Model
from .environment import Environment
from .evidence import EvidenceItem, EvidenceLedger, DisplayLevel
from .query_plan import Anchor, QueryPlan
from .retrieval import RetrievalResult
from .sufficiency import (
    FinishAction,
    ExpansionRequest,
    GateDecision,
    SufficiencyGate,
)
from .entity_extractor import Entity, EntityExtractor
from .multi_entity import MultiEntityOrchestrator
from accg.models import NodeId


@dataclass
class MsgRecord:
    """发给 LLM 或从 LLM 收到的单条消息"""
    role: str              # "system" | "user" | "assistant" | "tool"
    content: str           # 消息正文
    step: int = 0          # system/user=0, assistant/tool=1..N
    raw_json: str | None = None      # 工具原始 JSON（仅 tool）
    tool_name: str | None = None     # 工具名（仅 tool）
    tool_args: dict | None = None    # 工具参数（仅 tool）
    intercepted: bool = False        # 重复调用被拦截


@dataclass
class SynthesisRecord:
    """独立合成阶段"""
    prompt: str            # ANSWER_PROMPT 填好问题和证据后
    answer: str            # 模型返回


@dataclass
class RunResult:
    """Agent.run() 返回的完整运行轨迹"""
    answer: str
    error: str | None = None
    anchor_candidates: list[dict] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    messages: list[MsgRecord] = field(default_factory=list)
    synthesis: SynthesisRecord | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_selection_report: str | None = None
    model_requests: list[dict] = field(default_factory=list)
    query_plan: dict | None = None
    entities: list[dict] = field(default_factory=list)
    finish_draft: str | None = None

    @property
    def rounds(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @property
    def explorations(self) -> int:
        return sum(1 for m in self.messages if m.role == "tool" and not m.intercepted)

def _parse_graph_path_ids(result_json: str) -> list[tuple[str, int]]:
    """从 call_paths/transitive_* 返回结果中提取 (node_id, depth) 列表"""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    entries = []

    def _walk(obj):
        if isinstance(obj, dict):
            nid = obj.get("id") or obj.get("node_id")
            depth = obj.get("depth")
            if isinstance(nid, str) and NodeId.has_symbol(nid):
                entries.append((nid, depth if isinstance(depth, int) else 0))
            # 也走 edges / path 列表
            for key in ("edges", "path", "paths", "items", "results"):
                v = obj.get(key)
                if isinstance(v, list):
                    for item in v:
                        _walk(item)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    return entries


def _collect_node_ids(result_json: str) -> list[str]:
    """从工具返回 JSON 中收集所有 node id（用于 frontier）"""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    ids = []

    def _walk(obj):
        if isinstance(obj, dict):
            for key in ("id", "target_node_id", "source_node_id"):
                v = obj.get(key)
                if isinstance(v, str) and (NodeId.has_symbol(v) or v.endswith(".py")):
                    ids.append(v)
                    break
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    return ids


def _collect_endpoint_node_ids(result_json: str) -> list[str]:
    """从传递查询结果中提取端点节点，保持原始顺序并去重。"""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    entries = data if isinstance(data, list) else data.get(
        "items",
        data.get("results", []),
    )
    node_ids = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("endpoint_node_id") or entry.get("id")
        if isinstance(node_id, str) and node_id and node_id not in seen:
            seen.add(node_id)
            node_ids.append(node_id)
    return node_ids


def _collect_path_node_ids(result_json: str) -> list[str]:
    """从 call_paths 结果中提取完整路径节点，保持路径顺序并去重。"""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    entries = data if isinstance(data, list) else data.get(
        "items",
        data.get("results", []),
    )
    node_ids = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for node_id in entry.get("node_ids", []):
            if (
                isinstance(node_id, str)
                and NodeId.has_symbol(node_id)
                and node_id not in seen
            ):
                seen.add(node_id)
                node_ids.append(node_id)
    return node_ids


SYSTEM_PROMPT = """
你是一个代码分析 ReAct Agent。ACCG 已为当前项目构建了代码图，你可以在图结构和源码之间双向验证。用图定位"去哪读"，用源码验证"读到了什么"。

## 输出格式

THOUGHT: <当前已知什么、缺什么证据、下一步查什么>
ACTION: {"name": "<工具名>", "arguments": {<参数>}}

无依赖的平行查询可写多个 ACTION（最多 2 个）：
ACTION: {"name": "contextualize", "arguments": {"name": "A"}}
ACTION: {"name": "contextualize", "arguments": {"name": "B"}}
有依赖则必须分轮。

证据足够时输出 FINAL: <最终答案，用中文，引用证据>
- JSON 必须合法，不要编造或预测结果

## 节点类型与返回字段

contextualize 的返回字段取决于节点 type：

### FUNCTION / METHOD
- source_context(源码)、signature、docstring
- calls(调用了谁) / called_by(被谁调用)
- 工具: contextualize, transitive_callees, transitive_callers, call_paths, read_file
- transitive_* 和 call_paths 必须用完整 ID（格式: 文件路径::类名::方法名）

### CLASS
- source_context、docstring、methods(类中方法列表)
- inherits(parents + children) — 继承层次
- instantiated_by — 谁创建了该类的实例（含置信度和 via_class）
- 工具: contextualize, class_hierarchy, read_file
- CLASS 没有 calls/called_by。查使用者用 instantiated_by 或 contextualize 类名::__init__
- 禁止对 CLASS 调用 transitive_* / call_paths

## 环境: __CWD__
__GRAPH_STATUS__
"""

ANSWER_PROMPT = """\
你是一个代码分析专家。请根据以下 Agent 检索到的证据，回答用户的问题。

## 要求

- 每个关键论断必须引用具体的证据（文件路径、行号、置信度）
- 对比多个类/函数时，**必须**对比它们的继承层次（inherits/parents）和调用关系，指出差异背后的语义含义
- 如果证据之间存在关联，要明确指出它们如何串成链条
- 如果某个关键环节缺少证据，在答案末尾标注"[缺失证据: xxx]"
- 用中文回答，保持简洁、精确、技术化
- 结构：先给出核心结论一句话，再分段展开证据链条

## 用户问题

__QUESTION__

## Agent 收集的证据

__EVIDENCE__

## Agent 结束草稿（仅供参考，不属于证据）

__DRAFT__

## 限定搜索记录（仅用于说明搜索范围，不属于事实证据）

__SEARCH_SCOPE__

## 请回答
"""


class Agent:
    """Graph-guided ReAct Agent -- 图查询 + 只读文件工具"""

    MAX_EXPLORATION_DEPTH = 3
    QUERY_CANDIDATE_LIMIT = 24
    CANDIDATE_DISPLAY_LIMIT = 8

    def __init__(
        self,
        model: Model,
        env: Environment,
        graph_tool=None,
        max_steps: int = 15,
        answer_model=None,
        on_step: Callable[["MsgRecord"], None] | None = None,
        on_audit: Callable[[str], None] | None = None,
        reranker=None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.model = model
        self.env = env
        self.graph_tool = graph_tool
        self.max_steps = max_steps
        self.messages: list[dict] = []
        self.step_count = 0
        self._explored: set[str] = set()
        self._frontier: deque[str] = deque()
        self._ledger = EvidenceLedger()
        self.answer_model = answer_model or model
        self.on_step = on_step
        self.on_audit = on_audit
        self._trace: list[MsgRecord] = []
        self._retrieval_result: RetrievalResult | None = None
        self.last_model_requests: list[dict] = []
        self._full_tool_results: dict[str, str] = {}
        self.last_query_plan: dict = {}
        self._entity_extractor = entity_extractor
        self._orchestrator = (
            MultiEntityOrchestrator(graph_tool, reranker)
            if graph_tool and entity_extractor
            else None
        )
        self._gate = SufficiencyGate()
        self._expansion_count = 0
        self._expanded_relations: set[str] = set()
        self._latest_finish_draft = ""
        self._reranker = reranker

    def _emit(self, record: MsgRecord) -> None:
        self._trace.append(record)
        if self.on_step:
            self.on_step(record)

    @property
    def _evidence(self) -> list[str]:
        """兼容旧接口：返回证据的自然语言渲染列表。"""
        return [item.render(DisplayLevel.PREVIEW) for item in self._ledger.items()
                if item.kind != "error"]

    def _audit_model_request(self, stage: str, messages: list[dict]) -> str:
        """保存完整模型请求并输出人类可读审计信息。

        stage 示例: "exploration_step_3", "answer_synthesis"
        """
        saved = copy.deepcopy(messages)
        full_tool_results = copy.deepcopy(self._full_tool_results)

        # 精确统计上下文大小
        total_chars = 0
        cjk = 0
        for msg in saved:
            content = msg.get("content", "")
            total_chars += len(content)
            cjk += sum(1 for c in content if "一" <= c <= "鿿")
            for tc in msg.get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                total_chars += len(args)
                cjk += sum(1 for c in args if "一" <= c <= "鿿")
        est_tokens = (total_chars - cjk) // 4 + int(cjk / 1.5)

        # 人类可读输出
        parts = [
            f"── 发给大模型的完整内容 | {stage} "
            f"| msgs={len(saved)} chars={total_chars} "
            f"est_tokens={est_tokens} ──"
        ]
        for i, msg in enumerate(saved):
            role = msg.get("role", "?").upper()
            parts.append(f"\n消息 {i+1}/{len(saved)} | {role}")

            if role == "TOOL":
                call_id = msg.get("tool_call_id", "?")
                parts.append(f"  关联工具调用: {call_id}")
                content = full_tool_results.get(call_id, msg.get("content", ""))
                parts.append(f"  内容 ({len(content)} 字符):")
                parts.append(content)
            elif role == "ASSISTANT":
                tool_calls = msg.get("tool_calls", [])
                parts.append(f"  content: {msg.get('content', '')}")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    parts.append(f"  工具名称: {fn.get('name', '?')}")
                    parts.append(f"  工具参数: {fn.get('arguments', '?')}")
                    parts.append(f"  调用 ID: {tc.get('id', '?')}")
            else:
                content = msg.get("content", "")
                parts.append(f"  {content}")

        audit_text = "\n".join(parts)
        self.last_model_requests.append({
            "stage": stage,
            "messages": saved,
            "audit_text": audit_text,
            "full_tool_results": full_tool_results,
        })
        if self.on_audit:
            self.on_audit(audit_text)
        return audit_text

    def run(self, task: str) -> RunResult:
        """执行任务,返回包含完整轨迹的 RunResult"""
        self._trace = []
        self._retrieval_result = None
        self.messages = []
        self.step_count = 0
        self._explored.clear()
        self._frontier.clear()
        self._ledger = EvidenceLedger()
        self.last_model_requests.clear()
        self._full_tool_results.clear()
        self.last_query_plan = {}
        self._expansion_count = 0
        self._expanded_relations.clear()
        self._latest_finish_draft = ""
        graph_status = ""
        if self.graph_tool:
            try:
                graph_status = self.graph_tool.ensure_built()
            except Exception as e:
                return RunResult(answer="", error=f"图构建失败: {e}")
            if not self.graph_tool.is_ready:
                return RunResult(answer="", error="图工具未就绪")

        # 预分析：实体提取 + 多实体检索 / 单检索
        prelude = ""
        candidates = []
        query_plan = QueryPlan(query=task)
        retrieval_started_at = time.perf_counter()

        entities = []
        use_multi_entity = False
        if self._orchestrator and self.graph_tool:
            entities = self._entity_extractor.extract(task)
            query_plan.entities = [e.to_dict() for e in entities]
            use_multi_entity = len(entities) > 1

        if use_multi_entity:
            prelude_result = self._orchestrator.run(
                entities=entities,
                task=task,
                ledger=self._ledger,
                recommended_count=self._gate.recommended_anchor_count(
                    task, len(entities)
                ),
            )
            prelude += "\n\n[候选符号] 以下按实体分别检索:\n"
            prelude += prelude_result.text
            query_plan.prefetch_evidence_ids = list(
                prelude_result.prefetch_evidence_ids
            )
            for name, anchor_list in prelude_result.entity_anchors.items():
                for anchor_data in anchor_list:
                    anchor = Anchor.from_dict(anchor_data)
                    anchor.validation = {
                        "valid": True,
                        "reason": "exact_contextualize_result",
                    }
                    anchor.evidence_ids = anchor_data.get("evidence_ids", [])
                    anchor.display_level = anchor_data.get("display_level", "complete")
                    anchor.omitted_reason = anchor_data.get("omitted_reason", "")
                    query_plan.anchors.append(anchor)
            query_plan.rejected_anchors = prelude_result.rejected_anchors
            query_plan.diagnostics.extend(prelude_result.diagnostics)
            query_plan.diagnostics.append(
                f"多实体检索: {len(entities)} 个实体, "
                f"{prelude_result.anchor_count} 个有效锚点"
            )
            self._retrieval_result = RetrievalResult(
                candidates=[],
                stages_attempted=[],
                stages_succeeded=[],
                diagnostics=["多实体路径，见 query_plan.entities"],
            )
        elif self.graph_tool:
            try:
                search = getattr(self.graph_tool, "search", None)
                if search is None:
                    search = self.graph_tool.retrieve_query_candidates
                self._retrieval_result = (
                    search(
                        task,
                        limit=self.QUERY_CANDIDATE_LIMIT,
                        use_embeddings=getattr(
                            self.graph_tool,
                            "enable_embeddings",
                            False,
                        ),
                    )
                )
                candidates = [
                    candidate.to_dict()
                    for candidate in self._retrieval_result.candidates
                ]
                query_plan.candidates = list(
                    self._retrieval_result.candidates
                )
                query_plan.diagnostics.extend(
                    self._retrieval_result.diagnostics
                )
            except Exception as e:
                self._retrieval_result = RetrievalResult(
                    candidates=[],
                    stages_attempted=[],
                    stages_succeeded=[],
                    diagnostics=[f"候选检索失败: {e}"],
                    status="failed",
                )
                query_plan.diagnostics.append(f"候选检索失败: {e}")
            if candidates:
                # ── 在线重排 ──
                rerank_info = None
                if self._reranker is not None:
                    try:
                        rerank_result = self._reranker.rerank(task, candidates)
                        rerank_info = {
                            "relevant_ids": rerank_result.ranked_ids,
                            "reasoning": rerank_result.reasoning,
                            "elapsed_ms": round(rerank_result.elapsed_ms, 1),
                            "error": rerank_result.error,
                        }
                        query_plan.diagnostics.append(
                            f"重排完成: {len(rerank_result.ranked_ids)} 个相关候选 "
                            f"({rerank_result.elapsed_ms:.0f}ms)"
                            + (f", 原因: {rerank_result.reasoning}" if rerank_result.reasoning else "")
                        )
                        if rerank_result.passed:
                            # 将重排后的候选列表用于后续锚点选择
                            candidates = self._reranker.apply(task, candidates)
                    except Exception as e:
                        rerank_info = {"error": str(e)}
                        query_plan.diagnostics.append(f"重排失败: {e}")
                if rerank_info:
                    query_plan.rerank = rerank_info

                items = []
                for c in candidates[:self.CANDIDATE_DISPLAY_LIMIT]:
                    sources = ",".join(c.get("sources", []))
                    items.append(
                        f"  - {c['name']} ({c['type']}) {c['id']} "
                        f"[score={c['score']:.2f}; sources={sources}]"
                    )
                prelude = "\n\n[候选符号] 以下与问题语义最相关:\n" + "\n".join(items)
                if rerank_info and rerank_info.get("relevant_ids"):
                    prelude += (
                        "\n\n[重排] 小模型认为最相关的 5 个候选: "
                        + ", ".join(
                            next(
                                (
                                    c.get("name", cid)
                                    for c in candidates[:24]
                                    if c.get("id") == cid
                                ),
                                cid,
                            )
                            for cid in rerank_info["relevant_ids"][:5]
                        )
                    )

                ordered_anchors = self.graph_tool.select_query_anchors(
                    task,
                    candidates,
                    max_anchors=len(candidates),
                )
                target_count = min(
                    self._gate.recommended_anchor_count(task),
                    len(ordered_anchors),
                )
                for anchor_data in ordered_anchors:
                    if len(query_plan.anchors) >= target_count:
                        break
                    validation = self.graph_tool.validate_query_anchor(
                        anchor_data
                    )
                    if not validation.get("valid"):
                        query_plan.rejected_anchors.append({
                            "candidate": anchor_data,
                            "reason": validation.get("reason", "invalid"),
                            "message": validation.get("message", ""),
                            "suggestions": validation.get("suggestions", []),
                        })
                        continue

                    raw_prefetch = json.dumps(
                        self.graph_tool.inspect(anchor_data["id"]),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                    evidence_items = EvidenceItem.from_tool_result(
                        "query_graph",
                        {
                            "action": "contextualize",
                            "name": anchor_data["id"],
                        },
                        raw_prefetch,
                        step=0,
                    )
                    source_items = [
                        item for item in evidence_items
                        if item.kind == "source"
                        and item.node_id == anchor_data["id"]
                    ]
                    if not source_items:
                        query_plan.rejected_anchors.append({
                            "candidate": anchor_data,
                            "reason": "prefetch_without_source",
                            "message": "精确 inspect 未返回锚点源码",
                            "suggestions": [],
                        })
                        continue
                    for item in evidence_items:
                        self._ledger.add(item)

                    anchor = Anchor.from_dict(anchor_data)
                    anchor.validation = validation
                    anchor.evidence_ids = [
                        item.evidence_id for item in source_items
                    ]
                    anchor.prefetch_action = {
                        "interface": "inspect",
                        "adapter": "query_graph",
                        "action": "contextualize",
                        "name": anchor.id,
                    }
                    query_plan.anchors.append(anchor)
                    query_plan.prefetch_evidence_ids.extend(
                        anchor.evidence_ids
                    )
                    self._full_tool_results[
                        f"prefetch:{anchor.id}"
                    ] = raw_prefetch

                if query_plan.anchors:
                    prefetched = [
                        item
                        for item in self._ledger.source_items
                        if item.evidence_id
                        in set(query_plan.prefetch_evidence_ids)
                    ]
                    evidence_text, display_reports = (
                        self._ledger.render_prefetch_evidence(
                            prefetched
                        )
                    )
                    reports_by_id = {
                        report["evidence_id"]: report
                        for report in display_reports
                    }
                    for anchor in query_plan.anchors:
                        report = next(
                            (
                                reports_by_id[evidence_id]
                                for evidence_id in anchor.evidence_ids
                                if evidence_id in reports_by_id
                            ),
                            None,
                        )
                        if report is not None:
                            anchor.display_level = report["display_level"]
                            anchor.omitted_reason = report["omitted_reason"]
                    prelude += (
                        "\n\n[自动验证锚点的证据]\n"
                        + evidence_text
                    )
                if self._retrieval_result is not None:
                    query_plan.diagnostics.append(
                        f"成功预取 {len(query_plan.anchors)} 个锚点"
                    )

                accepted_ids = {
                    anchor.id for anchor in query_plan.anchors
                }
                rejected_ids = {
                    item.get("candidate", {}).get("id")
                    for item in query_plan.rejected_anchors
                }
                query_requests_tests = bool(
                    set(re.findall(r"[A-Za-z0-9_]+", task.lower()))
                    & {"test", "tests", "pytest", "fixture"}
                )
                for candidate in candidates:
                    candidate_id = candidate.get("id")
                    if (
                        candidate_id in accepted_ids
                        or candidate_id in rejected_ids
                    ):
                        continue
                    normalized_path = str(
                        candidate.get("file", "")
                    ).replace("\\", "/").lower()
                    path_parts = normalized_path.split("/")
                    basename = path_parts[-1] if path_parts else ""
                    is_test = (
                        "tests" in path_parts
                        or "test" in path_parts
                        or basename.startswith("test_")
                        or basename.endswith("_test.py")
                    )
                    reason = (
                        "low_quality_test_candidate"
                        if is_test and not query_requests_tests
                        else "max_anchor_limit"
                    )
                    query_plan.rejected_anchors.append({
                        "candidate": candidate,
                        "reason": reason,
                        "message": (
                            "问题未询问测试，测试候选不参与锚点选择"
                            if reason == "low_quality_test_candidate"
                            else "已达到锚点数量上限"
                        ),
                        "suggestions": [],
                    })

        if self._retrieval_result is not None:
            self._retrieval_result.duration_ms = (
                time.perf_counter() - retrieval_started_at
            ) * 1000
            query_plan.diagnostics.append(
                "候选检索与锚点预取总耗时 "
                f"{self._retrieval_result.duration_ms:.3f}ms"
            )
        self.last_query_plan = query_plan.to_dict()

        system_content = SYSTEM_PROMPT.replace("__CWD__", self.env.config.cwd).replace("__GRAPH_STATUS__", graph_status)
        user_content = f"任务: {task}{prelude}"

        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        self._emit(MsgRecord(role="system", content=system_content, step=0))
        self._emit(MsgRecord(role="user", content=user_content, step=0))

        recent_actions = []
        contextualized_symbols: set[str] = set()

        for _ in range(self.max_steps):
            self.step_count += 1

            self._audit_model_request(f"exploration_step_{self.step_count}", self.messages)
            response = self.model.query(self.messages)

            thought = response.get("content", "").strip()
            raw_content = response.get("raw_content", "")

            # 模型完成信号检测
            finish_action = FinishAction.from_content(raw_content)
            has_tools = bool(response["tool_calls"])
            model_done = (
                finish_action is not None
                or not has_tools
            )

            if model_done:
                if finish_action is not None:
                    self._latest_finish_draft = finish_action.draft
                draft = finish_action.draft if finish_action else thought
                self._emit(MsgRecord(role="assistant", content=raw_content, step=self.step_count))
                self.messages.append({"role": "assistant", "content": thought})

                # 证据充分性门控
                gate_decision = self._gate.evaluate(
                    question=task,
                    query_plan=self.last_query_plan,
                    evidence_items=self._ledger.items(),
                    draft=draft,
                    expansion_count=self._expansion_count,
                    expanded_relations=self._expanded_relations,
                )

                if gate_decision.passed:
                    return self._synthesize(task, candidates, draft=draft)

                # 门控未通过 —— 尝试受控扩展
                if (
                    gate_decision.expansion_requests
                    and self._expansion_count < SufficiencyGate.MAX_AUTO_EXPANSIONS
                ):
                    expanded_items = self._execute_expansions(
                        gate_decision.expansion_requests,
                        query_plan,
                    )
                    self._expansion_count += 1
                    self.last_query_plan = query_plan.to_dict()

                    # 压缩旧工具结果，保留最近轮次
                    self._compress_messages(keep_recent_turns=3)

                    # 通知模型证据不足，包含新增证据
                    gate_msg = self._format_gate_failure_message(
                        gate_decision,
                        draft,
                        expanded_items,
                    )
                    self.messages.append({
                        "role": "user",
                        "content": gate_msg,
                    })
                    self._emit(MsgRecord(
                        role="user", content=gate_msg,
                        step=self.step_count,
                    ))
                    continue

                # 无扩展可用 —— 返回错误
                error_msg = "证据不足: " + "; ".join(
                    gate_decision.missing_requirements
                )
                if finish_action and finish_action.draft:
                    error_msg = (
                        f"[门控未通过] {error_msg}。"
                        f"模型草稿: {finish_action.draft[:200]}"
                    )
                return RunResult(
                    answer="",
                    error=error_msg,
                    anchor_candidates=candidates,
                    retrieval=self._retrieval_result,
                    messages=list(self._trace),
                    query_plan=copy.deepcopy(self.last_query_plan),
                    entities=list(query_plan.entities),
                    finish_draft=self._latest_finish_draft or None,
                )

            # 有工具调用 —— 正常探索
            self._emit(MsgRecord(role="assistant", content=raw_content, step=self.step_count))
            self.messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response["tool_calls"],
            })

            for tc in response["tool_calls"]:
                tool_name = tc["function"]["name"]

                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                action_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                ctx_name = args.get("name") or args.get("symbol") or ""
                is_dup_ctx = (
                    tool_name == "query_graph"
                    and args.get("action") == "contextualize"
                    and ctx_name in contextualized_symbols
                )
                intercepted = (action_key in recent_actions[-5:]) or is_dup_ctx
                evidence_items = []

                if intercepted:
                    obs_raw = ""
                    obs_text = (
                        f"[重复调用拦截]\n"
                        f"你刚刚查过 {ctx_name or '这个符号'}，结果不会改变。\n"
                        f"请用已有证据输出 FINAL，或换个符号继续探索。"
                    )
                else:
                    recent_actions.append(action_key)
                    if len(recent_actions) > 5:
                        recent_actions = recent_actions[-5:]
                    obs_raw = self._execute_tool(tool_name, args)
                    # 创建结构化证据并写入账本
                    evidence_items = EvidenceItem.from_tool_result(
                        tool_name, args, obs_raw, self.step_count)
                    for ei in evidence_items:
                        self._ledger.add(ei)
                    self._register_dynamic_anchors(
                        evidence_items,
                        query_plan,
                    )
                    self.last_query_plan = query_plan.to_dict()
                    obs_text = self._ledger.render_for_observation(evidence_items)
                    if not obs_text:
                        obs_text = self._format_for_llm(obs_raw, tool_name, args)
                    if tool_name == "query_graph" and args.get("action") == "contextualize" and ctx_name:
                        contextualized_symbols.add(ctx_name)

                # 控制消息不进入账本，仅附加到本轮 observation。
                raw_json = obs_text if not intercepted else ""
                if not intercepted:
                    raw_json = obs_raw

                # 继承收敛分析
                if not intercepted and tool_name == "query_graph" and args.get("action") == "contextualize":
                    convergence = self._check_convergence(raw_json)
                    if convergence:
                        obs_text += "\n\n" + convergence

                # 追加已探索面包屑
                obs_text = self._track_exploration(tool_name, args, obs_text) if not intercepted else obs_text

                record_raw = raw_json if raw_json else None
                self._emit(MsgRecord(
                    role="tool", content=obs_text, step=self.step_count,
                    raw_json=record_raw, tool_name=tool_name, tool_args=args,
                    intercepted=intercepted,
                ))

                self.messages.append({
                    "role": "tool",
                    "content": obs_text,
                    "tool_call_id": tc["id"],
                })
                if not intercepted:
                    self._full_tool_results[tc["id"]] = obs_raw

        return RunResult(
            answer="[达到最大步数]",
            anchor_candidates=candidates,
            retrieval=self._retrieval_result,
            messages=list(self._trace),
            query_plan=copy.deepcopy(self.last_query_plan),
            entities=list(query_plan.entities),
            finish_draft=self._latest_finish_draft or None,
        )

    @staticmethod
    def _register_dynamic_anchors(
        evidence_items: list[EvidenceItem],
        query_plan: QueryPlan,
    ) -> None:
        """将模型精确定位到的完整源码登记为动态验证锚点。"""
        known_ids = {
            anchor.id
            for anchor in query_plan.anchors
        }
        for item in evidence_items:
            if item.kind != "source" or not item.complete or not item.node_id:
                continue
            payload = item.payload if isinstance(item.payload, dict) else {}
            node_type = str(payload.get("type", ""))
            file_path = str(item.file or payload.get("file", ""))
            if (
                item.node_id in known_ids
                or not node_type
                or not file_path
                or item.start_line is None
                or item.end_line is None
            ):
                continue
            anchor = Anchor(
                id=item.node_id,
                name=str(payload.get("name", item.node_id.split("::")[-1])),
                type=node_type,
                file=file_path,
                score=0.0,
                sources=["model_exploration"],
                selection_reason="模型通过精确图查询定位",
                covered_terms=[],
                candidate_sources=["model_exploration"],
            )
            anchor.validation = {
                "valid": True,
                "reason": "exact_contextualize_result",
            }
            anchor.evidence_ids = [item.evidence_id]
            anchor.prefetch_action = {
                "interface": "model_exploration",
                "adapter": "query_graph",
                "action": "contextualize",
                "name": item.node_id,
            }
            query_plan.anchors.append(anchor)
            known_ids.add(item.node_id)

    def _execute_expansions(
        self,
        requests: list[ExpansionRequest],
        query_plan: QueryPlan,
    ) -> list[EvidenceItem]:
        """执行受控关系扩展：遍历 → 写入账本 → 记录到 query_plan。"""
        if not self.graph_tool:
            return []
        observed_items: list[EvidenceItem] = []
        for req in requests:
            record = {
                "reason": req.reason,
                "action": req.action,
                "symbol": req.symbol,
                "target": req.target,
                "max_depth": req.max_depth,
                "min_confidence": req.min_confidence,
                "edge_types": list(req.edge_types),
                "status": "completed",
                "result_count": 0,
                "expanded_node_ids": [],
                "evidence_ids": [],
                "source_evidence_ids": [],
                "step": self.step_count,
            }
            try:
                items: list[EvidenceItem] = []
                expanded_node_ids = []
                if req.action == "shared_callers" and req.target:
                    endpoint_sets = []
                    subqueries = []
                    for symbol in (req.symbol, req.target):
                        kwargs = {
                            "symbol": symbol,
                            "max_depth": req.max_depth,
                            "min_confidence": req.min_confidence,
                        }
                        raw = self.graph_tool.execute_full(
                            "transitive_callers",
                            **kwargs,
                        )
                        tool_args = {
                            "action": "transitive_callers",
                            **kwargs,
                        }
                        items.extend(EvidenceItem.from_tool_result(
                            "query_graph",
                            tool_args,
                            raw,
                            step=self.step_count,
                        ))
                        endpoint_sets.append(
                            set(_collect_endpoint_node_ids(raw))
                        )
                        subqueries.append(kwargs)
                    shared_ids = (
                        set.intersection(*endpoint_sets)
                        if endpoint_sets
                        else set()
                    )
                    expanded_node_ids = sorted(shared_ids)
                    record["subqueries"] = subqueries
                    items.extend(
                        self._contextualize_expansion_nodes(
                            expanded_node_ids[:5],
                        )
                    )
                elif req.action == "contextualize":
                    tool_args = {
                        "action": "contextualize",
                        "name": req.symbol,
                        "limit": 1,
                    }
                    raw = self.graph_tool.execute_full(
                        "contextualize",
                        name=req.symbol,
                        limit=1,
                    )
                    items = EvidenceItem.from_tool_result(
                        "query_graph",
                        tool_args,
                        raw,
                        step=self.step_count,
                    )
                    expanded_node_ids = [req.symbol]
                elif req.action in (
                    "transitive_callers", "transitive_callees",
                    "call_paths",
                ):
                    kwargs: dict = {
                        "symbol": req.symbol,
                        "max_depth": req.max_depth,
                        "min_confidence": req.min_confidence,
                    }
                    if req.action == "call_paths" and req.target:
                        kwargs["source"] = req.symbol
                        kwargs["target"] = req.target
                    raw = self.graph_tool.execute_full(req.action, **kwargs)
                    tool_args = {
                        "action": req.action,
                        **kwargs,
                    }
                    items = EvidenceItem.from_tool_result(
                        "query_graph",
                        tool_args,
                        raw,
                        step=self.step_count,
                    )
                    if req.action == "call_paths":
                        expanded_node_ids = _collect_path_node_ids(raw)
                        items.extend(
                            self._contextualize_expansion_nodes(
                                expanded_node_ids[:5],
                            )
                        )
                    elif req.action in (
                        "transitive_callers",
                        "transitive_callees",
                    ):
                        expanded_node_ids = _collect_endpoint_node_ids(raw)
                        items.extend(
                            self._contextualize_expansion_nodes(
                                expanded_node_ids[:5],
                            )
                        )
                elif req.action == "class_hierarchy":
                    tool_args = {
                        "action": "class_hierarchy",
                        "class_name": req.symbol,
                    }
                    raw = self.graph_tool.execute_full(
                        "class_hierarchy",
                        class_name=req.symbol,
                    )
                    items = EvidenceItem.from_tool_result(
                        "query_graph",
                        tool_args,
                        raw,
                        step=self.step_count,
                    )
                else:
                    continue

                observed_items.extend(items)

                evidence_ids = []
                source_evidence_ids = []
                error_items = [
                    item for item in items
                    if item.kind == "error"
                ]
                for item in items:
                    result = self._ledger.add(item)
                    if result in ("added", "merged"):
                        evidence_ids.append(item.evidence_id)
                        if item.kind == "source":
                            source_evidence_ids.append(item.evidence_id)

                record["result_count"] = len(items)
                record["relation_result_count"] = sum(
                    1 for item in items
                    if item.kind == "relation"
                )
                record["expanded_node_ids"] = expanded_node_ids
                record["evidence_ids"] = evidence_ids
                record["source_evidence_ids"] = source_evidence_ids
                if error_items:
                    record["status"] = "failed"
                    record["error"] = "; ".join(
                        str(item.payload)
                        for item in error_items
                    )
                    query_plan.diagnostics.append(
                        f"扩展失败 [{req.action} {req.symbol}]: "
                        f"{record['error']}"
                    )
                self._expanded_relations.add(req.key)
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                query_plan.diagnostics.append(
                    f"扩展失败 [{req.action} {req.symbol}]: {exc}"
                )
            query_plan.relation_expansions.append(record)
        return observed_items

    def _contextualize_expansion_nodes(
        self,
        node_ids: list[str],
    ) -> list[EvidenceItem]:
        items = []
        for node_id in node_ids:
            context_args = {
                "action": "contextualize",
                "name": node_id,
                "limit": 1,
            }
            context_raw = self.graph_tool.execute_full(
                "contextualize",
                name=node_id,
                limit=1,
            )
            items.extend(EvidenceItem.from_tool_result(
                "query_graph",
                context_args,
                context_raw,
                step=self.step_count,
            ))
        return items

    def _compress_messages(self, keep_recent_turns: int = 2) -> None:
        """压缩消息历史：旧工具结果替换为证据 ID 摘要，保留最近 N 轮。

        移除被门控拒绝的 FINAL 消息，确保模型不会看到自己被驳回的结论。
        审计仍保存完整原始请求。
        """
        if len(self.messages) <= 2:
            return

        all_evidence_ids = [
            item.evidence_id for item in self._ledger.items()
        ]
        # 找到最后一条 assistant 消息（被门控拒绝的 FINAL）
        last_idx = len(self.messages) - 1
        while last_idx >= 2:
            role = self.messages[last_idx].get("role", "")
            if role == "assistant":
                break
            last_idx -= 1

        if last_idx < 2:
            return

        keep_end = last_idx
        keep_start = max(2, keep_end - keep_recent_turns * 3)
        while (
            keep_start > 2
            and self.messages[keep_start].get("role") == "tool"
        ):
            keep_start -= 1

        # 重建消息列表
        new_messages = list(self.messages[:2])  # system + user
        anchor_ids = [
            anchor.get("id", "")
            for anchor in self.last_query_plan.get("anchors", [])
            if anchor.get("id")
        ]
        expansion_count = len(
            self.last_query_plan.get("relation_expansions", [])
        )
        new_messages.append({
            "role": "user",
            "content": (
                f"[证据摘要] 前面探索已收集 {len(all_evidence_ids)} 条证据，"
                f"证据 ID: {', '.join(all_evidence_ids) or '无'}。"
                f"已验证锚点: {', '.join(anchor_ids) or '无'}。"
                f"已执行关系扩展: {expansion_count} 次。"
                "完整数据在证据账本中。"
                "你可以继续用工具探索更多证据，"
                "或输出 FINAL: <答案>。"
            ),
        })
        # 保留被拒 FINAL 之前的最近必要轮次
        if keep_start < keep_end:
            new_messages.extend(self.messages[keep_start:keep_end])

        self.messages = new_messages

    @staticmethod
    def _format_gate_failure_message(
        decision: GateDecision,
        draft: str,
        expanded_items: list[EvidenceItem],
    ) -> str:
        """构建门控未通过的通知消息，附加新证据提示。"""
        parts = [
            "[证据充分性检查未通过]",
            "",
            f"缺失项: {'; '.join(decision.missing_requirements)}",
        ]
        if decision.expansion_requests:
            parts.append(
                f"已自动扩展 {len(decision.expansion_requests)} 个关系，"
                "新增证据已写入账本，请重新评估证据是否充分。"
            )
        if expanded_items:
            parts.append("\n## 新增证据")
            parts.append(
                "\n---\n".join(
                    item.render(
                        DisplayLevel.COMPLETE
                        if item.kind == "source"
                        else (
                            DisplayLevel.PREVIEW
                            if item.kind == "relation"
                            else DisplayLevel.FOLD
                        )
                    )
                    for item in expanded_items
                )
            )
        else:
            parts.append("\n## 扩展结果\n限定范围内未获取到新增证据。")
        if draft:
            parts.append(
                f"\n你之前的草稿（参考，可修改）:\n{draft}"
            )
        parts.append(
            "\n## 请继续\n"
            "如果证据仍不足，请用标准格式继续探索：\n"
            "THOUGHT: <还需要什么证据>\n"
            "ACTION: {\"name\": \"<工具>\", \"arguments\": {...}}\n\n"
            "如果证据已足够，请输出: FINAL: <最终答案>"
        )
        return "\n".join(parts)

    @staticmethod
    def _check_convergence(raw_json: str) -> str | None:
        """检测 CLASS 节点 instantiated_by 中子类路径的汇聚模式"""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        results = data.get("results", [])
        if not results:
            return None

        r = results[0]
        if r.get("type") != "CLASS":
            return None

        inst = r.get("instantiated_by", [])
        via_classes: dict[str, list[str]] = {}
        for entry in inst:
            via = entry.get("via_class")
            if via:
                via_classes.setdefault(via, []).append(entry.get("name", "?"))

        if len(via_classes) < 3:
            return None

        total_callers = sum(len(v) for v in via_classes.values())
        subclass_names = list(via_classes.keys())
        sample = subclass_names[:4]
        more = f" ...共{len(subclass_names)}个" if len(subclass_names) > 4 else ""

        return (
            f"[继承收敛] {r.get('name','')}::__init__ 被 {len(subclass_names)} 个子类的"
            f" {total_callers} 个调用者汇聚（跳数=2: caller → 子类 → 当前类::__init__）\n"
            f"覆盖: {', '.join(sample)}{more}\n"
            f"汇聚到此的每个 via_class 都是一条独立证据，无需逐一验证各子类"
        )

    def _synthesize(
        self,
        task: str,
        candidates: list[dict],
        draft: str | None = None,
    ) -> RunResult:
        """将收集到的证据交给独立 prompt 合成最终答案"""
        if not self._ledger.has_synthesis_evidence:
            return RunResult(answer="", error="未收集到任何证据",
                             anchor_candidates=candidates,
                             retrieval=self._retrieval_result,
                             messages=list(self._trace),
                             query_plan=copy.deepcopy(self.last_query_plan),
                             entities=list(self.last_query_plan.get("entities", [])),
                             finish_draft=self._latest_finish_draft or None)

        # 从账本选择证据（预算内，逐项等级）
        selected = self._ledger.select_for_synthesis()
        evidence_text = self._ledger.render_selected_for_synthesis()
        selection_report = self._ledger.selection_report()
        if not evidence_text:
            return RunResult(answer="", error="所有证据超出合成预算",
                             anchor_candidates=candidates,
                             retrieval=self._retrieval_result,
                             messages=list(self._trace),
                             query_plan=copy.deepcopy(self.last_query_plan),
                             entities=list(self.last_query_plan.get("entities", [])),
                             finish_draft=self._latest_finish_draft or None)

        prompt = (
            ANSWER_PROMPT
            .replace("__QUESTION__", task)
            .replace("__EVIDENCE__", evidence_text)
            .replace("__DRAFT__", draft or "无")
            .replace(
                "__SEARCH_SCOPE__",
                self._render_search_scope(),
            )
        )

        messages = [{"role": "user", "content": prompt}]
        # 审计合成请求
        self._audit_model_request("answer_synthesis", messages)
        answer = self.answer_model.generate(messages)

        return RunResult(
            answer=answer,
            anchor_candidates=candidates,
            retrieval=self._retrieval_result,
            messages=list(self._trace),
            synthesis=SynthesisRecord(prompt=prompt, answer=answer),
            evidence=list(selected),
            evidence_selection_report=selection_report,
            model_requests=copy.deepcopy(self.last_model_requests),
            query_plan=copy.deepcopy(self.last_query_plan),
            entities=list(self.last_query_plan.get("entities", [])),
            finish_draft=self._latest_finish_draft or None,
        )

    def _render_search_scope(self) -> str:
        expansions = self.last_query_plan.get(
            "relation_expansions",
            [],
        )
        if not expansions:
            return "无"
        lines = []
        for record in expansions:
            target = (
                f" -> {record.get('target')}"
                if record.get("target")
                else ""
            )
            lines.append(
                "- "
                f"{record.get('action')} "
                f"{record.get('symbol')}{target}; "
                f"edges={record.get('edge_types', [])}; "
                f"max_depth={record.get('max_depth')}; "
                f"min_confidence={record.get('min_confidence')}; "
                f"status={record.get('status')}; "
                f"result_count={record.get('result_count')}"
            )
        return "\n".join(lines)

    def _execute_tool(self, name: str, args: dict) -> str:
        """执行工具并返回未裁剪结果，展示预算由证据账本负责。"""
        if name == "read_file":
            try:
                result = self.env.read_file(
                    path=args.get("path", ""),
                    start_line=args.get("start_line", 0),
                    end_line=args.get("end_line", 0),
                    context=args.get("context", 0),
                )
                return result
            except Exception as e:
                return f"[错误] read_file 失败: {e}"

        elif name == "list_dir":
            try:
                result = self.env.list_dir(path=args.get("path", ""))
                return result
            except Exception as e:
                return f"[错误] list_dir 失败: {e}"

        elif name == "query_graph":
            if not self.graph_tool:
                return '{"error":"图工具未初始化"}'
            if hasattr(self.graph_tool, "execute_full"):
                return self.graph_tool.execute_full(**args)
            return self.graph_tool.execute(**args)

        else:
            return f"[错误] 未知工具: {name}"

    def _track_exploration(self, tool_name: str, args: dict, obs_text: str) -> str:
        """探索状态记录：收集已探索符号并生成面包屑"""
        if tool_name != "query_graph":
            return obs_text

        # 尝试解析 JSON（可能已经是格式化文本）
        try:
            data = json.loads(obs_text)
        except json.JSONDecodeError:
            data = None

        if data is not None:
            path_entries = _parse_graph_path_ids(obs_text)
            for nid, _depth in path_entries:
                self._explored.add(nid)
            node_ids = _collect_node_ids(obs_text)
            for nid in node_ids:
                self._explored.add(nid)

        # 面包屑：无论 JSON 解析成败都追加
        action = args.get("action", "")
        if action == "contextualize":
            name = args.get("name", "")
            if name and name not in self._frontier:
                self._frontier.append(name)

        if self._frontier:
            recent = list(self._frontier)[-5:]
            obs_text += f"\n\n[已探索] {' → '.join(recent)}"

        return obs_text

    def _format_for_llm(self, obs_text: str, tool_name: str, args: dict) -> str:
        """将 JSON 工具结果转为紧凑自然语言"""
        if tool_name != "query_graph":
            return obs_text

        try:
            data = json.loads(obs_text)
        except json.JSONDecodeError:
            return obs_text

        if "error" in data and not isinstance(data.get("error"), dict):
            return f"[{data.get('action', '?')}] 错误: {data.get('error')}"

        action = args.get("action", "")
        return self._NARRATORS.get(action, lambda d, a: obs_text)(data, args)

    @staticmethod
    def _narrate_contextualize(data: dict, args: dict) -> str:
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

            # CLASS 节点的额外关系
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

    @staticmethod
    def _narrate_narrow_down(data, args) -> str:
        clues = data.get("clues_used", args.get("clues", []))
        results = data.get("results", [])
        lines = [f"[narrow_down] clues={clues} → 匹配{data.get('matched', 0)}/{data.get('total_candidates', 0)}个"]
        for r in results:
            lines.append(f"  {r.get('name','')} ({r.get('type','')}, {r.get('file','')}) relevance={r.get('relevance','?')}")
        return "\n".join(lines)

    @staticmethod
    def _narrate_calls(data, args) -> str:
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

    @staticmethod
    def _narrate_class_hierarchy(data, args) -> str:
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

    @staticmethod
    def _narrate_extract_clues(data, args) -> str:
        clues = data.get("clues", [])
        lines = [f"[extract_clues] 提取到{len(clues)}个可定位符号:"]
        for c in clues[:8]:
            lines.append(f"  {c.get('name','')} ({c.get('type','')}, {c.get('file','')})")
        if len(clues) > 8:
            lines.append(f"  ...共{len(clues)}个")
        return "\n".join(lines)

    @staticmethod
    def _narrate_module(data, args) -> str:
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
        "contextualize": _narrate_contextualize,
        "narrow_down": _narrate_narrow_down,
        "transitive_callers": _narrate_calls,
        "transitive_callees": _narrate_calls,
        "call_paths": _narrate_calls,
        "class_hierarchy": _narrate_class_hierarchy,
        "extract_clues": _narrate_extract_clues,
        "module_structure": _narrate_module,
        "module_tree": _narrate_module,
    }

