# -*- coding: utf-8 -*-
"""Agent 核心 -- 极简 ReAct 循环,支持 图查询 + 只读文件 双工具"""
import json
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from .model import Model
from .environment import Environment
from .retrieval import RetrievalResult
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

    @property
    def rounds(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @property
    def explorations(self) -> int:
        return sum(1 for m in self.messages if m.role == "tool" and not m.intercepted)

_FINAL_PATTERN = re.compile(r"FINAL[:\s]\s*(.+?)(?=\n(?:ACTION|THOUGHT):|\Z)", re.DOTALL)


def _extract_final(content: str) -> str | None:
    m = _FINAL_PATTERN.search(content)
    return m.group(1).strip() if m else None


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

## 请回答
"""


class Agent:
    """Graph-guided ReAct Agent -- 图查询 + 只读文件工具"""

    MAX_EXPLORATION_DEPTH = 3

    def __init__(self, model: Model, env: Environment, graph_tool=None, max_steps: int = 15, answer_model=None, on_step: Callable[["MsgRecord"], None] | None = None):
        self.model = model
        self.env = env
        self.graph_tool = graph_tool
        self.max_steps = max_steps
        self.messages: list[dict] = []
        self.step_count = 0
        self._explored: set[str] = set()
        self._frontier: deque[str] = deque()
        self._evidence: list[str] = []
        self.answer_model = answer_model or model
        self.on_step = on_step
        self._trace: list[MsgRecord] = []
        self._retrieval_result: RetrievalResult | None = None

    def _emit(self, record: MsgRecord) -> None:
        self._trace.append(record)
        if self.on_step:
            self.on_step(record)

    def run(self, task: str) -> RunResult:
        """执行任务,返回包含完整轨迹的 RunResult"""
        self._trace = []
        self._retrieval_result = None
        graph_status = ""
        if self.graph_tool:
            try:
                graph_status = self.graph_tool.ensure_built()
            except Exception as e:
                return RunResult(answer="", error=f"图构建失败: {e}")
            if not self.graph_tool.is_ready:
                return RunResult(answer="", error="图工具未就绪")

        # 预分析：确定性检索 + 可选 embedding 增强
        prelude = ""
        candidates = []
        if self.graph_tool:
            try:
                self._retrieval_result = (
                    self.graph_tool.retrieve_query_candidates(
                        task,
                        limit=8,
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
            except Exception as e:
                self._retrieval_result = RetrievalResult(
                    candidates=[],
                    stages_attempted=[],
                    stages_succeeded=[],
                    diagnostics=[f"候选检索失败: {e}"],
                    status="failed",
                )
            if candidates:
                items = []
                for c in candidates:
                    sources = ",".join(c.get("sources", []))
                    items.append(
                        f"  - {c['name']} ({c['type']}) {c['id']} "
                        f"[score={c['score']:.2f}; sources={sources}]"
                    )
                prelude = "\n\n[候选符号] 以下与问题语义最相关:\n" + "\n".join(items)

        system_content = SYSTEM_PROMPT.replace("__CWD__", self.env.config.cwd).replace("__GRAPH_STATUS__", graph_status)
        user_content = f"任务: {task}{prelude}"

        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        self._emit(MsgRecord(role="system", content=system_content, step=0))
        self._emit(MsgRecord(role="user", content=user_content, step=0))

        self.step_count = 0
        self._explored.clear()
        self._frontier.clear()
        self._evidence.clear()

        recent_actions = []
        contextualized_symbols: set[str] = set()

        for _ in range(self.max_steps):
            self.step_count += 1

            response = self.model.query(self.messages)

            thought = response.get("content", "").strip()
            raw_content = response.get("raw_content", "")
            finish_reason = response.get("finish_reason")

            # FINAL 文本标记（用户显式指令，最高优先级）
            final_text = _extract_final(raw_content)
            if final_text:
                self._emit(MsgRecord(role="assistant", content=raw_content, step=self.step_count))
                self.messages.append({"role": "assistant", "content": thought})
                # 有证据 → 合成；无证据 → 直接用 FINAL 文本
                if self._evidence:
                    return self._synthesize(task, candidates)
                return RunResult(
                    answer=final_text,
                    anchor_candidates=candidates,
                    retrieval=self._retrieval_result,
                    messages=list(self._trace),
                )

            # API 原生停牌信号（借鉴 OpenCode：finish_reason="stop" 即模型完成）
            if finish_reason == "stop" and not response["tool_calls"]:
                self._emit(MsgRecord(role="assistant", content=raw_content, step=self.step_count))
                self.messages.append({"role": "assistant", "content": thought})
                # 有证据 → 合成；无证据 → 直接用 thought
                if self._evidence:
                    return self._synthesize(task, candidates)
                if thought:
                    return RunResult(
                        answer=thought,
                        anchor_candidates=candidates,
                        retrieval=self._retrieval_result,
                        messages=list(self._trace),
                    )
                return RunResult(
                    answer="[模型未生成有效输出]",
                    anchor_candidates=candidates,
                    retrieval=self._retrieval_result,
                    messages=list(self._trace),
                )

            # 无工具调用（finish_reason 可能为 "length" / None）
            if not response["tool_calls"]:
                self._emit(MsgRecord(role="assistant", content=raw_content, step=self.step_count))
                self.messages.append({"role": "assistant", "content": thought})
                if self._evidence:
                    return self._synthesize(task, candidates)
                return RunResult(
                    answer=thought,
                    anchor_candidates=candidates,
                    retrieval=self._retrieval_result,
                    messages=list(self._trace),
                )

            # 有工具调用
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
                    obs_text = obs_raw
                    if tool_name == "query_graph" and args.get("action") == "contextualize" and ctx_name:
                        contextualized_symbols.add(ctx_name)

                # JSON 结果转自然语言
                raw_json = obs_text if not intercepted else ""
                obs_text = self._format_for_llm(obs_text, tool_name, args) if not intercepted else obs_text

                # 继承收敛分析
                if not intercepted and tool_name == "query_graph" and args.get("action") == "contextualize":
                    convergence = self._check_convergence(raw_json)
                    if convergence:
                        obs_text += "\n\n" + convergence

                # 追加已探索面包屑
                obs_text = self._track_exploration(tool_name, args, obs_text) if not intercepted else obs_text

                # 收集证据
                if obs_text and "[重复调用拦截]" not in obs_text:
                    self._evidence.append(obs_text)

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

        return RunResult(
            answer="[达到最大步数]",
            anchor_candidates=candidates,
            retrieval=self._retrieval_result,
            messages=list(self._trace),
        )

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

    def _synthesize(self, task: str, candidates: list[dict]) -> RunResult:
        """将收集到的证据交给独立 prompt 合成最终答案"""
        if not self._evidence:
            return RunResult(answer="", error="未收集到任何证据",
                             anchor_candidates=candidates,
                             retrieval=self._retrieval_result,
                             messages=list(self._trace))

        # 裁剪证据：每段最多 1500 字符，总共最多 6 段
        trimmed = []
        total = 0
        for e in self._evidence[-6:]:
            short = e[:1500]
            trimmed.append(short)
            total += len(short)
            if total > 8000:
                break

        evidence_text = "\n---\n".join(trimmed)
        prompt = ANSWER_PROMPT.replace("__QUESTION__", task).replace("__EVIDENCE__", evidence_text)

        messages = [{"role": "user", "content": prompt}]
        answer = self.answer_model.generate(messages)

        return RunResult(
            answer=answer,
            anchor_candidates=candidates,
            retrieval=self._retrieval_result,
            messages=list(self._trace),
            synthesis=SynthesisRecord(prompt=prompt, answer=answer),
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        """执行工具,返回 (obs_text, is_exit, submission)"""
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
            return self.graph_tool.execute(**args) if self.graph_tool else '{"error":"图工具未初始化"}'

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

