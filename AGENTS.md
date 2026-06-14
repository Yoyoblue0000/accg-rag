# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

基于 ACCG 代码图的 ReAct Agent，使用纯文本协议调用本地 Ollama 模型进行仓库级代码问答。依赖 [accg-core](https://github.com/Yoyoblue0000/accg-core) 提供图构建与查询能力。

## 架构

```
mini_agent/
  agent.py        — ReAct 循环、SYSTEM_PROMPT、ANSWER_PROMPT
  model.py        — LLM 接口：流式调用 + THOUGHT/ACTION 解析 + finish_reason 捕获
  graph_tool.py   — 图查询工具：9 种 action + EmbeddingRanker（磁盘缓存）
  environment.py  — 只读文件工具：read_file / list_dir
  sufficiency.py  — FinishAction 解析、确定性证据充分性门控、受控扩展计划
scripts/
  run_agent.py    — 单任务入口
  run_qa.py       — QA 批量评估入口（支持 --json、--id、即时写入）
  analyze_candidates.py — embedding 候选相关性分析
tests/
  test_agent_model.py      — model 层解析测试
  test_agent_graph_tool.py — 候选排序、锚点选择、预取、扩展、合成测试
  test_agent_evidence.py   — 证据账本与渲染测试
  test_sufficiency.py      — 门控通过/拒绝、实体过滤、否定语义测试
  test_multi_entity.py     — 多实体检索候选记录测试
  test_entity_extraction.py — 实体提取与去重测试
  test_query_plan.py       — 查询计划与锚点选择测试
  test_retrieval_metrics.py — 检索指标评估测试
```

## 常用命令

```bash
# 安装
uv venv && uv pip install -e .

# Agent 单任务
.venv/Scripts/python.exe scripts/run_agent.py "问题描述"

# QA 批量评估（服务器）
~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/requests_repo \
  --qa-path ~/program/test_repos/sweqa_requests.json \
  --model qwen2.5-coder:14b-instruct --limit 20

# 运行 Agent 测试
.venv/Scripts/python.exe -m pytest tests -v

# 代码质量
.venv/Scripts/python.exe -m ruff check mini_agent/ scripts/ --select E,F --ignore E501
.venv/Scripts/python.exe -m compileall -q mini_agent scripts
uv pip check
```

## 协议

纯文本 ReAct 协议，非 OpenAI function calling。LLM 输出：

```
THOUGHT: <推理>
ACTION: {"name": "<工具名>", "arguments": {<参数>}}
FINAL: <最终答案>
```

无依赖时可并行写多个 ACTION（最多 2 个）。model.py 解析层将图操作自动包装为 `query_graph(action=..., ...)`。

## 核心流程

```
Agent.run(task)
  │
  ├─ 1. 建图 + EmbeddingRanker.build_index（首次慢，磁盘缓存加速）
  ├─ 2. 级联检索候选 → 验证并预取 1-2 个主要锚点源码
  ├─ 3. ReAct 循环（max 15 步）
  │     └─ model.query() → THOUGHT/ACTION 解析 → 工具执行 → 证据收集
  ├─ 4. 完成请求
  │     ├─ FINAL 文本 → FinishAction(draft)
  │     └─ finish_reason="stop" / 无工具 → 仅表示本轮传输结束
  ├─ 5. SufficiencyGate 确定性检查
  │     ├─ 通过 → _synthesize()
  │     └─ 未通过 → 最多 2 次有界关系扩展并继续探索
  └─ 6. _synthesize()：问题 + 账本证据 + 最新草稿 + 搜索范围 → 最终答案
```

## 关键设计

- **传输结束与任务完成分离**：`finish_reason="stop"` 不绕过证据门控
- **FinishAction**：仅解析协议行 `FINAL: <draft>`，草稿独立保存且不作为事实证据
- **SufficiencyGate**：按单实体、比较、调用/数据流、继承、实例化和否定结论执行确定性规则
- **受控关系扩展**：最多 2 次、深度 ≤2、最低置信度 0.45，优先共享调用者并记录完整审计
- **模型历史压缩**：移除被拒 FINAL，旧工具结果替换为证据 ID 和查询计划摘要
- **EmbeddingRanker 磁盘缓存**：指纹校验，代码不变则直接从 `.accg/embeddings_*.json` 加载
- **on_step 回调**：每步即时输出，不等全部完成后一次性打印
- **两阶段合成**：Agent 收集证据 → `Model.generate()` 独立合成（ANSWER_PROMPT）
- **重复调用拦截**：最近 5 条 action 去重 + contextualize 符号去重

## 工具一览

| 工具 | 类型 | 说明 |
|---|---|---|
| contextualize | query_graph | 一次返回源码 + calls/called_by + inherits + instantiated_by |
| narrow_down | query_graph | 基于线索精简候选 |
| extract_clues | query_graph | 从源码提取可定位符号 |
| transitive_callers | query_graph | 传递调用者 |
| transitive_callees | query_graph | 传递被调用者 |
| call_paths | query_graph | 调用路径 |
| class_hierarchy | query_graph | 类继承层次 |
| module_tree | query_graph | 目录树 |
| module_structure | query_graph | 模块结构 |
| read_file | 文件 | 读文件（支持行号和上下文窗口） |
| list_dir | 文件 | 列出目录内容 |

## 服务器验证

Agent 依赖 Ollama 进行 LLM 推理和 embedding，本地 GPU 有限，所有 QA 批量测试必须在服务器上验证。

### 服务器环境

| 项目 | 详情 |
|---|---|
| 地址 | `ssh amd-jk6kg8k@10.67.8.138`（密钥 `~/.ssh/id_ed25519`） |
| 硬件 | AMD Ryzen AI MAX+ 395，128GB 统一内存，Radeon 8060S GPU（gfx1151） |
| Python | `~/.local/bin/uv run python` |
| Ollama API | `http://localhost:11434/v1` |
| GPU 驱动 | ROCm 7.2.3，amdgpu 6.16.13 |
| 可用显存 | ~111.5 GiB（统一内存架构） |

### GPU 模型兼容性

⚠️ **Qwen3 全系（含 MoE）在 gfx1151 上输出为空，不可用。**

| 模型 | 架构 | GPU | 速度 | 备注 |
|---|---|---|---|---|
| qwen2.5-coder:14b | Dense 14.8B | ✅ | 19 t/s | **当前默认** |
| qwen2.5:72b | Dense 72.7B | ✅ | 4.5 t/s | 太慢，不适合批量 |
| qwen2.5:14b | Dense 14.8B | ✅ | ~20 t/s | 通用版 |
| qwen3:30b | MoE 30.5B | ❌ | — | GPU 输出为空 |
| nomic-embed-text | 137M | ✅ | 15ms | Embedding，必须保留 |

### 同步+验证流程

```bash
# 正式部署：按 Git SHA 发布。确保本地工作区干净。
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && git fetch origin && git checkout <sha> && ~/.local/bin/uv pip install -e . && echo DEPLOY_OK"

# QA 全量（含门禁，必须加 --fail-on-error --prohibit-dirty --judge-model）
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && ~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/requests_repo \
  --qa-path ~/program/test_repos/sweqa_requests.json \
  --model qwen2.5-coder:14b-instruct --limit 20 \
  --judge-model qwen2.5:14b --judge-threshold 0.7 \
  --fail-on-error --prohibit-dirty"

# QA 单题 + verbose（调试用）
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && ~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/requests_repo \
  --qa-path ~/program/test_repos/sweqa_requests.json \
  --model qwen2.5-coder:14b-instruct --id 1 -v"

# 查看 GPU 状态
ssh amd-jk6kg8k@10.67.8.138 "rocm-smi"
# 查看 Ollama 推理日志
ssh amd-jk6kg8k@10.67.8.138 "journalctl -u ollama --no-pager --since '2 min ago'"
# 查看 QA 进度
ssh amd-jk6kg8k@10.67.8.138 "cat /tmp/qa_results.json | python3 -c 'import json; d=json.load(open(\"/tmp/qa_results.json\")); print(len(d), \"done\")'"
```

### 常见问题排查

| 症状 | 可能原因 | 解决 |
|---|---|---|
| QA 进程卡住 | 前次残留进程抢 GPU | `pkill -f run_qa.py` |
| 输出全为空 | 用了 qwen3 系列 | 换 `--model qwen2.5-coder:14b-instruct` |
| embedding 极慢 | 多模型抢 VRAM | `sudo systemctl restart ollama` |
| `finish_reason` 为 None | Ollama 版本太旧 | 升级到 0.24.0+ |

## 编码规范

- 注释和提交信息使用中文
- BFS 队列必须使用 `collections.deque`
- 集合成员检查使用 `set`，禁止对列表做 `in` 扫描
- 删除代码时同步清理相关的 import 和注释
