# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 ACCG 代码图的 ReAct Agent，使用纯文本协议调用本地 Ollama 模型进行仓库级代码问答。依赖 [accg-core](https://github.com/Yoyoblue0000/accg-core) 提供图构建与查询能力。

## 架构

```
mini_agent/
  agent.py        — ReAct 循环、SYSTEM_PROMPT、ANSWER_PROMPT、收敛分析
  model.py        — LLM 接口：流式调用 + THOUGHT/ACTION 解析 + finish_reason 捕获
  graph_tool.py   — 图查询工具：9 种 action + EmbeddingRanker（磁盘缓存）
  environment.py  — 只读文件工具：read_file / list_dir
scripts/
  run_agent.py    — 单任务入口
  run_qa.py       — QA 批量评估入口（支持 --json、--id、即时写入）
  analyze_candidates.py — embedding 候选相关性分析
tests/
  test_agent_model.py      — model 层解析测试（40 条）
  test_agent_graph_tool.py — 候选排序与预取测试
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
.venv/Scripts/python.exe -m pytest tests/test_agent_model.py -v
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
  ├─ 2. rank_candidates(task) → Top-8 语义候选注入 user 消息
  ├─ 3. ReAct 循环（max 15 步）
  │     └─ model.query() → THOUGHT/ACTION 解析 → 工具执行 → 证据收集
  ├─ 4. 终止判断
  │     ├─ FINAL 文本命中 → 合成
  │     ├─ finish_reason="stop" 无工具 → 合成
  │     └─ 无工具调用 → 合成（有证据时）
  └─ 5. _synthesize()：ANSWER_PROMPT + 证据 → 独立 LLM 调用 → 最终答案
```

## 关键设计

- **finish_reason 停牌**：从 Ollama 流式响应捕获 `finish_reason`，作为 API 原生停牌信号（借鉴 OpenCode）
- **FINAL 文本双保险**：正则 `FINAL[:\s]` 匹配冒号/换行/空格三种写法
- **EmbeddingRanker 磁盘缓存**：指纹校验，代码不变则直接从 `.accg/embeddings_*.pkl` 加载
- **on_step 回调**：每步即时输出，不等全部完成后一次性打印
- **两阶段合成**：Agent 收集证据 → `Model.generate()` 独立合成（ANSWER_PROMPT）
- **重复调用拦截**：最近 5 条 action 去重 + contextualize 符号去重
- **收敛分析**：≥3 个不同 via_class 时自动附加汇聚提示

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

## 服务器部署

```bash
# 同步代码
tar czf /tmp/accg_sync.tar.gz --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' .
scp -i ~/.ssh/id_ed25519 /tmp/accg_sync.tar.gz amd-jk6kg8k@10.67.8.138:~/program/accg-rag/
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && tar xzf accg_sync.tar.gz && rm accg_sync.tar.gz && ~/.local/bin/uv pip install -e ."

# 运行 QA
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && ~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/requests_repo \
  --qa-path ~/program/test_repos/sweqa_requests.json \
  --model qwen2.5-coder:14b-instruct --limit 20"
```

## 编码规范

- 注释和提交信息使用中文
- BFS 队列必须使用 `collections.deque`
- 集合成员检查使用 `set`，禁止对列表做 `in` 扫描
- 删除代码时同步清理相关的 import 和注释
